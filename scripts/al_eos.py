#!/usr/bin/env python3
"""Generate and optionally run fixed-volume fcc-Al SCFs for an EOS fit.

The default grid has twenty volumes from -8% to +8% around the relaxed
one-atom fcc primitive-cell volume.  Quantum ESPRESSO is launched only when
``--run`` is supplied; otherwise this script only writes the inputs and CSV
manifest.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PW = PROJECT_ROOT / ".qe" / "bin" / "pw.x"
DEFAULT_MPI = Path("/opt/homebrew/bin/mpirun")
DEFAULT_PSEUDO = PROJECT_ROOT / "Al.pbe-n-van.UPF"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "eos-runs"

# A2 convergence settings.
ECUTWFC_RY = 35.0
ECUTRHO_RY = 280.0
KMESH = 16
DEGAUSS_RY = 0.01

# From the completed vc-relax: one fcc primitive cell with one Al atom.
RELAXED_VOLUME_BOHR3 = 109.89829
BOHR_TO_ANGSTROM = 0.529177210903

FINAL_ENERGY_RE = re.compile(
    r"^\s*!\s+total\s+energy\s*=\s*"
    r"([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)\s+Ry",
    re.MULTILINE,
)
FINAL_PRESSURE_RE = re.compile(
    r"\bP=\s*([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)"
)


@dataclass(frozen=True)
class EOSPoint:
    """One fixed-volume SCF calculation for the equation of state."""

    scale: float
    volume_bohr3: float

    @property
    def alat_bohr(self) -> float:
        # V_primitive = a_conventional^3 / 4 for fcc Al.
        return (4.0 * self.volume_bohr3) ** (1.0 / 3.0)

    @property
    def tag(self) -> str:
        return f"scale-{self.scale:.4f}_volume-{self.volume_bohr3:09.4f}-bohr3"

    @property
    def prefix(self) -> str:
        return f"aluminum_eos_{self.tag.replace('.', 'p')}"


def make_input(point: EOSPoint, pseudo: Path, scratch_dir: Path) -> str:
    """Return a self-contained fixed-volume SCF input."""

    return f"""&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '{point.prefix}',
  tstress = .true.,
  tprnfor = .true.,
  pseudo_dir = '{pseudo.parent.resolve().as_posix()}/',
  outdir = '{scratch_dir.resolve().as_posix()}/'
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {point.alat_bohr:.10f},
  nat = 1,
  ntyp = 1,
  ecutwfc = {ECUTWFC_RY:.1f},
  ecutrho = {ECUTRHO_RY:.1f},
  occupations = 'smearing',
  smearing = 'methfessel-paxton',
  degauss = {DEGAUSS_RY:.2f}
/
&ELECTRONS
  diagonalization = 'david',
  mixing_mode = 'plain',
  conv_thr = 1.0d-8
/
ATOMIC_SPECIES
  Al  26.982  {pseudo.name}
ATOMIC_POSITIONS alat
  Al  0.00  0.00  0.00
K_POINTS automatic
  {KMESH} {KMESH} {KMESH}  0 0 0
"""


def parse_last(pattern: re.Pattern[str], contents: str, label: str) -> float:
    matches = list(pattern.finditer(contents))
    if not matches:
        raise ValueError(f"No {label} found in QE output")
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def successful_result(output_path: Path) -> tuple[float, float] | None:
    """Read a reusable normally completed SCF result, if it exists."""

    if not output_path.is_file():
        return None
    contents = output_path.read_text(encoding="utf-8", errors="replace")
    if (
        "JOB DONE." not in contents
        or "convergence has been achieved" not in contents
        or "convergence NOT achieved" in contents
    ):
        return None
    try:
        return (
            parse_last(FINAL_ENERGY_RE, contents, "final total energy"),
            parse_last(FINAL_PRESSURE_RE, contents, "pressure"),
        )
    except ValueError:
        return None


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    fields = (
        "scale",
        "volume_bohr3",
        "volume_ang3",
        "alat_bohr",
        "alat_ang",
        "ecutwfc_ry",
        "ecutrho_ry",
        "kmesh",
        "degauss_ry",
        "status",
        "energy_ry",
        "pressure_kbar",
        "pressure_gpa",
        "input_path",
        "output_path",
        "note",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--center-volume",
        type=float,
        default=RELAXED_VOLUME_BOHR3,
        help=f"Relaxed primitive-cell volume in bohr^3 (default: {RELAXED_VOLUME_BOHR3})",
    )
    parser.add_argument(
        "--span",
        type=float,
        default=0.08,
        help="Symmetric fractional volume range about the center (default: 0.08).",
    )
    parser.add_argument(
        "--npoints",
        type=int,
        default=20,
        help="Number of equally spaced volumes, at least 11 (default: 20).",
    )
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--pseudo", type=Path, default=DEFAULT_PSEUDO)
    parser.add_argument("--pw", type=Path, default=DEFAULT_PW)
    parser.add_argument("--mpi", type=Path, default=DEFAULT_MPI)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--run", action="store_true", help="Actually launch PWSCF.")
    parser.add_argument("--force", action="store_true", help="Rerun completed points.")
    args = parser.parse_args()

    if args.center_volume <= 0 or not math.isfinite(args.center_volume):
        parser.error("--center-volume must be positive and finite")
    if not 0 < args.span < 1 or not math.isfinite(args.span):
        parser.error("--span must be between 0 and 1")
    if args.npoints < 11:
        parser.error("--npoints must be at least 11")
    if args.nproc < 1:
        parser.error("--nproc must be at least 1")

    pseudo = args.pseudo.resolve()
    pw = args.pw.resolve()
    mpi = args.mpi.resolve()
    if not pseudo.is_file():
        parser.error(f"Pseudopotential not found: {pseudo}")
    if args.run and (not pw.is_file() or not mpi.is_file()):
        parser.error("QE executable or MPI launcher not found; check --pw and --mpi")

    scales = [
        1.0 - args.span + 2.0 * args.span * index / (args.npoints - 1)
        for index in range(args.npoints)
    ]
    points = [EOSPoint(scale, args.center_volume * scale) for scale in scales]
    runs_dir = args.runs_dir.resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"EOS grid: {len(points)} volumes from {scales[0]:.3f} to {scales[-1]:.3f} "
        f"of {args.center_volume:.5f} bohr^3; QE runs only with --run."
    )

    rows: list[dict[str, object]] = []
    any_failed = False
    for point in points:
        run_dir = runs_dir / point.tag
        input_path = run_dir / "Al.scf.inp"
        output_path = run_dir / "Al.scf.out"
        scratch_dir = run_dir / "qe-out"
        run_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(exist_ok=True)
        input_path.write_text(make_input(point, pseudo, scratch_dir), encoding="utf-8")

        result = None if args.force else successful_result(output_path)
        status, note = "generated", "QE not launched; add --run to execute this point."
        energy, pressure = None, None

        if result is not None:
            status, note = "cached", "Existing normally completed SCF output reused."
            energy, pressure = result
        elif args.run:
            command = [str(mpi), "-np", str(args.nproc), str(pw), "-in", input_path.name]
            with output_path.open("w", encoding="utf-8") as stream:
                completed = subprocess.run(
                    command,
                    cwd=run_dir,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            result = successful_result(output_path)
            if completed.returncode == 0 and result is not None:
                status, note = "success", ""
                energy, pressure = result
            else:
                status = "failed"
                note = f"QE exit code {completed.returncode}; inspect {output_path}"
                any_failed = True

        rows.append({
            "scale": point.scale,
            "volume_bohr3": point.volume_bohr3,
            "volume_ang3": point.volume_bohr3 * BOHR_TO_ANGSTROM**3,
            "alat_bohr": point.alat_bohr,
            "alat_ang": point.alat_bohr * BOHR_TO_ANGSTROM,
            "ecutwfc_ry": ECUTWFC_RY,
            "ecutrho_ry": ECUTRHO_RY,
            "kmesh": KMESH,
            "degauss_ry": DEGAUSS_RY,
            "status": status,
            "energy_ry": "" if energy is None else energy,
            "pressure_kbar": "" if pressure is None else pressure,
            "pressure_gpa": "" if pressure is None else pressure * 0.1,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "note": note,
        })

    result_path = runs_dir / "results.csv"
    write_results(result_path, rows)
    print(f"Wrote {result_path}")
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
