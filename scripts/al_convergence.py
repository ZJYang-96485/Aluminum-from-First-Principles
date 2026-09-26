#!/usr/bin/env python3
"""Generate and optionally run fixed-cell Al convergence studies with QE.

The script deliberately does not launch Quantum ESPRESSO unless ``--run`` is
provided. That makes it safe to inspect the generated inputs before spending
time on the 48-point cutoff/k-mesh grid or the subsequent smearing scans.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PW = PROJECT_ROOT / ".qe" / "bin" / "pw.x"
DEFAULT_MPI = Path("/opt/homebrew/bin/mpirun")
DEFAULT_PSEUDO = PROJECT_ROOT / "Al.pbe-n-van.UPF"
DEFAULT_RUNS_DIR = PROJECT_ROOT / "convergence-runs"

# Required fixed-cell study settings.
ALAT_BOHR = 9.0
ECUT_VALUES_RY = (15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0)
KMESH_VALUES = (4, 6, 8, 10, 12, 16)
BASE_DEGAUSS_RY = 0.05
SMEARING_VALUES_RY = (0.05, 0.02, 0.01)
ECUTRHO_RATIO = 8.0

FINAL_ENERGY_RE = re.compile(
    r"^\s*!\s+total\s+energy\s*=\s*"
    r"([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)\s+Ry",
    re.MULTILINE,
)


@dataclass(frozen=True)
class RunSpec:
    """One independent SCF calculation in a convergence study."""

    study: str
    ecutwfc_ry: float
    kmesh: int
    degauss_ry: float
    ecutrho_ratio: float

    @property
    def ecutrho_ry(self) -> float:
        return self.ecutwfc_ry * self.ecutrho_ratio

    @property
    def tag(self) -> str:
        return (
            f"ecut-{self.ecutwfc_ry:05.1f}-ry_"
            f"k-{self.kmesh:02d}x{self.kmesh:02d}x{self.kmesh:02d}_"
            f"degauss-{self.degauss_ry:.3f}-ry"
        )

    @property
    def prefix(self) -> str:
        return f"aluminum_{self.tag.replace('.', 'p')}"


def make_input(spec: RunSpec, pseudo: Path, scratch_dir: Path) -> str:
    """Render a self-contained QE input for a single fixed-cell SCF point."""

    pseudo_dir = pseudo.parent.resolve().as_posix()
    outdir = scratch_dir.resolve().as_posix()
    return f"""&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '{spec.prefix}',
  tstress = .true.,
  tprnfor = .true.,
  pseudo_dir = '{pseudo_dir}/',
  outdir = '{outdir}/'
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {ALAT_BOHR:.1f},
  nat = 1,
  ntyp = 1,
  ecutwfc = {spec.ecutwfc_ry:.6f},
  ecutrho = {spec.ecutrho_ry:.6f},
  occupations = 'smearing',
  smearing = 'methfessel-paxton',
  degauss = {spec.degauss_ry:.6f}
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
  {spec.kmesh} {spec.kmesh} {spec.kmesh}  0 0 0
"""


def parse_final_energy_text(contents: str) -> float:
    """Return the last QE total energy printed in ``contents``, in Ry."""

    matches = list(FINAL_ENERGY_RE.finditer(contents))
    if not matches:
        raise ValueError("No final '! total energy' line found")
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def read_qe_output(output: Path) -> str:
    return output.read_text(encoding="utf-8", errors="replace")


def is_converged_scf_output(contents: str) -> bool:
    """Accept only a normally completed, self-consistent QE calculation."""

    return (
        "JOB DONE." in contents
        and "convergence has been achieved" in contents
        and "convergence NOT achieved" not in contents
    )


def study_specs(args: argparse.Namespace) -> list[RunSpec]:
    if args.study == "grid":
        return [
            RunSpec(
                study="cutoff-kmesh",
                ecutwfc_ry=ecut,
                kmesh=kmesh,
                degauss_ry=BASE_DEGAUSS_RY,
                ecutrho_ratio=args.ecutrho_ratio,
            )
            for ecut in ECUT_VALUES_RY
            for kmesh in KMESH_VALUES
        ]

    if args.best_ecut is None:
        raise ValueError("The smearing study requires --best-ecut after reviewing the grid.")
    return [
        RunSpec(
            study="smearing",
            ecutwfc_ry=args.best_ecut,
            kmesh=kmesh,
            degauss_ry=degauss,
            ecutrho_ratio=args.ecutrho_ratio,
        )
        for degauss in SMEARING_VALUES_RY
        for kmesh in KMESH_VALUES
    ]


def paths_for(runs_dir: Path, spec: RunSpec) -> tuple[Path, Path, Path, Path]:
    run_dir = runs_dir / spec.study / spec.tag
    return (
        run_dir,
        run_dir / "Al.scf.inp",
        run_dir / "Al.scf.out",
        run_dir / "qe-out",
    )


def run_qe(
    *,
    spec: RunSpec,
    input_path: Path,
    output_path: Path,
    mpi: Path,
    pw: Path,
    nproc: int,
) -> tuple[str, float | None, str]:
    """Execute one point and return its state, energy, and explanatory note."""

    command = [str(mpi), "-np", str(nproc), str(pw), "-in", input_path.name]
    with output_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )

    contents = read_qe_output(output_path)
    try:
        energy = parse_final_energy_text(contents)
    except ValueError as error:
        return "failed", None, f"exit={completed.returncode}; {error}"

    if completed.returncode != 0:
        return "failed", None, f"exit={completed.returncode}"
    if not is_converged_scf_output(contents):
        return "failed", None, "QE did not report a normally converged SCF."
    return "success", energy, ""


def completed_result(output_path: Path) -> float | None:
    """Return an existing successful result, if the output is usable."""

    if not output_path.is_file():
        return None
    contents = read_qe_output(output_path)
    if not is_converged_scf_output(contents):
        return None
    try:
        return parse_final_energy_text(contents)
    except ValueError:
        return None


def manifest_row(
    spec: RunSpec,
    state: str,
    energy: float | None,
    note: str,
    input_path: Path,
    output_path: Path,
) -> dict[str, str | float | int]:
    return {
        "study": spec.study,
        "ecutwfc_ry": spec.ecutwfc_ry,
        "ecutrho_ry": spec.ecutrho_ry,
        "ecutrho_ratio": spec.ecutrho_ratio,
        "kmesh": spec.kmesh,
        "degauss_ry": spec.degauss_ry,
        "status": state,
        "energy_ry": "" if energy is None else energy,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "note": note,
    }


def write_manifest(path: Path, rows: Iterable[dict[str, str | float | int]]) -> None:
    fieldnames = (
        "study",
        "ecutwfc_ry",
        "ecutrho_ry",
        "ecutrho_ratio",
        "kmesh",
        "degauss_ry",
        "status",
        "energy_ry",
        "input_path",
        "output_path",
        "note",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def validate_run_environment(pw: Path, mpi: Path, pseudo: Path) -> None:
    missing: list[str] = []
    if not pw.is_file() or not (pw.stat().st_mode & 0o111):
        missing.append(f"QE executable: {pw}")
    if not mpi.is_file() or not (mpi.stat().st_mode & 0o111):
        missing.append(f"MPI launcher: {mpi}")
    if not pseudo.is_file():
        missing.append(f"Al pseudopotential: {pseudo}")
    if missing:
        raise FileNotFoundError("Missing required path(s):\n  " + "\n  ".join(missing))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate or run fixed-alat aluminum QE convergence calculations. "
            "QE runs only when --run is explicitly supplied."
        )
    )
    parser.add_argument("study", choices=("grid", "smearing"))
    parser.add_argument(
        "--best-ecut",
        type=float,
        help="Wavefunction cutoff (Ry) selected from the grid; required for smearing.",
    )
    parser.add_argument(
        "--ecutrho-ratio",
        type=float,
        default=ECUTRHO_RATIO,
        help=f"Fixed ecutrho / ecutwfc ratio for all points (default: {ECUTRHO_RATIO:g}).",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory for generated cases and results (default: {DEFAULT_RUNS_DIR}).",
    )
    parser.add_argument("--pseudo", type=Path, default=DEFAULT_PSEUDO)
    parser.add_argument("--pw", type=Path, default=DEFAULT_PW)
    parser.add_argument("--mpi", type=Path, default=DEFAULT_MPI)
    parser.add_argument("--nproc", type=int, default=2, help="Local MPI ranks (default: 2).")
    parser.add_argument("--run", action="store_true", help="Actually launch PWSCF for each point.")
    parser.add_argument("--force", action="store_true", help="Rerun points with completed output files.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan only; do not create inputs, outputs, or directories.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.nproc < 1:
        raise ValueError("--nproc must be at least 1")
    if not math.isfinite(args.ecutrho_ratio) or args.ecutrho_ratio < 8.0:
        raise ValueError("Use an ecutrho ratio of at least 8 for this ultrasoft pseudopotential.")
    if args.study == "smearing" and (
        args.best_ecut is None
        or not math.isfinite(args.best_ecut)
        or args.best_ecut <= 0.0
    ):
        raise ValueError("--best-ecut must be a positive finite cutoff for the smearing study.")
    if args.dry_run and args.run:
        raise ValueError("Choose either --dry-run or --run, not both.")

    specs = study_specs(args)
    runs_dir = args.runs_dir.resolve()
    pseudo = args.pseudo.resolve()
    pw = args.pw.resolve()
    mpi = args.mpi.resolve()

    print(
        f"Planned {len(specs)} {args.study} calculations: alat={ALAT_BOHR:.1f} bohr, "
        f"ecutrho/ecutwfc={args.ecutrho_ratio:g}."
    )
    if args.dry_run:
        for spec in specs:
            _, input_path, output_path, _ = paths_for(runs_dir, spec)
            print(f"  {spec.tag}\n    input:  {input_path}\n    output: {output_path}")
        return 0

    if args.run:
        validate_run_environment(pw, mpi, pseudo)
    elif not pseudo.is_file():
        raise FileNotFoundError(f"Al pseudopotential not found: {pseudo}")

    rows: list[dict[str, str | float | int]] = []
    failed = False
    for spec in specs:
        run_dir, input_path, output_path, scratch_dir = paths_for(runs_dir, spec)
        run_dir.mkdir(parents=True, exist_ok=True)
        scratch_dir.mkdir(exist_ok=True)
        input_path.write_text(make_input(spec, pseudo, scratch_dir), encoding="utf-8")

        state = "generated"
        energy: float | None = None
        note = "QE not launched; rerun with --run after reviewing inputs."
        if args.run:
            cached_energy = None if args.force else completed_result(output_path)
            if cached_energy is not None:
                state, energy, note = "cached", cached_energy, "Existing completed output retained."
            else:
                state, energy, note = run_qe(
                    spec=spec,
                    input_path=input_path,
                    output_path=output_path,
                    mpi=mpi,
                    pw=pw,
                    nproc=args.nproc,
                )
            failed = failed or state == "failed"

        rows.append(manifest_row(spec, state, energy, note, input_path, output_path))
        print(f"{state:9s} {spec.tag}" + ("" if energy is None else f"  {energy:.10f} Ry"))

    study_dir = runs_dir / specs[0].study
    manifest = study_dir / "results.csv"
    write_manifest(manifest, rows)
    print(f"Wrote manifest: {manifest}")
    if not args.run:
        print("No QE calculations were launched. Add --run when you are ready.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
