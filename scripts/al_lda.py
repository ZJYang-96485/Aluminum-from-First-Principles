#!/usr/bin/env python3
"""Prepare and optionally run the LDA Al convergence and EOS workflow.

This is deliberately safe by default: it writes inputs and manifests, but starts
Quantum ESPRESSO only when ``--run`` is supplied.  All LDA data live in
directories separate from the completed PBE A2/A3 workflow.

Workflow
--------
1. Briefly recheck cutoff, k mesh, and smearing with ``convergence``.
2. Run a short LDA ``vc-relax`` from a sensible starting lattice constant.
3. Generate/run a 20-point, +/-8 percent fixed-volume EOS around that result.

The supplied Al.pz-vbc.UPF is PZ-LDA and norm-conserving, so this driver uses
ecutrho = 4 * ecutwfc (rather than the 8x ratio used for the PBE ultrasoft PP).
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PW = PROJECT_ROOT / ".qe" / "bin" / "pw.x"
DEFAULT_MPI = Path("/opt/homebrew/bin/mpirun")
DEFAULT_PSEUDO = PROJECT_ROOT / "Al.pz-vbc.UPF"
DEFAULT_CONV_DIR = PROJECT_ROOT / "lda-convergence-runs"
DEFAULT_RELAX_DIR = PROJECT_ROOT / "lda-a3-relax"
DEFAULT_EOS_DIR = PROJECT_ROOT / "lda-eos-runs"

# A near-equilibrium starting point for fcc Al, used only for the brief check.
CONVERGENCE_ALAT_BOHR = 7.60
STARTING_ALAT_BOHR = 7.60
DEFAULT_DEGAUSS_RY = 0.01
ECUTRHO_RATIO_NC = 4.0
CUTOFFS_RY = (40.0, 50.0, 60.0, 70.0, 80.0)
# The 24^3 endpoint checks that 20^3 is genuinely converged rather than merely
# the best point in a too-short metallic k-point scan.
KMESHES = (12, 16, 20, 24)
SMEARINGS_RY = (0.05, 0.02, 0.01)
BOHR_TO_ANGSTROM = 0.529177210903
RY_TO_MEV = 13605.693122994

FINAL_ENERGY_RE = re.compile(
    r"^\s*!\s+total\s+energy\s*=\s*"
    r"([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)\s+Ry",
    re.MULTILINE,
)
FINAL_PRESSURE_RE = re.compile(
    r"\bP=\s*([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)"
)
FINAL_VOLUME_RE = re.compile(
    r"new\s+unit-cell\s+volume\s*=\s*"
    r"([-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[EeDd][-+]?\d+)?)\s+a\.u\.\^3",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Settings:
    """Numerical settings shared by a self-consistent calculation."""

    ecutwfc_ry: float
    kmesh: int
    degauss_ry: float
    ecutrho_ratio: float = ECUTRHO_RATIO_NC

    @property
    def ecutrho_ry(self) -> float:
        return self.ecutrho_ratio * self.ecutwfc_ry


@dataclass(frozen=True)
class EOSPoint:
    scale: float
    volume_bohr3: float

    @property
    def alat_bohr(self) -> float:
        # fcc primitive cell with one atom: V = a_conventional^3 / 4.
        return (4.0 * self.volume_bohr3) ** (1.0 / 3.0)

    @property
    def tag(self) -> str:
        return f"scale-{self.scale:.4f}_volume-{self.volume_bohr3:09.4f}-bohr3"


def ensure_run_environment(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    pseudo = args.pseudo.resolve()
    pw = args.pw.resolve()
    mpi = args.mpi.resolve()
    if not pseudo.is_file():
        raise FileNotFoundError(f"Pseudopotential not found: {pseudo}")
    if args.run:
        missing = [
            str(path)
            for path in (pw, mpi)
            if not path.is_file() or not (path.stat().st_mode & 0o111)
        ]
        if missing:
            raise FileNotFoundError("Missing executable(s): " + ", ".join(missing))
    return pseudo, pw, mpi


def qe_input(
    *,
    calculation: str,
    prefix: str,
    alat_bohr: float,
    settings: Settings,
    pseudo: Path,
    scratch_dir: Path,
) -> str:
    """Render a self-contained fcc-Al QE input using the supplied PZ-LDA PP."""

    relaxation_blocks = "&IONS\n/\n&CELL\n/\n" if calculation == "vc-relax" else ""
    return f"""&CONTROL
  calculation = '{calculation}',
  restart_mode = 'from_scratch',
  prefix = '{prefix}',
  tstress = .true.,
  tprnfor = .true.,
  pseudo_dir = '{pseudo.parent.resolve().as_posix()}/',
  outdir = '{scratch_dir.resolve().as_posix()}/'
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {alat_bohr:.10f},
  nat = 1,
  ntyp = 1,
  input_dft = 'PZ',
  ecutwfc = {settings.ecutwfc_ry:.6f},
  ecutrho = {settings.ecutrho_ry:.6f},
  occupations = 'smearing',
  smearing = 'methfessel-paxton',
  degauss = {settings.degauss_ry:.6f}
/
&ELECTRONS
  diagonalization = 'david',
  mixing_mode = 'plain',
  conv_thr = 1.0d-8
/
{relaxation_blocks}ATOMIC_SPECIES
  Al  26.982  {pseudo.name}
ATOMIC_POSITIONS alat
  Al  0.00  0.00  0.00
K_POINTS automatic
  {settings.kmesh} {settings.kmesh} {settings.kmesh}  0 0 0
"""


def parse_last(pattern: re.Pattern[str], text: str, name: str) -> float:
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"No {name} was found in QE output")
    return float(matches[-1].group(1).replace("D", "E").replace("d", "e"))


def completed_scf(output_path: Path) -> tuple[float, float] | None:
    """Return final energy and pressure only for normally converged SCFs."""

    if not output_path.is_file():
        return None
    text = output_path.read_text(encoding="utf-8", errors="replace")
    if (
        "JOB DONE." not in text
        or "convergence has been achieved" not in text
        or "convergence NOT achieved" in text
    ):
        return None
    try:
        return (
            parse_last(FINAL_ENERGY_RE, text, "final total energy"),
            parse_last(FINAL_PRESSURE_RE, text, "final pressure"),
        )
    except ValueError:
        return None


def completed_relax_volume(output_path: Path) -> float | None:
    """Return the final primitive-cell volume from a completed vc-relax."""

    if not output_path.is_file():
        return None
    text = output_path.read_text(encoding="utf-8", errors="replace")
    if "JOB DONE." not in text:
        return None
    try:
        return parse_last(FINAL_VOLUME_RE, text, "final unit-cell volume")
    except ValueError:
        return None


def launch_or_reuse(
    *,
    input_path: Path,
    output_path: Path,
    pw: Path,
    mpi: Path,
    nproc: int,
    run: bool,
    force: bool,
    result_reader,
) -> tuple[str, tuple[float, ...] | None, str]:
    """Use a valid existing result, or run QE only after explicit authorization."""

    result = None if force else result_reader(output_path)
    if result is not None:
        return "cached", result if isinstance(result, tuple) else (result,), "Existing completed output reused."
    if not run:
        return "generated", None, "QE not launched; add --run to execute this input."

    command = [str(mpi), "-np", str(nproc), str(pw), "-in", input_path.name]
    with output_path.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command,
            cwd=input_path.parent,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    result = result_reader(output_path)
    if completed.returncode == 0 and result is not None:
        return "success", result if isinstance(result, tuple) else (result,), ""
    return "failed", None, f"QE exit code {completed.returncode}; inspect {output_path}"


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_settings(settings: Settings) -> None:
    if settings.ecutwfc_ry <= 0 or settings.kmesh < 1 or settings.degauss_ry <= 0:
        raise ValueError("ecutwfc, kmesh, and degauss must all be positive")
    if settings.ecutrho_ratio < 4:
        raise ValueError("Use ecutrho / ecutwfc >= 4 for this norm-conserving workflow")


def settings_from_args(args: argparse.Namespace) -> Settings:
    settings = Settings(
        ecutwfc_ry=args.ecutwfc,
        kmesh=args.kmesh,
        degauss_ry=args.degauss,
        ecutrho_ratio=args.ecutrho_ratio,
    )
    validate_settings(settings)
    return settings


def convergence_specs(stage: str, args: argparse.Namespace) -> list[Settings]:
    if stage == "cutoff":
        return [Settings(ecut, 16, DEFAULT_DEGAUSS_RY, args.ecutrho_ratio) for ecut in CUTOFFS_RY]
    if args.ecutwfc is None:
        raise ValueError(f"convergence {stage} requires --ecutwfc after the cutoff check")
    if stage == "kmesh":
        return [Settings(args.ecutwfc, kmesh, DEFAULT_DEGAUSS_RY, args.ecutrho_ratio) for kmesh in KMESHES]
    if args.kmesh is None:
        raise ValueError("convergence smearing requires --kmesh after the k-mesh check")
    return [Settings(args.ecutwfc, args.kmesh, degauss, args.ecutrho_ratio) for degauss in SMEARINGS_RY]


def convergence_command(args: argparse.Namespace) -> int:
    pseudo, pw, mpi = ensure_run_environment(args)
    specs = convergence_specs(args.stage, args)
    root = args.runs_dir.resolve() / args.stage
    root.mkdir(parents=True, exist_ok=True)
    fields = (
        "stage", "ecutwfc_ry", "ecutrho_ry", "ecutrho_ratio", "kmesh", "degauss_ry",
        "alat_bohr", "status", "energy_ry", "pressure_kbar", "input_path", "output_path", "note",
    )
    rows: list[dict[str, object]] = []
    failed = False

    for settings in specs:
        validate_settings(settings)
        tag = (
            f"ecut-{settings.ecutwfc_ry:05.1f}-ry_"
            f"k-{settings.kmesh:02d}x{settings.kmesh:02d}x{settings.kmesh:02d}_"
            f"degauss-{settings.degauss_ry:.3f}-ry"
        )
        run_dir = root / tag
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "Al.scf.inp"
        output_path = run_dir / "Al.scf.out"
        scratch_dir = run_dir / "qe-out"
        scratch_dir.mkdir(exist_ok=True)
        input_path.write_text(
            qe_input(
                calculation="scf",
                prefix=f"lda_{args.stage}_{tag.replace('.', 'p')}",
                alat_bohr=CONVERGENCE_ALAT_BOHR,
                settings=settings,
                pseudo=pseudo,
                scratch_dir=scratch_dir,
            ),
            encoding="utf-8",
        )
        status, result, note = launch_or_reuse(
            input_path=input_path, output_path=output_path, pw=pw, mpi=mpi,
            nproc=args.nproc, run=args.run, force=args.force, result_reader=completed_scf,
        )
        failed = failed or status == "failed"
        energy, pressure = (result if result is not None else (None, None))
        rows.append({
            "stage": args.stage,
            "ecutwfc_ry": settings.ecutwfc_ry,
            "ecutrho_ry": settings.ecutrho_ry,
            "ecutrho_ratio": settings.ecutrho_ratio,
            "kmesh": settings.kmesh,
            "degauss_ry": settings.degauss_ry,
            "alat_bohr": CONVERGENCE_ALAT_BOHR,
            "status": status,
            "energy_ry": "" if energy is None else energy,
            "pressure_kbar": "" if pressure is None else pressure,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "note": note,
        })
    manifest = root / "results.csv"
    write_csv(manifest, rows, fields)
    print(f"Wrote {manifest}. QE was {'launched' if args.run else 'not launched'}.")
    return 1 if failed else 0


def summarize_command(args: argparse.Namespace) -> int:
    path = args.runs_dir.resolve() / args.stage / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No convergence manifest: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if row["status"] in {"success", "cached"}]
    if not rows:
        raise RuntimeError("No completed convergence results yet.")
    for row in rows:
        row["energy_ry"] = float(row["energy_ry"])
    # For cutoff and kmesh, the largest value is the local reference.  For
    # smearing, use the smallest width, which is closest to the T -> 0 limit.
    if args.stage == "cutoff":
        reference = max(rows, key=lambda r: float(r["ecutwfc_ry"]))
        order = lambda r: float(r["ecutwfc_ry"])
        label = "ecutwfc (Ry)"
        value = lambda r: float(r["ecutwfc_ry"])
    elif args.stage == "kmesh":
        reference = max(rows, key=lambda r: int(r["kmesh"]))
        order = lambda r: int(r["kmesh"])
        label = "k mesh"
        value = lambda r: int(r["kmesh"])
    else:
        reference = min(rows, key=lambda r: float(r["degauss_ry"]))
        order = lambda r: float(r["degauss_ry"])
        label = "degauss (Ry)"
        value = lambda r: float(r["degauss_ry"])
    print(f"Reference: {label} = {value(reference)}; E = {reference['energy_ry']:.12f} Ry")
    print(f"{'parameter':>16s}  {'E (Ry)':>18s}  {'|Delta E| (meV/atom)':>23s}")
    for row in sorted(rows, key=order):
        error_mev = abs(row["energy_ry"] - reference["energy_ry"]) * RY_TO_MEV
        print(f"{str(value(row)):>16s}  {row['energy_ry']:18.12f}  {error_mev:23.6f}")
    print("Use the smallest setting that remains within the stated 2 meV/atom target;"
          " inspect the full trend rather than relying on one accidental cancellation.")
    return 0


def relax_command(args: argparse.Namespace) -> int:
    pseudo, pw, mpi = ensure_run_environment(args)
    settings = settings_from_args(args)
    root = args.relax_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "Al.relax.inp"
    output_path = root / "Al.relax.out"
    scratch_dir = root / "qe-out"
    scratch_dir.mkdir(exist_ok=True)
    input_path.write_text(
        qe_input(
            calculation="vc-relax", prefix="aluminum_lda_relax",
            alat_bohr=args.starting_alat, settings=settings, pseudo=pseudo, scratch_dir=scratch_dir,
        ),
        encoding="utf-8",
    )
    status, result, note = launch_or_reuse(
        input_path=input_path, output_path=output_path, pw=pw, mpi=mpi,
        nproc=args.nproc, run=args.run, force=args.force, result_reader=completed_relax_volume,
    )
    if result is None:
        print(f"LDA relaxation status: {status}. {note}")
        return 1 if status == "failed" else 0
    volume = result[0]
    alat = (4.0 * volume) ** (1.0 / 3.0)
    print(f"LDA relaxed primitive volume: {volume:.8f} bohr^3")
    print(f"LDA relaxed conventional fcc alat: {alat:.8f} bohr = {alat * BOHR_TO_ANGSTROM:.8f} Angstrom")
    print("The EOS command will use this volume automatically with --center-from-relax.")
    return 0


def eos_command(args: argparse.Namespace) -> int:
    pseudo, pw, mpi = ensure_run_environment(args)
    settings = settings_from_args(args)
    if args.npoints < 11:
        raise ValueError("Use at least 11 EOS volumes")
    if not 0 < args.span < 1:
        raise ValueError("--span must be between 0 and 1")
    if args.center_from_relax:
        center = completed_relax_volume(args.relax_dir.resolve() / "Al.relax.out")
        if center is None:
            raise RuntimeError("No completed LDA relaxation found; run `relax --run` first.")
    elif args.center_volume is not None:
        center = args.center_volume
    else:
        raise ValueError("Provide --center-from-relax or --center-volume")
    if center <= 0:
        raise ValueError("Center volume must be positive")

    scales = [1.0 - args.span + 2.0 * args.span * i / (args.npoints - 1) for i in range(args.npoints)]
    points = [EOSPoint(scale, center * scale) for scale in scales]
    root = args.eos_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    fields = (
        "scale", "volume_bohr3", "volume_ang3", "alat_bohr", "alat_ang",
        "ecutwfc_ry", "ecutrho_ry", "kmesh", "degauss_ry", "status", "energy_ry",
        "pressure_kbar", "pressure_gpa", "input_path", "output_path", "note",
    )
    rows: list[dict[str, object]] = []
    failed = False
    for point in points:
        run_dir = root / point.tag
        run_dir.mkdir(parents=True, exist_ok=True)
        input_path = run_dir / "Al.scf.inp"
        output_path = run_dir / "Al.scf.out"
        scratch_dir = run_dir / "qe-out"
        scratch_dir.mkdir(exist_ok=True)
        input_path.write_text(
            qe_input(
                calculation="scf", prefix=f"aluminum_lda_eos_{point.tag.replace('.', 'p')}",
                alat_bohr=point.alat_bohr, settings=settings, pseudo=pseudo, scratch_dir=scratch_dir,
            ),
            encoding="utf-8",
        )
        status, result, note = launch_or_reuse(
            input_path=input_path, output_path=output_path, pw=pw, mpi=mpi,
            nproc=args.nproc, run=args.run, force=args.force, result_reader=completed_scf,
        )
        failed = failed or status == "failed"
        energy, pressure = (result if result is not None else (None, None))
        rows.append({
            "scale": point.scale,
            "volume_bohr3": point.volume_bohr3,
            "volume_ang3": point.volume_bohr3 * BOHR_TO_ANGSTROM**3,
            "alat_bohr": point.alat_bohr,
            "alat_ang": point.alat_bohr * BOHR_TO_ANGSTROM,
            "ecutwfc_ry": settings.ecutwfc_ry,
            "ecutrho_ry": settings.ecutrho_ry,
            "kmesh": settings.kmesh,
            "degauss_ry": settings.degauss_ry,
            "status": status,
            "energy_ry": "" if energy is None else energy,
            "pressure_kbar": "" if pressure is None else pressure,
            "pressure_gpa": "" if pressure is None else pressure * 0.1,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "note": note,
        })
    manifest = root / "results.csv"
    write_csv(manifest, rows, fields)
    print(f"Wrote {manifest} for {len(points)} LDA EOS points about {center:.8f} bohr^3.")
    return 1 if failed else 0


def shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pseudo", type=Path, default=DEFAULT_PSEUDO)
    parser.add_argument("--pw", type=Path, default=DEFAULT_PW)
    parser.add_argument("--mpi", type=Path, default=DEFAULT_MPI)
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--run", action="store_true", help="Actually launch Quantum ESPRESSO.")
    parser.add_argument("--force", action="store_true", help="Rerun completed cases.")


def numerical_arguments(parser: argparse.ArgumentParser, *, required: bool) -> None:
    parser.add_argument("--ecutwfc", type=float, required=required)
    parser.add_argument("--kmesh", type=int, required=required)
    parser.add_argument("--degauss", type=float, default=DEFAULT_DEGAUSS_RY)
    parser.add_argument("--ecutrho-ratio", type=float, default=ECUTRHO_RATIO_NC)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    convergence = commands.add_parser("convergence", help="Generate/run a brief LDA numerical recheck.")
    convergence.add_argument("stage", choices=("cutoff", "kmesh", "smearing"))
    convergence.add_argument("--runs-dir", type=Path, default=DEFAULT_CONV_DIR)
    convergence.add_argument("--ecutwfc", type=float)
    convergence.add_argument("--kmesh", type=int)
    convergence.add_argument("--ecutrho-ratio", type=float, default=ECUTRHO_RATIO_NC)
    shared_arguments(convergence)

    summary = commands.add_parser("summarize", help="Print convergence errors relative to the local reference.")
    summary.add_argument("stage", choices=("cutoff", "kmesh", "smearing"))
    summary.add_argument("--runs-dir", type=Path, default=DEFAULT_CONV_DIR)

    relax = commands.add_parser("relax", help="Generate/run an LDA vc-relax near the expected minimum.")
    relax.add_argument("--relax-dir", type=Path, default=DEFAULT_RELAX_DIR)
    relax.add_argument("--starting-alat", type=float, default=STARTING_ALAT_BOHR)
    numerical_arguments(relax, required=True)
    shared_arguments(relax)

    eos = commands.add_parser("eos", help="Generate/run the 20-point LDA fixed-volume EOS.")
    eos.add_argument("--eos-dir", type=Path, default=DEFAULT_EOS_DIR)
    eos.add_argument("--relax-dir", type=Path, default=DEFAULT_RELAX_DIR)
    eos.add_argument("--center-from-relax", action="store_true")
    eos.add_argument("--center-volume", type=float)
    eos.add_argument("--span", type=float, default=0.08)
    eos.add_argument("--npoints", type=int, default=20)
    numerical_arguments(eos, required=True)
    shared_arguments(eos)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "nproc", 1) < 1:
        parser.error("--nproc must be at least 1")
    try:
        if args.command == "convergence":
            return convergence_command(args)
        if args.command == "summarize":
            return summarize_command(args)
        if args.command == "relax":
            return relax_command(args)
        return eos_command(args)
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
