#!/usr/bin/env python3
"""Generate the A5 fcc-Al SCF and band-path inputs; never launches QE.

The path is the course-specified fcc path
Gamma-X-W-K-Gamma-L-U-W-L-K.  Points are allocated by Cartesian reciprocal
length, so consecutive points have approximately uniform physical spacing.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PSEUDO = PROJECT / "Al.pbe-n-van.UPF"
OUTDIR = PROJECT / "qe-out" / "a5-bands"
PREFIX = "aluminum_a5_bands"

# PBE Birch-Murnaghan minimum from A3: a0 = 4.02496285 Angstrom.
BOHR_TO_ANGSTROM = 0.529177210903
ALAT_BOHR = 4.02496285 / BOHR_TO_ANGSTROM

# Converged PBE settings selected in A2.
ECUTWFC_RY = 35.0
ECUTRHO_RY = 280.0  # 8 * ecutwfc for the ultrasoft Vanderbilt UPF.
KMESH = 16
DEGAUSS_RY = 0.01
NBND = 8
NPOINTS = 240

# Conventional Cartesian coordinates in units of 2*pi/a.  These are converted
# to QE's primitive reciprocal "crystal" coordinates below for ibrav = 2.
PATH = (
    ("Γ", (0.00, 0.00, 0.00)),
    ("X", (0.00, 1.00, 0.00)),
    ("W", (0.50, 1.00, 0.00)),
    ("K", (0.75, 0.75, 0.00)),
    ("Γ", (0.00, 0.00, 0.00)),
    ("L", (0.50, 0.50, 0.50)),
    ("U", (0.25, 1.00, 0.25)),
    ("W", (0.50, 1.00, 0.00)),
    ("L", (0.50, 0.50, 0.50)),
    ("K", (0.75, 0.75, 0.00)),
)


def sub(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def add(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return tuple(x + y for x, y in zip(a, b))  # type: ignore[return-value]


def scale(a: tuple[float, float, float], factor: float) -> tuple[float, float, float]:
    return tuple(factor * x for x in a)  # type: ignore[return-value]


def norm(a: tuple[float, float, float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def fcc_cartesian_to_crystal(k: tuple[float, float, float]) -> tuple[float, float, float]:
    """Convert 2*pi/a Cartesian fcc coordinates to QE ibrav=2 coefficients.

    QE's reciprocal primitive vectors are (-1,-1,1), (1,1,1), and
    (-1,1,-1), each in units of 2*pi/a.
    """

    x, y, z = k
    return ((z - x) / 2.0, (y + z) / 2.0, (y - x) / 2.0)


def allocate_intervals(lengths: list[float], npoints: int) -> list[int]:
    """Allocate exactly npoints - 1 intervals in proportion to length."""

    if npoints < len(lengths) + 1:
        raise ValueError(f"Need at least {len(lengths) + 1} points, got {npoints}.")

    total_intervals = npoints - 1
    total_length = sum(lengths)
    raw = [total_intervals * length / total_length for length in lengths]
    intervals = [max(1, math.floor(value)) for value in raw]
    missing = total_intervals - sum(intervals)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - math.floor(raw[i]), reverse=True)
    for i in order[:missing]:
        intervals[i] += 1
    return intervals


def make_kpath() -> tuple[list[tuple[float, float, float]], list[int], list[int]]:
    cartesian = [coordinate for _, coordinate in PATH]
    lengths = [norm(sub(stop, start)) for start, stop in zip(cartesian[:-1], cartesian[1:])]
    intervals = allocate_intervals(lengths, NPOINTS)

    points = [cartesian[0]]
    high_symmetry_indices = [0]
    for start, stop, nintervals in zip(cartesian[:-1], cartesian[1:], intervals):
        direction = sub(stop, start)
        for step in range(1, nintervals + 1):
            points.append(add(start, scale(direction, step / nintervals)))
        high_symmetry_indices.append(len(points) - 1)

    assert len(points) == NPOINTS
    return points, intervals, high_symmetry_indices


def write_inputs() -> None:
    if not PSEUDO.is_file():
        raise FileNotFoundError(f"Required pseudopotential not found: {PSEUDO}")

    kpoints_cart, intervals, special_indices = make_kpath()
    kpoints_crystal = [fcc_cartesian_to_crystal(point) for point in kpoints_cart]
    outdir = OUTDIR.resolve().as_posix() + "/"
    pseudo_dir = PSEUDO.parent.resolve().as_posix() + "/"

    scf_input = f"""&CONTROL
  calculation = 'scf',
  restart_mode = 'from_scratch',
  prefix = '{PREFIX}',
  pseudo_dir = '{pseudo_dir}',
  outdir = '{outdir}'
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {ALAT_BOHR:.10f},
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
  conv_thr = 1.0d-10,
  mixing_mode = 'plain'
/
ATOMIC_SPECIES
  Al  26.982  {PSEUDO.name}
ATOMIC_POSITIONS alat
  Al  0.00  0.00  0.00
K_POINTS automatic
  {KMESH} {KMESH} {KMESH}  0 0 0
"""

    kpoint_lines = "\n".join(f"  {kx: .10f}  {ky: .10f}  {kz: .10f}  1.0" for kx, ky, kz in kpoints_crystal)
    bands_input = f"""&CONTROL
  calculation = 'bands',
  verbosity = 'high',
  prefix = '{PREFIX}',
  pseudo_dir = '{pseudo_dir}',
  outdir = '{outdir}'
/
&SYSTEM
  ibrav = 2,
  celldm(1) = {ALAT_BOHR:.10f},
  nat = 1,
  ntyp = 1,
  ecutwfc = {ECUTWFC_RY:.1f},
  ecutrho = {ECUTRHO_RY:.1f},
  occupations = 'smearing',
  smearing = 'methfessel-paxton',
  degauss = {DEGAUSS_RY:.2f},
  nbnd = {NBND},
  nosym = .true.
/
&ELECTRONS
  diagonalization = 'david',
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Al  26.982  {PSEUDO.name}
ATOMIC_POSITIONS alat
  Al  0.00  0.00  0.00
K_POINTS crystal
{len(kpoints_crystal)}
{kpoint_lines}
"""

    (PROJECT / "Al.band.scf.inp").write_text(scf_input)
    (PROJECT / "Al.band.inp").write_text(bands_input)

    distance = 0.0
    rows = []
    special_lookup = {index: PATH[position][0] for position, index in enumerate(special_indices)}
    for index, (cartesian, crystal) in enumerate(zip(kpoints_cart, kpoints_crystal)):
        if index:
            distance += norm(sub(cartesian, kpoints_cart[index - 1]))
        rows.append(
            {
                "index": index,
                "segment": next(i for i, endpoint in enumerate(special_indices[1:]) if index <= endpoint),
                "label": special_lookup.get(index, ""),
                "distance_2pi_over_a": f"{distance:.10f}",
                "kx_cart_2pi_over_a": f"{cartesian[0]:.10f}",
                "ky_cart_2pi_over_a": f"{cartesian[1]:.10f}",
                "kz_cart_2pi_over_a": f"{cartesian[2]:.10f}",
                "k1_crystal": f"{crystal[0]:.10f}",
                "k2_crystal": f"{crystal[1]:.10f}",
                "k3_crystal": f"{crystal[2]:.10f}",
            }
        )
    csv_path = PROJECT / "a5-bands-kpath.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    segment_report = ", ".join(
        f"{PATH[i][0]}–{PATH[i + 1][0]}: {count}" for i, count in enumerate(intervals)
    )
    spacings = [
        norm(sub(kpoints_cart[i], kpoints_cart[i - 1])) for i in range(1, len(kpoints_cart))
    ]
    print(f"Wrote {PROJECT / 'Al.band.scf.inp'}")
    print(f"Wrote {PROJECT / 'Al.band.inp'}")
    print(f"Wrote {csv_path}")
    print(f"Generated {len(kpoints_cart)} fcc-path points ({segment_report}).")
    print(
        "Reciprocal spacing (2π/a): "
        f"min = {min(spacings):.6f}, max = {max(spacings):.6f}; "
        f"a = {ALAT_BOHR:.8f} bohr."
    )


if __name__ == "__main__":
    write_inputs()
