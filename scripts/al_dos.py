#!/usr/bin/env python3
"""Generate A5 PBE NSCF and DOS inputs; this script never launches QE.

The NSCF calculation reuses the completed A5 SCF charge density at the PBE
equilibrium lattice constant.  It samples a uniform 24 x 24 x 24 mesh and
uses the tetrahedron method, appropriate for a metallic total DOS.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from al_bands import (
    ALAT_BOHR,
    ECUTRHO_RY,
    ECUTWFC_RY,
    OUTDIR,
    PREFIX,
    PROJECT,
    PSEUDO,
)


DEFAULT_KMESH = 24
DEFAULT_NBND = 12
DEFAULT_EMIN_EV = -5.0
DEFAULT_EMAX_EV = 35.0
DEFAULT_DELTA_E_EV = 0.02


def write_inputs(
    kmesh: int,
    nbnd: int,
    emin_ev: float,
    emax_ev: float,
    delta_e_ev: float,
) -> tuple[Path, Path, Path]:
    """Write the fixed-potential NSCF and ``dos.x`` input files."""
    if kmesh < 24:
        raise ValueError("The assignment requires an NSCF mesh of at least 24 x 24 x 24.")
    if nbnd < 8:
        raise ValueError("Use at least the eight bands retained for the A5 band structure.")
    if not emin_ev < emax_ev:
        raise ValueError("Require Emin < Emax.")
    if delta_e_ev <= 0.0:
        raise ValueError("DeltaE must be positive.")
    if not PSEUDO.is_file():
        raise FileNotFoundError(f"Required pseudopotential not found: {PSEUDO}")

    outdir = OUTDIR.resolve().as_posix() + "/"
    pseudo_dir = PSEUDO.parent.resolve().as_posix() + "/"
    fildos = (PROJECT / "qe-results" / "Al.dos.dat").resolve().as_posix()
    nscf_path = PROJECT / "Al.dos.nscf.inp"
    dos_path = PROJECT / "Al.dos.inp"

    # ``from_scratch`` begins the NSCF diagonalization afresh while reading the
    # charge density already present under this same prefix/outdir pair.
    nscf_input = f"""&CONTROL
  calculation = 'nscf',
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
  occupations = 'tetrahedra',
  nbnd = {nbnd}
/
&ELECTRONS
  diagonalization = 'david',
  diago_full_acc = .true.
/
ATOMIC_SPECIES
  Al  26.982  {PSEUDO.name}
ATOMIC_POSITIONS alat
  Al  0.00  0.00  0.00
K_POINTS automatic
  {kmesh} {kmesh} {kmesh}  0 0 0
"""

    # Do not specify degauss here: this makes dos.x retain tetrahedron
    # integration rather than silently falling back to Gaussian broadening.
    dos_input = f"""&DOS
  prefix = '{PREFIX}',
  outdir = '{outdir}',
  bz_sum = 'tetrahedra',
  Emin = {emin_ev:.2f},
  Emax = {emax_ev:.2f},
  DeltaE = {delta_e_ev:.3f},
  fildos = '{fildos}'
/
"""

    nscf_path.write_text(nscf_input)
    dos_path.write_text(dos_input)
    return nscf_path, dos_path, Path(fildos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmesh", type=int, default=DEFAULT_KMESH)
    parser.add_argument("--nbnd", type=int, default=DEFAULT_NBND)
    parser.add_argument("--emin", type=float, default=DEFAULT_EMIN_EV)
    parser.add_argument("--emax", type=float, default=DEFAULT_EMAX_EV)
    parser.add_argument("--delta-e", type=float, default=DEFAULT_DELTA_E_EV)
    arguments = parser.parse_args()

    nscf_path, dos_path, fildos_path = write_inputs(
        arguments.kmesh,
        arguments.nbnd,
        arguments.emin,
        arguments.emax,
        arguments.delta_e,
    )
    print(f"Wrote {nscf_path}")
    print(f"Wrote {dos_path}")
    print(f"Expected DOS table: {fildos_path}")
    print(
        f"NSCF mesh: {arguments.kmesh} x {arguments.kmesh} x {arguments.kmesh}; "
        f"nbnd = {arguments.nbnd}; tetrahedron DOS."
    )
    print(f"Reuse the completed SCF density with prefix '{PREFIX}' in {OUTDIR}.")


if __name__ == "__main__":
    main()
