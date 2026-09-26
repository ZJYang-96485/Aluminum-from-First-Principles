# Quantum ESPRESSO: local run path

This project has a native Apple-silicon Quantum ESPRESSO 7.6 installation:

```zsh
./.qe/bin/pw.x
```

Use the project wrapper rather than typing machine-specific MPI paths:

```zsh
QE_NP=2 ./run-qe.zsh scf
QE_NP=2 ./run-qe.zsh relax
QE_NP=2 ./run-qe.zsh band
```

`QE_NP` chooses the number of local MPI ranks; the wrapper defaults to 4. It
uses the matching Homebrew Open MPI installation and writes output to
`qe-results/` and temporary QE data to `qe-out/`.

## Input layout

All supplied inputs now use `pseudo_dir = './'`, which resolves the bundled
`Al.pbe-n-van.UPF`, and `outdir = './qe-out/'`. Run the wrapper from any
directory; it changes into the project directory before starting QE.

## Standard calculation order

1. Converge `ecutwfc`, the k-point mesh, and (because Al is a metal) the
   smearing width together. Retain separate output files for each point.
2. Run `relax` or `vc-relax` with the selected settings. Confirm forces tend
   to zero and record the final coordinates/cell.
3. Put that relaxed structure in a final SCF input and run `scf`.
4. Run `band` only after the matching SCF data are present in `qe-out/`.
5. Extract the *last* `! total energy` line and inspect `JOB DONE.`; do not
   treat an intermediate SCF energy as the result.

The bundled PBE Al pseudopotential is ultrasoft. The supplied values
(`ecutwfc = 15 Ry`, `4x4x4` mesh, and `degauss = 0.05 Ry`) are a runnable
baseline, not a converged result. In particular, include and converge
`ecutrho` (an ultrasoft potential commonly needs roughly 8-12 times
`ecutwfc`) before reporting physical results.

## Fixed-cell convergence driver

`scripts/al_convergence.py` generates a 48-point fixed-cell convergence grid
using `alat = 9.0 bohr`, eight cutoffs (15-50 Ry), six cubic meshes
(4x4x4 through 16x16x16), `degauss = 0.05 Ry`, and `ecutrho = 8 * ecutwfc`.
It never runs QE unless `--run` is explicit. After selecting a cutoff from
that grid, use its `smearing` mode to make the required 0.05, 0.02, and 0.01
Ry k-point scans. Generated cases and results live in `convergence-runs/`,
which is ignored by Git.

## Installation validation: H2

The walkthrough-compatible H2 test is separate from the Al project:

```zsh
QE_NP=2 ./run-h2-test.zsh
```

It uses the same `H_US.van` pseudopotential, 10 bohr cubic cell, 20 Ry
cutoff, and 1.4 bohr bond length as the walkthrough. On this installation it
gave `-2.31089939 Ry`, within `2.94e-6 Ry` of the walkthrough reference
`-2.31089645 Ry`. Its scratch and output files are intentionally ignored by
Git.

## Verified baseline launch

`QE_NP=2 ./run-qe.zsh scf` completed successfully on this machine. Its output
is `qe-results/Al.scf.out`; the final baseline free energy was
`-12.40765029 Ry` after five SCF iterations. That number only verifies the
installation and input path, not convergence.
