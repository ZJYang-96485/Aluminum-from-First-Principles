#!/bin/zsh
# Run a Quantum ESPRESSO calculation from this project on this Mac.

set -euo pipefail

project_dir=${0:A:h}
qe_bin="$project_dir/.qe/bin"
pw="$qe_bin/pw.x"
mpi="/opt/homebrew/bin/mpirun"
ranks="${QE_NP:-4}"
case_name="${1:-scf}"

if [[ ! -x "$pw" ]]; then
  print -u2 "Quantum ESPRESSO is not installed at $pw"
  exit 1
fi

if [[ ! -x "$mpi" ]]; then
  print -u2 "Open MPI is not installed at $mpi"
  exit 1
fi

case "$case_name" in
  scf)
    input="Al.scf.inp"
    output="qe-results/Al.scf.out"
    ;;
  relax)
    input="Al.relax.inp"
    output="qe-results/Al.relax.out"
    ;;
  band|bands)
    input="Al.band.inp"
    output="qe-results/Al.band.out"
    ;;
  *)
    print -u2 "Usage: ${0:t} [scf|relax|band]"
    exit 2
    ;;
esac

cd "$project_dir"
mkdir -p qe-out qe-results
"$mpi" -np "$ranks" "$pw" -in "$input" > "$output" 2>&1
print "Finished $case_name calculation: $project_dir/$output"
