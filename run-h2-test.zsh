#!/bin/zsh
# Run the Module 1 H2 SCF validation case with the project-local QE build.

set -euo pipefail

project_dir=${0:A:h}
test_dir="$project_dir/h2-test"
pw="$project_dir/.qe/bin/pw.x"
mpi="/opt/homebrew/bin/mpirun"
ranks="${QE_NP:-2}"

if [[ ! -x "$pw" || ! -x "$mpi" ]]; then
  print -u2 "QE or Open MPI is not installed at the expected project-local path."
  exit 1
fi

cd "$test_dir"
mkdir -p qe-out
"$mpi" -np "$ranks" "$pw" -in H2.scf.inp > H2.scf.out 2>&1
print "Finished H2 validation: $test_dir/H2.scf.out"
