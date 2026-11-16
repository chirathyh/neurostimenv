#!/usr/bin/env bash
set -euo pipefail
repository=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repository"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
python_bin="${EEGREPR_PYTHON:-python}"
command=("$python_bin" -u -m experiments.eeg_representation.h1.v6.extended_cli "$@")
printf 'Invoked:'
printf ' %q' "${command[@]}"
printf '\n'
exec "${command[@]}"
