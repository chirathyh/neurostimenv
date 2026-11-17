#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
printf 'Executing:'
printf ' %q' python -m experiments.eeg_representation.h1.v6_pilot.cli "$@"
printf '\n'
exec python -m experiments.eeg_representation.h1.v6_pilot.cli "$@"
