#!/usr/bin/env bash
set -euo pipefail
# Explicit stages only. CUDA determinism must be configured before Python starts.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
SCRIPT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SCRIPT_ROOT"
MODE="${1:?Usage: run_h1_h4_v4.sh MODE ROOT [cpu|cuda] [CLI options]}"
RESULT_ROOT="${2:-/data/u7041472/eegrepr-results/evaluation-v4}"
DEVICE="${3:-cpu}"
shift "$(( $# >= 3 ? 3 : $# ))"
case "$MODE" in
 doctor|controls|audit-existing|qualify-existing|geometry-existing|plan|timing-gate|generate|derive|fit|score|retention|replication|report|export) ;;
 *) echo "Unsupported mode: $MODE (there is no all mode)" >&2; exit 2;;
esac
PROFILE=bounded
case "$MODE" in audit-existing|qualify-existing|geometry-existing) PROFILE=existing;; timing-gate) PROFILE=timing_gate;; esac
PYTHON_BIN="${EEGREPR_PYTHON:-python}"
CMD=("$PYTHON_BIN" -m experiments.eeg_representation.h1.v4.cli "$MODE" --root "$RESULT_ROOT" --device "$DEVICE" --config "configs/eeg_representation/v4/$PROFILE.yaml" "$@")
printf 'Invoking:'
printf ' %q' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
