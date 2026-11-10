#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
OUT="${1:?Usage: run_training_gpu.sh OUTPUT_ROOT [CONFIG] [PROTOCOL]}"
CONFIG="${2:-configs/eeg_representation/pilot.yaml}"
PROTOCOL="${3:-leave-task-label-out}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.eeg_representation.cli fit-suite --config "$CONFIG" --output-root "$OUT" --protocol "$PROTOCOL" --device cuda --resume
python -m experiments.eeg_representation.cli evaluate --config "$CONFIG" --output-root "$OUT" --protocol "$PROTOCOL" --device cuda --resume
python -m experiments.eeg_representation.cli summarize --config "$CONFIG" --output-root "$OUT" --resume
