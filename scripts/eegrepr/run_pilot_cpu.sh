#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_DIR"
OUT="${1:?Usage: run_pilot_cpu.sh OUTPUT_ROOT [WORKERS]}"
WORKERS="${2:-1}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m experiments.eeg_representation.cli plan --config configs/eeg_representation/pilot.yaml --output-root "$OUT"
python -m experiments.eeg_representation.cli generate --config configs/eeg_representation/pilot.yaml --output-root "$OUT" --workers "$WORKERS" --resume
python -m experiments.eeg_representation.cli derive --config configs/eeg_representation/pilot.yaml --output-root "$OUT" --resume
# Forks are intentionally a separate command: 64/128 futures are a distinct campaign.
