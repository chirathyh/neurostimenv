#!/usr/bin/env bash
# Run from the implementation repository; use an already activated parent venv.
set -euo pipefail
MODE=${1:-help}
BASE=${2:-/data/u7041472/eegrepr-results/h1-information-v2}
DEVICE=${3:-cpu}
PYTHON=${EEGREPR_PYTHON:-python}
CLI=("$PYTHON" -m experiments.eeg_representation.h1.cli)
CONFIGS=configs/eeg_representation/h1
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MAIN_PATH="$PWD"
if [[ ! -f "$CONFIGS/analytic_smoke.yaml" ]]; then
  echo 'Run this script from the H1 implementation repository root.' >&2; exit 2
fi
run_stage() {
  local stage=$1
  shift
  local log="$BASE/logs/$NAME-$stage-$(date -u +%Y%m%dT%H%M%S)-$$.log"
  mkdir -p "$BASE/logs"
  "${CLI[@]}" "$stage" --config "$CONFIG" --output-root "$OUT" --resume "$@" 2>&1 | tee "$log"
}
select_config() {
  NAME=$1
  CONFIG="$CONFIGS/$NAME.yaml"
  OUT="$BASE/$NAME"
}
finish() {
  run_stage h1-derive
  run_stage h1-benchmark --device "$DEVICE"
  run_stage h1-replication
  if [[ "$NAME" != simulator_smoke ]]; then run_stage h1-compression --device "$DEVICE"; fi
  run_stage h1-scale
  run_stage h1-summarize
  if [[ ! -e "$BASE/$NAME-evidence.zip" ]]; then
    run_stage h1-export --archive "$BASE/$NAME-evidence.zip"
  else
    echo "Existing immutable archive retained: $BASE/$NAME-evidence.zip"
  fi
}
case "$MODE" in
  help|-h|--help)
    cat <<'HELP'
Usage: bash scripts/eegrepr/run_h1.sh MODE [OUTPUT_BASE] [cpu|cuda]
Modes: doctor, controls, smoke, simulator-smoke, tests, gate,
       plan-bounded, bounded, plan-scale, scale,
       plan-expanded, expanded, plan-fixed-budget, fixed-budget,
       plan-compression, compression, plan-targeted-scale, targeted-scale,
       plan-noise, noise, plan-confirmation, confirmation
Defaults: /data/u7041472/eegrepr-results/h1-information-v2, cpu.
Activate the existing repository-parent Python environment first.
The gate runs 8 new full-network continuations; generation is serial.
All large modes require a compatible completed gate at BASE/timing_gate.
Read the separate plan output BEFORE explicitly choosing a large run mode.
No-argument invocation prints this help and starts no experiment.
Outputs are immutable: resume only with identical source/config, never edit files in place.
HELP
    ;;
  doctor)
    "${CLI[@]}" doctor
    "${CLI[@]}" h1-audit --pilot-root /data/u7041472/eegrepr-results/core-pilot --verify-raw
    ;;
  tests)
    EEGREPR_RUN_NEURON=1 "$PYTHON" -m unittest discover -s tests -p 'test_eegrepr*.py' -v
    "$PYTHON" -m unittest -v tests/test_online_stimulation.py
    ;;
  controls)
    select_config analytic_smoke; run_stage h1-controls
    ;;
  smoke|simulator-smoke)
    if [[ "$MODE" == smoke ]]; then select_config analytic_smoke; else select_config simulator_smoke; fi
    run_stage h1-plan
    run_stage h1-controls
    run_stage h1-generate
    run_stage h1-forks
    finish
    ;;
  gate)
    select_config timing_gate; run_stage h1-plan; run_stage h1-forks
    # This prints the measured gate and extrapolated bounded/expanded resource costs.
    select_config bounded_development; run_stage h1-plan --timing-root "$BASE/timing_gate"
    select_config expanded_reference; run_stage h1-plan --timing-root "$BASE/timing_gate"
    ;;
  plan-*|bounded|scale|expanded|fixed-budget|compression|targeted-scale|noise|confirmation)
    ACTION=${MODE#plan-}
    case "$ACTION" in
      bounded) select_config bounded_development;;
      scale) select_config scale_core;;
      expanded) select_config expanded_reference;;
      fixed-budget) select_config fixed_budget_extension;;
      compression) select_config compression;;
      targeted-scale) select_config scale_targeted;;
      noise) select_config input_noise_extension;;
      confirmation) select_config confirmation_plan;;
      *) echo "Unknown mode: $MODE" >&2; exit 2;;
    esac
    run_stage h1-plan --timing-root "$BASE/timing_gate"
    if [[ "$MODE" != plan-* ]]; then
      EXTRA=()
      if [[ "$ACTION" == expanded || "$ACTION" == fixed-budget || "$ACTION" == confirmation ]]; then EXTRA=(--execute-large); fi
      run_stage h1-generate --timing-root "$BASE/timing_gate" "${EXTRA[@]}"
      run_stage h1-forks --timing-root "$BASE/timing_gate" "${EXTRA[@]}"
      finish
    fi
    ;;
  *) echo "Unknown mode: $MODE; use help" >&2; exit 2;;
esac
