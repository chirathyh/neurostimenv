#!/usr/bin/env bash
# Recovery only: no simulation mode is exposed by this wrapper.
set -euo pipefail
mode=${1:-help}
if [[ "$mode" == help || "$mode" == --help || "$mode" == -h ]]; then
    printf '%s\n' 'Usage: bash scripts/eegrepr/recover_v6_reporting_pilot.sh check|run [ROOT] [SOURCE_ZIP] [cuda|cpu]'
    printf '%s\n' 'Uses saved upstream pilot evidence. check verifies files; run resumes fitting/scoring/report/export with native execution prohibited.'
    exit 0
fi
if [[ "$mode" != check && "$mode" != run ]]; then
    printf 'Unknown recovery mode: %s\n' "$mode" >&2
    exit 2
fi
pilot_root=${2:-/data/u7041472/eegrepr-results/v6-reporting-pilot-v1}
source_zip=${3:-/data/u7041472/eegrepr-results/evaluation-v4/bounded-prospective-standard.zip}
device=${4:-cuda}
if [[ "$device" != cuda && "$device" != cpu ]]; then
    printf '%s\n' 'Device must match the frozen pilot: cuda or cpu; no automatic substitution.' >&2
    exit 2
fi
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$repo_root"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
args=(python -u -X faulthandler -m experiments.eeg_representation.h1.v6_pilot.recovery --root "$pilot_root" --source-archive "$source_zip" --device "$device")
if [[ "$mode" == check ]]; then args+=(--check-only); else args+=(--execute); fi
printf 'Executing:'
printf ' %q' "${args[@]}"
printf '\n'
if [[ "$mode" == check ]]; then exec "${args[@]}"; fi
# Match the normal pilot's fail-fast root checks before creating its log directory.
if [[ ! -f "$pilot_root/pilot_provenance.json" ]]; then
    printf 'Existing pilot provenance is required: %s\n' "$pilot_root" >&2
    exit 2
fi
mkdir -p "$pilot_root/logs"
log="$pilot_root/logs/recovery-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
set +e
"${args[@]}" 2>&1 | tee "$log"
statuses=("${PIPESTATUS[@]}")
set -e
printf 'Recovery exit code: %s\nLog: %s\n' "${statuses[0]}" "$log"
if (( statuses[0] != 0 )); then exit "${statuses[0]}"; fi
exit "${statuses[1]}"
