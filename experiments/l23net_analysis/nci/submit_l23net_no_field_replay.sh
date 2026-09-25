#!/bin/bash

set -euo pipefail

SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd "${SCRIPT_DIRECTORY}/../../.." && pwd)
WORKER="${SCRIPT_DIRECTORY}/run_l23net_no_field_replay.sh"
NCI_PROJECT=${NCI_PROJECT:-sj53}

if [[ ! -f "${WORKER}" ]]; then
    echo "Missing worker script: ${WORKER}." >&2
    exit 2
fi
cd "${REPOSITORY}"
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked files have uncommitted changes; submit a committed experiment." >&2
    exit 2
fi
EXPECTED_COMMIT=$(git rev-parse HEAD)

JOB_ID=$(qsub \
    -P "${NCI_PROJECT}" \
    -v "EXPECTED_COMMIT=${EXPECTED_COMMIT}" \
    -o "${SCRIPT_DIRECTORY}/out_l23net_no_field_replay.txt" \
    -e "${SCRIPT_DIRECTORY}/err_l23net_no_field_replay.txt" \
    "${WORKER}")

echo "Submitted deterministic L23Net no-field replay gate."
echo "Job: ${JOB_ID}"
echo "Commit: ${EXPECTED_COMMIT}"
echo "Resources: 624 CPUs, 200 GB, 00:40:00"
echo "Result directory will be:"
echo "/g/data/ny83/ch9972/NeuroStim/neurostimenv/results/l23net_no_field_replay_${JOB_ID}"
