#!/bin/bash
# Submit exactly two paired structures, in project priority order, then analysis.
set -euo pipefail
SCRIPT_DIRECTORY=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPOSITORY=$(cd "${SCRIPT_DIRECTORY}/../../.." && pwd)
FIRST_PROJECT=${1:-sj53}
SECOND_PROJECT=${2:-fa32}
for PROJECT_CODE in "${FIRST_PROJECT}" "${SECOND_PROJECT}"; do
    case "${PROJECT_CODE}" in sj53|fa32|ny83) ;; *) echo "Unsupported project ${PROJECT_CODE}" >&2; exit 2 ;; esac
done
cd "${REPOSITORY}"
git diff --quiet && git diff --cached --quiet || { echo 'Commit tracked changes before submission.' >&2; exit 2; }
EXPECTED_COMMIT=$(git rev-parse HEAD)
G1A_REPORT=${G1A_REPORT:-${REPOSITORY}/results/l23net_no_field_replay_179858770.gadi-pbs/l23net_no_field_replay_comparison.json}
G1A_SHA=$(python3 "${SCRIPT_DIRECTORY}/g1b_preflight.py" prerequisite "${G1A_REPORT}")
command -v qsub >/dev/null || { echo 'qsub is unavailable; run on Gadi login node.' >&2; exit 2; }
SUITE_DIRECTORY=$(mktemp -d "${REPOSITORY}/results/l23net_g1b_$(date +%Y%m%d_%H%M%S)_XXXXXX")
mkdir -p "${SUITE_DIRECTORY}/seed_7101" "${SUITE_DIRECTORY}/seed_7102"
cp "${G1A_REPORT}" "${SUITE_DIRECTORY}/g1a_prerequisite.json"
python3 - "${SUITE_DIRECTORY}" "${EXPECTED_COMMIT}" "${G1A_SHA}" "${FIRST_PROJECT}" "${SECOND_PROJECT}" <<'PY'
import json
from pathlib import Path
import sys
directory, commit, g1a, first, second = sys.argv[1:]
manifest = {'commit': commit, 'g1a_sha256': g1a, 'pairs': [
    {'seed': 7101, 'project': first}, {'seed': 7102, 'project': second}],
    'resources_per_pair': {'ncpus': 624, 'mem_gb': 256, 'walltime': '02:30:00'},
    'maximum_pair_reservation_ksu': 3.12, 'summary_project': first}
(Path(directory)/'submission.json').write_text(json.dumps(manifest, indent=2)+'\n')
PY
echo "Suite: ${SUITE_DIRECTORY}"
echo "Each pair: 624 ranks, 256GB, 02:30:00; maximum reservation 3.12 KSU."
echo "Projects: seed 7101 -> ${FIRST_PROJECT}, seed 7102 -> ${SECOND_PROJECT}; summary -> ${FIRST_PROJECT}."
echo 'Keep the repository and virtual environment unchanged until all three jobs finish.'
JOB_IDS=()
for PAIR_SEED in 7101 7102; do
    PROJECT_CODE=${FIRST_PROJECT}
    if [[ "${PAIR_SEED}" == 7102 ]]; then PROJECT_CODE=${SECOND_PROJECT}; fi
    if ! JOB_ID=$(qsub -P "${PROJECT_CODE}" \
        -v "EXPECTED_COMMIT=${EXPECTED_COMMIT},PAIR_SEED=${PAIR_SEED},SUITE_DIRECTORY=${SUITE_DIRECTORY}" \
        -o "${SUITE_DIRECTORY}/seed_${PAIR_SEED}/pbs.out" -e "${SUITE_DIRECTORY}/seed_${PAIR_SEED}/pbs.err" \
        "${SCRIPT_DIRECTORY}/run_l23net_g1b_pair.sh"); then
        echo "Submission failed for seed ${PAIR_SEED}. Already submitted jobs: ${JOB_IDS[*]:-none}." >&2
        echo "Manifest: ${SUITE_DIRECTORY}/submission.json. Do not blindly rerun the entire suite." >&2
        exit 1
    fi
    JOB_IDS+=("${JOB_ID}")
    python3 - "${SUITE_DIRECTORY}" "${PAIR_SEED}" "${JOB_ID}" <<'PY'
import json
from pathlib import Path
import sys
directory, seed, job = sys.argv[1:]
p = Path(directory)/'submission.json'
data = json.loads(p.read_text())
for row in data['pairs']:
    if row['seed'] == int(seed): row['job_id'] = job
p.write_text(json.dumps(data, indent=2)+'\n')
PY
    echo "Submitted ${PAIR_SEED} on ${PROJECT_CODE}: ${JOB_ID}"
done
if ! SUMMARY_JOB=$(qsub -P "${FIRST_PROJECT}" -W "depend=afterany:${JOB_IDS[0]}:${JOB_IDS[1]}" \
    -v "EXPECTED_COMMIT=${EXPECTED_COMMIT},SUITE_DIRECTORY=${SUITE_DIRECTORY}" \
    -o "${SUITE_DIRECTORY}/summary_pbs.out" -e "${SUITE_DIRECTORY}/summary_pbs.err" \
    "${SCRIPT_DIRECTORY}/run_l23net_g1b_summary.sh"); then
    echo "Summary submission failed; pair jobs ${JOB_IDS[*]} are already submitted." >&2
    echo "Keep ${SUITE_DIRECTORY}/submission.json and run only the summary after the pairs finish." >&2
    exit 1
fi
echo "${SUMMARY_JOB}" > "${SUITE_DIRECTORY}/summary_job_id.txt"
echo "Summary job: ${SUMMARY_JOB}"
echo "After completion: cat '${SUITE_DIRECTORY}/g1b_summary.md'"
echo "For status/debugging: bash '${SCRIPT_DIRECTORY}/collect_l23net_g1b_status.sh' '${SUITE_DIRECTORY}'"
