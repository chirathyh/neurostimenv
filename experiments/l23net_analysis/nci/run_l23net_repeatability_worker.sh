#!/bin/bash
#PBS -l software=python
#PBS -l wd
set -euo pipefail
: "${SUITE_DIRECTORY:?Use submit_l23net_repeatability.py}"
: "${WAVE_JOB_NAME:?Missing worker name}"
: "${STRUCTURE_SEED:?Missing circuit seed}"
: "${HISTORY_SEED:?Missing OU history seed}"
: "${WAVE_ARM:?Missing stimulation arm}"
: "${EXPECTED_COMMIT:?Missing commit}"
RESULT_DIRECTORY="${SUITE_DIRECTORY}/${WAVE_JOB_NAME}"
exec > >(tee -a "${RESULT_DIRECTORY}/job_live.log") 2>&1
HEARTBEAT_PID=""
capture_exit() {
    local code=$?
    if [[ -n "${HEARTBEAT_PID}" ]]; then
        kill "${HEARTBEAT_PID}" 2>/dev/null || true
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
    echo "${code}" > "${RESULT_DIRECTORY}/worker_exit_code.txt"
    qstat -fx "${PBS_JOBID}" > "${RESULT_DIRECTORY}/qstat_at_exit.txt" 2>&1 || true
    echo "End $(date --iso-8601=seconds) | exit ${code}"
}
trap capture_exit EXIT
echo "Start $(date --iso-8601=seconds) | ${PBS_JOBID} | ${WAVE_JOB_NAME} | OU seed ${HISTORY_SEED}"
cd "${PBS_O_WORKDIR:?Missing checkout}"
REPOSITORY="$PWD"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo 'Checkout changed.'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Dirty tracked checkout.'; exit 2; }
git rev-parse HEAD > "${RESULT_DIRECTORY}/git_commit.txt"
cp "${PBS_NODEFILE}" "${RESULT_DIRECTORY}/pbs_nodefile.txt"
sort -u "${PBS_NODEFILE}" > "${RESULT_DIRECTORY}/allocated_nodes.txt"
[[ "$(wc -l < "${PBS_NODEFILE}")" == 624 && "$(wc -l < "${RESULT_DIRECTORY}/allocated_nodes.txt")" == 13 ]] || {
    echo 'Expected 624 slots on 13 nodes.'; exit 2;
}

module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
export MAIN_PATH="${REPOSITORY}" PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 experiments/l23net_analysis/nci/g1b_preflight.py environment "${RESULT_DIRECTORY}"
python3 - "${SUITE_DIRECTORY}" <<'PY'
import sys
from experiments.l23net_analysis.repeatability_protocol import check_manifest
check_manifest(sys.argv[1])
PY

MECHANISM_SOURCE="${REPOSITORY}/setup/circuits/L23Net/mod"
MECHANISM_BUILD="/g/data/ny83/ch9972/NeuroStim/nci_mechanisms/l23net_repeatability_${PBS_JOBID}"
mkdir -p "${MECHANISM_BUILD}"
find "${MECHANISM_SOURCE}" -maxdepth 1 -type f -name '*.mod' -print0 | sort -z | xargs -0 sha256sum > "${RESULT_DIRECTORY}/mechanism_sha256.txt"
(cd "${MECHANISM_BUILD}" && nrnivmodl "${MECHANISM_SOURCE}")
export L23NET_MECHANISM_PATH="${MECHANISM_BUILD}"
[[ -s "${MECHANISM_BUILD}/x86_64/.libs/libnrnmech.so" ]] || { echo 'Compiled library missing.'; exit 2; }
mpirun -np 13 --map-by ppr:1:node --bind-to core \
    python3 experiments/l23net_analysis/nci/g1b_preflight.py mechanisms "${RESULT_DIRECTORY}" < /dev/null

printf 'timestamp\twalltime\tcpupercent\tmem\tcompleted_ms\n' > "${RESULT_DIRECTORY}/heartbeat.tsv"
(
    while true; do
        STATE=$(qstat -fx "${PBS_JOBID}" 2>/dev/null | awk -F' = ' '/resources_used.walltime =/{w=$2} /resources_used.cpupercent =/{c=$2} /resources_used.mem =/{m=$2} END{printf "%s\t%s\t%s",w,c,m}' || true)
        PROGRESS=$(python3 - "${RESULT_DIRECTORY}" <<'PY'
import json
from pathlib import Path
import sys

try:
    report = json.loads((Path(sys.argv[1]) / 'l23net_s1_run.json').read_text())
    print(report.get('completed_simulated_ms', 0))
except (OSError, ValueError):
    print(0)
PY
)
        printf '%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "${STATE}" "${PROGRESS}" >> "${RESULT_DIRECTORY}/heartbeat.tsv"
        sleep 60
    done
) &
HEARTBEAT_PID=$!

mpirun -np 624 --map-by ppr:48:node --bind-to core \
    python3 experiments/l23net_analysis/run_l23net_repeatability.py \
    experiment.name="l23net_repeatability_${WAVE_JOB_NAME}" experiment.dir="${RESULT_DIRECTORY}" hydra.run.dir="${RESULT_DIRECTORY}/hydra" \
    experiment.seed="${STRUCTURE_SEED}" experiment.debug=false experiment.plot=false experiment.tqdm=false \
    env=hl23net analysis=l23net_dose_response analysis.mode=discovery analysis.condition=mdd analysis.arm="${WAVE_ARM}" \
    analysis.dose_suite="${SUITE_DIRECTORY}" analysis.s0_gate="${SUITE_DIRECTORY}/s0_gate.json" \
    env.simulation.MDD=true env.simulation.DRUG=false env.ts.apply=true \
    env.simulation.duration=60000 env.simulation.obs_win_len=1000 env.network.dt=0.025 \
    env.network.celsius=34 env.network.tstart=0 env.network.v_init=-80 env.network.syn_activity=true \
    env.online.temperature_mode=configured ++env.online.l23net_history_seed="${HISTORY_SEED}" < /dev/null
