#!/bin/bash
#PBS -q normal
#PBS -l ncpus=624
#PBS -l mem=256GB
#PBS -l walltime=02:30:00
#PBS -l jobfs=10GB
#PBS -l storage=gdata/ny83
#PBS -l software=python
#PBS -l wd
#PBS -N l23_s1
set -euo pipefail
: "${SUITE_DIRECTORY:?Use submit_l23net_s1.py}"
: "${EXPECTED_COMMIT:?Missing commit}"
: "${JOB_INDEX:?Missing worker index}"
cd "${PBS_O_WORKDIR:?Missing checkout}"
REPOSITORY="$PWD"
module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
export MAIN_PATH="${REPOSITORY}" PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
read -r JOB_NAME SEED CONDITION ARM STAGE < <(python3 - "${SUITE_DIRECTORY}" "${JOB_INDEX}" <<'PY'
import json,pathlib,sys
m=json.loads((pathlib.Path(sys.argv[1])/'submission.json').read_text())
j=m['jobs'][int(sys.argv[2])];r=j['run']
print(j['name'],r['seed'],r['condition'],r['arm'],m['stage'])
PY
)
RESULT_DIRECTORY="${SUITE_DIRECTORY}/${JOB_NAME}"
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
    echo "End: $(date --iso-8601=seconds); exit=${code}"
}
trap capture_exit EXIT
echo "Start: $(date --iso-8601=seconds); ${PBS_JOBID}; ${JOB_NAME}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo 'Checkout changed.'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Dirty tracked checkout.'; exit 2; }
git rev-parse HEAD > "${RESULT_DIRECTORY}/git_commit.txt"
cp "${PBS_NODEFILE}" "${RESULT_DIRECTORY}/pbs_nodefile.txt"
sort -u "${PBS_NODEFILE}" > "${RESULT_DIRECTORY}/allocated_nodes.txt"
[[ "$(wc -l < "${PBS_NODEFILE}")" == 624 && "$(wc -l < "${RESULT_DIRECTORY}/allocated_nodes.txt")" == 13 ]] || exit 2
python3 experiments/l23net_analysis/nci/g1b_preflight.py environment "${RESULT_DIRECTORY}"
python3 -m pip check
python3 - "${SUITE_DIRECTORY}" <<'PY'
import json,pathlib,sys
from experiments.l23net_analysis.s0_protocol import load_gate
from experiments.l23net_analysis.s1_analysis import code_hashes,load_qualification
p=pathlib.Path(sys.argv[1]);m=json.loads((p/'submission.json').read_text());g=load_gate(p/'s0_gate.json')
assert code_hashes()==m['code_sha256'] and g['sha256']==m['s0_sha256']
if m['stage']=='discovery':load_qualification(p/'qualification.json',g['sha256'])
PY
MECHANISM_SOURCE="${REPOSITORY}/setup/circuits/L23Net/mod"
MECHANISM_BUILD="/g/data/ny83/ch9972/NeuroStim/nci_mechanisms/l23net_s1_${PBS_JOBID}"
mkdir -p "${MECHANISM_BUILD}"
find "${MECHANISM_SOURCE}" -maxdepth 1 -type f -name '*.mod' -print0 | sort -z | xargs -0 sha256sum > "${RESULT_DIRECTORY}/mechanism_sha256.txt"
(cd "${MECHANISM_BUILD}" && nrnivmodl "${MECHANISM_SOURCE}")
export L23NET_MECHANISM_PATH="${MECHANISM_BUILD}"
[[ -s "${MECHANISM_BUILD}/x86_64/.libs/libnrnmech.so" ]] || exit 2
mpirun -np 13 --map-by ppr:1:node --bind-to core \
    python3 experiments/l23net_analysis/nci/g1b_preflight.py mechanisms "${RESULT_DIRECTORY}" < /dev/null
python3 -m unittest -v tests.test_l23net_s0_s1 tests.test_online_stimulation
echo -e 'timestamp\twalltime\tcpupercent\tmem\tcompleted_ms' > "${RESULT_DIRECTORY}/heartbeat.tsv"
(
    while true; do
        STATE=$(qstat -fx "${PBS_JOBID}" 2>/dev/null | awk -F' = ' '/resources_used.walltime =/{w=$2} /resources_used.cpupercent =/{c=$2} /resources_used.mem =/{m=$2} END{printf "%s\t%s\t%s",w,c,m}' || true)
        PROGRESS=$(python3 - "${RESULT_DIRECTORY}" <<'PY'
import json,pathlib,sys
try:print(json.loads((pathlib.Path(sys.argv[1])/'l23net_s1_run.json').read_text()).get('completed_simulated_ms',0))
except (OSError,ValueError):print(0)
PY
)
        echo -e "$(date --iso-8601=seconds)\t${STATE}\t${PROGRESS}" >> "${RESULT_DIRECTORY}/heartbeat.tsv"
        sleep 60
    done
) &
HEARTBEAT_PID=$!
MDD=true
if [[ "${CONDITION}" == reference ]]; then MDD=false; fi
mpirun -np 624 --map-by ppr:48:node --bind-to core \
    python3 experiments/l23net_analysis/run_l23net_s1.py \
    experiment.name="s1_${JOB_NAME}" experiment.dir="${RESULT_DIRECTORY}" hydra.run.dir="${RESULT_DIRECTORY}/hydra" \
    experiment.seed="${SEED}" experiment.debug=false experiment.plot=false experiment.tqdm=false \
    env=hl23net analysis=l23net_s1 analysis.mode="${STAGE}" analysis.condition="${CONDITION}" analysis.arm="${ARM}" \
    analysis.s0_gate="${SUITE_DIRECTORY}/s0_gate.json" analysis.qualification="${SUITE_DIRECTORY}/qualification.json" \
    env.simulation.MDD="${MDD}" env.simulation.DRUG=false env.ts.apply=true \
    env.simulation.duration=60000 env.simulation.obs_win_len=1000 env.network.dt=0.025 \
    env.network.celsius=34 env.network.tstart=0 env.network.v_init=-80 env.network.syn_activity=true \
    env.online.temperature_mode=configured < /dev/null
