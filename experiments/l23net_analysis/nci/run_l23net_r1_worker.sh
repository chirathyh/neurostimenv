#!/bin/bash
#PBS -q normal
#PBS -l ncpus=624
#PBS -l mem=256GB
#PBS -l walltime=02:30:00
#PBS -l jobfs=10GB
#PBS -l storage=gdata/ny83
#PBS -l software=python
#PBS -l wd
#PBS -N l23_r1
set -euo pipefail
: "${SUITE_DIRECTORY:?Use submit_l23net_r1.py}"
: "${EXPECTED_COMMIT:?Missing commit}"
: "${JOB_INDEX:?Missing worker index}"
REPOSITORY=${PBS_O_WORKDIR:?Missing submission directory}
cd "${REPOSITORY}"
JOB_NAME=$(python3 - "${SUITE_DIRECTORY}" "${JOB_INDEX}" <<'PY'
import json, pathlib, sys
print(json.loads((pathlib.Path(sys.argv[1])/'submission.json').read_text())['jobs'][int(sys.argv[2])]['name'])
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
    if (( code != 0 )); then echo failed > "${RESULT_DIRECTORY}/current_phase.txt"; fi
    qstat -fx "${PBS_JOBID}" > "${RESULT_DIRECTORY}/qstat_at_exit.txt" 2>&1 || true
    echo "End: $(date --iso-8601=seconds); exit=${code}"
}
trap capture_exit EXIT
echo preflight > "${RESULT_DIRECTORY}/current_phase.txt"
echo "Start: $(date --iso-8601=seconds); job=${PBS_JOBID}; ${JOB_NAME}"
module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo 'Submitted commit changed.'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Dirty tracked worktree.'; exit 2; }
export MAIN_PATH="${REPOSITORY}" PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
git rev-parse HEAD > "${RESULT_DIRECTORY}/git_commit.txt"
cp "${PBS_NODEFILE}" "${RESULT_DIRECTORY}/pbs_nodefile.txt"
sort -u "${PBS_NODEFILE}" > "${RESULT_DIRECTORY}/allocated_nodes.txt"
[[ "$(wc -l < "${PBS_NODEFILE}")" == 624 && "$(wc -l < "${RESULT_DIRECTORY}/allocated_nodes.txt")" == 13 ]] || { echo 'Expected 624 CPUs on 13 nodes.'; exit 2; }
python3 experiments/l23net_analysis/nci/g1b_preflight.py environment "${RESULT_DIRECTORY}"
python3 -m pip check
MECHANISM_SOURCE="${REPOSITORY}/setup/circuits/L23Net/mod"
MECHANISM_BUILD="/g/data/ny83/ch9972/NeuroStim/nci_mechanisms/l23net_r1_${PBS_JOBID}"
mkdir -p "${MECHANISM_BUILD}"
find "${MECHANISM_SOURCE}" -maxdepth 1 -type f -name '*.mod' -print0 | sort -z | xargs -0 sha256sum > "${RESULT_DIRECTORY}/mechanism_sha256.txt"
echo compiling_mechanisms > "${RESULT_DIRECTORY}/current_phase.txt"
(cd "${MECHANISM_BUILD}" && nrnivmodl "${MECHANISM_SOURCE}")
export L23NET_MECHANISM_PATH="${MECHANISM_BUILD}"
[[ -s "${MECHANISM_BUILD}/x86_64/.libs/libnrnmech.so" ]] || exit 2
mpirun -np 13 --map-by ppr:1:node --bind-to core \
    python3 experiments/l23net_analysis/nci/g1b_preflight.py mechanisms "${RESULT_DIRECTORY}"
python3 -m unittest -v tests/test_online_stimulation.py tests/test_online_recording_optimizations.py tests/test_l23net_r1.py

python3 - "${SUITE_DIRECTORY}" "${JOB_INDEX}" "${RESULT_DIRECTORY}" <<'PY'
import json, pathlib, sys
from experiments.l23net_analysis.r1_protocol import verify_g1b, sha256
suite, index, output = pathlib.Path(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
manifest = json.loads((suite/'submission.json').read_text())
verify_g1b(suite/'g1b_prerequisite.json')
if manifest['stage'] == 'extension':
    for name, digest in manifest['core_hashes'].items():
        if sha256(pathlib.Path(manifest['core_directory'])/name) != digest:
            raise ValueError('Frozen core changed: '+name)
job = manifest['jobs'][index]
(output/'runs.txt').write_text(''.join(f"{r['seed']} {r['condition']} {job['cohort']}\n" for r in job['runs']))
PY
echo -e 'timestamp\twalltime\tcpupercent\tmem\tphase\tcompleted_ms' > "${RESULT_DIRECTORY}/heartbeat.tsv"
(
    while true; do
        STATE=$(qstat -fx "${PBS_JOBID}" 2>/dev/null | awk -F' = ' '/resources_used.walltime =/{w=$2} /resources_used.cpupercent =/{c=$2} /resources_used.mem =/{m=$2} END{printf "%s\t%s\t%s",w,c,m}' || true)
        PHASE=$(tr -d '\n' < "${RESULT_DIRECTORY}/current_phase.txt")
        PROGRESS=$(python3 - "${RESULT_DIRECTORY}" "${PHASE}" <<'PY'
import json, pathlib, sys
try:
    print(json.loads((pathlib.Path(sys.argv[1])/sys.argv[2]/'l23net_r1_run.json').read_text()).get('completed_simulated_ms', 0))
except (OSError, ValueError):
    print(0)
PY
)
        echo -e "$(date --iso-8601=seconds)\t${STATE}\t${PHASE}\t${PROGRESS}" >> "${RESULT_DIRECTORY}/heartbeat.tsv"
        sleep 60
    done
) &
HEARTBEAT_PID=$!
while read -r SEED CONDITION COHORT; do
    MDD=false
    if [[ "${CONDITION}" == mdd ]]; then MDD=true; fi
    PHASE="seed_${SEED}/${CONDITION}"
    echo "${PHASE}" > "${RESULT_DIRECTORY}/current_phase.txt"
    OUTPUT="${RESULT_DIRECTORY}/${PHASE}"
    mpirun -np 624 --map-by ppr:48:node --bind-to core \
        python3 experiments/l23net_analysis/run_l23net_r1.py \
        experiment.name="r1_${COHORT}_${SEED}_${CONDITION}" \
        experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
        experiment.seed="${SEED}" experiment.debug=false experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_r1 analysis.cohort="${COHORT}" analysis.condition="${CONDITION}" analysis.env_seed=0 \
        env.simulation.MDD="${MDD}" env.simulation.DRUG=false env.ts.apply=false \
        env.simulation.duration=28000 env.simulation.obs_win_len=1000 \
        env.network.dt=0.025 env.network.celsius=34 env.network.tstart=0 env.network.v_init=-80 \
        env.network.syn_activity=true env.online.temperature_mode=configured < /dev/null
done < "${RESULT_DIRECTORY}/runs.txt"
echo completed > "${RESULT_DIRECTORY}/current_phase.txt"
