#!/bin/bash
#PBS -q normal
#PBS -l ncpus=624
#PBS -l mem=256GB
#PBS -l walltime=02:30:00
#PBS -l jobfs=10GB
#PBS -l storage=gdata/ny83
#PBS -l software=python
#PBS -l wd
#PBS -N l23_g1b

set -euo pipefail
: "${SUITE_DIRECTORY:?Use submit_l23net_g1b.sh}"
: "${PAIR_SEED:?Missing pair seed}"
: "${EXPECTED_COMMIT:?Missing submitted commit}"
case "${PAIR_SEED}" in 7101|7102) ;; *) exit 2 ;; esac
REPOSITORY=/g/data/ny83/ch9972/NeuroStim/neurostimenv
RESULT_DIRECTORY="${SUITE_DIRECTORY}/seed_${PAIR_SEED}"
mkdir -p "${RESULT_DIRECTORY}"
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
    echo "End: $(date --iso-8601=seconds); exit=${code}; results=${RESULT_DIRECTORY}"
}
trap capture_exit EXIT
echo preflight > "${RESULT_DIRECTORY}/current_phase.txt"
echo "Start: $(date --iso-8601=seconds); job=${PBS_JOBID}; project=${PROJECT:-unknown}; seed=${PAIR_SEED}"
module load python3/3.10.4
module load openmpi/5.0.5
source /g/data/ny83/ch9972/NeuroStim/bin/activate
cd "${REPOSITORY}"
[[ "$(git rev-parse HEAD)" == "${EXPECTED_COMMIT}" ]] || { echo 'Submitted commit changed.'; exit 2; }
git diff --quiet && git diff --cached --quiet || { echo 'Tracked worktree is dirty.'; exit 2; }
export MAIN_PATH="${REPOSITORY}" PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
git rev-parse HEAD > "${RESULT_DIRECTORY}/git_commit.txt"
cp "${PBS_NODEFILE}" "${RESULT_DIRECTORY}/pbs_nodefile.txt"
sort -u "${PBS_NODEFILE}" > "${RESULT_DIRECTORY}/allocated_nodes.txt"
ALLOCATED_CPUS=$(wc -l < "${PBS_NODEFILE}")
ALLOCATED_NODES=$(wc -l < "${RESULT_DIRECTORY}/allocated_nodes.txt")
[[ "${ALLOCATED_CPUS}" == 624 && "${ALLOCATED_NODES}" == 13 ]] || { echo 'Expected 624 CPUs across 13 normal nodes.'; exit 2; }
echo "Allocated CPUs/nodes: ${ALLOCATED_CPUS}/${ALLOCATED_NODES}; MPI ranks: 624; memory: 256GB"
python3 experiments/l23net_analysis/nci/g1b_preflight.py environment "${RESULT_DIRECTORY}"
python3 -m pip check
MECHANISM_SOURCE="${REPOSITORY}/setup/circuits/L23Net/mod"
MECHANISM_BUILD="/g/data/ny83/ch9972/NeuroStim/nci_mechanisms/l23net_g1b_${PBS_JOBID}"
mkdir -p "${MECHANISM_BUILD}"
find "${MECHANISM_SOURCE}" -maxdepth 1 -type f -name '*.mod' -print0 | sort -z | xargs -0 sha256sum > "${RESULT_DIRECTORY}/mechanism_sha256.txt"
echo compiling_mechanisms > "${RESULT_DIRECTORY}/current_phase.txt"
(cd "${MECHANISM_BUILD}" && nrnivmodl "${MECHANISM_SOURCE}")
export L23NET_MECHANISM_PATH="${MECHANISM_BUILD}"
[[ -s "${MECHANISM_BUILD}/x86_64/.libs/libnrnmech.so" ]] || { echo 'Missing shared mechanism build.'; exit 2; }
mpirun -np 13 --map-by ppr:1:node --bind-to core \
    python3 experiments/l23net_analysis/nci/g1b_preflight.py mechanisms "${RESULT_DIRECTORY}"
python3 -m unittest -v tests/test_online_stimulation.py tests/test_online_recording_optimizations.py tests/test_l23net_replay_validation.py tests/test_l23net_g1b.py
echo fixed_step_regression > "${RESULT_DIRECTORY}/current_phase.txt"
mpirun -np 2 --map-by ppr:2:node --bind-to core \
    python3 tests/mpi_online_fixed_step_regression.py --duration-ms 28000 --dt-ms 0.025

echo -e 'timestamp\tjob_state\twalltime\tcpupercent\tmem\tphase\treference_ms\tmdd_ms' > "${RESULT_DIRECTORY}/heartbeat.tsv"
(
    while true; do
        PBS_STATE=$(qstat -fx "${PBS_JOBID}" 2>/dev/null | awk -F' = ' '/job_state =/{s=$2} /resources_used.walltime =/{w=$2} /resources_used.cpupercent =/{c=$2} /resources_used.mem =/{m=$2} END{printf "%s\t%s\t%s\t%s",s,w,c,m}' || true)
        PHASE=$(tr -d '\n' < "${RESULT_DIRECTORY}/current_phase.txt")
        PROGRESS=$(python3 experiments/l23net_analysis/nci/g1b_preflight.py progress "${RESULT_DIRECTORY}")
        echo -e "$(date --iso-8601=seconds)\t${PBS_STATE}\t${PHASE}\t${PROGRESS}" >> "${RESULT_DIRECTORY}/heartbeat.tsv"
        sleep 60
    done
) &
HEARTBEAT_PID=$!

for CONDITION in reference mdd; do
    echo "${CONDITION}" > "${RESULT_DIRECTORY}/current_phase.txt"
    MDD=false
    if [[ "${CONDITION}" == mdd ]]; then MDD=true; fi
    OUTPUT="${RESULT_DIRECTORY}/${CONDITION}"
    # Separate mpirun processes guarantee a fresh NEURON interpreter per condition.
    mpirun -np 624 --map-by ppr:48:node --bind-to core \
        python3 experiments/l23net_analysis/run_l23net_g1b.py \
        experiment.name="g1b_${PAIR_SEED}_${CONDITION}" \
        experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
        experiment.seed="${PAIR_SEED}" experiment.debug=false experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_g1b analysis.condition="${CONDITION}" analysis.env_seed=0 \
        env.simulation.MDD="${MDD}" env.simulation.DRUG=false env.ts.apply=false \
        env.simulation.duration=28000 env.simulation.obs_win_len=1000 \
        env.network.dt=0.025 env.network.celsius=34 env.network.tstart=0 env.network.v_init=-80 \
        env.network.syn_activity=true env.online.temperature_mode=configured
done
echo analyzing > "${RESULT_DIRECTORY}/current_phase.txt"
python3 experiments/l23net_analysis/analyze_l23net_g1b.py --pair "${RESULT_DIRECTORY}"
python3 - "${RESULT_DIRECTORY}" <<'PY'
import json
from pathlib import Path
import sys
p = Path(sys.argv[1])
status = json.loads((p / 'g1b_pair_summary.json').read_text())['status']
(p / 'current_phase.txt').write_text(status + '\n')
print('G1B pair completed: ' + status)
PY
