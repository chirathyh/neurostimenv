#!/bin/bash
#PBS -P sj53
#PBS -q normal
#PBS -l walltime=00:40:00
#PBS -l ncpus=624
#PBS -l mem=200GB
#PBS -l jobfs=10GB
#PBS -l storage=gdata/ny83
#PBS -l software=python
#PBS -l wd
#PBS -N l23_replay
#PBS -o out_l23net_no_field_replay.txt
#PBS -e err_l23net_no_field_replay.txt

# Gate G1A: two independent six-second full-network zero-field replays and an
# exact artifact comparison. The historical path is not a bitwise oracle: it
# has a different temperature/lifecycle and uses h.fadvance rather than the
# corrected ParallelContext.psolve MPI integration path.

set -euo pipefail

module load python3/3.10.4
module load openmpi/5.0.5

REPOSITORY=/g/data/ny83/ch9972/NeuroStim/neurostimenv
VIRTUAL_ENV_ACTIVATE=/g/data/ny83/ch9972/NeuroStim/bin/activate
EXPECTED_COMMIT=${EXPECTED_COMMIT:-}
MPI_RANKS=624
EXPECTED_NCPUS=624
REQUESTED_MEMORY_GB=200
DURATION_MS=6000
WINDOW_MS=1000
EXPERIMENT_SEED=10

if [[ -z "${EXPECTED_COMMIT}" ]]; then
    echo "EXPECTED_COMMIT was not supplied; use submit_l23net_no_field_replay.sh." >&2
    exit 2
fi
if [[ ! -d "${REPOSITORY}/.git" ]]; then
    echo "Repository not found at ${REPOSITORY}." >&2
    exit 2
fi
if [[ ! -f "${VIRTUAL_ENV_ACTIVATE}" ]]; then
    echo "Virtual environment not found at ${VIRTUAL_ENV_ACTIVATE}." >&2
    exit 2
fi
if [[ ! -r "${PBS_NODEFILE}" ]]; then
    echo "PBS node file is unavailable: ${PBS_NODEFILE}." >&2
    exit 2
fi

cd "${REPOSITORY}"
ACTUAL_COMMIT=$(git rev-parse HEAD)
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
    echo "HEAD ${ACTUAL_COMMIT} does not equal submitted commit ${EXPECTED_COMMIT}." >&2
    exit 2
fi
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Tracked files in the NCI worktree have uncommitted changes." >&2
    exit 2
fi

source "${VIRTUAL_ENV_ACTIVATE}"
export MAIN_PATH="${REPOSITORY}"
export PYTHONPATH="${REPOSITORY}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

RESULT_DIRECTORY="${REPOSITORY}/results/l23net_no_field_replay_${PBS_JOBID}"
FIRST_DIRECTORY="${RESULT_DIRECTORY}/replay_a"
SECOND_DIRECTORY="${RESULT_DIRECTORY}/replay_b"
COMPARISON_REPORT="${RESULT_DIRECTORY}/l23net_no_field_replay_comparison.json"
MECHANISM_SOURCE="${REPOSITORY}/setup/circuits/L23Net/mod"
# PBS_JOBFS is node-local; all nodes need a shared mechanism library.
MECHANISM_BUILD_ROOT=/g/data/ny83/ch9972/NeuroStim/nci_mechanisms
MECHANISM_BUILD="${MECHANISM_BUILD_ROOT}/l23net_${PBS_JOBID}"

mkdir -p \
    "${RESULT_DIRECTORY}" \
    "${FIRST_DIRECTORY}" \
    "${SECOND_DIRECTORY}" \
    "${MECHANISM_BUILD}"
RUN_LOG="${RESULT_DIRECTORY}/job_live.log"
HEARTBEAT_LOG="${RESULT_DIRECTORY}/heartbeat.tsv"
PHASE_FILE="${RESULT_DIRECTORY}/current_phase.txt"
exec > >(tee -a "${RUN_LOG}") 2>&1

HEARTBEAT_PID=""
capture_job_state() {
    if [[ -n "${HEARTBEAT_PID}" ]] && kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        kill "${HEARTBEAT_PID}" 2>/dev/null || true
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
    qstat -fx "${PBS_JOBID}" > "${RESULT_DIRECTORY}/qstat_at_exit.txt" 2>&1 || true
}
trap capture_job_state EXIT

sort -u "${PBS_NODEFILE}" > "${RESULT_DIRECTORY}/allocated_nodes.txt"
cp "${PBS_NODEFILE}" "${RESULT_DIRECTORY}/pbs_nodefile.txt"
git rev-parse HEAD > "${RESULT_DIRECTORY}/git_commit.txt"
find "${MECHANISM_SOURCE}" -maxdepth 1 -type f -name '*.mod' -print0 \
    | sort -z \
    | xargs -0 sha256sum > "${RESULT_DIRECTORY}/mechanism_sha256.txt"

ALLOCATED_NODES=$(wc -l < "${RESULT_DIRECTORY}/allocated_nodes.txt")
ALLOCATED_CPUS=$(wc -l < "${PBS_NODEFILE}")
if (( ALLOCATED_CPUS != EXPECTED_NCPUS )); then
    echo "PBS allocated ${ALLOCATED_CPUS} CPUs; expected ${EXPECTED_NCPUS}." >&2
    exit 2
fi
if (( MPI_RANKS % ALLOCATED_NODES != 0 )); then
    echo "${MPI_RANKS} ranks do not divide across ${ALLOCATED_NODES} nodes." >&2
    exit 2
fi
RANKS_PER_NODE=$((MPI_RANKS / ALLOCATED_NODES))

echo "Start: $(date --iso-8601=seconds)"
echo "PBS job: ${PBS_JOBID}"
echo "Repository: ${REPOSITORY}"
echo "Branch: $(git branch --show-current)"
echo "Commit: ${ACTUAL_COMMIT}"
echo "Allocated nodes: ${ALLOCATED_NODES}"
echo "Allocated CPUs/MPI ranks: ${ALLOCATED_CPUS}/${MPI_RANKS}"
echo "Ranks per node: ${RANKS_PER_NODE}"
echo "Requested memory: ${REQUESTED_MEMORY_GB} GiB"
echo "Replay duration/window: ${DURATION_MS}/${WINDOW_MS} ms"
echo "Results: ${RESULT_DIRECTORY}"

python3 - "${RESULT_DIRECTORY}/environment_versions.json" <<'PY'
import json
import sys

import LFPy
import h5py
import mpi4py
import neuron
import numpy
import scipy

versions = {
    "LFPy": LFPy.__version__,
    "NEURON": neuron.__version__,
    "NumPy": numpy.__version__,
    "SciPy": scipy.__version__,
    "h5py": h5py.__version__,
    "mpi4py": mpi4py.__version__,
    "python": sys.version,
}
expected = {
    "LFPy": "2.3",
    "NEURON": "8.2.3",
    "NumPy": "1.26.3",
    "SciPy": "1.11.4",
    "mpi4py": "3.1.5",
}
errors = [
    f"{name}: expected {value}, found {versions.get(name)}"
    for name, value in expected.items()
    if versions.get(name) != value
]
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump({"versions": versions, "expected": expected, "errors": errors}, stream, indent=2, sort_keys=True)
    stream.write("\n")
print("Environment versions: " + json.dumps(versions, sort_keys=True))
if errors:
    raise SystemExit("Incompatible or unfrozen environment: " + "; ".join(errors))
PY
python3 -m pip check

echo compiling_mechanisms > "${PHASE_FILE}"
(
    cd "${MECHANISM_BUILD}"
    nrnivmodl "${MECHANISM_SOURCE}"
)
export L23NET_MECHANISM_PATH="${MECHANISM_BUILD}"
if [[ ! -s "${MECHANISM_BUILD}/x86_64/.libs/libnrnmech.so" ]]; then
    echo "L23Net mechanism build did not create libnrnmech.so." >&2
    exit 2
fi

mpirun \
    -np "${ALLOCATED_NODES}" \
    --map-by ppr:1:node \
    --bind-to core \
    python3 -c '
import os
import socket
from mpi4py import MPI
import neuron
from neuron import h

comm = MPI.COMM_WORLD
path = os.environ["L23NET_MECHANISM_PATH"]
if not neuron.load_mechanisms(path):
    raise RuntimeError(f"rank {comm.rank} on {socket.gethostname()} could not load {path}")
probe = h.Section(name=f"mechanism_probe_rank_{comm.rank}")
probe.insert("tonic")
probe.insert("Ih")
hosts = comm.allgather(socket.gethostname())
if len(set(hosts)) != comm.size:
    raise RuntimeError(f"one-rank-per-node preflight did not cover all nodes: {hosts}")
print(f"MECHANISM_PREFLIGHT rank={comm.rank} host={socket.gethostname()}", flush=True)
h.delete_section(sec=probe)
'

echo unit_tests > "${PHASE_FILE}"
python3 -m unittest -v \
    tests/test_online_stimulation.py \
    tests/test_online_recording_optimizations.py \
    tests/test_l23net_replay_validation.py

echo fixed_step_regression > "${PHASE_FILE}"
mpirun \
    -np 2 \
    --map-by ppr:2:node \
    --bind-to core \
    python3 tests/mpi_online_fixed_step_regression.py \
        --duration-ms "${DURATION_MS}" \
        --dt-ms 0.025

echo -e "timestamp\tjob_state\twalltime\tcpupercent\tmem\tphase\treplay_a_ms\treplay_b_ms" \
    > "${HEARTBEAT_LOG}"
(
    while true; do
        PBS_STATE=$(qstat -fx "${PBS_JOBID}" 2>/dev/null \
            | awk -F' = ' '/job_state =/{s=$2} /resources_used.walltime =/{w=$2} /resources_used.cpupercent =/{c=$2} /resources_used.mem =/{m=$2} END{printf "%s\t%s\t%s\t%s", s,w,c,m}')
        PHASE=$(tr -d '\n' < "${PHASE_FILE}" 2>/dev/null || echo unknown)
        PROGRESS=$(python3 - "${FIRST_DIRECTORY}" "${SECOND_DIRECTORY}" <<'PY'
import json
from pathlib import Path
import sys

values = []
for directory in sys.argv[1:]:
    path = Path(directory) / "l23net_no_field_replay.json"
    if not path.exists():
        values.append("0")
        continue
    try:
        values.append(str(json.loads(path.read_text()).get("completed_simulated_ms", 0)))
    except Exception:
        values.append("unreadable")
print("\t".join(values))
PY
)
        echo -e "$(date --iso-8601=seconds)\t${PBS_STATE}\t${PHASE}\t${PROGRESS}" \
            >> "${HEARTBEAT_LOG}"
        sleep 60
    done
) &
HEARTBEAT_PID=$!

run_replay() {
    local label=$1
    local output_directory=$2
    echo "replay_${label}" > "${PHASE_FILE}"
    mpirun \
        -np "${MPI_RANKS}" \
        --map-by "ppr:${RANKS_PER_NODE}:node" \
        --bind-to core \
        python3 experiments/l23net_analysis/run_l23net_no_field_replay.py \
            experiment.name="l23net_no_field_replay_${label}" \
            experiment.dir="${output_directory}" \
            hydra.run.dir="${output_directory}/hydra" \
            experiment.seed="${EXPERIMENT_SEED}" \
            experiment.debug=false \
            experiment.tqdm=false \
            experiment.plot=false \
            env=hl23net \
            analysis=l23net_no_field_replay \
            env.simulation.MDD=false \
            env.simulation.DRUG=false \
            env.simulation.duration="${DURATION_MS}" \
            env.simulation.obs_win_len="${WINDOW_MS}" \
            env.network.dt=0.025 \
            env.network.celsius=34.0 \
            env.network.syn_activity=true \
            env.online.temperature_mode=configured \
            env.ts.apply=false \
            analysis.duration_ms="${DURATION_MS}" \
            analysis.window_ms="${WINDOW_MS}" \
            analysis.resource_request.ncpus="${ALLOCATED_CPUS}" \
            analysis.resource_request.mpi_ranks="${MPI_RANKS}" \
            analysis.resource_request.memory_gb="${REQUESTED_MEMORY_GB}"
}

run_replay a "${FIRST_DIRECTORY}"
run_replay b "${SECOND_DIRECTORY}"

echo comparing > "${PHASE_FILE}"
python3 experiments/l23net_analysis/compare_l23net_no_field_replays.py \
    --first "${FIRST_DIRECTORY}" \
    --second "${SECOND_DIRECTORY}" \
    --output "${COMPARISON_REPORT}"

kill "${HEARTBEAT_PID}" 2>/dev/null || true
wait "${HEARTBEAT_PID}" 2>/dev/null || true
HEARTBEAT_PID=""
echo passed > "${PHASE_FILE}"

echo "End: $(date --iso-8601=seconds)"
echo "L23Net deterministic no-field replay gate PASSED."
echo "Comparison: ${COMPARISON_REPORT}"
