#!/bin/bash
# New cohort/condition routing over two MPI ranks; not a phenotype qualification.
set -euo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPOSITORY}"
source /home/chirath/Documents/depression-simulator/bin/activate
export MAIN_PATH="${REPOSITORY}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUTPUT_ROOT=$(mktemp -d "${REPOSITORY}/results/r1_local_XXXXXX")
echo "Local R1 validation: ${OUTPUT_ROOT}"
for SPEC in '8101 reference calibration' '8201 reference primary' '8201 mdd primary' '8301 reference extension' '8301 mdd extension'; do
    read -r SEED CONDITION COHORT <<< "${SPEC}"
    MDD=false
    if [[ "${CONDITION}" == mdd ]]; then MDD=true; fi
    OUTPUT="${OUTPUT_ROOT}/seed_${SEED}/${CONDITION}"
    mkdir -p "${OUTPUT}"
    mpirun -np 2 python3 experiments/l23net_analysis/run_l23net_r1.py \
        experiment.name="r1_local_${SEED}_${CONDITION}" \
        experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
        experiment.seed="${SEED}" experiment.debug=true experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_r1 analysis.cohort="${COHORT}" analysis.condition="${CONDITION}" \
        analysis.require_full_network=false analysis.duration_ms=2000 analysis.window_ms=1000 \
        analysis.resource_request.mpi_ranks=2 analysis.resource_request.ncpus=2 analysis.resource_request.memory_gb=8 \
        env.simulation.MDD="${MDD}" env.simulation.DRUG=false env.ts.apply=false \
        env.simulation.duration=2000 env.simulation.obs_win_len=1000 env.network.dt=0.025 \
        env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
        > "${OUTPUT}/run.log" 2>&1
done
python3 - "${OUTPUT_ROOT}" <<'PY'
import json, pathlib, sys
from experiments.l23net_analysis.r1_analysis import audit_run
from experiments.l23net_analysis.g1b_analysis import pairing_errors
root = pathlib.Path(sys.argv[1])
reports = {}
for seed, condition, cohort in ((8101,'reference','calibration'), (8201,'reference','primary'), (8201,'mdd','primary'), (8301,'reference','extension'), (8301,'mdd','extension')):
    row, report, _ = audit_run(root/f'seed_{seed}'/condition, seed, condition, cohort, debug=True)
    reports[seed,condition] = report
    assert report['artifacts']['trace_summary']['committed_samples'] == 80000
    assert report['artifacts']['trace_summary']['committed_windows'] == 2
for seed in (8201,8301):
    assert not pairing_errors(reports[seed,'reference'], reports[seed,'mdd'])
print('PASS: five two-rank trajectories; exact sample/window counts, finite traces, zero field, paired construction, cleanup.')
PY
