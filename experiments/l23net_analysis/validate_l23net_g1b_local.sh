#!/bin/bash
# Reduced local construction/pairing/continuation checks, not G1B qualification.
set -euo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPOSITORY}"
source /home/chirath/Documents/depression-simulator/bin/activate
export MAIN_PATH="${REPOSITORY}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
OUTPUT_ROOT=$(mktemp -d "${REPOSITORY}/results/g1b_local_XXXXXX")
echo "Local validation: ${OUTPUT_ROOT}"
for SEED in 7101 7102; do
    DURATION=6000
    if [[ "${SEED}" == 7102 ]]; then DURATION=1000; fi
    for CONDITION in reference mdd; do
        MDD=false
        if [[ "${CONDITION}" == mdd ]]; then MDD=true; fi
        OUTPUT="${OUTPUT_ROOT}/seed_${SEED}/${CONDITION}"
        mkdir -p "${OUTPUT}"
        mpirun -np 2 python experiments/l23net_analysis/run_l23net_g1b.py \
            experiment.name="g1b_local_${SEED}_${CONDITION}" \
            experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
            experiment.seed="${SEED}" experiment.debug=true experiment.plot=false experiment.tqdm=false \
            env=hl23net analysis=l23net_g1b analysis.require_full_network=false \
            analysis.condition="${CONDITION}" analysis.duration_ms="${DURATION}" analysis.window_ms=1000 \
            analysis.resource_request.mpi_ranks=2 analysis.resource_request.ncpus=2 analysis.resource_request.memory_gb=8 \
            env.simulation.MDD="${MDD}" env.simulation.DRUG=false env.ts.apply=false \
            env.simulation.duration="${DURATION}" env.simulation.obs_win_len=1000 env.network.dt=0.025 \
            env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
            > "${OUTPUT}/run.log" 2>&1
    done
    python experiments/l23net_analysis/analyze_l23net_g1b.py --pair "${OUTPUT_ROOT}/seed_${SEED}" --debug-smoke
done
echo "Local paired smoke completed: ${OUTPUT_ROOT}"
