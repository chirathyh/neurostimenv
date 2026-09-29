#!/bin/bash
# Four-second engineering episodes only; cannot qualify production S1.
set -euo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPOSITORY}"
source /home/chirath/Documents/depression-simulator/bin/activate
export MAIN_PATH="${REPOSITORY}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
OUTPUT_ROOT=$(mktemp -d "${REPOSITORY}/results/s1_local_XXXXXX")
echo "S1_LOCAL_RESULTS=${OUTPUT_ROOT}"
for SPEC in '1 sham' '2 sham' '2 fixed_theta' '2 fixed_alpha' '2 fixed_low_beta' '2 transverse_alpha'; do
    read -r RANKS ARM <<< "${SPEC}"
    OUTPUT="${OUTPUT_ROOT}/r${RANKS}_${ARM}"
    mkdir -p "${OUTPUT}"
    mpirun -np "${RANKS}" python3 experiments/l23net_analysis/run_l23net_s1.py \
        experiment.name="s1_local_${ARM}" experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
        experiment.seed=8451 experiment.debug=true experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_s1 analysis.mode=debug analysis.arm="${ARM}" \
        analysis.require_full_network=false analysis.duration_ms=4000 analysis.window_ms=1000 \
        analysis.excluded_ms=0 analysis.stim_start_ms=1000 analysis.stim_stop_ms=3000 analysis.ramp_ms=250 \
        analysis.resource_request.mpi_ranks="${RANKS}" analysis.resource_request.ncpus="${RANKS}" analysis.resource_request.memory_gb=8 \
        env.simulation.MDD=true env.simulation.DRUG=false env.ts.apply=true \
        env.simulation.duration=4000 env.simulation.obs_win_len=1000 env.network.dt=0.025 \
        env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
        > "${OUTPUT}/run.log" 2>&1 < /dev/null
    echo "Completed ${RANKS} ranks / ${ARM}"
done
python3 - "${OUTPUT_ROOT}" <<'PY'
import pathlib,sys
from experiments.l23net_analysis.s1_analysis import audit_run,compare_prefix
root=pathlib.Path(sys.argv[1])
sham=audit_run(root/'r2_sham',None,debug=True)
audit_run(root/'r1_sham',None,debug=True)
for arm in ('fixed_theta','fixed_alpha','fixed_low_beta','transverse_alpha'):
    p=root/f'r2_{arm}';r=audit_run(p,None,debug=True)
    compare_prefix(root/'r2_sham',p,sham,r)
    assert r['window_protocol']['nonzero_field_samples']>0
    assert r['window_protocol']['max_field_v_per_m']>.39
    assert r['window_protocol']['tissue_coupling_observed']
    assert r['artifacts']['trace_summary']['committed_samples']==160000
print('PASS: serial sham, two-rank sham/three frequencies/orientation, exact paired prehistory, streamed traces, envelope, tissue coupling, removal and cleanup.')
PY
