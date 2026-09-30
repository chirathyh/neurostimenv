#!/bin/bash
# Reduced 4-s engineering checks; these cannot qualify production efficacy.
set -euo pipefail
REPOSITORY=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "${REPOSITORY}"
source /home/chirath/Documents/depression-simulator/bin/activate
export MAIN_PATH="${REPOSITORY}" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
OUTPUT_ROOT=$(mktemp -d "${REPOSITORY}/results/s2_pilot_local_XXXXXX")
echo "S2_PILOT_LOCAL_RESULTS=${OUTPUT_ROOT}"
for SPEC in '1 sham pilot' '1 fixed_low_beta pilot' '2 sham pilot' '2 fixed_low_beta pilot' '2 fixed_low_beta legacy'; do
    read -r RANKS ARM KIND <<< "${SPEC}"
    OUTPUT="${OUTPUT_ROOT}/r${RANKS}_${ARM}_${KIND}"
    mkdir -p "${OUTPUT}"
    RUNNER=run_l23net_s2_pilot.py
    CONFIG=l23net_s2_pilot
    if [[ "${KIND}" == legacy ]]; then RUNNER=run_l23net_s1.py; CONFIG=l23net_s1; fi
    mpirun -np "${RANKS}" python3 "experiments/l23net_analysis/${RUNNER}" \
        experiment.name="s2_local_${ARM}" experiment.dir="${OUTPUT}" hydra.run.dir="${OUTPUT}/hydra" \
        experiment.seed=8501 experiment.debug=true experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis="${CONFIG}" analysis.mode=debug analysis.condition=mdd analysis.arm="${ARM}" \
        analysis.require_full_network=false analysis.duration_ms=4000 analysis.window_ms=1000 \
        analysis.excluded_ms=0 analysis.stim_start_ms=1000 analysis.stim_stop_ms=3000 analysis.ramp_ms=250 \
        analysis.resource_request.mpi_ranks="${RANKS}" analysis.resource_request.ncpus="${RANKS}" analysis.resource_request.memory_gb=8 \
        env.simulation.MDD=true env.simulation.DRUG=false env.ts.apply=true \
        env.simulation.duration=4000 env.simulation.obs_win_len=1000 env.network.dt=0.025 \
        env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
        > "${OUTPUT}/run.log" 2>&1 < /dev/null
    echo "Completed ${RANKS} ranks / ${ARM} / ${KIND}"
done
python3 - "${OUTPUT_ROOT}" <<'PY'
import pathlib,sys,h5py,numpy as np
from experiments.l23net_analysis.s1_analysis import audit_run,compare_prefix
root=pathlib.Path(sys.argv[1])
for ranks in (1,2):
    sham=root/f'r{ranks}_sham_pilot';active=root/f'r{ranks}_fixed_low_beta_pilot'
    a,b=audit_run(sham,None,debug=True),audit_run(active,None,debug=True)
    compare_prefix(sham,active,a,b)
    assert b['window_protocol']['max_field_v_per_m']>.39
    assert b['window_protocol']['tissue_coupling_observed']
    for report in (a,b):
        assert report['artifacts']['trace_summary']['committed_samples']==160000
        assert report['build']['effective_celsius_by_rank']['minimum']==34
        assert report['build']['effective_celsius_by_rank']['maximum']==34
        assert report['completed_simulated_ms']==4000
old=root/'r2_fixed_low_beta_legacy';new=root/'r2_fixed_low_beta_pilot'
a,b=audit_run(old,None,debug=True),audit_run(new,None,debug=True)
assert a['artifacts']['trace_summary']==b['artifacts']['trace_summary']
assert a['seed_manifest']==b['seed_manifest'] and a['structure']==b['structure']
assert [w['spikes'] for w in a['windows']]==[w['spikes'] for w in b['windows']]
print('PASS: 1/2-MPI sham and active, exact paired baseline, finite traces, correct field/envelope/removal, temperature, sparse spikes and cleanup. New two-rank active trace is bit-identical to original S1.')
PY
