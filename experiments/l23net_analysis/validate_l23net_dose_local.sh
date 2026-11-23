#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/chirath/Documents/depression-simulator/bin/activate
unset DISPLAY
export MAIN_PATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p results
DOSE_SMOKE_DIR=$(mktemp -d "$PWD/results/l23net_dose_local_XXXXXX")
printf 'DOSE_SMOKE_DIR=%s\n' "$DOSE_SMOKE_DIR"
for DOSE_CASE in '1 sham' '1 dose_08' '2 sham' '2 dose_02' '2 dose_04' '2 dose_08'; do
    read -r DOSE_RANKS DOSE_ARM <<< "$DOSE_CASE"
    DOSE_RUN="$DOSE_SMOKE_DIR/r${DOSE_RANKS}_${DOSE_ARM}"
    mkdir -p "$DOSE_RUN"
    if ! mpirun -np "$DOSE_RANKS" python experiments/l23net_analysis/run_l23net_dose_response.py \
        experiment.name="l23net_dose_smoke_${DOSE_ARM}" experiment.dir="$DOSE_RUN" hydra.run.dir="$DOSE_RUN/hydra" \
        experiment.seed=8601 experiment.debug=true experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_dose_response analysis.mode=debug analysis.arm="$DOSE_ARM" \
        analysis.require_full_network=false analysis.duration_ms=500 analysis.window_ms=100 \
        analysis.excluded_ms=100 analysis.stim_start_ms=200 analysis.stim_stop_ms=400 analysis.ramp_ms=25 \
        analysis.resource_request.mpi_ranks="$DOSE_RANKS" analysis.resource_request.ncpus="$DOSE_RANKS" analysis.resource_request.memory_gb=8 \
        env.simulation.MDD=true env.simulation.DRUG=false env.ts.apply=true \
        env.simulation.duration=500 env.simulation.obs_win_len=100 env.network.dt=0.025 \
        env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
        > "$DOSE_RUN/run.log" 2>&1 < /dev/null; then
        tail -n 75 "$DOSE_RUN/run.log"
        exit 1
    fi
done
python - "$DOSE_SMOKE_DIR" <<'PY'
from pathlib import Path
import sys
import numpy as np
from experiments.l23net_analysis.analyze_l23net_dose_response import audit_run
from experiments.l23net_analysis.dose_response_protocol import AMPLITUDES
from experiments.l23net_analysis.s1_analysis import compare_prefix

root = Path(sys.argv[1])
for ranks in (1, 2):
    sham = root / f'r{ranks}_sham'
    sham_report, _, _ = audit_run(sham)
    coupling_per_unit_field = []
    for arm in (('dose_08',) if ranks == 1 else ('dose_02', 'dose_04', 'dose_08')):
        active = root / f'r{ranks}_{arm}'
        report, _, _ = audit_run(active)
        compare_prefix(sham, active, sham_report, report)
        assert np.isclose(report['window_protocol']['max_field_v_per_m'], AMPLITUDES[arm], atol=1e-6)
        assert report['window_protocol']['tissue_coupling_observed']
        assert report['artifacts']['trace_summary']['committed_samples'] == 20000
        assert report['build']['effective_celsius_by_rank']['minimum'] == 34
        assert report['build']['effective_celsius_by_rank']['maximum'] == 34
        assigned = max(window['assigned_extracellular_peak_mV'] for window in report['windows'])
        coupling_per_unit_field.append(assigned / AMPLITUDES[arm])
        print(f'PASS {ranks} ranks / {arm}, peak assigned extracellular voltage {assigned:.6g} mV')
    assert np.allclose(coupling_per_unit_field, coupling_per_unit_field[0], rtol=0, atol=1e-12)
print('PASS serial and two-rank traces, analytic field and removal, spikes and exact matching')
PY
