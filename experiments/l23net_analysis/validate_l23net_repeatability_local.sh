#!/bin/bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
source /home/chirath/Documents/depression-simulator/bin/activate
unset DISPLAY
export MAIN_PATH="$PWD" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
mkdir -p results
SMOKE_DIR=$(mktemp -d "$PWD/results/l23net_repeatability_local_XXXXXX")
printf 'SMOKE_DIR=%s\n' "$SMOKE_DIR"

for CASE in '2 legacy sham' '2 a sham' '2 a dose_02' '2 a dose_04' '2 b sham' '2 b dose_04' '1 a dose_04'; do
    read -r RANKS HISTORY ARM <<< "$CASE"
    RUN_DIR="$SMOKE_DIR/r${RANKS}_${HISTORY}_${ARM}"
    mkdir -p "$RUN_DIR"
    HISTORY_OVERRIDE=()
    case "$HISTORY" in
        a) HISTORY_OVERRIDE=(++env.online.l23net_history_seed=3346111) ;;
        b) HISTORY_OVERRIDE=(++env.online.l23net_history_seed=3346112) ;;
    esac
    if ! mpirun -np "$RANKS" python experiments/l23net_analysis/run_l23net_repeatability.py \
        experiment.name="l23net_repeatability_smoke_${ARM}" experiment.dir="$RUN_DIR" hydra.run.dir="$RUN_DIR/hydra" \
        experiment.seed=8601 experiment.debug=true experiment.plot=false experiment.tqdm=false \
        env=hl23net analysis=l23net_dose_response analysis.mode=debug analysis.arm="$ARM" \
        analysis.require_full_network=false analysis.duration_ms=500 analysis.window_ms=100 \
        analysis.excluded_ms=100 analysis.stim_start_ms=200 analysis.stim_stop_ms=400 analysis.ramp_ms=25 \
        analysis.resource_request.mpi_ranks="$RANKS" analysis.resource_request.ncpus="$RANKS" analysis.resource_request.memory_gb=8 \
        env.simulation.MDD=true env.simulation.DRUG=false env.ts.apply=true \
        env.simulation.duration=500 env.simulation.obs_win_len=100 env.network.dt=0.025 \
        env.debug_n_neurons.PYR=2 env.debug_n_neurons.SST=1 env.debug_n_neurons.PV=1 env.debug_n_neurons.VIP=1 \
        "${HISTORY_OVERRIDE[@]}" > "$RUN_DIR/run.log" 2>&1 < /dev/null; then
        tail -n 75 "$RUN_DIR/run.log"
        exit 1
    fi
done

python - "$SMOKE_DIR" <<'PY'
from pathlib import Path
import sys

import numpy as np
from experiments.l23net_analysis.analyze_l23net_dose_response import audit_run
from experiments.l23net_analysis.analyze_l23net_repeatability import prestimulation_hash
from experiments.l23net_analysis.dose_response_protocol import AMPLITUDES
from experiments.l23net_analysis.s1_analysis import compare_prefix

root = Path(sys.argv[1])
reports = {path.name: audit_run(path)[0] for path in root.iterdir()}
for name, report in reports.items():
    assert report['artifacts']['trace_summary']['committed_samples'] == 20000, name
    assert report['build']['effective_celsius_by_rank']['minimum'] == 34, name
    assert report['build']['effective_celsius_by_rank']['maximum'] == 34, name
    assert prestimulation_hash(root / name / 'l23net_s1_trace.h5', report) == (
        report['window_protocol']['prestimulation_eeg_dipole_sha256']), name
    if name.endswith('sham'):
        continue
    arm = name.split('_', 2)[2]
    protocol = report['window_protocol']
    assert protocol['decision']['delivered_arm'] == arm, (name, protocol['decision'])
    assert np.isclose(protocol['max_field_v_per_m'], AMPLITUDES[arm], atol=1e-6), name
    assert protocol['tissue_coupling_observed'], name

for history in ('a', 'b'):
    sham_name = f'r2_{history}_sham'
    sham = reports[sham_name]
    for name, active in reports.items():
        if not name.startswith(f'r2_{history}_dose_'):
            continue
        compare_prefix(root / sham_name, root / name, sham, active)
        assert sham['build_audit']['background_history'] == active['build_audit']['background_history'], name
    print(f'PASS two-rank history {history}: matching stream, EEG, dipole and spike prefixes across arms')

legacy, a, b = (reports[f'r2_{history}_sham'] for history in ('legacy', 'a', 'b'))
assert a['structure'] == b['structure'] == legacy['structure']
assert len({r['build_audit']['invariant_sha256'] for r in (legacy, a, b)}) == 1
assert len({r['window_protocol']['prestimulation_eeg_dipole_sha256'] for r in (legacy, a, b)}) == 3
assert a['build_audit']['background_history']['stream_sha256'] != b['build_audit']['background_history']['stream_sha256']
assert 'background_history' not in legacy['build_audit']
print('PASS distinct prestimulation histories on identical structure and OU parameters; active delivery in serial and MPI')
PY
