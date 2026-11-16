"""Isolated integration-step adapter; the historical simulator remains unchanged.

The continuous exogenous schedules and 1-ms phase grid do not depend on solver dt.
Fine integrations replay the full prefix and retain the same 16-kHz right-edge
output samples. Only separate serial worker processes may call this adapter.
"""
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import os
import numpy as np

OUTPUT_DT_MS = .0625
ALLOWED_DT_MS = (.0625, .03125, .015625)


def common_output_grid(trial, integration_dt_ms):
    """Sample already-integrated output without changing Fourier normalization."""
    dt = float(integration_dt_ms)
    if dt not in ALLOWED_DT_MS:
        raise ValueError('Unregistered integration dt')
    stride = round(OUTPUT_DT_MS / dt)
    time = np.asarray(trial['time_ms'], dtype=np.float64)
    expected = np.arange(1, len(time)+1, dtype=np.float64)*dt
    if not np.allclose(time, expected, rtol=0, atol=1e-7):
        raise ValueError('Integration recorder must use uniform (0,stop] timestamps')
    if len(time) % stride:
        raise ValueError('Episode ends off the common output grid')
    result = dict(trial)
    for key in ('time_ms', 'eeg_v', 'dipole_na_um'):
        result[key] = np.asarray(trial[key], dtype=np.float64)[stride-1::stride].copy()
    # Exact common grid avoids floating accumulation in downstream support tests.
    result['time_ms'] = np.arange(1, len(result['time_ms'])+1, dtype=np.float64)*OUTPUT_DT_MS
    result['metadata'] = dict(trial['metadata'], integration_dt_ms=dt,
        integration_sample_count=len(time), native_dt_ms=OUTPUT_DT_MS,
        native_sample_count=len(result['time_ms']), output_grid_hz=16000.,
        numerical_adapter='v6 coupled exogenous full-prefix replay; no checkpoint restoration',
        output_sampling='retain every integer solver stride on the common (0,stop] 16-kHz grid; no interpolation')
    return result


def coupling_fingerprint(trial):
    from ...simulator import array_hash
    p = trial['privileged']
    keys = ('external_event_times_ms', 'external_event_offsets', 'phase_time_ms', 'phase_rad',
            'structure_geometry', 'synapse_parameters', 'recurrent_netcons', 'external_netcons')
    missing = [k for k in keys if k not in p]
    if missing:
        raise ValueError('Coupled forcing audit absent: '+str(missing))
    return {k:array_hash(p[k]) for k in keys}


def verify_coupling(trials):
    """Integration trajectories may differ; forcing, structure and output grid may not."""
    if len(trials)<2:
        raise ValueError('At least two integration levels required')
    reference=coupling_fingerprint(trials[0])
    for trial in trials[1:]:
        if coupling_fingerprint(trial)!=reference:
            raise ValueError('Integration comparison rerolled exogenous input or changed physical structure')
        if not np.array_equal(trial['time_ms'],trials[0]['time_ms']):
            raise ValueError('Numerical comparison lacks a common physical output grid')
    return dict(passed=True, levels=len(trials), forcing_hashes=reference,
                trajectory_identity_required=False, output_grid_hz=16000.)


def generate_trial(config, spec):
    from ... import simulator as legacy
    dt=float(config['simulation'].get('native_dt_ms',OUTPUT_DT_MS))
    if dt not in ALLOWED_DT_MS:
        raise ValueError('Integration dt must be .0625, .03125 or .015625 ms')
    root=Path(__file__).resolve().parents[4]
    with legacy._LOCK:
        previous=os.environ.get('MAIN_PATH');os.environ['MAIN_PATH']=str(root)
        try:
            # Imports happen only in the NEURON path, after MAIN_PATH is set.
            from env.models.neuron import env_online
            original_validate=legacy._validate
            original_env=env_online.OnlineNeuronEnv
            def validated(c,s):
                check=deepcopy(c);check['simulation']['native_dt_ms']=OUTPUT_DT_MS
                _,_,total,phase_dt=original_validate(check,s)
                if not np.isclose(total/dt,round(total/dt),rtol=0,atol=1e-8):
                    raise ValueError('Episode end is not aligned to integration step')
                return c['simulation'],dt,total,phase_dt
            def environment(args,*a,**kw):
                args.env.network.dt=dt
                # networkParameters.dt and cellParameters.dt interpolate this field.
                return original_env(args,*a,**kw)
            with patch.object(legacy,'_validate',validated),patch.object(env_online,'OnlineNeuronEnv',environment):
                trial=legacy.generate_trial(config,spec)
            trial=common_output_grid(trial,dt)
            trial['metadata']['coupled_forcing_hashes']=coupling_fingerprint(trial)
            return trial
        finally:
            if previous is None:os.environ.pop('MAIN_PATH',None)
            else:os.environ['MAIN_PATH']=previous


def coupled_adapter_control(destination=None):
    """Tiny actual-NEURON adapter audit, not the 768-episode qualification panel."""
    from ...schema import DEFAULTS
    from ...simulator import generate_trial as original_generate
    from ...forks import validate_fork_prefixes
    from ...storage import atomic_json
    config=deepcopy(DEFAULTS)
    config['simulation'].update(native_dt_ms=.0625,duration_s=.055,burn_in_s=.005,
        phase_grid_dt_ms=1.,population_e=2,population_i=1,external_synapses_per_cell=4)
    spec=dict(structure_seed=1,structure_id='1',trial_seed=42,phase_seed=10,future_seed=3,
        future_phase_seed=5,future_start_ms=30.,regime=dict(name='Ralpha',frequency_hz=10.,modulation_depth=.04))
    # Original path has its own scoped MAIN_PATH handling.
    original=original_generate(config,spec);trials=[]
    for dt in ALLOWED_DT_MS:
        q=deepcopy(config);q['simulation']['native_dt_ms']=dt
        trials.append(generate_trial(q,spec))
    if not np.array_equal(original['eeg_v'],trials[0]['eeg_v']):
        raise AssertionError('Base-dt adapter altered frozen generator EEG')
    if not np.array_equal(original['dipole_na_um'],trials[0]['dipole_na_um']):
        raise AssertionError('Base-dt adapter altered frozen generator dipole')
    coupled=verify_coupling(trials)
    fine=deepcopy(config);fine['simulation']['native_dt_ms']=ALLOWED_DT_MS[-1]
    other=generate_trial(fine,{**spec,'future_seed':4,'future_phase_seed':6})
    fork=validate_fork_prefixes([trials[-1],other],30.)
    result=dict(status='complete',passed=True,kind='small actual NEURON adapter engineering control',
        circuits=1,population_e=2,population_i=1,episode_ms=60.,episodes=5,
        base_dt_bitwise_matches_original=True,integration_dts_ms=list(ALLOWED_DT_MS),
        common_samples=len(trials[0]['time_ms']),coupled_forcing=coupled,fine_dt_fork=fork,
        distinction='This verifies routing, common schedules/output support and fork replay. It does not establish full-network dt convergence.')
    if destination is not None:atomic_json(Path(destination),result)
    return result
