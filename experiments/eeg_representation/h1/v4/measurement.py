"""Declared causal acquisition operators, with measured rather than nominal response.

A0 is the frozen legacy implementation. A1 is a box projection and A2 is a
new causal acquisition. No sample is backdated and no zero-phase filter is used.
"""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy import signal
from ..acquisition import causal_history as legacy_history, filter_kernel
from ..targets import canonical_hash
from ..artifacts import atomic_json, atomic_npz

ACQUISITIONS = ('A0', 'A1', 'A2')
STARTUP_TOLERANCE = 1e-4


def coefficient_hash(array):
    return hashlib.sha256(np.asarray(array, dtype='<f8').tobytes()).hexdigest()


@lru_cache(maxsize=8)
def _design(acquisition_id, native_dt_ms=.0625, rate_hz=500.):
    if acquisition_id not in ACQUISITIONS:
        raise ValueError(f'Unknown acquisition {acquisition_id}')
    fs = 1000 / native_dt_ms
    width = round(fs / rate_hz)
    if not np.isclose(width * rate_hz, fs):
        raise ValueError('Native/output rate must have an integer ratio')
    if acquisition_id == 'A0':
        return 'fir', filter_kernel(fs, (.5, 30.), 2049)
    if acquisition_id == 'A1':
        return 'fir', np.ones(width, dtype=np.float64) / width
    return 'sos', signal.butter(3, (4., 30.), btype='bandpass', fs=fs, output='sos')


def apply_prefix(values, acquisition_id, native_dt_ms=.0625, rate_hz=500.):
    kind, coefficients = _design(acquisition_id, native_dt_ms, rate_hz)
    x = np.asarray(values, dtype=float)
    return (signal.sosfilt(coefficients, x) if kind == 'sos'
            else signal.lfilter(coefficients, [1.], x))


@lru_cache(maxsize=8)
def _characterization(acquisition_id, native_dt_ms=.0625, rate_hz=500.):
    fs = 1000 / native_dt_ms
    kind, coefficients = _design(acquisition_id, native_dt_ms, rate_hz)
    # Dense low-frequency grid plus the complete anti-alias stop region.
    freq = np.unique(np.r_[np.linspace(0, 60, 1201),
                           np.geomspace(60.001, fs / 2, 500), 250., fs / 2])
    if kind == 'sos':
        _, response = signal.sosfreqz(coefficients, worN=freq, fs=fs)
    else:
        _, response = signal.freqz(coefficients, worN=freq, fs=fs)
    # Zero response has no defined phase. Exclude it before differentiating;
    # otherwise the arbitrary DC phase contaminates the first valid SOS delay.
    valid = np.abs(response) > 1e-8
    phase = np.full(len(freq),np.nan);delay = np.full(len(freq),np.nan)
    phase[valid] = np.unwrap(np.angle(response[valid]))
    if kind=='fir':
        # Both qualified FIRs are symmetric; the off-zero delay is exact even
        # when transition/stop-band phase wraps at a sign change of the response.
        delay[valid]=(len(coefficients)-1)*native_dt_ms/2
    elif valid.sum()>=2:
        delay[valid]=-np.gradient(phase[valid],2*np.pi*freq[valid])*1000
    # Phase and delay are unidentifiable at response zeros. Store null, not NaN.
    phase_list = [float(a) if good else None for a, good in zip(phase, valid)]
    delay_list = [float(a) if good else None for a, good in zip(delay, valid)]
    impulse = np.zeros(round(20 * fs)); impulse[0] = 1.
    ir = apply_prefix(impulse, acquisition_id, native_dt_ms, rate_hz)
    step = apply_prefix(np.ones_like(impulse), acquisition_id, native_dt_ms, rate_hz)
    dc = float(abs(response[0]))
    residue = abs(step - float(response[0].real))
    # Infinite-support IIR: report an explicit finite-horizon tolerance diagnostic.
    tail = np.cumsum(np.abs(ir)[::-1])[::-1]
    norm = float(tail[0])
    bad_step = np.flatnonzero(residue > STARTUP_TOLERANCE)
    bad_tail = np.flatnonzero(tail > STARTUP_TOLERANCE * max(norm, 1e-30))
    settle = (int(bad_step[-1]) + 1) * native_dt_ms if len(bad_step) else 0.
    memory = (int(bad_tail[-1]) + 1) * native_dt_ms if len(bad_tail) else 0.
    passband = (freq >= 6) & (freq <= 24)
    stopband = freq >= 250
    gain = abs(response)
    peak = float(gain.max())
    pass_atten = float(-20 * np.log10(gain[passband].min() / peak))
    stop_atten = float(-20 * np.log10(max(gain[stopband].max() / peak, 1e-300)))
    result = {
        'acquisition_id': acquisition_id, 'coefficient_kind': kind,
        'coefficient_sha256': coefficient_hash(coefficients),
        'coefficients': coefficients.tolist(), 'native_dt_ms': native_dt_ms,
        'native_rate_hz': fs, 'sample_rate_hz': rate_hz,
        'dc_gain': dc, 'peak_gain': peak,
        'worst_6_24_hz_attenuation_db_relative_peak': pass_atten,
        'minimum_at_or_above_250_hz_attenuation_db_relative_peak': stop_atten,
        'frequency_hz': freq.tolist(), 'gain': gain.tolist(),
        'phase_rad': phase_list, 'group_delay_ms_by_frequency': delay_list,
        'availability': 'unshifted completed-sample right edge; no backdating',
        'startup': 'zero filter state before first native sample; filter the full native prefix once',
        'unit_step_startup_tolerance': STARTUP_TOLERANCE,
        'unit_step_settling_ms': settle,
        'effective_impulse_l1_memory_ms': memory,
        'effective_memory_relative_l1_tolerance': STARTUP_TOLERANCE,
        'memory_diagnostic_horizon_ms': 20000.,
        'residue_at_diagnostic_horizon': float(residue[-1]),
        'impulse_tail_at_diagnostic_horizon': float(abs(ir[-1])),
        'time_support_ms': None if kind == 'sos' else (len(coefficients)-1)*native_dt_ms,
        'support_kind': 'infinite IIR; tolerance-based effective memory is not finite support' if kind == 'sos' else 'finite FIR',
        'nominal_design': {'A0': 'frozen 2049-tap Hamming nominal .5--30 Hz FIR; high-pass not realized',
                           'A1': 'completed nonoverlapping native 2-ms box means; not ideal anti-aliasing',
                           'A2': 'third-order Butterworth 4--30 Hz bandpass (six poles), causal SOS'}[acquisition_id],
    }
    result['measurement_criteria'] = {
        'dc_gain_le_1e_6': dc <= 1e-6,
        'gain_6_24_within_3db_of_peak': pass_atten <= 3.,
        'attenuation_ge250_at_least_40db': stop_atten >= 40.,
        'unit_step_settles_in_diagnostic_horizon': bool(residue[-1] <= STARTUP_TOLERANCE),
    }
    result['design_measurement_pass'] = all(result['measurement_criteria'].values()) if acquisition_id == 'A2' else None
    result['operator_hash'] = canonical_hash({k: result[k] for k in (
        'acquisition_id','coefficient_sha256','native_dt_ms','sample_rate_hz','availability','startup')})
    return result


def metadata(acquisition_id, native_dt_ms=.0625, rate_hz=500.):
    # Return a copy so callers cannot mutate a cached qualification.
    return json.loads(json.dumps(_characterization(acquisition_id, native_dt_ms, rate_hz)))


def causal_history(eeg_v, time_ms, *, cutoff_ms, history_s=4., acquisition_id='A0', rate_hz=500.):
    x = np.asarray(eeg_v, dtype=float).reshape(-1)
    t = np.asarray(time_ms, dtype=float)
    if x.shape != t.shape or len(t) < 2:
        raise ValueError('Expected finite native single-channel EEG and matching timestamps')
    prefix = t <= cutoff_ms + 1e-9
    tp, xp = t[prefix], x[prefix]
    if len(tp)<2 or not np.isfinite(xp).all():
        raise ValueError('Expected finite observed prefix')
    dt = float(tp[1] - tp[0])
    if not np.allclose(np.diff(tp), dt, rtol=0, atol=1e-7):
        raise ValueError('Observed native timestamps must be uniform')
    if not len(tp) or not np.isclose(tp[-1], cutoff_ms, rtol=0, atol=1e-7):
        raise ValueError('Unavailable cutoff')
    meta = metadata(acquisition_id, dt, rate_hz)
    count = round(history_s * rate_hz)
    desired = cutoff_ms - history_s * 1000 + np.arange(1, count+1) * (1000/rate_hz)
    idx = np.searchsorted(tp, desired - 1e-8)
    if np.any(idx >= len(tp)) or not np.allclose(tp[idx], desired, rtol=0, atol=1e-7):
        raise ValueError('History grid is unavailable')
    if acquisition_id == 'A0':
        out, rel, _ = legacy_history(x, t, cutoff_ms=cutoff_ms, history_s=history_s, rate_hz=rate_hz)
    elif acquisition_id == 'A1':
        # Explicit block means, so completed bins agree bit-for-bit with native targets.
        width = round((1000/rate_hz) / dt)
        start = idx[0] - width + 1
        if start < 0:
            raise ValueError('Completed bin has insufficient native support')
        out = xp[start:idx[-1]+1].reshape(count, width).mean(axis=1)
        rel = desired - cutoff_ms
    else:
        if not meta['design_measurement_pass']:
            raise ValueError('A2 measurement qualification failed; no outcome-based redesign permitted')
        elapsed = desired[0] - tp[0]
        if elapsed < meta['unit_step_settling_ms']:
            raise ValueError(f'A2 retained input precedes measured settling ({elapsed:g} < {meta["unit_step_settling_ms"]:g} ms); extend burn-in')
        out = apply_prefix(xp, acquisition_id, dt, rate_hz)[idx]
        rel = desired - cutoff_ms
    meta['retained_first_absolute_ms'] = float(desired[0])
    meta['retained_last_absolute_ms'] = float(desired[-1])
    meta['retained_startup_elapsed_ms'] = float(desired[0]-tp[0])
    meta['retained_startup_qualified'] = bool(desired[0]-tp[0] >= meta['unit_step_settling_ms'])
    return out, rel, meta


def same_sensor_future_decomposition(eeg_v, time_ms, *, cutoff_ms, acquisition_id, native_dt_ms=.0625):
    """Split a filtered future into inherited past response and post-cutoff input.

    Outputs retain every native timestamp. The sum is the full acquired signal;
    neither the inherited component nor IIR memory is called neural prediction.
    """
    x = np.asarray(eeg_v, dtype=float).reshape(-1)
    t = np.asarray(time_ms, dtype=float)
    if x.shape != t.shape:
        raise ValueError('Mismatched native arrays')
    past = np.where(t <= cutoff_ms + 1e-9, x, 0.)
    future = np.where(t > cutoff_ms + 1e-9, x, 0.)
    inherited = apply_prefix(past, acquisition_id, native_dt_ms)
    post = apply_prefix(future, acquisition_id, native_dt_ms)
    whole = apply_prefix(x, acquisition_id, native_dt_ms)
    return {'time_ms': t, 'whole': whole, 'inherited_past': inherited,
            'post_cutoff_contribution': post,
            'maximum_additivity_error': float(np.max(abs(whole-inherited-post))),
            'interpretation': 'same-sensor future includes inherited filter state; not purely post-cutoff neural content'}


def qualify(root=None):
    """Measurement-only qualification, independent of model outcomes."""
    result = {'status': 'complete', 'design_selected_without_model_outcomes': True,
              'conditions': {a: metadata(a) for a in ACQUISITIONS}}
    result['passed'] = result['conditions']['A2']['design_measurement_pass']
    if root is not None:
        out = Path(root)/'measurement'
        atomic_json(out/'qualification.json', result)
        atomic_npz(out/'coefficients.npz', **{a: _design(a)[1] for a in ACQUISITIONS})
    return result
