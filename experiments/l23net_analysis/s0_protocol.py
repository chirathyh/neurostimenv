"""Prospective full-spectrum measurement contract (no NEURON dependency)."""
import copy
import json
from pathlib import Path

import numpy as np
from scipy import signal

from experiments.l23net_analysis.replay_validation import canonical_json_sha256
from experiments.l23net_analysis.r1_protocol import sha256

BANDS = {"theta": (4., 8.), "alpha": (8., 12.), "low_beta": (12., 16.)}
FIXED = {"theta": 6., "alpha": 10., "low_beta": 14.}
PROTOCOL = {
    "version": 1, "bands_hz": BANDS, "excluded_ms": 8000,
    "baseline_ms": 20000, "stimulation_ms": 22000, "ramp_ms": 1000,
    "washout_ms": 10000, "duration_ms": 60000, "outcome_ms": [29000, 49000],
    "native_dt_ms": .025, "analysis_fs_hz": 250., "lowpass_hz": 80.,
    "lowpass_order": 8, "welch_segment_s": 4., "welch_overlap": .5,
    "peak_prominence_db": 3., "peak_half_difference_hz": .5,
    "frequency_minimum_coverage": .8, "phase_history_s": 2.,
    "phase_scoring_halfwidth_s": 1., "phase_horizons_s": [1., 4.],
    "phase_audit_boundaries_s": [16., 20., 23.],
    "phase_fit_minimum_r2": .1, "phase_maximum_mae_rad": float(np.pi/4),
    "phenotype_minimum_sensitivity": .8, "phenotype_minimum_specificity": .8,
    "amplitude_v_per_m": .4, "fixed_frequencies_hz": FIXED,
    "discovery_seeds": [8401, 8402, 8403], "qualification_seed": 8451,
    "screening_rule": "equal-weight signed standardized three-band score above empirical calibration 95th percentile",
    "outcome_rule": "RMS standardized log10 theta/alpha/low-beta distance; positive benefit=sham distance minus active distance",
    "phase_rule": "One-time phase requires a qualified forecast horizon spanning the entire active block; otherwise blocked.",
}


class CausalEEG:
    """Continuous anti-alias filtering; decimation aligned to (start, stop]."""
    def __init__(self, dt_ms):
        self.fs = 1000./float(dt_ms)
        self.factor = int(round(self.fs/PROTOCOL['analysis_fs_hz']))
        if self.factor < 1 or not np.isclose(self.fs/self.factor, PROTOCOL['analysis_fs_hz']):
            raise ValueError('Native sample rate must be an integer multiple of 250 Hz')
        self.sos = signal.butter(PROTOCOL['lowpass_order'], PROTOCOL['lowpass_hz'], fs=self.fs, output='sos')
        self.state = np.zeros((len(self.sos), 2))
        self.samples = 0

    def append(self, values):
        values = np.asarray(values, float).reshape(-1)
        if not np.isfinite(values).all():
            raise ValueError('Nonfinite EEG')
        y, self.state = signal.sosfilt(self.sos, values, zi=self.state)
        indices = np.arange(self.samples+1, self.samples+1+len(y))
        self.samples += len(y)
        take = indices % self.factor == 0
        return indices[take]/self.fs, y[take]


def spectrum(values):
    x = np.asarray(values, float)
    n = int(PROTOCOL['welch_segment_s']*PROTOCOL['analysis_fs_hz'])
    if x.ndim != 1 or len(x) < n or not np.isfinite(x).all():
        raise ValueError('PSD needs at least four finite seconds')
    return signal.welch(x, fs=PROTOCOL['analysis_fs_hz'], window='hann', nperseg=n,
                        noverlap=n//2, detrend='constant', scaling='density', average='mean')


def log_powers(values, exclude_hz=None):
    f, p = spectrum(values)
    powers = []
    for lo, hi in BANDS.values():
        mask = (f >= lo) & (f <= hi)
        ff, pp = f[mask], p[mask]
        # Excluded quadrature removes each affected trapezoid, never bridges the gap.
        keep = np.ones(len(ff)-1, bool)
        if exclude_hz is not None:
            a, b = exclude_hz
            keep &= ~((ff[:-1] < b) & (ff[1:] > a))
        power = np.sum((np.diff(ff)*(pp[:-1]+pp[1:])/2)[keep])
        if power <= 0:
            raise ValueError('Nonpositive integrated band power')
        powers.append(np.log10(power))
    return np.asarray(powers)


def epoch(times, values, start, stop):
    times, values = np.asarray(times), np.asarray(values)
    x = values[(times > start+1e-9) & (times <= stop+1e-9)]
    if len(x) != round((stop-start)*PROTOCOL['analysis_fs_hz']):
        raise ValueError(f'Incomplete analysis epoch ({start}, {stop}]')
    return x


def fit_target(log_values):
    x = np.asarray(log_values, float)
    sd = x.std(axis=0, ddof=1)
    floor = max(.01, .1*float(np.median(sd)))
    target = {'mean_log10': x.mean(axis=0).tolist(), 'scale_log10': np.maximum(sd, floor).tolist(),
              'n_structures': len(x), 'scale_floor': floor}
    scores = ((x-target['mean_log10'])/target['scale_log10']).mean(axis=1)
    target['screen_cutoff'] = float(np.quantile(scores, .95))
    target['screen_caveat'] = 'Empirical calibration cutoff, not a finite-sample 95% specificity guarantee.'
    return target


def scores(log_values, target):
    z = (np.asarray(log_values)-target['mean_log10'])/target['scale_log10']
    return {'signed_score': float(np.mean(z)), 'distance': float(np.sqrt(np.mean(z*z))),
            'eligible': bool(np.mean(z) > target['screen_cutoff'])}


def carrier(values, band):
    """Local peak, not an unconditional argmax on a descending PSD."""
    f, p = spectrum(values)
    db = 10*np.log10(np.maximum(p, np.finfo(float).tiny))
    # Smooth three adjacent 0.25-Hz bins, without zero-padding resolution claims.
    smooth = np.convolve(db, np.ones(3)/3, mode='same')
    indices, props = signal.find_peaks(smooth, prominence=PROTOCOL['peak_prominence_db'])
    lo, hi = BANDS[band]
    valid = [j for j, i in enumerate(indices) if lo < f[i] < hi]
    if not valid:
        return {'accepted': False, 'frequency_hz': None, 'prominence_db': 0.}
    j = max(valid, key=lambda j: props['prominences'][j])
    return {'accepted': True, 'frequency_hz': float(f[indices[j]]),
            'prominence_db': float(props['prominences'][j])}


def stable_carrier(values, band):
    full = carrier(values, band)
    half = len(values)//2
    a, b = carrier(values[:half], band), carrier(values[half:], band)
    accepted = full['accepted'] and a['accepted'] and b['accepted']
    accepted = accepted and max(abs(a['frequency_hz']-b['frequency_hz']),
                               abs(a['frequency_hz']-full['frequency_hz']),
                               abs(b['frequency_hz']-full['frequency_hz'])) <= PROTOCOL['peak_half_difference_hz']
    return {**full, 'accepted': bool(accepted), 'halves': [a, b]}


def phase_fit(times, values, frequency_hz, at_s):
    """OLS cosine phase at at_s; input data, never the fit, defines causality."""
    x = np.asarray(values, float)
    t = np.asarray(times, float)-at_s
    if len(x) < 4 or x.std() == 0:
        return {'phase_rad': 0., 'r2': 0.}
    # Scaling avoids precision loss with ~1e-9 V EEG and large absolute times.
    y = x/x.std()
    angle = 2*np.pi*frequency_hz*t
    nuisance = np.column_stack([np.ones(len(t)), t])
    design = np.column_stack([np.cos(angle), np.sin(angle), nuisance])
    coef = np.linalg.lstsq(design, y, rcond=None)[0]
    null = nuisance @ np.linalg.lstsq(nuisance, y, rcond=None)[0]
    denom = np.sum((y-null)**2)
    r2 = max(0., 1.-np.sum((y-design@coef)**2)/denom) if denom > 0 else 0.
    return {'phase_rad': float(np.arctan2(-coef[1], coef[0])), 'r2': float(r2)}


def wrap_phase(x):
    return np.angle(np.exp(1j*x))


def phase_audit(times, values, band):
    rows = []
    for boundary in PROTOCOL['phase_audit_boundaries_s']:
        past = epoch(times, values, 8, boundary)
        pick = stable_carrier(past, band)
        if not pick['accepted']:
            rows.append({'boundary_s': boundary, 'accepted': False, 'reason': 'carrier'})
            continue
        f = pick['frequency_hz']
        selection = (times > boundary-2+1e-9) & (times <= boundary+1e-9)
        fitted = phase_fit(times[selection], values[selection], f, boundary)
        for horizon in PROTOCOL['phase_horizons_s']:
            center = boundary+horizon
            future = (times > center-1+1e-9) & (times <= center+1+1e-9)
            reference = phase_fit(times[future], values[future], f, center)
            error = float(wrap_phase(fitted['phase_rad']+2*np.pi*f*horizon-reference['phase_rad']))
            rows.append({'boundary_s': boundary, 'horizon_s': horizon, 'frequency_hz': f,
                         'accepted': fitted['r2'] >= PROTOCOL['phase_fit_minimum_r2'],
                         'error_rad': error, 'fit_r2': fitted['r2'], 'scoring_r2': reference['r2']})
    return rows


def load_gate(path):
    """Self-contained frozen S0 bundle; no laptop-specific upstream paths needed."""
    bundle = json.loads(Path(path).read_text())
    payload = copy.deepcopy(bundle)
    digest = payload.pop('sha256')
    if canonical_json_sha256(payload) != digest:
        raise ValueError('S0 bundle content hash mismatch')
    if canonical_json_sha256(bundle['protocol']) != canonical_json_sha256(PROTOCOL):
        raise ValueError('S0 protocol differs from this runner')
    if not bundle['technical_passed'] or not bundle['gates']['full_spectrum_phenotype']:
        raise ValueError('S0 full-spectrum phenotype gate failed')
    return bundle
