"""Serializable physical-time targets shared by fitting, forks and scoring.

Primary waveform is a declared future-block box-average projection, not ideal
anti-aliasing and not the original native waveform. Native samples use (a,b].
No target filtering changes the availability time of an input sample.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import hashlib
import json
import numpy as np
from scipy import signal

TARGET_VERSION = 'h1-target-2.0'
OPERATORS = ('native_waveform', 'bin_average_waveform', 'rms', 'log_total_power',
             'log_bandpower', 'full_complex_fft', 'magnitude_spectrum')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def aligned(value, step, name):
    if not np.isfinite(value) or not np.isclose(value / step, round(value / step), rtol=0, atol=1e-8):
        raise ValueError(f'{name}={value} must align exactly to {step} ms; no rounding')


@dataclass(frozen=True)
class TargetSpec:
    operator: str = 'bin_average_waveform'
    cutoff_ms: float = 6000.
    history_s: float = 4.
    lead_ms: float = 10.
    duration_ms: float = 128.
    output_grid_hz: float = 500.
    native_dt_ms: float = .0625
    protocol_name: str = 'native_future_common_box_projection_v2'
    source_signal: str = 'native_clean_neural_eeg'
    units: str = 'V'
    input_acquisition: dict = field(default_factory=dict)
    normalization_id: str = 'physical-unfitted'
    context_id: str = 'panel'
    population_id: str = 'unspecified'
    split_id: str = 'evaluation'
    nuisance: dict = field(default_factory=dict)
    code_version: str = TARGET_VERSION
    passband_hz: tuple = (4., 30.)
    detrend: str = 'constant'
    window: str = 'hann'
    power_floor_v2: float = 1e-40
    support_convention: str = '(cutoff+lead,cutoff+lead+duration]'
    output_projection: str = 'nonoverlapping native-sample box means; not ideal anti-aliasing'

    def __post_init__(self):
        if self.operator=='native_waveform':
            object.__setattr__(self,'output_projection','identity native samples at declared right-edge timestamps')
            if self.protocol_name=='native_future_common_box_projection_v2':
                object.__setattr__(self,'protocol_name','native_future_waveform_v2')
        if self.operator not in OPERATORS:
            raise ValueError(f'Unsupported target operator: {self.operator}')
        if self.history_s <= 0 or self.cutoff_ms < 1000*self.history_s or self.lead_ms <= 0 or self.duration_ms <= 0:
            raise ValueError('Invalid history/cutoff/strictly-positive future support')
        if self.native_dt_ms <= 0 or self.output_grid_hz <= 0:
            raise ValueError('Invalid target grid')
        for name in ('cutoff_ms', 'lead_ms', 'duration_ms'):
            aligned(getattr(self, name), self.native_dt_ms, name)
        if self.operator != 'native_waveform':
            step = 1000/self.output_grid_hz
            aligned(step, self.native_dt_ms, 'output step')
            aligned(self.lead_ms, step, 'lead')
            aligned(self.duration_ms, step, 'duration')
        if self.source_signal != 'native_clean_neural_eeg':
            raise ValueError('This physical-target registry accepts native clean EEG only')
        if self.support_convention != '(cutoff+lead,cutoff+lead+duration]':
            raise ValueError('Unknown target support convention')
        if self.detrend != 'constant' or self.window != 'hann':
            raise ValueError('Only the explicitly qualified constant-detrend/Hann spectral operator is supported')
        if not 0 < self.passband_hz[0] < self.passband_hz[1] < self.output_grid_hz/2:
            raise ValueError('Spectral passband must lie strictly inside the target Nyquist interval')

    @property
    def dimension(self):
        n = round(self.duration_ms / self.native_dt_ms) if self.operator == 'native_waveform' else round(self.duration_ms*self.output_grid_hz/1000)
        if self.operator in ('rms', 'log_total_power', 'log_bandpower'): return 1
        return 2*n if self.operator == 'full_complex_fft' else n

    def to_dict(self):
        out = asdict(self)
        out['target_dimension'] = self.dimension
        out['physical_support_ms'] = [self.cutoff_ms+self.lead_ms, self.cutoff_ms+self.lead_ms+self.duration_ms]
        out['loss'] = 'mean squared error per real output coordinate'
        if self.operator == 'full_complex_fft':
            out['fft_packing'] = 'sqrt(2)*concat(real,imag) of full orthonormal FFT; per-coordinate MSE is preserved'
        return out

    @classmethod
    def from_dict(cls, value):
        return cls(**{k:v for k,v in value.items() if k in cls.__dataclass_fields__})

    @property
    def fingerprint(self): return canonical_hash(self.to_dict())

    @property
    def physical_fingerprint(self):
        keys = ('operator','lead_ms','duration_ms','output_grid_hz','native_dt_ms',
                'source_signal','units','passband_hz','detrend','window','power_floor_v2',
                'support_convention','output_projection','code_version')
        return canonical_hash({k:getattr(self,k) for k in keys})


def require_same_target(a, b):
    a = a.fingerprint if isinstance(a, TargetSpec) else a
    b = b.fingerprint if isinstance(b, TargetSpec) else b
    if a != b: raise ValueError('Incompatible target/context/acquisition/normalization fingerprints; risk subtraction rejected')


def operator_from_projected(waveform, spec):
    """Apply the registered operator to physical box-projected future samples."""
    w = np.asarray(waveform, dtype=np.float64)
    if w.ndim != 1 or not np.isfinite(w).all(): raise ValueError('Expected finite single-channel target')
    if spec.operator in ('native_waveform', 'bin_average_waveform'): return w.copy()
    power = float(np.mean(w*w))
    if spec.operator == 'rms': return np.array([np.sqrt(power)])
    if spec.operator == 'log_total_power': return np.array([np.log(max(power,spec.power_floor_v2))])
    if spec.operator == 'log_bandpower':
        freq, psd = signal.periodogram(w,fs=spec.output_grid_hz,window=spec.window,detrend=spec.detrend,scaling='density')
        keep=(freq>=spec.passband_hz[0]) & (freq<=spec.passband_hz[1])
        if not np.any(keep): raise ValueError('Target duration has no bins in the declared spectral band')
        band = float(np.sum(psd[keep]) * spec.output_grid_hz/len(w))
        return np.array([np.log(max(band,spec.power_floor_v2))])
    f=np.fft.fft(w,norm='ortho')
    if spec.operator == 'full_complex_fft': return np.sqrt(2)*np.concatenate([f.real,f.imag])
    if spec.operator == 'magnitude_spectrum': return np.abs(f)
    raise AssertionError(spec.operator)


def extract_target(eeg_v, time_ms, spec):
    x=np.asarray(eeg_v,dtype=np.float64).reshape(-1)
    t=np.asarray(time_ms,dtype=np.float64)
    if x.shape != t.shape or not np.isfinite(x).all() or np.any(np.diff(t)<=0):
        raise ValueError('Invalid native target arrays')
    start=spec.cutoff_ms+spec.lead_ms
    n=round(spec.duration_ms/spec.native_dt_ms)
    idx=int(np.searchsorted(t,start+spec.native_dt_ms/2))
    expected=start+np.arange(1,n+1)*spec.native_dt_ms
    if idx+n>len(t) or not np.allclose(t[idx:idx+n],expected,rtol=0,atol=1e-8):
        raise ValueError('Native target support is absent or misaligned')
    native=x[idx:idx+n]
    if spec.operator=='native_waveform': return native.copy()
    width=round(1000/spec.output_grid_hz/spec.native_dt_ms)
    return operator_from_projected(native.reshape(-1,width).mean(axis=1),spec)


def physical_risk(standardized_risk, scale):
    if not np.isfinite(scale) or scale<=0: raise ValueError('Positive affine scale required')
    return np.asarray(standardized_risk)*scale**2
