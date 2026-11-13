"""Explicit physical targets for the v4 H1--H4 campaign.

T3 is a one-second, future-only block projection, available after the complete
block; its 54 coordinates are frequency/components, not samples in time.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import numpy as np
from ..targets import canonical_hash, aligned

FAMILIES = ('T0','T1','T2_mean_square','T2_ac_variance','T2_log_mean_square','T2_log_ac_variance','T3')


@dataclass(frozen=True)
class TargetSpec:
    family: str = 'T0'
    lead_ms: float = 10.
    duration_ms: float = 128.
    cutoff_ms: float = 6000.
    native_dt_ms: float = .0625
    bin_ms: float = 2.
    log_reference_v2: float = 1.
    log_floor_v2: float = 1e-40

    def __post_init__(self):
        if self.family not in FAMILIES:
            raise ValueError('Unknown target family')
        for key in ('lead_ms','duration_ms','cutoff_ms','bin_ms'):
            aligned(getattr(self,key), self.native_dt_ms, key)
        if self.lead_ms <= 0 or self.duration_ms <= 0 or self.log_floor_v2 <= 0 or self.log_reference_v2 <= 0:
            raise ValueError('Invalid target support or physical log convention')
        if self.family == 'T3' and (self.duration_ms != 1000. or self.native_dt_ms != .0625):
            raise ValueError('T3 is frozen to a 1000-ms, 16-kHz native block')
        if self.family != 'T3':
            aligned(self.duration_ms, self.bin_ms, 'block duration/bin width')

    @property
    def dimension(self):
        if self.family == 'T3': return 54
        if self.family.startswith('T2'): return 1
        return round(self.duration_ms/self.bin_ms)

    @property
    def units(self):
        return 'ln(V^2 / 1 V^2)' if 'log_' in self.family else ('V^2' if self.family.startswith('T2') else 'V')

    @property
    def id(self):
        return f'{self.family}_l{self.lead_ms:g}_d{self.duration_ms:g}'

    def to_dict(self):
        out = asdict(self)
        out.update({'dimension': self.dimension, 'target_dimension': self.dimension,
                    'units': self.units,
                    'risk_units': 'squared dimensionless log units' if 'log_' in self.family else ('V^4' if self.family.startswith('T2') else 'V^2'),
                    'source_signal': 'native_clean_neural_eeg',
                    'support_convention': '(cutoff+lead, cutoff+lead+duration]',
                    'physical_support_ms': [self.cutoff_ms+self.lead_ms,self.cutoff_ms+self.lead_ms+self.duration_ms],
                    'availability_ms': self.cutoff_ms+self.lead_ms+self.duration_ms,
                    'loss': 'mean squared error per declared real coordinate',
                    'projection': 'future-only orthonormal real Fourier block coordinates 4..30 Hz' if self.family=='T3' else 'native nonoverlapping 2-ms box means; not ideal anti-aliasing',
                    'fourier_basis': 'sqrt(2/N) cos/sin(2*pi*k*n/N), n=0..15999; phase origin is first included native sample' if self.family=='T3' else None,
                    'fourier_mse_caution': 'per-54-coordinate coefficient MSE is not per-16000-sample full waveform MSE' if self.family=='T3' else None,
                    'axis': 'frequency_hz, component' if self.family=='T3' else ('right-edge time' if self.family in ('T0','T1') else 'one scalar'),
                    'centering': 'within-future block, not a causal preprocessing of input' if self.family in ('T1','T2_ac_variance','T2_log_ac_variance') else 'none',
                    'log_convention': 'natural logarithm of max(power, physical floor)/reference; fixed before fitting; count activations'})
        return out

    @property
    def fingerprint(self): return canonical_hash(self.to_dict())


def native_block(eeg_v, time_ms, spec):
    x = np.asarray(eeg_v, dtype=np.float64).reshape(-1)
    t = np.asarray(time_ms, dtype=np.float64)
    if x.shape != t.shape:
        raise ValueError('Invalid finite single-channel native target data')
    n = round(spec.duration_ms/spec.native_dt_ms)
    start = spec.cutoff_ms+spec.lead_ms
    idx = np.searchsorted(t, start+spec.native_dt_ms/2)
    expected = start+np.arange(1,n+1)*spec.native_dt_ms
    if idx+n > len(t) or not np.allclose(t[idx:idx+n],expected,rtol=0,atol=1e-8):
        raise ValueError('Native target support is absent or misaligned')
    block=x[idx:idx+n]
    if not np.isfinite(block).all():raise ValueError('Nonfinite target block')
    return block.copy()


def from_binned(waveform, spec):
    y = np.asarray(waveform,dtype=np.float64)
    if y.shape[-1] != round(spec.duration_ms/spec.bin_ms):
        raise ValueError('Binned block does not match declared target support')
    centered = y-y.mean(axis=-1,keepdims=True)
    if spec.family == 'T0': return y.copy()
    if spec.family == 'T1': return centered
    if spec.family == 'T3': raise ValueError('T3 requires native 16-kHz future samples; binned data cannot recover it')
    power = np.mean(centered**2 if 'ac_variance' in spec.family else y**2,axis=-1,keepdims=True)
    if 'log_' in spec.family:
        return np.log(np.maximum(power,spec.log_floor_v2)/spec.log_reference_v2)
    return power


def extract_target(eeg_v,time_ms,spec):
    y = native_block(eeg_v,time_ms,spec)
    if spec.family == 'T3':
        # ortho DFT -> independent real sine/cos axes, exactly unit-norm basis rows.
        transformed = np.fft.rfft(y,norm='ortho')
        f = np.arange(4,31)
        # DFT convention uses -sin; negate imaginary coordinates for +sin basis.
        return np.stack((np.sqrt(2)*transformed[f].real,-np.sqrt(2)*transformed[f].imag),axis=-1).reshape(-1)
    return from_binned(y.reshape(-1,round(spec.bin_ms/spec.native_dt_ms)).mean(axis=1),spec)


def floor_activations(waveform,spec):
    if 'log_' not in spec.family:return 0
    raw_family = spec.family.replace('log_','')
    raw = from_binned(waveform,TargetSpec(**{**asdict(spec),'family':raw_family}))
    return int(np.count_nonzero(raw < spec.log_floor_v2))


def axes(spec):
    if spec.family == 'T3':
        return {'target_time_ms':np.array([],dtype=float),
                'target_frequency_hz':np.repeat(np.arange(4,31,dtype=float),2),
                'target_component':np.tile(np.array(['cos','sin']),27)}
    return {'target_time_ms':spec.lead_ms+np.arange(1,spec.dimension+1)*spec.bin_ms if spec.family in ('T0','T1') else np.array([],dtype=float)}


def full_real_fft(waveform):
    """Invertible n-real-coordinate orthonormal packing; Parseval is exact."""
    y = np.asarray(waveform,dtype=float)
    n = y.shape[-1]
    z = np.fft.rfft(y,axis=-1,norm='ortho')
    parts = [z[...,:1].real]
    stop = -1 if n%2==0 else None
    interior = z[...,1:stop]
    parts.append(np.stack((np.sqrt(2)*interior.real,-np.sqrt(2)*interior.imag),axis=-1).reshape(*y.shape[:-1],-1))
    if n%2==0:parts.append(z[...,-1:].real)
    return np.concatenate(parts,axis=-1)


def inverse_full_real_fft(coordinates):
    c = np.asarray(coordinates,dtype=float);n=c.shape[-1]
    z = np.zeros((*c.shape[:-1],n//2+1),dtype=complex)
    z[...,0]=c[...,0]
    interior=c[...,1:-1 if n%2==0 else None].reshape(*c.shape[:-1],-1,2)
    z[...,1:-1 if n%2==0 else None]=(interior[...,0]-1j*interior[...,1])/np.sqrt(2)
    if n%2==0:z[...,-1]=c[...,-1]
    return np.fft.irfft(z,n=n,axis=-1,norm='ortho')


def default_specs(cutoff_ms=6000.):
    values=[TargetSpec(family=family,lead_ms=lead,duration_ms=duration,cutoff_ms=cutoff_ms)
            for family in ('T0','T1') for lead in (2.,10.) for duration in (32.,128.)]
    values += [TargetSpec(family=family,cutoff_ms=cutoff_ms) for family in FAMILIES if family.startswith('T2')]
    return values+[TargetSpec(family='T3',duration_ms=1000.,cutoff_ms=cutoff_ms)]
