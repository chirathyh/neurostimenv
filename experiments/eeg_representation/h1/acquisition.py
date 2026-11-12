"""Causal common-passband acquisition with honest availability timestamps."""
from __future__ import annotations
import hashlib
import numpy as np
from scipy import signal
from .targets import aligned


def filter_kernel(native_fs=16000., passband=(.5,30.), taps=2049):
    if taps<3 or taps%2!=1: raise ValueError('Odd FIR tap count required')
    if not 0<passband[0]<passband[1]<native_fs/2: raise ValueError('Invalid passband')
    # Declared finite FIR; nominal cutoffs are not claims of sharp .5-Hz resolution.
    return signal.firwin(int(taps),passband,pass_zero=False,fs=native_fs,window='hamming')


def metadata(rate_hz=500., native_dt_ms=.0625, passband=(.5,30.), taps=2049):
    native_fs=1000/native_dt_ms
    aligned(native_fs/rate_hz,1.,'native/input integer ratio')
    if rate_hz/2<=passband[1]: raise ValueError('Input rate cannot preserve the common passband')
    h=filter_kernel(native_fs,passband,taps)
    return {'kind':'causal_fir_common_passband','sample_rate_hz':float(rate_hz),
        'native_dt_ms':native_dt_ms,'passband_hz':list(passband),'taps':len(h),
        'kernel_sha256':hashlib.sha256(h.astype('<f8').tobytes()).hexdigest(),
        'support_ms':(len(h)-1)*native_dt_ms,'group_delay_ms':(len(h)-1)*native_dt_ms/2,
        'availability':'unshifted right-edge native acquisition timestamp',
        'filter_design':'Hamming-window FIR; nominal passband edges, finite transition width',
        'phase':'linear phase applied causally; zero initial state before episode'}


def causal_history(eeg_v,time_ms,*,cutoff_ms,history_s,rate_hz=500.,passband=(.5,30.),taps=2049):
    t=np.asarray(time_ms,dtype=float);x=np.asarray(eeg_v,dtype=float).reshape(-1)
    if x.shape!=t.shape or len(t)<2: raise ValueError('Invalid observation arrays')
    native_dt=float(t[1]-t[0]);meta=metadata(rate_hz,native_dt,passband,taps)
    aligned(cutoff_ms,1000/rate_hz,'input cutoff')
    aligned(history_s*1000,1000/rate_hz,'input history')
    # Slicing BEFORE filtering makes future perturbations mechanically unavailable.
    prefix=t<=cutoff_ms+1e-9
    tp=t[prefix];xp=x[prefix]
    if not len(tp) or not np.isclose(tp[-1],cutoff_ms,rtol=0,atol=1e-9):raise ValueError('Unavailable cutoff')
    y=signal.lfilter(filter_kernel(1000/native_dt,passband,taps),[1.],xp)
    step=1000/rate_hz
    desired=cutoff_ms-history_s*1000+np.arange(1,round(history_s*rate_hz)+1)*step
    indices=np.searchsorted(tp,desired-1e-8)
    if np.any(indices>=len(tp)) or not np.allclose(tp[indices],desired,rtol=0,atol=1e-7):raise ValueError('Input history grid unavailable')
    if desired[0]<=meta['support_ms']:raise ValueError('History overlaps initial filter transient; increase burn-in')
    return y[indices],desired-cutoff_ms,meta


def fit_range(training_values, bits):
    x=np.asarray(training_values,dtype=float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite training acquisition range')
    if bits not in (4,8,12,16):raise ValueError('Unsupported acquisition bit depth')
    low,high=float(x.min()),float(x.max())
    if high<=low:high=low+max(abs(low)*1e-6,1e-15)
    return {'bits':int(bits),'low_v':low,'high_v':high,'fit_split':'train','method':'global training minimum/maximum, endpoint-inclusive uniform levels'}


def quantize(values, quantizer):
    x=np.asarray(values,dtype=float);q=quantizer
    low,high=q['low_v'],q['high_v'];levels=2**q['bits']-1
    clipped=(x<low)|(x>high)
    code=np.rint((np.clip(x,low,high)-low)/(high-low)*levels).astype(np.uint16)
    return low+code.astype(float)*(high-low)/levels,code,float(clipped.mean())
