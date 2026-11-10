"""Causal physical-signal processing; no simulator imports or per-window scaling."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy import signal


def causal_resample(eeg_v, time_ms, target_fs=500.0):
    """Return a causal FIR-filtered stream at grid-aligned right endpoints.

    Filtering starts at native sample 0 with zero state. Retain burn-in through
    this operation, then crop. Time stamps report *availability*, not a falsely
    delay-corrected time. No sample after a timestamp contributes to its value.
    The linear-phase FIR has a known delay, retained in the metadata.
    """
    x = np.asarray(eeg_v, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    t = np.asarray(time_ms, dtype=np.float64)
    if x.ndim != 2 or t.ndim != 1 or len(t) != len(x) or len(t) < 2:
        raise ValueError("Expected time-major EEG (samples,channels) and >=2 times.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(t)):
        raise ValueError("Nonfinite physical signal.")
    dt = float(t[1] - t[0])
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-8, atol=1e-9):
        raise ValueError("Native sample times must be uniformly increasing.")
    native_fs = 1000.0 / dt
    ratio = int(round(native_fs / float(target_fs)))
    if ratio < 1 or not np.isclose(native_fs, float(target_fs)*ratio):
        raise ValueError("Target rate must divide the native sampling rate.")
    if ratio == 1:
        return x.copy(), t.copy(), {"method":"identity", "causal":True,
            "sample_rate_hz":float(target_fs), "group_delay_ms":0.0,
            "initial_state":"identity", "timestamp":"availability"}
    taps = signal.firwin(64*ratio+1, 0.45*target_fs, fs=native_fs,
                        window=("kaiser", 8.6))
    # Padding one sample makes output k=1 occur at native sample ratio-1;
    # upfirdn avoids computing all discarded native-rate filtered samples.
    filtered = signal.upfirdn(taps, np.pad(x, ((1,0),(0,0))), up=1,
                             down=ratio, axis=0)[1:1+len(t)//ratio]
    times = t[ratio-1::ratio]
    return filtered, times, {
        "method":"causal_kaiser_fir_decimation", "causal":True,
        "native_sample_rate_hz":native_fs, "sample_rate_hz":float(target_fs),
        "decimation_factor":ratio, "fir_taps":taps.tolist(),
        "group_delay_ms":(len(taps)-1)*dt/2,
        "filter_support_ms":(len(taps)-1)*dt,
        "cutoff_hz":0.45*target_fs, "initial_state":"zero_before_first_native_sample",
        "timestamp":"availability; group delay is not subtracted",
        "final_history_v":x[-(len(taps)-1):].tolist(),
    }


@dataclass(frozen=True)
class PhysicalNormalizer:
    """Float64 channel moments fitted exclusively on training records."""
    mean_v: np.ndarray
    scale_v: np.ndarray
    count: int
    fitted_split: str = "train"

    @classmethod
    def fit(cls, signals: Iterable[np.ndarray], *, split="train"):
        if split != "train":
            raise ValueError("Normalization statistics may only fit training data.")
        count, mean, m2 = 0, None, None
        for values in signals:
            x = np.asarray(values, dtype=np.float64)
            if x.ndim == 1: x = x[:, None]
            if not len(x): continue
            if not np.isfinite(x).all(): raise ValueError("Nonfinite training EEG.")
            local_mean = x.mean(0, dtype=np.float64)
            local_m2 = np.sum((x-local_mean)**2, axis=0, dtype=np.float64)
            if mean is None:
                count, mean, m2 = len(x), local_mean, local_m2
            else:
                delta = local_mean - mean
                total = count + len(x)
                m2 = m2 + local_m2 + delta**2*count*len(x)/total
                mean = mean + delta*len(x)/total
                count = total
        if mean is None or count < 2: raise ValueError("Insufficient training samples.")
        scale = np.sqrt(m2/count)
        if np.any(scale == 0) or not np.isfinite(scale).all():
            raise ValueError("Genuinely zero-variance/nonfinite channel; no denominator floor is used.")
        return cls(mean, scale, count)

    def transform(self, values):
        return ((np.asarray(values,dtype=np.float64)-self.mean_v)/self.scale_v).astype(np.float32)

    def to_dict(self):
        return {"mean_v":self.mean_v.tolist(),"scale_v":self.scale_v.tolist(),
                "count":self.count,"fitted_split":self.fitted_split,
                "physical_dtype":"float64","model_dtype":"float32"}


def add_observation_noise(eeg_v, *, snr_db, reference_power_v2, seed):
    """Independent white Gaussian noise with train-pool channel reference power.

    SNR = 10 log10(reference channel power / noise variance). It is a label-blind
    reference SNR, not an assertion that each realized trial has that SNR.
    """
    x = np.asarray(eeg_v,dtype=np.float64)
    power = np.asarray(reference_power_v2,dtype=np.float64)
    if np.any(power <= 0) or not np.isfinite(power).all():
        raise ValueError("Reference channel powers must be finite and positive.")
    variance = power*10.0**(-float(snr_db)/10.0)
    noise = np.random.default_rng(int(seed)).normal(size=x.shape)*np.sqrt(variance)
    return x+noise, {"kind":"white_gaussian_training_reference_snr",
        "reference_power_v2":power.tolist(),"noise_covariance_v2":np.diag(np.atleast_1d(variance)).tolist(),
        "snr_db":float(snr_db),"seed":int(seed),"independent_of_latent_trial":True}
