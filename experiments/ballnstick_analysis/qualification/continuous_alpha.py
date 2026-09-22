"""Continuous carrier evidence. No hidden frequency/state is an estimator input."""
from __future__ import annotations

import numpy as np
from scipy import signal


def multitaper(x, fs, nw, k):
    x = np.asarray(x, dtype=float)
    if x.size < fs * 2 or not np.all(np.isfinite(x)):
        raise ValueError("Need at least two seconds of finite EEG")
    tapers, ratios = signal.windows.dpss(x.size, nw, Kmax=k, sym=False, return_ratios=True)
    nfft = max(x.size, int(32 * fs))  # interpolation, NOT additional spectral resolution
    spectra = [signal.periodogram(x, fs, window=t, nfft=nfft, detrend="constant")[1] for t in tapers]
    f = np.fft.rfftfreq(nfft, 1 / fs)
    return f, np.average(spectra, axis=0, weights=ratios)


def residual_db(f, psd):
    fit = ((f >= 4) & (f <= 7)) | ((f >= 13) & (f <= 20))
    coefficients = np.polyfit(np.log10(f[fit]), np.log10(np.maximum(psd[fit], 1e-300)), 1)
    background = 10 ** np.polyval(coefficients, np.log10(np.maximum(f, .01)))
    return 10 * np.log10(np.maximum(psd, 1e-300) / background)


def estimate(x, fs, parameters):
    """Smooth continuous evidence; agreement is graded across whole-record windows.

    Discovery may rank predeclared parameter sets using labels. This function
    itself receives ONLY EEG, sampling rate, and already-declared parameters.
    """
    f, psd = multitaper(x, fs, parameters["nw"], parameters["k"])
    grid = np.linspace(8, 12, 161)
    width = parameters["evidence_sigma_hz"]
    kernels = np.exp(-.5 * ((f[None, :] - grid[:, None]) / width) ** 2)
    kernels[:, (f < 7) | (f > 13)] = 0
    kernels /= kernels.sum(axis=1, keepdims=True)
    residual = residual_db(f, psd)
    whole = kernels @ residual
    win = min(len(x), int(parameters["window_s"] * fs))
    evidence = []
    for start in range(0, len(x) - win + 1, max(1, win // 2)):
        wf, wp = multitaper(x[start:start + win], fs, 2, 3)
        evidence.append(kernels @ np.interp(f, wf, residual_db(wf, wp)))
    evidence = np.asarray(evidence)
    # Average signed evidence, not a collection of independently selected peaks.
    pooled = .5 * whole + .5 * np.mean(evidence, axis=0)
    best = int(np.argmax(pooled))
    frequency = float(grid[best])
    distant = np.abs(grid - frequency) >= parameters["competitor_separation_hz"]
    prominence = float(pooled[best])
    margin = float(prominence - np.max(pooled[distant]))
    # Graded location stability; neighboring frequency bins are not competitors.
    weights = np.exp(np.clip((evidence - evidence.max(axis=1, keepdims=True)) / 2, -50, 0))
    centers = (weights @ grid) / weights.sum(axis=1)
    temporal_sd = float(np.std(centers))
    accepted = bool(np.std(x) > 0 and prominence >= parameters["minimum_evidence_db"] and
                    margin >= parameters["minimum_margin_db"] and
                    temporal_sd <= parameters["maximum_temporal_sd_hz"])
    return {"frequency_hz": frequency, "accepted": accepted,
        "fallback": None if accepted else "sham", "evidence_db": prominence,
        "margin_db": margin, "temporal_sd_hz": temporal_sd,
        "f_hz": f, "psd_v2_per_hz": psd, "residual_db": residual,
        "grid_hz": grid, "pooled_evidence_db": pooled, "window_evidence_db": evidence,
        "nominal_full_record_half_bandwidth_hz": parameters["nw"] * fs / len(x)}


def baseline_scaled_noise(neural, baseline_count, rho, rms_fraction, seed):
    """Stationary AR(1); normalize ONCE using prestimulation samples only."""
    neural = np.asarray(neural, dtype=float)
    if neural.ndim != 1 or not (0 < baseline_count <= len(neural)) or not 0 <= rho < 1 or rms_fraction < 0:
        raise ValueError("Noise requires a flat EEG, a valid baseline, 0<=rho<1, and nonnegative scale")
    rng = np.random.default_rng(seed)
    innovations = rng.normal(size=len(neural)) * np.sqrt(1 - rho * rho)
    innovations[0] /= np.sqrt(1 - rho * rho)  # stationary and prefix-invariant
    unit = signal.lfilter([1], [1, -rho], innovations)
    baseline = np.asarray(neural[:baseline_count])
    centered = baseline - np.mean(baseline)
    scale = rms_fraction * np.sqrt(np.mean(centered ** 2)) / np.sqrt(np.mean(unit[:baseline_count] ** 2))
    return np.asarray(neural) + scale * unit, unit, float(scale)


def phase_at_boundary(raw, fs, frequency, stop_s, history_s):
    """Causal regression at a known frequency, using only the preceding tail.

    Trend + both quadratures reduce DC/trend leakage; no Hilbert transform or
    future-dependent filtering or post-boundary samples.
    """
    n = int(round(history_s * fs))
    tail = np.asarray(raw[-n:], dtype=float)
    if len(tail) != n:
        raise ValueError("Insufficient causal history")
    t = stop_s - history_s + (np.arange(n) + 1) / fs
    # Regression directly on raw samples; no phase shift from resampling.
    theta = 2 * np.pi * frequency * t
    design = np.column_stack([np.ones(n), np.linspace(-1, 1, n), np.cos(theta), np.sin(theta)])
    beta = np.linalg.lstsq(design, tail, rcond=None)[0]
    amplitude = np.hypot(beta[2], beta[3])
    phase = np.angle(np.exp(1j * (2 * np.pi * frequency * stop_s + np.arctan2(-beta[3], beta[2]))))
    rms = np.std(tail)
    return float(phase), float(amplitude / max(rms, 1e-300))


def contexts(design, stage):
    specs = []
    for s in range(int(design["structures_per_stage"])):
        structure = int(design[f"{stage}_structure_start"]) + s
        rng = np.random.default_rng(np.random.SeedSequence([structure, int(design["frequency_seed"])]))
        carriers = [rng.uniform(8, 10), rng.uniform(10, 12)]
        for carrier_index, frequency in enumerate(carriers):
            # Same carrier and private-event random namespace across D levels.
            for diffusion in design["diffusion_levels"]:
                specs.append({"id": f"{stage}_s{s:02d}_c{carrier_index}_D{diffusion:g}",
                    "stage": stage, "structure_seed": structure,
                    "drive_seed": structure + 1100000 + carrier_index * 1000,
                    "noise_seed": structure + 2200000 + carrier_index * 1000,
                    "carrier_hz": float(frequency), "D": float(diffusion), "state": "A"})
        specs.append({"id": f"{stage}_s{s:02d}_B", "stage": stage,
            "structure_seed": structure, "drive_seed": structure + 1100000,
            "noise_seed": structure + 2200000, "carrier_hz": 10., "D": 0., "state": "B"})
    return specs
