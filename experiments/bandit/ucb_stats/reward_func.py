import numpy as np
import scipy.stats as stats
import scipy.signal as ss
from scipy.fft import fft


def bandpower(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])


def reward_func_simple(eeg_top, fs):

    # Define frequency bands (in Hz)
    FREQ_BANDS = {
        "theta": (4, 8),
        "alpha": (8, 12),
        "beta": (12, 16),
    }
    # Define weights for each band (you can tune these)
    BAND_WEIGHTS = {
        "theta": 1.0,
        "alpha": 1.0,
        "beta": 0.25,
    }
    HEALTHY = {
         "theta": 1.4401906273680963e-19,
        "alpha": 1.3428366620075392e-19,
        "beta": 5.953389227817006e-20,
    }
    DEPRESSION = {
        "theta": 1.8922723690645009e-19,
        "alpha": 1.8980323747960788e-19,
        "beta": 9.913579537680276e-20,
    }

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    eeg_top = ss.filtfilt(b, a, eeg_top, axis=-1)

    freqs, psd = ss.welch(eeg_top, fs=fs, nperseg=int(fs/2))
    psd = psd.flatten()

    total_reward = 0

    for band, limits in FREQ_BANDS.items():
        calc_power = bandpower(freqs, psd, limits)

        norm_power = (calc_power - HEALTHY[band]) / (HEALTHY[band])
        norm_power = norm_power**2

        # Compute reward (higher when closer to HEALTHY)
        reward = -1 * norm_power

        # Weighted sum of rewards from all bands
        total_reward += BAND_WEIGHTS[band] * reward

    return total_reward
