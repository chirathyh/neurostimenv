import numpy as np
import scipy.stats as stats
import scipy.signal as ss
from scipy.fft import fft
# import pywt
# from spectrum import aryule

# Example 1-channel EEG signal (replace with actual EEG data)
# Assuming EEG is a 1D numpy array (1-channel signal)
# eeg_signal = np.random.randn(1000)  # Replace with your EEG data
# fs = 1000  # Sampling frequency (in Hz), modify according to your data

DEPRESSION = 4.781662697628607e-19
HEALTHY = 3.3783662121573373e-19
TARGET_BAND = (4, 16)


def bandpower(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])


def reward_func_simple(eeg_top, fs):

    freqs, psd = ss.welch(eeg_top, fs=fs, nperseg=int(fs/2))
    psd = psd.flatten()
    calc_power = bandpower(freqs, psd, TARGET_BAND)

    norm_power = (calc_power - DEPRESSION) / (HEALTHY - DEPRESSION)
    norm_power = np.clip(norm_power, 0, 1)  # Ensure within bounds
    # Compute reward (higher when closer to HEALTHY)
    reward = 1 - abs(norm_power - 1)
    return reward


def reward_func_simple_old(eeg_top, fs):
    target_peak_freq = 8 # Hz
    frequencies, psd = ss.welch(eeg_top, fs=fs, nperseg=int(fs/4))
    psd = psd.flatten()
    peak_index = np.argmax(psd)
    peak_frequency = frequencies[peak_index]
    peak_power = psd[peak_index]
    reward = -1 * (peak_frequency - target_peak_freq)**2
    return reward


def feature_space(eeg, fs, ts=None):
    obs = {}
    tf = time_domain_features(eeg)
    ff = frequency_domain_features(eeg, fs)
    obs.update(tf)
    obs.update(ff)
    if not None:
        tsf = {"stimAmplitude": ts[0], "stimFreq": ts[1]}
        obs.update(tsf)
    return obs

# Time-Domain Features
def time_domain_features(signal):
    mean = np.mean(signal)
    variance = np.var(signal)
    std_dev = np.std(signal)
    skewness = stats.skew(signal, axis=1)
    kurtosis = stats.kurtosis(signal, axis=1)
    zcr = ((signal[:-1] * signal[1:]) < 0).sum() / len(signal)

    return {
        "mean": mean,
        "var": variance,
        "stdDev": std_dev,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "zeroCrossingRate": zcr
    }


# Frequency-Domain Features
def frequency_domain_features(eeg, fs):
    freqs, psd = ss.welch(eeg, fs=fs, nperseg=int(fs/4))
    psd = psd.flatten()

    # Frequency bands
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 12)
    beta_band = (12, 30)
    gamma_band = (30, 100)

    def bandpower(freqs, psd, band):
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        return np.trapz(psd[idx_band], freqs[idx_band])

    delta_power = bandpower(freqs, psd, delta_band)
    theta_power = bandpower(freqs, psd, theta_band)
    alpha_power = bandpower(freqs, psd, alpha_band)
    beta_power = bandpower(freqs, psd, beta_band)
    gamma_power = bandpower(freqs, psd, gamma_band)

    dominant_freq = freqs[np.argmax(psd)]

    return {
        "deltaPower": delta_power,
        "thetaPower": theta_power,
        "alphaPower": alpha_power,
        "betaPower": beta_power,
        "gammaPower": gamma_power,
        "dominantFreq": dominant_freq
    }


# Non-Linear Features
def non_linear_features(signal):
    # Approximate Entropy
    def approx_entropy(signal, m, r):
        N = len(signal)
        def _phi(m):
            x = np.array([signal[i:i + m] for i in range(N - m + 1)])
            C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=0) / (N - m + 1)
            return np.sum(np.log(C)) / (N - m + 1)
        return _phi(m) - _phi(m + 1)

    ap_entropy = approx_entropy(signal, 2, 0.2 * np.std(signal))

    # Fractal Dimension via Higuchi Method
    def higuchi_fd(signal, k_max=10):
        Lmk = np.zeros(k_max)
        N = len(signal)
        for k in range(1, k_max + 1):
            Lm = 0
            for m in range(k):
                Lm += np.sum(np.abs(signal[m + k::k] - signal[m::k])) / k
            Lmk[k - 1] = Lm / ((N - m) // k)
        return np.log(np.sum(Lmk) / k_max) / np.log(1 / k_max)

    fractal_dim = higuchi_fd(signal)

    return {
        "Approximate Entropy": ap_entropy,
        "Fractal Dimension": fractal_dim
    }

# nf = features.non_linear_features(eeg_top.flatten())
            # print(nf)

            # 2. Phase Locking Value (PLV)
            #plv = np.mean(np.abs(np.angle(ss.hilbert(eeg_top))))

            # 3. Functional Connectivity - Spectral Coherence
            # from mne_connectivity import spectral_connectivity
            # con, _, _, _, _ = spectral_connectivity([eeg_top], method='coh', sfreq=self.sampling_rate, fmin=8, fmax=12, faverage=True)
            # connectivity = np.mean(con)
            # 4. Brain Network Features - Coherence as a proxy
            # network_coherence = connectivity

            # 5. Amplitude Modulation - Standard deviation of the signal as a proxy
            # amplitude_modulation = np.std(eeg_top)
            #
            # # 6. Cross-Frequency Coupling (simplified - alpha-beta coupling)
            # analytic_signal = ss.hilbert(eeg_top)
            # instantaneous_phase = np.angle(analytic_signal)
            # cfc = np.mean(np.diff(instantaneous_phase))
            #
            # # 7. Hjorth Parameters (Activity, Mobility, Complexity)
            # activity = np.var(eeg_top)
            # mobility = np.sqrt(np.var(np.diff(eeg_top)) / activity)
            # complexity = np.sqrt(np.var(np.diff(np.diff(eeg_top))) / np.var(np.diff(eeg_top)))
            #
            # # 8. Sample Entropy
            # # sampen = sample_entropy(eeg_signal, 2, 0.2 * np.std(eeg_signal))
            #
            # # 9. Permutation Entropy
            # # perm_entropy = permutation_entropy(eeg_signal, order=3, delay=1, normalize=True)
            #
            # # 10. Alpha Peak Frequency (APF) - Dominant frequency in alpha band
            # alpha_idx = np.logical_and(frequencies >= 8, frequencies <= 12)
            # alpha_peak_freq = frequencies[np.argmax(psd[alpha_idx])]



# # Wavelet Features
# def wavelet_features(signal, wavelet='db4'):
#     coeffs = pywt.wavedec(signal, wavelet, level=5)
#     delta_coeff = coeffs[-1]
#     theta_coeff = coeffs[-2]
#     alpha_coeff = coeffs[-3]
#     beta_coeff = coeffs[-4]
#
#     delta_power = np.sum(delta_coeff**2)
#     theta_power = np.sum(theta_coeff**2)
#     alpha_power = np.sum(alpha_coeff**2)
#     beta_power = np.sum(beta_coeff**2)
#
#     return {
#         "Wavelet Delta Power": delta_power,
#         "Wavelet Theta Power": theta_power,
#         "Wavelet Alpha Power": alpha_power,
#         "Wavelet Beta Power": beta_power
#     }
#
# # Autoregressive (AR) Model Coefficients
# def ar_coefficients(signal, order=4):
#     ar_coeff, _, _ = aryule(signal, order)
#     return {f"AR Coefficient {i+1}": coeff for i, coeff in enumerate(ar_coeff)}
#
# # Extract Features
# features = {}
# features.update(time_domain_features(eeg_signal))
# features.update(frequency_domain_features(eeg_signal, fs))
# features.update(non_linear_features(eeg_signal))
# features.update(wavelet_features(eeg_signal))
# features.update(ar_coefficients(eeg_signal))
#
# # Print all extracted features
# for feature, value in features.items():
#     print(f"{feature}: {value}")
