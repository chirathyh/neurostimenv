import pandas as pd
import numpy as np
import scipy.signal as ss
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import shap
from sklearn.metrics import classification_report
import pywt

# Constants
dt = 0.025  # Time step
fs = (1 / dt) * 1000  # Sampling frequency
nperseg = int(fs / 2)
transient = 2000  #
t1 = int(transient / dt)

SAMPLES = 70

# Frequency bands
DELTA_BAND = (0.5, 4)
THETA_BAND = (4, 8)
ALPHA_BAND = (8, 12)
BETA_BAND = (12, 30)
GAMMA_BAND = (30, 100)


def bandpower(freqs, psd, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.trapz(psd[idx_band], freqs[idx_band])


def spectral_entropy(psd):
    # Normalized power spectral density
    psd_norm = psd / np.sum(psd)
    # Entropy calculation
    return -np.sum(psd_norm * np.log2(psd_norm))


def hurst_exponent(time_series):
    # Calculate the Hurst exponent using the rescaled range analysis method
    N = len(time_series)
    T = np.arange(1, N + 1)
    Y = np.cumsum(time_series - np.mean(time_series))
    R = np.max(Y) - np.min(Y)
    S = np.std(time_series)
    return np.log(R/S) / np.log(N)


def wavelet_features(EEG_signal, fs):
    # Perform Continuous Wavelet Transform (CWT) using the Morlet wavelet
    coefficients, frequencies = pywt.cwt(EEG_signal, scales=np.arange(1, 128), wavelet='morl', sampling_period=1/fs)

    # Compute wavelet power by squaring the coefficients
    wavelet_power = np.abs(coefficients)**2

    # Calculate the mean wavelet power across scales
    mean_wavelet_power = np.mean(wavelet_power, axis=-1)

    # Calculate wavelet entropy
    wavelet_entropy = -np.sum(wavelet_power * np.log(wavelet_power), axis=-1)

    # We can also include other features such as the standard deviation of the wavelet power
    std_wavelet_power = np.std(wavelet_power, axis=-1)

    return mean_wavelet_power, wavelet_entropy, std_wavelet_power


def extract_features(file_path):
    file_list = [file_path + str(i) + ".csv" for i in range(10, SAMPLES)]
    # Filter coefficients
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    features = []

    for file in file_list:
        print(f"Processing {file}...")
        EEG = np.loadtxt(file, delimiter=",")  # Load EEG data
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)  # Filter the EEG signal
        # Compute frequency domain features (e.g., PSD)
        freqs, psd = ss.welch(EEG_filt[t1:], fs=fs, nperseg=nperseg)

        # Extract time domain features
        mean_eeg = np.mean(EEG_filt)
        var_eeg = np.var(EEG_filt)
        skewness_eeg = stats.skew(EEG_filt, axis=-1)
        kurtosis_eeg = stats.kurtosis(EEG_filt, axis=-1)

        # Power in different frequency bands
        delta_power = bandpower(freqs, psd, DELTA_BAND)
        theta_power = bandpower(freqs, psd, THETA_BAND)
        alpha_power = bandpower(freqs, psd, ALPHA_BAND)
        beta_power = bandpower(freqs, psd, BETA_BAND)
        gamma_power = bandpower(freqs, psd, GAMMA_BAND)

        # Additional frequency domain features
        peak_frequency = freqs[np.argmax(psd)]  # Frequency with maximum power
        alpha_beta_ratio = alpha_power / beta_power if beta_power != 0 else np.nan  # Alpha to Beta ratio
        spectral_entropy_val = spectral_entropy(psd)  # Spectral Entropy
        hurst_exp = hurst_exponent(EEG_filt)  # Hurst Exponent

        # Wavelet-based features
        mean_wavelet_power, wavelet_entropy, std_wavelet_power = wavelet_features(EEG_filt, fs)

        # Append all features for this file
        features.append([mean_eeg, var_eeg, skewness_eeg, kurtosis_eeg, psd.mean(),
                         delta_power, theta_power, alpha_power, beta_power, gamma_power,
                         peak_frequency, alpha_beta_ratio, spectral_entropy_val, hurst_exp,
                         np.mean(mean_wavelet_power), np.mean(wavelet_entropy), np.mean(std_wavelet_power)])

    # Convert to NumPy array for easy manipulation
    features = np.array(features)

    return features


def process_eeg(file_path):
    features = extract_features(file_path)
    return features


feature_names = ['Mean', 'Variance', 'Skewness', 'Kurtosis', 'Mean PSD',
                 'deltaPower', 'thetaPower', 'alphaPower', 'betaPower', 'gammaPower',
                 'peak_frequency', 'alpha_beta_ratio', 'spectral_entropy_val', 'hurst_exp',
                 'mean_wavelet_power', 'wavelet_entropy', 'std_wavelet_power']


# Process both depression and healthy EEG datasets
features_mdd = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
features_healthy = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")

# Combine features from both groups and create labels
X = np.vstack([features_mdd, features_healthy])  # Features
y = np.array([1] * len(features_mdd) + [0] * len(features_healthy))  # Labels: 1 = MDD, 0 = Healthy

# Check the shape of X to confirm correct dimensions
print(f"Shape of X: {X.shape}")

# exit()

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Make prediction on the testing data
y_pred = rf.predict(X_test)
print(classification_report(y_pred, y_test))

# Perform Shapley analysis
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary plot


shap.summary_plot(shap_values[:,:,1], X_test, feature_names=feature_names)
shap.summary_plot(shap_values[:,:,0], X_test, feature_names=feature_names)

#
# explainer = shap.Explainer(rf)
# shap_values = explainer.shap_values(X_test)
# shap.summary_plot(shap_values[:,:,1], X_test)


