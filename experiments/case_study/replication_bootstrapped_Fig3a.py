import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np

# Constants
dt = 0.025  # Time step
fs = (1 / dt) * 1000  # Sampling frequency
nperseg = int(fs / 2)
transient = 2000  #
t1 = int(transient / dt)

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
})


def bootstrap_ci(data, num_bootstraps=1000, ci=95):
    """
    Compute the bootstrapped mean and confidence interval.

    Parameters:
    - data: NumPy array (each row is a PSD sample, columns are frequency bins)
    - num_bootstraps: Number of bootstrap resamples
    - ci: Confidence interval percentage (default: 95%)

    Returns:
    - mean_psd: Bootstrapped mean PSD
    - ci_lower: Lower bound of the confidence interval
    - ci_upper: Upper bound of the confidence interval
    """
    data = np.array(data)  # Ensure data is a NumPy array
    num_samples = data.shape[0]

    # Bootstrapped means
    boot_means = np.zeros((num_bootstraps, data.shape[1]))

    for i in range(num_bootstraps):
        # Resample with replacement
        resample_idx = np.random.choice(num_samples, num_samples, replace=True)
        resample = data[resample_idx, :]
        boot_means[i, :] = np.mean(resample, axis=0)

    # Compute bootstrapped mean and confidence interval
    mean_psd = np.mean(boot_means, axis=0)
    ci_lower = np.percentile(boot_means, (100 - ci) / 2, axis=0)
    ci_upper = np.percentile(boot_means, 100 - (100 - ci) / 2, axis=0)

    return mean_psd, ci_lower, ci_upper

def process_eeg(file_path):
    file_list = [file_path + str(i) + ".csv" for i in range(10, 70)]

    # Filter coefficients
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    # Lists to store PSDs
    all_psd = []
    all_freqs = None

    for file in file_list:
        print(f"Processing {file}...")

        # Load EEG data
        EEG = np.loadtxt(file, delimiter=",")

        # Filter the EEG signal
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # Compute PSD using Welch's method
        freqs, psd = ss.welch(EEG_filt[t1:], fs=fs, nperseg=nperseg)

        # Store results
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    # Convert to NumPy array
    all_psd = np.array(all_psd)

    # Compute bootstrapped mean and 95% CI
    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)

    return all_freqs, avg_psd, ci_lower, ci_upper

# Load / Process both depression and healthy EEG datasets
all_freqs, avg_psd, ci_lower, ci_upper = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")

# Plot the average PSD with bootstrapped 95% Confidence Interval
plt.figure(figsize=(10, 5))

# Depression group
plt.plot(all_freqs, avg_psd, color='r', label="Depression Baseline")
plt.fill_between(all_freqs, ci_lower, ci_upper, color='r', alpha=0.3)

# Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', label="Healthy Baseline")
plt.fill_between(all_freqs_h, ci_lower_h, ci_upper_h, color='k', alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)

# Add text annotations for frequency bands (Greek notation)
plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
         fontsize=14, color='black', ha='center', fontweight='bold')
plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
         fontsize=14, color='black', ha='center', fontweight='bold')
plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.xlim(4, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
# plt.title("Average Power Spectral Density (PSD) with Bootstrapped 95% CI")
plt.legend()
plt.grid(True)
plt.show()
