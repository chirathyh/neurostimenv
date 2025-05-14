import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t

# Constants
dt = 0.025  # Time step
fs = (1 / dt) * 1000  # Sampling frequency
nperseg = int(fs / 2)
transient = 2000  # in seconds
t1 = int(transient / dt)


def process_eeg(file_path, start=10, end=70):
    file_list = []
    for i in range(start, end):
        file_list.append(file_path+str(i)+".csv")

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
        freqs, psd = ss.welch(EEG_filt, fs=fs, nperseg=nperseg)

        # Store results
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    # Convert to NumPy array
    all_psd = np.array(all_psd)

    # Compute the average PSD across all EEGs
    avg_psd = np.mean(all_psd, axis=0)

    # Compute standard error of the mean (SEM)
    sem_psd = np.std(all_psd, axis=0, ddof=1) / np.sqrt(len(file_list))

    # Compute 95% confidence interval (using t-distribution)
    ci_95 = t.ppf(0.975, df=len(file_list)-1) * sem_psd

    return all_freqs, avg_psd, ci_95


# Process both depression and healthy EEG datasets
all_freqs, avg_psd, ci_95 = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
all_freqs_h, avg_psd_h, ci_95_h = process_eeg(file_path="../../data/feature_analysis/mdd1/EEG_MDD_", start=70, end=130)

# Plot the average PSD with 95% Confidence Interval
plt.figure(figsize=(10, 5))

# Depression group
plt.plot(all_freqs, avg_psd, color='r', label="Depression: N=1,000")
plt.fill_between(all_freqs, avg_psd - ci_95, avg_psd + ci_95, color='r', alpha=0.3)

# Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', label="Depression: N=100")
plt.fill_between(all_freqs_h, avg_psd_h - ci_95_h, avg_psd_h + ci_95_h, color='k', alpha=0.3)

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

# Formatting
plt.xlim(4, 30)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
plt.title("Average Power Spectral Density (PSD) with 95% CI")
plt.legend()
plt.grid(True)

plt.show()
