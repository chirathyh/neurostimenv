import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t

dt = 0.025
fs = (1 / dt) * 1000
nperseg = int(fs/2)

transient = 4000  # in seconds L23Net uses : 2000

t1 = int(transient/dt)
print("Sampling Rate:", fs)


def process(filepath):
    EEG = np.loadtxt(filepath, delimiter=",")
    print("Loaded data:\n", EEG)
    # signal filtering
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

    EEG_freq, EEG_ps = ss.welch(EEG_filt, fs=fs, nperseg=nperseg)
    #EEG_freq, EEG_ps = ss.welch(EEG_filt[t1:], fs=fs, nperseg=nperseg)
    return EEG_freq, EEG_ps



def process_eeg(file_path):
    file_list = []
    for i in range(10, 70):
        file_list.append(file_path+str(i)+".csv")

    # Filter coefficients
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    # Lists to store PSDs
    all_psd = []
    all_freqs = None

    for file in file_list:
        print(f"Processing {file}...")
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
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


def process_bandit_testing(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    print(sorted(csv_files))
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    all_psd = []
    all_freqs = None
    for file in sorted(csv_files):
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
        freqs, psd = ss.welch(EEG_filt, fs=fs, nperseg=nperseg)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)
    all_psd = np.array(all_psd)
    avg_psd = np.mean(all_psd, axis=0)
    sem_psd = np.std(all_psd, axis=0, ddof=1) / np.sqrt(len(csv_files))
    ci_95 = t.ppf(0.975, df=len(csv_files)-1) * sem_psd
    return all_freqs, avg_psd, ci_95


all_freqs_b, avg_psd_b, ci_95_b = process_bandit_testing(folder_path="../../data/bandit/hbandit1/testing")
all_freqs, avg_psd, ci_95 = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
all_freqs_h, avg_psd_h, ci_95_h = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")


# Plot EEG signal and PSD
plt.figure(figsize=(10, 5))
colors = ['royalblue', 'mediumseagreen', 'darkorchid', 'deepskyblue', 'limegreen', 'blueviolet']

# Depression group
plt.plot(all_freqs, avg_psd, color='r', label="Depression")
plt.fill_between(all_freqs, avg_psd - ci_95, avg_psd + ci_95, color='r', alpha=0.3)

# Healthy group
plt.plot(all_freqs_h, avg_psd_h, color='k', label="Healthy")
plt.fill_between(all_freqs_h, avg_psd_h - ci_95_h, avg_psd_h + ci_95_h, color='k', alpha=0.3)

# bandit results
plt.plot(all_freqs_b, avg_psd_b, color='g', label="Bandit Stimulation: 0.05mA, 8Hz")
plt.fill_between(all_freqs_b, avg_psd_b - ci_95_b, avg_psd_b + ci_95_b, color='g', alpha=0.3)

# Add vertical lines at 8 Hz and 12 Hz
plt.axvline(x=8, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=12, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=16, color='gray', linestyle='--', alpha=0.7)

# Add text annotations for frequency bands (Greek notation)
plt.text(6.5, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\theta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(10, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\alpha$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.text(20, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05, r"$\beta$",
         fontsize=14, color='black', ha='center', fontweight='bold')

plt.xlim(4, 50)
# plt.ylim(0, 1e-19)
plt.xlabel("Frequency (Hz)")
plt.ylabel(r'$PSD(\text{V}^2/\text{Hz})$')
plt.title("Power Spectral Density (PSD)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
