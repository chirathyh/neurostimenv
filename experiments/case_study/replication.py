import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from experiments.case_study.configs import get_configs
from experiments.case_study import reward_func
from experiments.case_study.bootstrap import bootstrap_ci

dt, fs, nperseg, _, t1 = get_configs()


def process_eeg(file_path):
    file_list = [file_path + str(i) + ".csv" for i in range(10, 70)]
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')  # Filter coefficients
    # Lists to store PSDs
    all_psd = []
    all_freqs = None
    for file in file_list:
        print(f"Processing {file}...")
        EEG = np.loadtxt(file, delimiter=",")  # Load EEG data
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)  # Filter the EEG signal
        freqs, psd = ss.welch(EEG_filt[t1:], fs=fs, nperseg=nperseg)  # Compute PSD using Welch's method
        # Store results
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    all_psd = np.array(all_psd)  # Convert to NumPy array
    avg_psd, ci_lower, ci_upper = bootstrap_ci(all_psd)  # Compute bootstrapped mean and 95% CI

    return all_freqs, avg_psd, ci_lower, ci_upper


def get_rewards(file_path):
    file_list = []
    reward_values = []

    file_list = [file_path + str(i) + ".csv" for i in range(10, 70)]
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')  # Filter coefficients

    for file in file_list:
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)
        EEG_segment = EEG_filt#[0:x1*8]
        reward_value = reward_func.reward_func_simple(np.array(EEG_segment), fs)
        reward_values.append(reward_value)

    return reward_values
