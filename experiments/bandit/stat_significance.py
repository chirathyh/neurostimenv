import os
import glob

import os
import re
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from scipy.signal import stft
from scipy import stats
from env.eeg import features

dt = 0.025
fs = (1 / dt) * 1000

nperseg = int(fs/2)
noverlap = int(nperseg*0.5) #nperseg-1 #nperseg // 2

transient = 4000  # ms; first 4s is removed from the EEG (triansient phase)
t1 = int(transient/dt)
print("Sampling Rate:", fs)
print("npserg", nperseg)


plt.rcParams.update({
    # Base font size for small text (ticks, annotations)
    'font.size': 14,
    # Axis titles
    'axes.titlesize': 16,
    # Axis labels (xlabel, ylabel)
    'axes.labelsize': 16,
    # Tick labels
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # Legend text
    'legend.fontsize': 14,
    # Figure title (if you ever use suptitle)
    'figure.titlesize': 18,
})


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # print(sorted(csv_files))
    # print(sorted(reward_files))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        #print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]

        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000/dt)

        if rew < -1.5:
            continue
        # selection_reward = features.reward_func_simple(np.array(EEG_filt[0:x1]), fs)
        # if selection_reward >= -0.09264591737143694 :
        #     continue

        # different protocol stages
        if segment == 1:
            EEG_segment = EEG_filt[0:x1]
        elif segment == 2:
            EEG_segment = EEG_filt[ x1 : x1*2]
        elif segment == 3:
            EEG_segment = EEG_filt[x1*2 : x1*3]
        elif segment == 4:
            EEG_segment = EEG_filt[x1*3 : x1*4]
        elif segment == 5:
            EEG_segment = EEG_filt[x1*4 : ]
        elif segment == -1:
            EEG_segment = EEG_filt
        else:
            print("select segment")
            exit()

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    return all_freqs, all_psd

# Calculate Cohen's d effect size
def cohen_d(x, y):
    """
    Compute Cohen's d pooled effect size for two independent samples.
    """
    nx, ny = len(x), len(y)
    pooled_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    return (np.mean(x) - np.mean(y)) / pooled_sd



SELECTED_ARM = 1
all_freqs_seg5, all_psdb_seg5 = process_bandit_testing(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=5)
all_freqs_seg1, all_psd_seg1 = process_bandit_testing(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=1)

freqs = all_freqs_seg5


def get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=4, high=8 ):
    theta_idx = np.where((freqs >= low) & (freqs <= high))[0]
    # Compute integrated power in the theta band for each sample using trapezoidal integration
    def theta_band_power(psd, freqs, idx):
        # psd: 1D PSD array; freqs: 1D frequency array; idx: indices for theta band
        return np.trapz(psd[idx], freqs[idx])
    segment5 = np.array([
        theta_band_power(psd, freqs, theta_idx)
        for psd in all_psdb_seg5
    ])
    segment1 = np.array([
        theta_band_power(psd, freqs, theta_idx)
        for psd in all_psd_seg1
    ])

    print(len(segment1))
    t_stat, p_value = stats.ttest_ind(segment1, segment5, equal_var=False)
    effect_size = cohen_d(segment1, segment5)
    # Display results
    print(f"Healthy group theta power (mean ± SD): {segment5.mean()} ± {segment5.std(ddof=1)}")
    print(f"Depression group theta power (mean ± SD): {segment1.mean()} ± {segment1.std(ddof=1)}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4e}")
    print(f"Cohen's d: {effect_size:.4f}")

print("4-8Hz")
get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=4, high=8 )
print("\n")

print("8-12Hz")
get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=8, high=12 )
print("\n")

print("12-16Hz")
get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=12, high=16 )
print("\n")
