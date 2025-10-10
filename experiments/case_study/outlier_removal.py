import os
import glob

import os
import re
import sys
from decouple import config
import statistics

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)


import statsmodels
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t
from scipy.signal import stft
from env.eeg import features

from experiments.case_study.data_loader import process_bandit_testing

from experiments.bandit.stats.mad_z_score import mad_outlier_mask
from experiments.bandit.stats.trim_upper_quantile import trim_upper_quantile
from experiments.bandit.stats.evt import ev_outlier_mask
from experiments.bandit.stats.sensitivity import sensitivity_analysis_mad


plt.rcParams.update({
    'font.size': 14,  # Base font size for small text (ticks, annotations)
    'axes.titlesize': 16,  # Axis titles
    'axes.labelsize': 16,  # Axis labels (xlabel, ylabel)
    'xtick.labelsize': 14,  # Tick labels
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,  # Figure title (if you ever use suptitle)
})


SELECTED_ARM = 1
SEGEMENT = 1
AMP = [1, 2, 4, 2, 2, 15]  # mA
FREQ = [8, 8, 8, 10, 40, 77.5]  # Hz

print("\nUsing pre-processed EEG (bandpass filter)")
reward_values, reward_values_final_segment = process_bandit_testing(folder_path="../../../data/bandit/simnibsbandit3/training",
                                                                    selected_arm=SELECTED_ARM, segment=5,
                                                                    preprocessed_eeg=True, filter=False, filter_threshold=1)
x = -1 * np.asarray(reward_values_final_segment)
mask, modz = mad_outlier_mask(x, thresh=2.5, one_sided='upper')
x_filtered = x[mask]
print(f"Removed {np.sum(~mask)} outliers based on MAD rule; kept {x_filtered.size}/{x.size}.")
print(np.sort(x_filtered))

df_sensitivity = sensitivity_analysis_mad(x, thresholds=[2, 2.5, 3.0], one_sided='upper')
print(df_sensitivity)

# print("\nUsing Raw EEG segments")
# reward_values, reward_values_final_segment = process_bandit_testing(folder_path="../../../data/bandit/simnibsbandit3/training",
#                                                                     selected_arm=SELECTED_ARM, segment=5,
#                                                                     preprocessed_eeg=False, filter=False, filter_threshold=1)
# x = -1 * np.asarray(reward_values_final_segment)
# mask, modz = mad_outlier_mask(x, thresh=2.5, one_sided='upper')
# x_filtered = x[mask]
# print(f"Removed {np.sum(~mask)} outliers based on MAD rule; kept {x_filtered.size}/{x.size}.")
# print(np.sort(x_filtered))

# quantile-based removal
# x_trimmed, threshold = trim_upper_quantile(x, quantile=0.90)
# print(f"Removed values > {threshold:.4g}. Kept {len(x_trimmed)}/{len(x)} samples.")

# extreme value threshold:
# mask = ev_outlier_mask(x, threshold_quantile=0.85, tail_prob_thresh=0.05)
# x_filtered = x[mask]
# print(f"Removed {np.sum(~mask)}/{x.size} points as extreme tail outliers.")
# print(np.sort(x_filtered))

# print(np.sort(x))




