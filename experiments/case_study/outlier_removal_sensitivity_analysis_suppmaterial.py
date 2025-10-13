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

from experiments.case_study.data_loader import process_bandit_testing
from experiments.case_study.mad_z_score import mad_outlier_mask
from experiments.case_study.sensitivity import sensitivity_analysis_mad


plt.rcParams.update({
    'font.size': 14,  # Base font size for small text (ticks, annotations)
    'axes.titlesize': 16,  # Axis titles
    'axes.labelsize': 16,  # Axis labels (xlabel, ylabel)
    'xtick.labelsize': 14,  # Tick labels
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 18,  # Figure title (if you ever use suptitle)
})

BANDIT_DATA = "data/bandit/training"
SELECTED_ARM = 1
SEGEMENT = 1
AMP = [1, 2, 4, 2, 2, 15]  # mA
FREQ = [8, 8, 8, 10, 40, 77.5]  # Hz

print("\nRemoved Outliers")
reward_values, reward_values_final_segment = process_bandit_testing(folder_path=BANDIT_DATA,
                                                                    selected_arm=SELECTED_ARM, segment=5,
                                                                    preprocessed_eeg=True, filter=False, filter_threshold=1)
x = -1 * np.asarray(reward_values_final_segment)
mask, modz = mad_outlier_mask(x, thresh=2.5, one_sided='upper')
x_filtered = x[mask]
print(f"Removed {np.sum(~mask)} outliers based on MAD rule; kept {x_filtered.size}/{x.size}.")
print(np.sort(x_filtered))

print("\nSensitivity Analysis")
df_sensitivity = sensitivity_analysis_mad(x, thresholds=[2, 2.5, 3.0], one_sided='upper')
print(df_sensitivity)




