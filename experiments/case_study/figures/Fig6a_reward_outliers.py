import sys
import os
import glob
from decouple import config
MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)
from experiments.case_study import reward_func
import scipy.signal as ss
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t
from experiments.case_study.configs import get_configs

dt, fs, nperseg, _, t1 = get_configs()

plt.rcParams.update({
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
})


def process_bandit_testing(folder_path, selected_arm=1):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    reward_values, reward_values_final_segment = [], []

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        df = pd.read_csv(reward_file)
        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # different protocol stages
        x1 = int(1000/dt)
        reward_value = reward_func.reward_func_simple(np.array(EEG_filt[0:x1]), fs)
        reward_values.append(reward_value)

        reward_final = reward_func.reward_func_simple(np.array(EEG_filt[x1*4:]), fs)
        reward_values_final_segment.append(reward_final)

    return reward_values, reward_values_final_segment


SELECTED_ARM = 1
BANDIT_RESULTS="../data/bandit/training"

reward_values, reward_values_final_segment = process_bandit_testing(folder_path=BANDIT_RESULTS, selected_arm=SELECTED_ARM)

print("Rewards for first 1-second segment")
print(len(reward_values))
print(reward_values)
print("Rewards for final 1-second segment")
print(len(reward_values_final_segment))
print(reward_values_final_segment)


fig = plt.figure(figsize=(10, 5))
plt.hist(reward_values_final_segment, bins=120, alpha=0.5, label='EEG Segment 5')
plt.axvline(x=-1.82668915, color='black', linestyle='--', linewidth=1.5, label='2.5 MAD (55/63 samples)')
plt.legend()
plt.xlabel('Reward')
plt.ylabel('Frequency')
# plt.title('Overlaid Histograms of x and y')
plt.tight_layout()
fig.savefig('png/Fig6a.png', dpi=600, bbox_inches='tight', pad_inches=0.02, facecolor='auto', transparent=False)
plt.show()


