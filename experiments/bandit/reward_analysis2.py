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



def bootstrap_ci(data, num_bootstraps=8, ci=95):
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
    file_list = []
    reward_values = []
    for i in range(10, 70):
        file_list.append(file_path+str(i)+".csv")

    # Filter coefficients
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')


    for file in file_list:
        #print(f"Processing {file}...")
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000/dt)
        EEG_segment = EEG_filt[0:x1*8]
        reward_value = features.reward_func_simple(np.array(EEG_segment), fs)
        reward_values.append(reward_value)

    return reward_values


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # print(sorted(csv_files))
    # print(sorted(reward_files))

    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')
    reward_values = []
    reward_values_final_segment = []
    reward_values_final_segment3 = []

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        #print(file, reward_file)
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]

        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue
        #print(rew)
        # if rew > -2:
        #     continue
        # print(file, reward_file)
        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        # different protocol stages
        x1 = int(1000/dt)
        # if segment == 1:
        #     EEG_segment = EEG_filt[0:x1]
        # elif segment == 2:
        #     EEG_segment = EEG_filt[ x1 : x1*2]
        # elif segment == 3:
        #     EEG_segment = EEG_filt[x1*2 : x1*3]
        # elif segment == 4:
        #     EEG_segment = EEG_filt[x1*3 : x1*4]
        # elif segment == 5:
        #     EEG_segment = EEG_filt[x1*4 : ]
        # elif segment == -1:
        #     EEG_segment = EEG_filt
        # else:
        #     print("select segment")
        #     exit()

        reward_value = features.reward_func_simple(np.array(EEG_filt[0:x1]), fs)
        reward_values.append(reward_value)

        reward_final = features.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        reward_values_final_segment.append(reward_final)

        reward_final3 = features.reward_func_simple(np.array(EEG_filt[x1*2 : x1*3]), fs)
        reward_values_final_segment3.append(reward_final3)

    return reward_values, reward_values_final_segment, reward_values_final_segment3


def save_calculated_psd_healthy_mdd(all_freqs, avg_psd, ci_lower, ci_upper,
                                    all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h):
    np.save('../../data/bandit/pre_calculated/mdd_all_freqs.npy', all_freqs)
    np.save('../../data/bandit/pre_calculated/mdd_avg_psd.npy', avg_psd)
    np.save('../../data/bandit/pre_calculated/mdd_ci_lower.npy', ci_lower)
    np.save('../../data/bandit/pre_calculated/mdd_ci_upper.npy', ci_upper)
    np.save('../../data/bandit/pre_calculated/healthy_all_freqs.npy', all_freqs_h)
    np.save('../../data/bandit/pre_calculated/healthy_avg_psd.npy', avg_psd_h)
    np.save('../../data/bandit/pre_calculated/healthy_ci_lower.npy', ci_lower_h)
    np.save('../../data/bandit/pre_calculated/healthy_ci_upper.npy', ci_upper_h)


def load_calculated_psd_healthy_mdd():
    all_freqs = np.load('../../data/bandit/pre_calculated/mdd_all_freqs.npy')
    avg_psd = np.load('../../data/bandit/pre_calculated/mdd_avg_psd.npy')
    ci_lower = np.load('../../data/bandit/pre_calculated/mdd_ci_lower.npy')
    ci_upper = np.load('../../data/bandit/pre_calculated/mdd_ci_upper.npy')
    all_freqs_h = np.load('../../data/bandit/pre_calculated/healthy_all_freqs.npy')
    avg_psd_h = np.load('../../data/bandit/pre_calculated/healthy_avg_psd.npy')
    ci_lower_h = np.load('../../data/bandit/pre_calculated/healthy_ci_lower.npy')
    ci_upper_h = np.load('../../data/bandit/pre_calculated/healthy_ci_upper.npy')
    return all_freqs, avg_psd, ci_lower, ci_upper, all_freqs_h, avg_psd_h, ci_lower_h, ci_upper_h

#### MAIN CODE ###

SELECTED_ARM = 1
SEGEMENT = 1
AMP = [1, 2, 4, 2, 2, 15]  # mA
FREQ = [8, 8, 8, 10, 40, 77.5]  # Hz

# reward_values_mdd = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
# reward_values_healthy = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")
#
# print(min(reward_values_healthy))
# print(max(reward_values_healthy))
# exit()
# -0.09264591737143694
# -0.0008928692071909251

# print(len(reward_values_mdd))
# print(len(reward_values_healthy))
# #
# plt.figure(figsize=(10, 5))
# plt.hist(reward_values_mdd, bins=10, alpha=0.5, label='MDD', orientation='horizontal')  # semi-transparent bars for x
# plt.hist(reward_values_healthy, bins=10, alpha=0.5, label='Healthy', orientation='horizontal')  # same bins for y
# plt.axhline(y=-0.09264591737143694, color='r', label='Cutoff value', linestyle='--', alpha=0.7)
# plt.legend()                              # show labels
# plt.xlabel('Frequency')
# plt.ylabel('Reward/Score')
#
# plt.show()
# exit()

reward_values, reward_values_final_segment, reward_values_final_segment3 = process_bandit_testing(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=5)

print(len(reward_values))
print(len(reward_values_final_segment))
print(reward_values)
print(reward_values_final_segment)


# plt.figure()
# plt.hist(reward_values, bins=30, alpha=0.5, label='Segment 1', orientation='horizontal')  # semi-transparent bars for x
# plt.hist(reward_values_final_segment, bins=30, alpha=0.5, label='Segment 5', orientation='horizontal')  # same bins for y
# plt.legend()                              # show labels
# plt.xlabel('Frequency')
# plt.ylabel('Reward')
# plt.title('Overlaid Histograms of x and y')
# plt.show()
# exit()

plt.figure(figsize=(8, 10))

# Plot line segments
up = 0
for a_val, b_val, c_val in zip(reward_values, reward_values_final_segment, reward_values_final_segment3):
    if a_val >= -0.09264591737143694:  # a_val >= b_val and
        c = 'r'
    elif a_val >= b_val:
        c = 'k'
        up += 1
    else:
        c ='g'
    plt.plot([0, 1, 2], [a_val, c_val, b_val], color=c, alpha=0.6, marker='o',            # draw circles at each data point
                    markersize=6,          # size of the circles
                    markerfacecolor=c,     # fill color same as line
                    markeredgecolor=c )

# Customize plot
plt.xlim(-0.1, 1.1)
plt.xticks([0, 1, 2], ['Segment 1', 'Segment 3', 'Segment 5'])
plt.ylabel('Reward')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(up)

