import os
import glob

import os
import re
import sys
from decouple import config
import statistics

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
    'font.size': 16,
    'axes.titlesize': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
    'figure.titlesize': 18,
})



def compute_thresholds(x, n_sd=2.5):
    """
    Given a list or iterable x, compute:
      - mean
      - sample standard deviation (SD; ddof=1)
      - lower threshold = mean - n_sd * SD
      - upper threshold = mean + n_sd * SD

    Raises ValueError if x is empty.
    For a single-element list, SD is set to 0.0 and thresholds == mean.
    """
    # Ensure x is a sequence
    x_list = list(x)
    if len(x_list) == 0:
        raise ValueError("Input list is empty; cannot compute mean/SD.")
    mean_val = statistics.mean(x_list)
    if len(x_list) == 1:
        sd_val = 0.0
    else:
        # statistics.stdev uses sample SD (ddof=1)
        sd_val = statistics.stdev(x_list)
    lower = mean_val - n_sd * sd_val
    upper = mean_val + n_sd * sd_val
    return mean_val, sd_val, lower, upper



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
        EEG_segment = EEG_filt#[0:x1*8]
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

    return reward_values, reward_values_final_segment


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

reward_values_mdd = process_eeg(file_path="../../data/feature_analysis/mdd/EEG_MDD_")
reward_values_healthy = process_eeg(file_path="../../data/feature_analysis/healthy/EEG_HEALTHY_")
#
# print(min(reward_values_healthy))
# print(max(reward_values_healthy))
# exit()
# -0.09264591737143694
# -0.0008928692071909251

# print(len(reward_values_mdd))
# print(len(reward_values_healthy))
#
plt.figure(figsize=(10, 5))
plt.hist(reward_values_mdd, color='red', bins=10, label='Depression', orientation='horizontal')  # semi-transparent bars for x alpha=0.5,
plt.hist(reward_values_healthy, color='black', bins=10, label='Healthy', orientation='horizontal')  # same bins for y alpha=0.5,
plt.axhline(y=-0.09264591737143694, color='green', label='Cutoff value', linestyle='--', alpha=0.7)
plt.legend()                              # show labels
plt.xlabel('Frequency')
plt.ylabel('Reward/Score')

plt.show()
exit()

def quantile_filter(x, lower_q=0.01, upper_q=0.99):
    """
    Returns the lower and upper bounds based on quantile cutoffs.
    Useful for trimming outliers in skewed data.

    Args:
        x: list or np.array
        lower_q: Lower quantile (e.g., 0.01 for 1st percentile)
        upper_q: Upper quantile (e.g., 0.99 for 99th percentile)
    """
    x = np.asarray(x, dtype=float) * -1
    lower_bound = np.quantile(x, lower_q)
    upper_bound = np.quantile(x, upper_q)
    mask = (x >= lower_bound) & (x <= upper_bound)
    return lower_bound, upper_bound, mask


reward_values, reward_values_final_segment = process_bandit_testing(folder_path="../../data/bandit/simnibsbandit3/training", selected_arm=SELECTED_ARM, segment=5)

print(len(reward_values))
print(len(reward_values_final_segment))
print(reward_values)
print(reward_values_final_segment)


# Q1 = np.quantile(reward_values_final_segment, 0.25)
# Q3 = np.quantile(reward_values_final_segment, 0.75)
# IQR = Q3 - Q1
# print(1.5 * IQR)
# exit()


# print("\n=== Quantile-based Filtering ===")
# # low_q, high_q, mask_q = quantile_filter(reward_values, lower_q=0.05, upper_q=0.75)
# # print(f"95% range: {low_q:.4f} - {high_q:.4f}")
#
# low_q, high_q, mask_q = quantile_filter(reward_values_final_segment, lower_q=0.05, upper_q=0.85)
# print(f"95% range: {low_q:.4f} - {high_q:.4f}")
# exit()
#
#
# mean_val, sd_val, lower, upper = compute_thresholds(reward_values_final_segment, n_sd=2.5)
# print(f"Mean = {mean_val:.4f}")
# print(f"Sample SD = {sd_val:.4f}")
# print(f"Lower threshold (mean - 2.5 SD) = {lower:.4f}")
# print(f"Upper threshold (mean + 2.5 SD) = {upper:.4f}")

# take the log transform
# reward_values = np.log(np.asarray(reward_values) * -1)
# reward_values = np.log(np.asarray(reward_values) * -1)
# reward_values_final_segment = np.log(np.asarray(reward_values_final_segment) * -1)


plt.figure(figsize=(10, 5))
#plt.hist(reward_values, bins=30, alpha=0.5, label='Segment 1')  # semi-transparent bars for x  , orientation='horizontal'
plt.hist(reward_values_final_segment, bins=120, alpha=0.5, label='EEG Segment 5')  # , orientation='horizontal' ; same bins for y

# plt.axvline(x=-1.3929, color='red', linestyle='--', linewidth=1.5, label='75% (47/63 samples)')
# plt.axvline(x=-1.7268, color='blue', linestyle='--', linewidth=1.5, label='85% (53/63 samples)')
# plt.axvline(x=-2.0399, color='green', linestyle='--', linewidth=1.5, label='90% (56/g3 samples)')
plt.axvline(x=-1.82668915, color='black', linestyle='--', linewidth=1.5, label='2.5 MAD (55/63 samples)')


plt.legend()                              # show labels
plt.xlabel('Reward')
plt.ylabel('Frequency')
# plt.title('Overlaid Histograms of x and y')
plt.show()
exit()

plt.figure(figsize=(5, 10))

# Plot line segments
up = 0
for a_val, b_val in zip(reward_values, reward_values_final_segment):
    if a_val >= -0.09264591737143694:  # a_val >= b_val and
        c = 'r'
    elif a_val >= b_val:
        c = 'k'
        up += 1
    else:
        c ='g'
    plt.plot([0, 1], [a_val, b_val], color=c, alpha=0.6, marker='o',            # draw circles at each data point
                    markersize=6,          # size of the circles
                    markerfacecolor=c,     # fill color same as line
                    markeredgecolor=c )

# Customize plot
plt.xlim(-0.1, 1.1)
plt.xticks([0, 1], ['Segment 1', 'Segment 5'])
plt.ylabel('Reward')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print(up)

