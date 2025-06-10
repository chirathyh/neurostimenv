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
#
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


def medcouple(x):
    """
    Compute the medcouple statistic, a robust measure of skewness.
    Adapted from Hubert & Vandervelken (2008) and Coe (2020).
    x : 1D array-like, must be sorted
    Returns: float (medcouple value)
    """
    x = np.sort(np.asarray(x))
    n = len(x)
    median = np.median(x)

    # Split into lower and upper halves relative to median
    x_low = x[x <= median]
    x_high = x[x >= median]

    # Create all pairwise kernel values
    def kernel(u, v):
        return (v + u - 2*median) / (v - u) if v != u else np.sign(v - median)

    # Vectorized kernel computation
    Z = []
    for i in range(len(x_low)):
        for j in range(len(x_high)):
            u = x_low[i]
            v = x_high[j]
            if v != u:
                Z.append((v + u - 2*median) / (v - u))
            else:
                Z.append(np.sign(v - median))
    Z = np.array(Z)
    return np.median(Z)


def winsorize_iqr_skewed(x, k=1.5):
    """
    Detect and winsorize outliers based on adjusted 1.5*IQR rule
    for skewed data using the medcouple statistic.

    Parameters:
    x : list or 1D numpy array
    k : float, multiplier for IQR (default 1.5)

    Returns:
    winsorized_x : numpy array with outliers winsorized
    lower_fence, upper_fence : floats, used fences for winsorization
    medcouple_val : float, skewness measure
    """
    x = np.asarray(x) * -1

    print("DATA")
    print(x)

    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1

    mc = medcouple(x)

    print("MC manual")
    print(mc)
    #
    # print("MC library")
    # def _medcouple_1d(y):
    #     """
    #     Calculates the medcouple robust measure of skew.
    #
    #     Parameters
    #     ----------
    #     y : array_like, 1-d
    #         Data to compute use in the estimator.
    #
    #     Returns
    #     -------
    #     mc : float
    #         The medcouple statistic
    #
    #     Notes
    #     -----
    #     The current algorithm requires a O(N**2) memory allocations, and so may
    #     not work for very large arrays (N>10000).
    #
    #     .. [*] M. Hubert and E. Vandervieren, "An adjusted boxplot for skewed
    #        distributions" Computational Statistics & Data Analysis, vol. 52, pp.
    #        5186-5201, August 2008.
    #     """
    #
    #     # Parameter changes the algorithm to the slower for large n
    #
    #     y = np.squeeze(np.asarray(y))
    #     if y.ndim != 1:
    #         raise ValueError("y must be squeezable to a 1-d array")
    #
    #     y = np.sort(y)
    #
    #     n = y.shape[0]
    #     if n % 2 == 0:
    #         mf = (y[n // 2 - 1] + y[n // 2]) / 2
    #     else:
    #         mf = y[(n - 1) // 2]
    #
    #     z = y - mf
    #     lower = z[z <= 0.0]
    #     upper = z[z >= 0.0]
    #     upper = upper[:, None]
    #     standardization = upper - lower
    #     is_zero = np.logical_and(lower == 0.0, upper == 0.0)
    #     standardization[is_zero] = np.inf
    #     spread = upper + lower
    #     h = spread / standardization
    #     # GH5395
    #     num_ties = np.sum(lower == 0.0)
    #     if num_ties:
    #         # Replacements has -1 above the anti-diagonal, 0 on the anti-diagonal,
    #         # and 1 below the anti-diagonal
    #         replacements = np.ones((num_ties, num_ties)) - np.eye(num_ties)
    #         replacements -= 2 * np.triu(replacements)
    #         # Convert diagonal to anti-diagonal
    #         replacements = np.fliplr(replacements)
    #         # Always replace upper right block
    #         h[:num_ties, -num_ties:] = replacements
    #
    #     return np.median(h)
    # mc_new = _medcouple_1d(x)
    # print(mc_new)
    # exit()

    # Adjust fences based on medcouple (Hubert & Van der Veeken 2008)
    if mc > 0:
        # lower_fence = q1 - k * np.exp(-4 * mc) * iqr
        # upper_fence = q3 + k * np.exp(3 * mc) * iqr

        # clipped
        adj_lower = np.clip(np.exp(-4 * mc), 0.5, 2)
        adj_upper = np.clip(np.exp(3 * mc), 0.5, 4)
        lower_fence = q1 - k * adj_lower * iqr
        upper_fence = q3 + k * adj_upper * iqr

    else:
        lower_fence = q1 - k * np.exp(-3 * mc) * iqr
        upper_fence = q3 + k * np.exp(4 * mc) * iqr

    winsorized_x = np.copy(x)
    winsorized_x[winsorized_x < lower_fence] = lower_fence
    winsorized_x[winsorized_x > upper_fence] = upper_fence

    return winsorized_x, lower_fence, upper_fence, mc


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

print("\n=== IQR MC based Filtering ===")
winsorized_x, lf, uf, mc_val = winsorize_iqr_skewed(reward_values_final_segment)
print(f"Medcouple (skewness measure): {mc_val:.4f}")
print(f"Lower fence: {lf:.2f}, Upper fence: {uf:.2f}")
print(f"Winsorized data: {winsorized_x}")
print(len(winsorized_x))

exit()

print("\n=== Quantile-based Filtering ===")
# low_q, high_q, mask_q = quantile_filter(reward_values, lower_q=0.05, upper_q=0.75)
# print(f"95% range: {low_q:.4f} - {high_q:.4f}")

low_q, high_q, mask_q = quantile_filter(reward_values_final_segment, lower_q=0.05, upper_q=0.85)
print(f"95% range: {low_q:.4f} - {high_q:.4f}")
exit()


mean_val, sd_val, lower, upper = compute_thresholds(reward_values_final_segment, n_sd=2.5)
print(f"Mean = {mean_val:.4f}")
print(f"Sample SD = {sd_val:.4f}")
print(f"Lower threshold (mean - 2.5 SD) = {lower:.4f}")
print(f"Upper threshold (mean + 2.5 SD) = {upper:.4f}")


exit()



plt.figure()
plt.hist(reward_values, bins=30, alpha=0.5, label='Segment 1', orientation='horizontal')  # semi-transparent bars for x
plt.hist(reward_values_final_segment, bins=30, alpha=0.5, label='Segment 5', orientation='horizontal')  # same bins for y
plt.legend()                              # show labels
plt.xlabel('Frequency')
plt.ylabel('Reward')
plt.title('Overlaid Histograms of x and y')
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

