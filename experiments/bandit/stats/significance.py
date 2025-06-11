import os
import glob
import sys
from decouple import config

MAIN_PATH = config('MAIN_PATH')
sys.path.insert(1, MAIN_PATH)

import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as ss
import numpy as np
from scipy.stats import t, shapiro, ttest_ind, mannwhitneyu
from scipy.signal import stft
from scipy import stats
from env.eeg import features  # if needed elsewhere

from experiments.bandit.stats.configs import get_configs
dt, fs, nperseg, _, t1 = get_configs()


def process_bandit_testing(folder_path, selected_arm=1, segment=4):
    """
    Process EEG files and return frequencies and list of PSD arrays for chosen segment.
    """
    csv_files = glob.glob(os.path.join(folder_path, "EEG_BANDIT_*.csv"))
    reward_files = glob.glob(os.path.join(folder_path, "STIM_BANDIT_*.csv"))
    # Design bandpass filter
    b, a = ss.butter(N=2, Wn=[.1, 100.], btype='bandpass', fs=fs, output='ba')

    all_psd = []
    all_freqs = None

    for file, reward_file in zip(sorted(csv_files), sorted(reward_files)):
        df = pd.read_csv(reward_file)
        rew = df['Reward'].values[0]
        arm = df['Arm'].values[0]
        if arm != selected_arm:
            continue

        EEG = np.loadtxt(file, delimiter=",")
        EEG_filt = ss.filtfilt(b, a, EEG[t1:], axis=-1)

        x1 = int(1000 / dt)
        # Filtering criteria; example: skip very negative rewards

        rew = features.reward_func_simple(np.array(EEG_filt[x1*4 : ]), fs)
        if rew < -1.82668915:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
            continue
        # if rew < -2.00017065:  # 75% 1.3929; 80%: 1.5272; 78% 1.5133; 90%: 2.0399; 85% 1.7268
        #     continue

        # Select segment
        if segment == 1:
            EEG_segment = EEG_filt[0:x1]
        elif segment == 2:
            EEG_segment = EEG_filt[x1: x1*2]
        elif segment == 3:
            EEG_segment = EEG_filt[x1*2: x1*3]
        elif segment == 4:
            EEG_segment = EEG_filt[x1*3: x1*4]
        elif segment == 5:
            EEG_segment = EEG_filt[x1*4:]
        elif segment == -1:
            EEG_segment = EEG_filt
        else:
            raise ValueError("Invalid segment number")

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg)  #, noverlap=noverlap
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    return all_freqs, all_psd


def cohen_d(x, y):
    """
    Compute Cohen's d pooled effect size for two independent samples.
    """
    nx, ny = len(x), len(y)
    # If too small or zero variance, handle carefully
    var_x = np.var(x, ddof=1)
    var_y = np.var(y, ddof=1)
    if nx + ny - 2 <= 0 or var_x + var_y == 0:
        return np.nan
    pooled_var = ((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2)
    pooled_sd = np.sqrt(pooled_var)
    return (np.mean(x) - np.mean(y)) / pooled_sd


def rank_biserial_u(u_stat, n1, n2):
    """
    Compute rank-biserial correlation from Mann-Whitney U statistic.
    Formula: r = 1 - (2U)/(n1*n2)
    See: Kerby (2014) “The Simple Difference Formula: An approach to teaching nonparametric correlation”
    """
    return 1 - (2 * u_stat) / (n1 * n2)


def get_stats(freqs, all_psd_group1, all_psd_group2, low=4, high=8, alpha=0.05, plot=False):
    """
    Compute band power in [low, high] Hz for two groups of PSD arrays,
    then perform:
      - Normality checks (Shapiro-Wilk)
      - Welch's t-test + Cohen's d
      - Mann-Whitney U test + rank-biserial effect size
    Optionally plot histograms/Q-Q plots if plot=True.
    """
    # Identify frequency indices
    theta_idx = np.where((freqs >= low) & (freqs <= high))[0]
    if len(theta_idx) == 0:
        raise ValueError(f"No frequencies in the range {low}-{high} Hz")

    def band_power_list(all_psd):
        return np.array([np.trapz(psd[theta_idx], freqs[theta_idx]) for psd in all_psd])

    data1 = band_power_list(all_psd_group1)
    data2 = band_power_list(all_psd_group2)

    n1, n2 = len(data1), len(data2)
    print(f"Sample sizes: group1 = {n1}, group2 = {n2}")
    print(f"Group1 ({low}-{high}Hz) mean ± SD: {np.mean(data1):.4g} ± {np.std(data1, ddof=1):.4g}")
    print(f"Group2 ({low}-{high}Hz) mean ± SD: {np.mean(data2):.4g} ± {np.std(data2, ddof=1):.4g}")

    # Normality checks: Shapiro-Wilk (note: small samples limit power; if n>5000, skip or use other)
    def do_shapiro(arr, label):
        if len(arr) < 3:
            print(f"  Shapiro-Wilk for {label}: sample size too small to test normality reliably.")
            return
        try:
            stat, p = shapiro(arr)
            print(f"  Shapiro-Wilk {label}: W={stat:.4f}, p={p:.4f} ({'reject normality' if p < alpha else 'fail to reject normality'})")
        except Exception as e:
            print(f"  Shapiro-Wilk {label}: could not compute ({e})")
    do_shapiro(data1, "group1")
    do_shapiro(data2, "group2")

    if plot:
        # Histogram + Q-Q plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.ravel()
        axes[0].hist(data1, bins='auto', edgecolor='black')
        axes[0].set_title(f'Group1 Histogram ({low}-{high}Hz)')
        stats.probplot(data1, dist="norm", plot=axes[1])
        axes[1].set_title('Group1 Q–Q plot')
        axes[2].hist(data2, bins='auto', edgecolor='black')
        axes[2].set_title(f'Group2 Histogram ({low}-{high}Hz)')
        stats.probplot(data2, dist="norm", plot=axes[3])
        axes[3].set_title('Group2 Q–Q plot')
        plt.tight_layout()
        plt.show()

    # Welch's t-test
    t_stat, p_t = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
    d = cohen_d(data1, data2)
    print("\nWelch's t-test:")
    print(f"  t = {t_stat:.4f}, p = {p_t:.4e} ({'significant' if p_t < alpha else 'ns'})")
    print(f"  Cohen's d = {d:.4f}")

    # Mann-Whitney U test
    # Note: If there are ties, default method handles them; alternative='two-sided' tests difference in distributions.
    try:
        u_stat, p_u = mannwhitneyu(data1, data2, alternative='two-sided')
        r_rb = rank_biserial_u(u_stat, n1, n2)
        print("\nMann–Whitney U test:")
        print(f"  U = {u_stat}, p = {p_u:.4e} ({'significant' if p_u < alpha else 'ns'})")
        print(f"  Rank-biserial effect size r = {r_rb:.4f}  (positive means median(data1) > median(data2))")
    except ValueError as e:
        print(f"\nMann–Whitney U test could not be performed: {e}")

    # Return a dict of results if further processing is needed
    return {
        'data1': data1,
        'data2': data2,
        't_stat': t_stat, 'p_t': p_t, 'cohen_d': d,
        'u_stat': u_stat if 'u_stat' in locals() else None,
        'p_u': p_u if 'p_u' in locals() else None,
        'rank_biserial': r_rb if 'r_rb' in locals() else None
    }

if __name__ == "__main__":
    SELECTED_ARM = 1
    base_folder = "../../../data/bandit/simnibsbandit3/training"
    all_freqs_seg5, all_psdb_seg5 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=5)
    all_freqs_seg1, all_psd_seg1 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=1)

    freqs = all_freqs_seg5
    if freqs is None:
        raise RuntimeError("No PSD data found. Check folder paths and file patterns.")

    # Example: check 4-8 Hz
    print("=== 4–8 Hz band ===")
    res_4_8 = get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=4, high=8, plot=False)
    print("\n=== 8–12 Hz band ===")
    res_8_12 = get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=8, high=12, plot=False)
    print("\n=== 12–16 Hz band ===")
    res_12_16 = get_stats(freqs, all_psd_seg1, all_psdb_seg5, low=12, high=16, plot=False)
