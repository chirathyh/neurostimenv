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
from scipy.stats import shapiro, ttest_ind, mannwhitneyu
from scipy import stats

# Sampling parameters
dt = 0.025
fs = (1 / dt) * 1000  # adjust as before
nperseg = int(fs / 2)
noverlap = int(nperseg * 0.5)

transient = 4000  # ms; first 4s removed
t1 = int(transient / dt)

print("Sampling Rate:", fs)
print("nperseg:", nperseg)

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
        # if rew < -1.5:
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

        freqs, psd = ss.welch(EEG_segment, fs=fs, nperseg=nperseg, noverlap=noverlap)
        if all_freqs is None:
            all_freqs = freqs
        all_psd.append(psd)

    return all_freqs, all_psd

def cohen_d(x, y):
    """
    Compute Cohen's d pooled effect size for two independent samples.
    """
    nx, ny = len(x), len(y)
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
    r = 1 - (2U)/(n1*n2)
    """
    return 1 - (2 * u_stat) / (n1 * n2)

def get_stats(freqs, all_psd_group1, all_psd_group2,
              low=4, high=8,
              alpha=0.05,
              plot=False,
              remove_outliers=False,
              outlier_thresh=2.5):
    """
    Compute band power in [low, high] Hz for two groups of PSD arrays,
    then optionally remove outliers beyond ±outlier_thresh*SD for each group,
    and perform:
      - Normality checks (Shapiro-Wilk)
      - Welch's t-test + Cohen's d
      - Mann-Whitney U test + rank-biserial effect size
    Optionally plot histograms/Q-Q plots if plot=True.

    Returns a dict with both “original” and “filtered” results, so you can compare.
    """
    # Identify frequency indices
    idx_band = np.where((freqs >= low) & (freqs <= high))[0]
    if len(idx_band) == 0:
        raise ValueError(f"No frequencies in the range {low}-{high} Hz")

    def band_power_list(all_psd):
        return np.array([np.trapz(psd[idx_band], freqs[idx_band]) for psd in all_psd])

    # Original data arrays
    data1_orig = band_power_list(all_psd_group1)
    data2_orig = band_power_list(all_psd_group2)

    results = {
        'low': low, 'high': high,
        'data1_orig': data1_orig,
        'data2_orig': data2_orig,
        'n1_orig': len(data1_orig),
        'n2_orig': len(data2_orig),
    }

    def do_analysis(data1, data2, label_suffix=""):
        """
        Run tests on provided data arrays and print summary.
        Returns a dict of results.
        """
        n1, n2 = len(data1), len(data2)
        print(f"\n--- Analysis{label_suffix} ---")
        print(f"Sample sizes: group1 = {n1}, group2 = {n2}")

        if n1 == 0 or n2 == 0:
            print("  One of the groups has zero samples after filtering; skipping statistical tests.")
            return {
                'n1': n1, 'n2': n2,
                'shapiro1': None, 'shapiro2': None,
                't_stat': None, 'p_t': None, 'cohen_d': None,
                'u_stat': None, 'p_u': None, 'rank_biserial': None
            }

        mean1, sd1 = np.mean(data1), np.std(data1, ddof=1) if n1 > 1 else (np.mean(data1), np.nan)
        mean2, sd2 = np.mean(data2), np.std(data2, ddof=1) if n2 > 1 else (np.mean(data2), np.nan)
        print(f"Group1 ({low}-{high}Hz) mean ± SD: {mean1:.4g} ± {sd1:.4g}")
        print(f"Group2 ({low}-{high}Hz) mean ± SD: {mean2:.4g} ± {sd2:.4g}")

        # Normality checks
        def shapiro_check(arr, name):
            if len(arr) < 3:
                print(f"  Shapiro-Wilk for {name}: sample size {len(arr)} too small to test normality reliably.")
                return None
            try:
                stat, p = shapiro(arr)
                print(f"  Shapiro-Wilk {name}: W={stat:.4f}, p={p:.4f} ({'reject normality' if p < alpha else 'fail to reject normality'})")
                return (stat, p)
            except Exception as e:
                print(f"  Shapiro-Wilk {name}: error ({e})")
                return None

        shapiro1 = shapiro_check(data1, "group1")
        shapiro2 = shapiro_check(data2, "group2")

        if plot:
            # Histogram + Q-Q plot
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            axes = axes.ravel()
            axes[0].hist(data1, bins='auto', edgecolor='black')
            axes[0].set_title(f'Group1 Histogram ({low}-{high}Hz){label_suffix}')
            stats.probplot(data1, dist="norm", plot=axes[1])
            axes[1].set_title(f'Group1 Q–Q plot{label_suffix}')
            axes[2].hist(data2, bins='auto', edgecolor='black')
            axes[2].set_title(f'Group2 Histogram ({low}-{high}Hz){label_suffix}')
            stats.probplot(data2, dist="norm", plot=axes[3])
            axes[3].set_title(f'Group2 Q–Q plot{label_suffix}')
            plt.tight_layout()
            plt.show()

        # Welch's t-test
        try:
            t_stat, p_t = ttest_ind(data1, data2, equal_var=False, nan_policy='omit')
        except Exception as e:
            t_stat, p_t = None, None
            print(f"  Welch's t-test error: {e}")
        d = cohen_d(data1, data2) if (n1 > 1 and n2 > 1) else np.nan
        print("\n  Welch's t-test:")
        if t_stat is None:
            print("    Could not compute t-test.")
        else:
            sig = 'significant' if (p_t is not None and p_t < alpha) else 'ns'
            print(f"    t = {t_stat:.4f}, p = {p_t:.4e} ({sig})")
            print(f"    Cohen's d = {d:.4f}")

        # Mann-Whitney U test
        try:
            u_stat, p_u = mannwhitneyu(data1, data2, alternative='two-sided')
            r_rb = rank_biserial_u(u_stat, n1, n2)
            print("\n  Mann–Whitney U test:")
            sig_u = 'significant' if p_u < alpha else 'ns'
            print(f"    U = {u_stat}, p = {p_u:.4e} ({sig_u})")
            print(f"    Rank-biserial r = {r_rb:.4f}  (positive means median(group1) > median(group2))")
        except Exception as e:
            u_stat, p_u, r_rb = None, None, None
            print(f"  Mann–Whitney U test error: {e}")

        return {
            'n1': n1, 'n2': n2,
            'shapiro1': shapiro1, 'shapiro2': shapiro2,
            't_stat': t_stat, 'p_t': p_t, 'cohen_d': d,
            'u_stat': u_stat, 'p_u': p_u, 'rank_biserial': r_rb
        }

    # First, analysis on original data
    orig_results = do_analysis(data1_orig, data2_orig, label_suffix=" (original)")

    results.update({'orig': orig_results})

    # If requested, remove outliers per group
    if remove_outliers:
        # For group1
        mean1, sd1 = np.mean(data1_orig), np.std(data1_orig, ddof=1) if len(data1_orig) > 1 else (np.mean(data1_orig), np.nan)
        if np.isnan(sd1) or sd1 == 0:
            mask1 = np.ones_like(data1_orig, dtype=bool)
        else:
            mask1 = np.abs(data1_orig - mean1) <= outlier_thresh * sd1
        data1_filt = data1_orig[mask1]
        removed1 = len(data1_orig) - len(data1_filt)
        # For group2
        mean2, sd2 = np.mean(data2_orig), np.std(data2_orig, ddof=1) if len(data2_orig) > 1 else (np.mean(data2_orig), np.nan)
        if np.isnan(sd2) or sd2 == 0:
            mask2 = np.ones_like(data2_orig, dtype=bool)
        else:
            mask2 = np.abs(data2_orig - mean2) <= outlier_thresh * sd2
        data2_filt = data2_orig[mask2]
        removed2 = len(data2_orig) - len(data2_filt)

        print(f"\nOutlier removal: threshold = ±{outlier_thresh} SD from mean")
        print(f"  Group1: removed {removed1} / {len(data1_orig)} samples")
        print(f"  Group2: removed {removed2} / {len(data2_orig)} samples")

        filt_results = do_analysis(data1_filt, data2_filt, label_suffix=f" (filtered ±{outlier_thresh}SD)")
        results.update({
            'data1_filt': data1_filt,
            'data2_filt': data2_filt,
            'n1_filt': len(data1_filt),
            'n2_filt': len(data2_filt),
            'filt': filt_results
        })
    else:
        print("\nOutlier removal not requested; using original data only.")

    return results

if __name__ == "__main__":
    SELECTED_ARM = 1
    base_folder = "../../data/bandit/simnibsbandit3/training"
    all_freqs_seg5, all_psdb_seg5 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=5)
    all_freqs_seg1, all_psd_seg1 = process_bandit_testing(folder_path=base_folder, selected_arm=SELECTED_ARM, segment=1)

    freqs = all_freqs_seg5
    if freqs is None:
        raise RuntimeError("No PSD data found. Check folder paths and file patterns.")

    # Example: 4–8 Hz without removal
    print("=== 4–8 Hz band, without outlier removal ===")
    res_4_8_orig = get_stats(freqs, all_psd_seg1, all_psdb_seg5,
                             low=4, high=8,
                             plot=False,
                             remove_outliers=False)

    # Example: 4–8 Hz with removal at 2.5 SD
    print("\n=== 4–8 Hz band, with outlier removal ±2.5 SD ===")
    res_4_8_filt = get_stats(freqs, all_psd_seg1, all_psdb_seg5,
                             low=4, high=8,
                             plot=False,
                             remove_outliers=True,
                             outlier_thresh=2.)

    # Similarly for other bands:
    print("\n=== 8–12 Hz band, with outlier removal ±2.5 SD ===")
    res_8_12 = get_stats(freqs, all_psd_seg1, all_psdb_seg5,
                         low=8, high=12,
                         plot=False,
                         remove_outliers=True,
                         outlier_thresh=2.)

    print("\n=== 12–16 Hz band, with outlier removal ±2.5 SD ===")
    res_12_16 = get_stats(freqs, all_psd_seg1, all_psdb_seg5,
                          low=12, high=16,
                          plot=False,
                          remove_outliers=True,
                          outlier_thresh=2.)
