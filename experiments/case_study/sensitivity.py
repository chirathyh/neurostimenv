import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats


def mad_outlier_mask(arr, thresh=3.5, one_sided='upper'):
    """
    Compute modified Z-scores based on median/MAD and return a boolean mask
    of non-outliers, along with the modified Z-scores.
    - x: array-like numerical data
    - thresh: threshold for modified Z-score (e.g., 2.5, 3.0, 3.5)
    - one_sided: 'upper' to flag only large positive deviations; 'both' for two-sided.
    """
    # arr = np.asarray(x, dtype=float)
    n = arr.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        # If all values identical or MAD=0, no outliers by this rule
        modz = np.zeros_like(arr)
        mask = np.ones_like(arr, dtype=bool)
        return mask, modz
    modz = 0.6745 * (arr - med) / mad
    if one_sided == 'upper':
        mask = modz <= thresh
    elif one_sided == 'both':
        mask = np.abs(modz) <= thresh
    else:
        raise ValueError("one_sided must be 'upper' or 'both'")
    return mask, modz


def sensitivity_analysis_mad(x, thresholds=[2.5, 3.0, 3.5], one_sided='upper'):
    """
    Perform sensitivity analysis for MAD-based outlier removal across multiple thresholds.
    Returns a DataFrame summarizing removal counts and basic stats before/after.
    Also plots histograms of the original and filtered distributions for each threshold.
    """
    arr = np.asarray(x, dtype=float)
    n = arr.size
    # Prepare results
    results = []
    for thresh in thresholds:
        mask, modz = mad_outlier_mask(arr, thresh=thresh, one_sided=one_sided)
        arr_filtered = arr[mask]
        removed_count = np.sum(~mask)
        kept_count = np.sum(mask)
        # Compute summary stats
        mean_before = np.nan if n == 0 else np.mean(arr)
        median_before = np.nan if n == 0 else np.median(arr)
        std_before = np.nan if n == 0 else np.std(arr, ddof=1) if n > 1 else 0.0
        mean_after = np.nan if arr_filtered.size == 0 else np.mean(arr_filtered)
        median_after = np.nan if arr_filtered.size == 0 else np.median(arr_filtered)
        std_after = np.nan if arr_filtered.size <= 1 else np.std(arr_filtered, ddof=1)
        results.append({
            'threshold': thresh,
            'removed_count': removed_count,
            'kept_count': kept_count,
            'pct_removed': (removed_count / n * 100) if n > 0 else np.nan,
            'mean_before': mean_before,
            'median_before': median_before,
            'std_before': std_before,
            'mean_after': mean_after,
            'median_after': median_after,
            'std_after': std_after
        })

        # Plot histogram comparison
        plt.figure(figsize=(6, 4))
        plt.hist(arr, bins=30, alpha=0.5, label='Original', edgecolor='black')
        if arr_filtered.size > 0:
            plt.hist(arr_filtered, bins=30, alpha=0.5, label=f'Filtered (thresh={thresh})', edgecolor='black')
        plt.title(f'MAD Outlier Removal Sensitivity (threshold={thresh})')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

    df_results = pd.DataFrame(results)
    return df_results

# Example usage:
# Replace this with your actual data array x
# x = np.array([...])
# For demonstration, if x is empty, the results DataFrame will be empty or show NaNs
#x = np.array([])  # User should replace with their data array

# Perform sensitivity analysis

