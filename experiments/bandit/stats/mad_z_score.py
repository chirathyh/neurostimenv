import numpy as np


def mad_outlier_mask(x, thresh=3.5, one_sided='upper'):
    """
    Compute modified Z-scores based on median/MAD.
    Flags outliers where |modified_z| > thresh.
    If one_sided == 'upper', only flags x_i > median.
    Returns mask of non-outliers, array of modified z-scores.
    """
    # x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.array([], dtype=bool), np.array([])
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        # If all values equal or MAD zero: no outliers by this rule
        modz = np.zeros_like(x)
        mask = np.ones_like(x, dtype=bool)
        return mask, modz
    modz = 0.6745 * (x - med) / mad
    if one_sided == 'upper':
        mask = modz <= thresh
    elif one_sided == 'both':
        mask = np.abs(modz) <= thresh
    else:
        raise ValueError("one_sided must be 'upper' or 'both'")
    return mask, modz


