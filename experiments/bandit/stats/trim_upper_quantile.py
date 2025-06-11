import numpy as np


def trim_upper_quantile(x, quantile=0.95):
    """
    Remove values above the specified upper quantile.
    Returns filtered data and the threshold used.
    """
    if x.size == 0:
        return x, None
    thresh = np.quantile(x, quantile)
    mask = x <= thresh
    return x[mask], thresh

