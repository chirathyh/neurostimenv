import numpy as np
from scipy import stats


def ev_outlier_mask(x, threshold_quantile=0.8, tail_prob_thresh=0.05):
    """
    Identify outliers in the upper tail using GPD tail modeling.
    Returns mask of non-outliers.
    """
    # x = np.asarray(x, dtype=float)
    if x.size < 10:
        # Too few points for reliable EVT fit
        return np.ones_like(x, dtype=bool)
    # Step 1: choose threshold u
    u = np.quantile(x, threshold_quantile)
    exceedances = x[x > u] - u
    if exceedances.size < 5:
        # Not enough tail data to fit
        return np.ones_like(x, dtype=bool)
    # Step 2: fit GPD: scipy's fit returns (shape, loc, scale)
    # For GPD, loc should be 0 since we shifted by u
    c, loc, scale = stats.genpareto.fit(exceedances, floc=0)
    # Step 3: for each x_i > u, compute tail probability P(X > x_i)
    # P(X > x_i) = P(X > u) * P(Y > x_i - u | Y ~ GPD)
    # Estimate P(X > u) as proportion in data:
    p_exceed = exceedances.size / x.size
    # Survival function of GPD at y = x_i - u: 1 - CDF
    def tail_prob(xi):
        if xi <= u:
            return 1.0
        y = xi - u
        # GPD survival: stats.genpareto.sf(y, c, loc=0, scale=scale)
        return p_exceed * stats.genpareto.sf(y, c, loc=0, scale=scale)
    # Vectorize
    tail_probs = np.array([tail_prob(xi) for xi in x])
    # Mask non-outliers
    mask = tail_probs >= tail_prob_thresh
    return mask


