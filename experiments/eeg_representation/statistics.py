"""Structure-level inference and finite-fork estimators.

No function treats windows, all pairwise future differences, or optimizer
replicates as extra independent circuit structures.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

import numpy as np


def _finite(values: object, name: str = "values") -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not result.size or not np.isfinite(result).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return result


def per_state_fork_variance(futures: object, *, future_axis: int = 1) -> np.ndarray:
    """Unbiased trace(sample covariance)/d for each state.

    Input convention is (state, independent future, *target dimensions).
    Returns one scalar per state; it does not average states or structures.
    """
    x = _finite(futures, "futures")
    if x.ndim < 2 or future_axis % x.ndim == 0:
        raise ValueError("need a distinct state and future axis")
    x = np.moveaxis(x, future_axis, 1)
    if x.shape[1] < 2:
        raise ValueError("at least two independent futures are required")
    variance = x.var(axis=1, ddof=1)
    return variance.reshape(x.shape[0], -1).mean(axis=1)


def independent_pair_variance(futures: object, *, future_axis: int = 1) -> np.ndarray:
    """Disjoint-pair 0.5*mean squared difference, separately for every state.

    Drops the last future when the count is odd. Returns (state, pair) so
    Monte Carlo uncertainty remains distinguishable from state variability.
    """
    x = _finite(futures, "futures")
    if x.ndim < 2 or future_axis % x.ndim == 0:
        raise ValueError("need a distinct state and future axis")
    x = np.moveaxis(x, future_axis, 1)
    count = x.shape[1] // 2
    if not count:
        raise ValueError("at least two independent futures are required")
    differences = x[:, : 2 * count : 2] - x[:, 1 : 2 * count : 2]
    return 0.5 * np.square(differences.reshape(x.shape[0], count, -1)).mean(axis=-1)


def leave_one_out_oracle(futures: object, *, future_axis: int = 1) -> np.ndarray:
    """For each scored future, the mean of all *other* independent futures.

    Its expected scored MSE is R* times n/(n-1), not R*. This finite-ensemble
    cost must remain labelled; multiply by (n-1)/n for a variance estimator.
    """
    x = _finite(futures, "futures")
    n = x.shape[future_axis]
    if n < 2:
        raise ValueError("at least two independent futures are required")
    return (x.sum(axis=future_axis, keepdims=True) - x) / (n - 1)


def fork_summary(futures: object, *, seed: int = 0, n_bootstrap: int = 1000) -> dict:
    """Per-state future Monte Carlo intervals; no population/structure claim.

    Bootstrap resamples future indices independently inside each fixed state.
    Between-state or between-structure inference is intentionally separate.
    """
    x = _finite(futures, "futures")
    estimates = per_state_fork_variance(x)
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least two")
    rng = np.random.default_rng(seed)
    bounds = []
    for state in x:
        replicates = []
        for _ in range(n_bootstrap):
            sample = state[rng.integers(0, len(state), len(state))]
            replicates.append(float(sample.var(axis=0, ddof=1).mean()))
        bounds.append(np.quantile(replicates, [0.025, 0.975]).tolist())
    return {
        "per_state_variance": estimates.tolist(),
        "per_state_future_mc_ci95": bounds,
        "mean_over_states": float(estimates.mean()),
        "independent_pair_estimates": independent_pair_variance(x).tolist(),
        "states": len(x),
        "futures_per_state": x.shape[1],
        "interpretation": "full-causal-state conditional variance; lower bound on EEG-history Bayes risk",
        "interval_unit": "independent futures conditional on a fixed state; not circuit population uncertainty",
    }


def normalized_log_budget_auc(budgets: Sequence[float], scores: Sequence[float]) -> float:
    """Trapezoidal BA area divided by the frozen log-budget range."""
    n = _finite(budgets, "budgets")
    y = _finite(scores, "scores")
    if n.ndim != 1 or y.shape != n.shape or len(n) < 2:
        raise ValueError("at least two corresponding one-dimensional budgets and scores are required")
    if np.any(n <= 0) or np.any(np.diff(n) <= 0):
        raise ValueError("budgets must be positive and strictly increasing")
    if np.any((y < 0) | (y > 1)):
        raise ValueError("balanced accuracy must lie in [0, 1]")
    log_n = np.log(n)
    return float(np.trapz(y, log_n) / (log_n[-1] - log_n[0]))


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    """BH adjusted p-values, preserving declared family order."""
    p = _finite(p_values, "p_values")
    if p.ndim != 1 or np.any((p < 0) | (p > 1)):
        raise ValueError("p_values must be a vector in [0, 1]")
    order = np.argsort(p, kind="stable")
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def exact_sign_flip(values: Sequence[float], *, alternative: str = "two-sided") -> dict:
    """Exact paired structure-mean sign-flip test, including observed pattern.

    Requires exchangeability/symmetry of the paired null; not a test justified
    merely by a large number of repeated EEG windows. Hard-limited at 20
    structures to avoid silently changing exact inference to Monte Carlo.
    """
    x = _finite(values)
    if x.ndim != 1 or len(x) > 20:
        raise ValueError("exact sign flips require 1..20 structure-level differences")
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValueError("invalid alternative")
    observed = float(x.mean())
    tolerance = np.finfo(float).eps * max(float(np.abs(x).max()), np.finfo(float).tiny) * 16
    extreme = 0
    for signs in product((-1, 1), repeat=len(x)):
        statistic = float(np.dot(signs, x) / len(x))
        if alternative == "two-sided":
            extreme += abs(statistic) >= abs(observed) - tolerance
        elif alternative == "greater":
            extreme += statistic >= observed - tolerance
        else:
            extreme += statistic <= observed + tolerance
    return {"estimate": observed, "p_value": extreme / (2 ** len(x)), "n_structures": len(x),
            "patterns": 2 ** len(x), "alternative": alternative, "method": "exact_sign_flip"}


def paired_cluster_interval(
    left: Sequence[float], right: Sequence[float], clusters: Sequence[object], *,
    seed: int = 0, n_bootstrap: int = 5000, confidence: float = 0.95,
) -> dict:
    """Paired equal-structure-weight contrast and circuit-cluster bootstrap.

    Average all paired repetitions inside each structure first. Optimizer
    seeds and windows therefore cannot inflate the number of independent units.
    """
    a, b = _finite(left, "left"), _finite(right, "right")
    labels = np.asarray(clusters)
    if a.ndim != 1 or a.shape != b.shape or labels.shape != a.shape:
        raise ValueError("left, right, clusters must be matching vectors")
    if not 0 < confidence < 1 or n_bootstrap < 2:
        raise ValueError("invalid confidence or bootstrap count")
    groups = np.unique(labels)
    differences = np.array([(a[labels == group] - b[labels == group]).mean() for group in groups])
    if len(groups) < 2:
        return {"estimate": float(differences.mean()), "ci": None, "n_structures": 1,
                "status": "insufficient_structures", "structure_differences": differences.tolist()}
    rng = np.random.default_rng(seed)
    samples = rng.choice(differences, (n_bootstrap, len(groups)), replace=True).mean(axis=1)
    tail = (1 - confidence) / 2
    return {"estimate": float(differences.mean()), "ci": np.quantile(samples, [tail, 1 - tail]).tolist(),
            "confidence": confidence, "n_structures": len(groups),
            "structure_differences": differences.tolist(), "method": "paired_equal_weight_circuit_bootstrap"}
