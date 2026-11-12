"""Finite operational code rates and task-specific held-out readouts (H1c).

A B-bit code is an operational rate upper bound. Neither latent dimension nor
H(T)-held-out CE estimates a universal percentage of EEG information. Decoder
lower-bound differences are never interpreted as certified information loss.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import warnings

import numpy as np
from scipy.special import logsumexp

from ..storage import atomic_json, file_hash, jsonable

VERSION = "h1-compression-v1"
DEFAULTS = dict(dimensions=[8, 32, 128], code_bits=[32, 128, 512],
                quantile_range=[.005, .995], max_scalar_bits=16,
                representations=["feature18", "pca", "untrained", "masked_mse", "forecast"],
                readouts=["linear", "flexible"], label_budgets=[8, 32, "all"],
                regularization=[.01, 1., 100.], flexible_features=64, seed=7301)


def _hash(value):
    return hashlib.sha256(json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _npz(path, **arrays):
    from .artifacts import atomic_npz
    return atomic_npz(path, **arrays)


def entropy_bits(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or np.any(p < 0) or not np.isfinite(p).all() or not np.isclose(p.sum(), 1):
        raise ValueError("Expected a finite probability vector summing to one")
    return float(-np.sum(p[p > 0] * np.log2(p[p > 0])))


def fano_necessary_bits(classes, error):
    """Balanced K-class necessary MI bound for error <= delta <= 1-1/K."""
    if type(classes) is not int or classes < 2 or not 0 <= error <= 1 - 1 / classes:
        raise ValueError("Fano requires K>=2 and 0<=error<=1-1/K")
    h = entropy_bits([error, 1 - error])
    return float(max(0., math.log2(classes) - h - error * math.log2(classes - 1)))


def gaussian_channel(variance, noise_variance):
    if variance <= 0 or noise_variance <= 0 or not np.isfinite([variance, noise_variance]).all():
        raise ValueError("Finite positive Gaussian variances required")
    bits = .5 * math.log2(1 + variance / noise_variance)
    distortion = variance * noise_variance / (variance + noise_variance)
    return dict(mutual_information_bits=bits, posterior_mse=distortion,
                gaussian_rd_bound=variance * 2 ** (-2 * bits),
                scope="scalar Gaussian source, additive independent Gaussian channel, posterior mean decoder")


def gaussian_rate_distortion(variance, information_bits):
    if variance <= 0 or information_bits < 0 or not np.isfinite([variance, information_bits]).all():
        raise ValueError("Gaussian RD requires positive variance and nonnegative finite bits")
    return float(variance * 2 ** (-2 * information_bits))


def _discrete_entropy(rows):
    rows = np.asarray(rows)
    if rows.ndim == 1:
        rows = rows[:, None]
    _, counts = np.unique(rows, axis=0, return_counts=True)
    return entropy_bits(counts / counts.sum())


def ten_bit_counterexample():
    v = np.asarray(list(itertools.product([0, 1], repeat=10)), dtype=np.uint8)
    t = v[:, :1]
    outputs = {}
    for name, r in {"nuisance_9_bits": v[:, 1:], "task_3_bits": v[:, :3],
                    "nuisance_equal_rate_3_bits": v[:, 1:4]}.items():
        h = _discrete_entropy(r)
        mi = _discrete_entropy(t) + h - _discrete_entropy(np.concatenate([t, r], axis=1))
        outputs[name] = dict(code_bits=r.shape[1], input_mi_bits=h, input_fraction=h / 10,
                             task_mi_bits=mi, optimal_task_ce_bits=1 - mi)
    return dict(enumerated_inputs=len(v), input_entropy_bits=10., task_entropy_bits=1.,
                representations=outputs,
                scope="exact finite ten independent fair bits; fractions are not continuous EEG MI estimates")


def variance_only_counterexample():
    # The complete context identifies variance, but both conditional means are 0.
    task = np.repeat([0, 1], 2)
    y = np.array([-1., 1., -3., 3.])
    return dict(task_entropy_bits=1., conditional_means=[0., 0.], conditional_variances=[1., 9.],
                unconditional_variance=float(np.mean(y ** 2)),
                constant_representation_optimal_mse=float(np.mean(y ** 2)),
                class_representation_optimal_mse=float(np.mean(y ** 2)),
                constant_task_ce_bits=1., class_task_ce_bits=0.,
                constant_future_discrete_ce_bits=2., class_future_discrete_ce_bits=1.,
                exact_joint_atoms=[dict(task=int(t), future=float(z), probability=.25) for t, z in zip(task, y)],
                interpretation="Equal optimal mean-MSE does not imply equal retention of a variance-defined task or equal distributional prediction.")


def exact_controls(output=None):
    gaussian = gaussian_channel(1., .25)
    result = dict(schema_version=VERSION, ten_bit=ten_bit_counterexample(),
                  variance_only=variance_only_counterexample(), gaussian=gaussian,
                  fano=[dict(classes=k, error=e, necessary_bits=fano_necessary_bits(k, e))
                        for k in (2, 4, 8) for e in (0., .1, 1 - 1 / k)],
                  gaussian_rd=[dict(information_bits=b, distortion=gaussian_rate_distortion(1., b))
                               for b in (0., .5, 1., 2., 4.)])
    if not np.isclose(gaussian["posterior_mse"], gaussian["gaussian_rd_bound"], rtol=1e-13):
        raise AssertionError("Gaussian MI/RD identity failed")
    if output is not None:
        output = Path(output)
        atomic_json(output / "exact_controls.json", result)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
        labels = ["Nuisance 9", "Task 3", "Nuisance 3"]
        examples = list(result["ten_bit"]["representations"].values())
        x = np.arange(3)
        axes[0].bar(x - .18, [r["input_mi_bits"] for r in examples], .36, label="Input MI (bits)")
        axes[0].bar(x + .18, [r["task_mi_bits"] for r in examples], .36, label="Task MI (bits)")
        axes[0].set(xticks=x, xticklabels=labels, ylabel="Exact bits", title="Finite-bit counterexample")
        axes[0].legend(fontsize=8)
        b = np.linspace(0, 4, 101)
        axes[1].semilogy(b, 2 ** (-2 * b))
        axes[1].scatter([gaussian["mutual_information_bits"]], [gaussian["posterior_mse"]])
        axes[1].set(xlabel="Information (bits)", ylabel="Scalar Gaussian MSE", title="Gaussian control only")
        for ext in ("png", "pdf"):
            fig.savefig(output / f"exact_controls.{ext}", dpi=160)
        plt.close(fig)
    return result


def allocate_bits(dimension, budget_bits, max_scalar_bits=16):
    if type(dimension) is not int or type(budget_bits) is not int or dimension <= 0:
        raise ValueError("Positive integer dimension and integer budget required")
    if budget_bits < dimension:
        raise ValueError("Unsupported: at least one bit per retained coordinate is required")
    base, extra = divmod(budget_bits, dimension)
    bits = np.full(dimension, base, dtype=np.int64)
    bits[:extra] += 1
    if bits.max() > max_scalar_bits or max_scalar_bits > 16:
        raise ValueError("Unsupported: this numeric scalar code supports 1..16 bits per coordinate")
    if budget_bits % 8:
        raise ValueError("Unsupported: byte-aligned budgets required, no silent padding bits")
    return bits


def fit_quantizer(training, budget_bits, quantile_range=(.005, .995), max_scalar_bits=16):
    z = np.asarray(training, dtype=float)
    if z.ndim != 2 or z.shape[0] < 2 or not np.isfinite(z).all():
        raise ValueError("Quantizer requires finite [training examples, dimension] data")
    low, high = map(float, quantile_range)
    if not 0 <= low < high <= 1:
        raise ValueError("Invalid training-only quantile range")
    bits = allocate_bits(z.shape[1], int(budget_bits), max_scalar_bits)
    lower, upper = np.quantile(z, [low, high], axis=0)
    constant = upper == lower
    # Constant coordinates have one used code. They still occupy their declared bits.
    return dict(schema_version=VERSION, bits=bits, lower=lower, upper=upper,
                constant=constant, budget_bits=int(budget_bits), training_examples=len(z),
                quantile_range=[low, high], fit_split="train", levels=(2 ** bits - 1),
                stochastic_latent_noise="none", rate_interpretation="fixed-length upper bound, not joint entropy")


def quantize(values, codebook):
    z = np.asarray(values, dtype=float)
    lower, upper, levels = (np.asarray(codebook[k]) for k in ("lower", "upper", "levels"))
    if z.ndim != 2 or z.shape[1] != len(lower) or not np.isfinite(z).all():
        raise ValueError("Invalid latent array for fixed codebook")
    width = upper - lower
    scaled = (np.clip(z, lower, upper) - lower) / np.where(width > 0, width, 1)
    codes = np.rint(scaled * levels).astype(np.uint16)
    decoded = lower + codes.astype(float) / levels * width
    clipping = (z < lower) | (z > upper)
    report = dict(coordinate_clipping_fraction=float(clipping.mean()),
                  examples_clipped_fraction=float(clipping.any(axis=1).mean()),
                  per_coordinate_clipping_fraction=clipping.mean(axis=0).tolist(),
                  latent_dequantization_mse=float(np.mean((z - decoded) ** 2)))
    return codes, decoded, report


def pack_codes(codes, bits):
    codes = np.asarray(codes)
    bits = np.asarray(bits, dtype=int)
    if codes.ndim != 2 or len(bits) != codes.shape[1] or np.any(bits < 1) or np.any(bits > 16) or bits.sum() % 8:
        raise ValueError("Invalid fixed-rate integer code or non-byte-aligned budget")
    if np.any(codes < 0) or np.any(codes >= 2 ** bits) or np.any(codes != np.floor(codes)):
        raise ValueError("Code outside allocated coordinate range")
    unpacked = np.concatenate([((codes[:, i, None].astype(np.uint64) >> np.arange(b - 1, -1, -1, dtype=np.uint64)) & 1)
                               for i, b in enumerate(bits)], axis=1)
    return np.packbits(unpacked.astype(np.uint8), axis=1, bitorder="big")


def unpack_codes(packed, bits):
    packed, bits = np.asarray(packed, dtype=np.uint8), np.asarray(bits, dtype=int)
    if packed.ndim != 2 or bits.sum() != packed.shape[1] * 8 or np.any(bits < 1) or np.any(bits > 16):
        raise ValueError("Packed byte count differs from exact code budget")
    stream = np.unpackbits(packed, axis=1, bitorder="big")
    cols, offset = [], 0
    for b in bits:
        cols.append(stream[:, offset:offset + b] @ (2 ** np.arange(b - 1, -1, -1)))
        offset += b
    return np.stack(cols, axis=1).astype(np.uint16)


def fit_projection(training, dimension):
    z = np.asarray(training, dtype=float)
    if z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("Finite latent matrix required")
    if dimension > z.shape[1] or dimension > z.shape[0] - 1:
        raise ValueError("Unsupported: requested dimension exceeds source dimension or training sample rank")
    center = z.mean(axis=0)
    _, singular, vt = np.linalg.svd(z - center, full_matrices=False)
    return dict(center=center, components=vt[:dimension], singular_values=singular[:dimension],
                requested_dimension=dimension, numerical_rank=int(np.linalg.matrix_rank(z - center)),
                fit_split="train", transform="training-only PCA rotation/truncation, no padding")


def apply_projection(values, projection):
    return (np.asarray(values) - projection["center"]) @ projection["components"].T


def validate_splits(panel):
    groups, contexts = {}, {}
    for name in ("train", "development", "evaluation"):
        block = panel[name]
        x, y, mask = np.asarray(block["X"]), np.asarray(block["Y"]), np.asarray(block["future_mask"], bool)
        if x.ndim != 2 or y.ndim != 3 or y.shape[:2] != mask.shape or len(x) != len(y):
            raise ValueError(f"Invalid panel shapes: {name}")
        if not np.isfinite(x).all() or not np.isfinite(y[mask]).all() or np.any(mask.sum(axis=1) < 1):
            raise ValueError(f"Nonfinite/empty panel: {name}")
        groups[name], contexts[name] = set(map(str, block["structure_id"])), set(map(str, block["context_id"]))
        if len(contexts[name]) != len(x):
            raise ValueError("Context IDs must uniquely identify prefixes within each split")
    for a, b in itertools.combinations(groups, 2):
        if groups[a] & groups[b] or contexts[a] & contexts[b]:
            raise ValueError(f"Circuit/context split leakage: {a}/{b}")
    return {k: len(v) for k, v in groups.items()}


def _features_fit(x, kind, config, seed):
    x = np.asarray(x, float)
    center, scale = x.mean(0), x.std(0)
    scale = np.where(scale > 1e-12, scale, 1.)
    metadata = dict(center=center, scale=scale, kind=kind)
    if kind == "flexible":
        rng = np.random.default_rng(seed)
        standardized = (x - center) / scale
        sample = standardized[:min(256, len(x))]
        squared = np.sum((sample[:, None] - sample[None]) ** 2, axis=-1)
        positive = squared[squared > 0]
        width = math.sqrt(float(np.median(positive))) if len(positive) else 1.
        count = int(config["flexible_features"])
        metadata.update(weights=rng.normal(size=(x.shape[1], count)) / max(width, 1e-12),
                        phase=rng.uniform(0, 2 * np.pi, count), width=width, seed=seed)
    elif kind != "linear":
        raise ValueError(f"Unknown readout: {kind}")
    return metadata


def _features_apply(x, fitted):
    z = (np.asarray(x) - fitted["center"]) / fitted["scale"]
    if fitted["kind"] == "flexible":
        z = np.sqrt(2 / len(fitted["phase"])) * np.cos(z @ fitted["weights"] + fitted["phase"])
    return z


def _fit_head(x, y, kind, alpha):
    if kind == "classification":
        from sklearn.linear_model import LogisticRegression
        classes = np.unique(y)
        if len(classes) < 2:
            raise ValueError("Classification readout requires >=2 training classes")
        model = LogisticRegression(C=1 / float(alpha), solver="lbfgs", max_iter=2000)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            model.fit(x, y)
        return dict(kind=kind, classes=model.classes_, coef=model.coef_, intercept=model.intercept_,
                    warnings=[str(w.message) for w in captured], iterations=model.n_iter_.tolist())
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    xcenter, ycenter = x.mean(0), y.mean(0)
    xx, yy = x - xcenter, y - ycenter
    # A fixed regularizer grid is shared by direct and latent readouts.
    if x.shape[1] <= len(x):
        coef = np.linalg.solve(xx.T @ xx + alpha * np.eye(xx.shape[1]), xx.T @ yy)
    else:
        coef = xx.T @ np.linalg.solve(xx @ xx.T + alpha * np.eye(len(xx)), yy)
    return dict(kind=kind, coef=coef, intercept=ycenter - xcenter @ coef, warnings=[])


def _predict_head(x, head):
    if head["kind"] == "classification":
        logits = x @ head["coef"].T + head["intercept"]
        if logits.shape[1] == 1:
            logits = np.concatenate([np.zeros_like(logits), logits], axis=1)
        return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    return x @ head["coef"] + head["intercept"]


def _score(y, prediction, kind, classes=None):
    if kind == "classification":
        indexes = {value: i for i, value in enumerate(classes)}
        if any(value not in indexes for value in y):
            raise ValueError("Unseen class in held-out fold: cannot silently omit/reshape logits")
        correct = np.array([indexes[value] for value in y])
        return -np.log2(np.maximum(prediction[np.arange(len(y)), correct], np.finfo(float).tiny))
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    return np.mean((prediction - y) ** 2, axis=1)


def _select_head(x, y, dev_x, dev_y, kind, readout, config, seed):
    feature_map = _features_fit(x, readout, config, seed)
    a, b = _features_apply(x, feature_map), _features_apply(dev_x, feature_map)
    choices = []
    for alpha in config["regularization"]:
        head = _fit_head(a, y, kind, float(alpha))
        predicted = _predict_head(b, head)
        score = float(np.mean(_score(dev_y, predicted, kind, head.get("classes"))))
        choices.append((score, float(alpha), head))
    _, alpha, head = min(choices, key=lambda q: (q[0], q[1]))
    return feature_map, head, dict(selected_regularization=alpha,
                                  candidates=[dict(regularization=a, development_loss=s) for s, a, _ in choices],
                                  tuning_split="development; circuit-disjoint from fit and evaluation")


def nested_crossfit(x, y, structures, kind, readout, config, seed):
    """Outer training-circuit holdout, inner training-circuit regularizer tuning.

    No outer held-out labels enter its own feature scaling or model selection.
    Embeddings/quantizers are frozen upstream unsupervised train-only objects;
    the audit concerns supervised readout fitting, not a second encoder training.
    """
    structures = np.asarray(structures)
    unique = np.unique(structures)
    if len(unique) < 3:
        return dict(status="unavailable", reason="nested supervised crossfit needs >=3 training circuits"), {}
    prediction = None
    fold_records = []
    for outer in unique:
        outer_train, outer_test = structures != outer, structures == outer
        candidates = []
        for alpha in config["regularization"]:
            fold_losses = []
            for inner in unique[unique != outer]:
                train = outer_train & (structures != inner)
                valid = outer_train & (structures == inner)
                try:
                    transform = _features_fit(x[train], readout, config, seed)
                    head = _fit_head(_features_apply(x[train], transform), y[train], kind, alpha)
                    pred = _predict_head(_features_apply(x[valid], transform), head)
                    fold_losses.append(float(np.mean(_score(y[valid], pred, kind, head.get("classes")))))
                except ValueError as exc:
                    return dict(status="unavailable", reason=str(exc), outer_structure=str(outer)), {}
            candidates.append((float(np.mean(fold_losses)), float(alpha)))
        _, alpha = min(candidates)
        transform = _features_fit(x[outer_train], readout, config, seed)
        head = _fit_head(_features_apply(x[outer_train], transform), y[outer_train], kind, alpha)
        try:
            pred = _predict_head(_features_apply(x[outer_test], transform), head)
            loss = _score(y[outer_test], pred, kind, head.get("classes"))
        except ValueError as exc:
            return dict(status="unavailable", reason=str(exc), outer_structure=str(outer)), {}
        if prediction is None:
            prediction = np.zeros((len(x), pred.shape[1]))
        if prediction.shape[1] != pred.shape[1]:
            return dict(status="unavailable", reason="training fold class-set mismatch"), {}
        prediction[outer_test] = pred
        fold_records.append(dict(outer_structure=str(outer), loss=float(np.mean(loss)),
                                 selected_regularization=alpha, n_heldout=int(outer_test.sum())))
    return dict(status="complete", folds=fold_records, mean_circuit_loss=float(np.mean([f["loss"] for f in fold_records])),
                conditioning="frozen upstream train-only representation; nested supervised decoder audit"), {"crossfit_prediction": prediction}


def _balanced_subset(y, structures, budget, kind, seed):
    if budget == "all" or budget is None:
        return np.arange(len(y))
    budget = int(budget)
    if budget < 2:
        raise ValueError("Readout label budget must be >=2 independent contexts")
    rng = np.random.default_rng(seed)
    labels = np.asarray(y) if kind == "classification" else np.zeros(len(y), int)
    buckets = {}
    for i, (label, group) in enumerate(zip(labels, structures)):
        buckets.setdefault((str(label), str(group)), []).append(i)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    keys = sorted(buckets, key=lambda key: (key[1], key[0]))
    while len(selected) < min(budget, len(y)):
        changed = False
        for key in keys:
            if buckets[key] and len(selected) < budget:
                selected.append(buckets[key].pop())
                changed = True
        if not changed:
            break
    return np.asarray(sorted(selected), int)


def _task_arrays(panel):
    tasks = {"waveform_reconstruction": ("regression", "X"),
             "future_mean": ("forecast", "Y"), "rhythm": ("classification", "task_rhythm"),
             "carrier": ("classification", "task_carrier"),
             "phase_sin": ("regression", "task_phase_sin"),
             "phase_cos": ("regression", "task_phase_cos"),
             "log_power": ("regression", "task_log_power")}
    declared = panel["manifest"].get("available_mechanism_tasks", {})
    for name, kind in declared.items():
        if kind not in {"regression", "classification"}:
            raise ValueError("Mechanism tasks need an explicit kind")
        tasks[name] = (kind, "task_" + name)
    result = {}
    for name, (kind, key) in tasks.items():
        if all(key in panel[s] for s in ("train", "development", "evaluation")):
            result[name] = (kind, key)
    return result


def _task_view(block, key, kind):
    if kind == "forecast":
        # Fit to one independent future per prefix; evaluation later scores every future.
        first = np.argmax(block["future_mask"], axis=1)
        y = block["Y"][np.arange(len(first)), first]
    else:
        y = np.asarray(block[key])
    mask = np.isfinite(y).all(axis=1) if y.ndim > 1 else np.isfinite(y)
    if key == "task_carrier":
        mask &= y >= 0
    return y, mask


def _circuit_scores(loss, structures):
    return [dict(structure_id=str(s), n_contexts=int(np.sum(structures == s)),
                 mean_loss=float(np.mean(loss[structures == s]))) for s in np.unique(structures)]


def _evaluate_readout(panel, matrices, task, kind, key, readout, budget, config, destination):
    views = {s: _task_view(panel[s], key, kind) for s in ("train", "development", "evaluation")}
    selected = _balanced_subset(views["train"][0][views["train"][1]], panel["train"]["structure_id"][views["train"][1]], budget,
                                "classification" if kind == "classification" else "regression", config["seed"])
    masks = {s: np.flatnonzero(views[s][1]) for s in views}
    masks["train"] = masks["train"][selected]
    if min(map(len, masks.values())) < 2:
        return dict(status="unsupported", reason="fewer than two valid independent contexts in a split")
    x = {s: matrices[s][masks[s]] for s in views}
    y = {s: views[s][0][masks[s]] for s in views}
    structure = {s: panel[s]["structure_id"][masks[s]] for s in views}
    if len(np.unique(structure["train"])) < 2:
        return dict(status="unsupported", reason="readout needs at least two fit circuits")
    if kind == "classification":
        classes = np.unique(y["train"])
        if len(classes) < 2 or any(set(np.unique(y[s])) - set(classes) for s in ("development", "evaluation")):
            return dict(status="unsupported", reason="class-set mismatch or a constant training class", class_set=classes.tolist())
    head_kind = "classification" if kind == "classification" else "regression"
    fit_started = time.monotonic()
    transform, head, tuning = _select_head(x["train"], y["train"], x["development"], y["development"],
                                         head_kind, readout, config, config["seed"])
    fit_seconds = time.monotonic() - fit_started
    inference_started = time.monotonic()
    pred = _predict_head(_features_apply(x["evaluation"], transform), head)
    inference_seconds = time.monotonic() - inference_started
    dev_pred = _predict_head(_features_apply(x["development"], transform), head)
    arrays = dict(prediction=pred, observed_target=y["evaluation"],
                  context_id=panel["evaluation"]["context_id"][masks["evaluation"]],
                  structure_id=structure["evaluation"], evaluation_index=masks["evaluation"],
                  train_context_id=panel["train"]["context_id"][masks["train"]])
    result = dict(status="complete", task=task, kind=kind, readout=readout, label_budget=budget,
                  fit_seconds=fit_seconds, inference_seconds=inference_seconds,
                  actual_training_contexts=len(x["train"]), training_circuits=len(np.unique(structure["train"])),
                  evaluation_contexts=len(x["evaluation"]), tuning=tuning, warnings=head["warnings"],
                  readout_parameter_count=int(np.size(head["coef"]) + np.size(head["intercept"])),
                  comparison_scope="decoder accessibility; approximation error is not absence of retained information")
    if kind == "forecast":
        all_y = panel["evaluation"]["Y"][masks["evaluation"]]
        future_mask = panel["evaluation"]["future_mask"][masks["evaluation"]]
        per_future = np.mean((all_y - pred[:, None]) ** 2, axis=-1)
        loss = (per_future * future_mask).sum(1) / future_mask.sum(1)
        # A homoskedastic diagonal Gaussian is declared and calibrated on dev, never test.
        dev_y = y["development"]
        if dev_y.ndim == 1:
            dev_y = dev_y[:, None]
        variance = np.mean((dev_pred - dev_y) ** 2, axis=0)
        floor = max(float(np.mean(np.var(y["train"], axis=0))) * 1e-8, 1e-12)
        variance = np.maximum(variance, floor)
        from .scoring import gaussian_scores
        proper = gaussian_scores(all_y, pred, np.broadcast_to(variance, pred.shape), future_mask,
                                 draws=int(config.get("distribution_draws", 16)), seed=int(config["seed"]))
        result.update(gaussian_nll_nats_per_dimension=float(np.nanmean(proper["gaussian_nll"])),
                      gaussian_crps=float(np.nanmean(proper["marginal_crps"])),
                      gaussian_energy=float(np.nanmean(proper["energy"])),
                      marginal_95_coverage=float(np.nanmean(proper["coverage_95"])),
                      distribution="homoskedastic diagonal Gaussian; variance calibrated on development residuals", variance_floor=floor)
        arrays.update(target_futures=all_y, future_mask=future_mask, per_future_mse=per_future,
                      gaussian_variance=variance, future_id=panel["evaluation"]["future_id"][masks["evaluation"]])
        arrays.update({"proper_" + name: values for name, values in proper.items()})
        baseline = np.mean(y["train"], axis=0)
        r0 = np.mean(((all_y - baseline) ** 2), axis=-1)[future_mask].mean()
        result.update(r0=float(r0), nmse=float(loss.mean() / r0) if r0 > 0 else None,
                      skill=float(1 - loss.mean() / r0) if r0 > 0 else None)
    else:
        loss = _score(y["evaluation"], pred, head_kind, head.get("classes"))
    if kind == "classification":
        classes = head["classes"]
        counts = np.array([(y["evaluation"] == t).sum() for t in classes])
        entropy = entropy_bits(counts / counts.sum())
        result.update(classes=classes.tolist(), class_counts=counts.tolist(),
                      task_ce_bits=float(loss.mean()), empirical_task_entropy_bits=entropy,
                      task_mi_lower_bound_estimate_bits=float(entropy - loss.mean()),
                      mi_estimate_caveat="plug-in held-out estimate of decoder-dependent H(T)-CE lower bound; finite-sample uncertainty, not a certified bound or MI-loss difference",
                      accuracy=float(np.mean(classes[np.argmax(pred, axis=1)] == y["evaluation"])))
        arrays["probabilities"] = pred
        arrays["logits"] = np.log(np.maximum(pred, np.finfo(float).tiny))
        arrays["classes"] = classes
        error = 1 - result["accuracy"]
        result["fano_necessary_bits_at_observed_error"] = fano_necessary_bits(len(classes), error) if np.all(counts == counts[0]) and error <= 1 - 1 / len(classes) else None
    if kind not in {"classification", "forecast"}:
        target = np.asarray(y["evaluation"])
        if target.ndim == 1:
            target = target[:, None]
        baseline = np.mean(y["train"], axis=0)
        r0 = float(np.mean((target - baseline) ** 2))
        result.update(r0=r0, nmse=float(loss.mean() / r0) if r0 > 0 else None,
                      skill=float(1 - loss.mean() / r0) if r0 > 0 else None)
    if key == "X":
        result["physical_mse"] = float(loss.mean() * panel["manifest"]["normalization"]["x_scale"] ** 2)
        result["loss_units"] = "training-standardized observed history squared units; physical_mse in V^2"
    elif key == "Y":
        result["physical_mse"] = float(loss.mean() * panel["manifest"]["normalization"]["y_scale"] ** 2)
    result.update(mean_context_loss=float(loss.mean()), per_circuit=_circuit_scores(loss, structure["evaluation"]),
                  loss_units=("bits" if kind == "classification" else
                              "training-standardized observed history squared units" if key == "X" else
                              "training-standardized future target squared units" if key == "Y" else
                              "natural-log(V^2) squared units" if key == "task_log_power" else "dimensionless squared units"),
                  task_definition=panel["manifest"].get("task_definitions", {}).get(task, key),
                  supervised_target_source=key,
                  task_timing="future TargetSpec" if key == "Y" else "available/predecision history only")
    arrays["per_context_loss"] = loss
    crossfit_started = time.monotonic()
    crossfit, crossfit_arrays = nested_crossfit(x["train"], y["train"], structure["train"], head_kind, readout, config, config["seed"])
    result["nested_crossfit_seconds"] = time.monotonic() - crossfit_started
    result["nested_supervised_crossfit"] = crossfit
    arrays.update(crossfit_arrays)
    _npz(destination / "predictions.npz", **arrays)
    model_arrays = {"coef": head["coef"], "intercept": head["intercept"],
                    **{"map_" + k: v for k, v in transform.items() if isinstance(v, np.ndarray)}}
    if "classes" in head:
        model_arrays["classes"] = head["classes"]
    _npz(destination / "readout.npz", **model_arrays)
    result["feature_map"] = {k: v for k, v in transform.items() if not isinstance(v, np.ndarray)}
    atomic_json(destination / "scores.json", result)
    return result


def _identity(config, manifests):
    return _hash(dict(schema=VERSION, config=config, panels=manifests, code_sha256={p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}))


def _verify_completed(root, marker, identity, resume):
    if not marker.exists():
        return None
    if not resume:
        raise FileExistsError(f"Existing completed analysis; use compatible --resume: {marker}")
    saved = json.loads(marker.read_text())
    if saved.get("identity") != identity:
        raise ValueError("Compression/scale resume identity mismatch")
    for row in saved["artifacts"]:
        path = root / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or file_hash(path) != row["sha256"]:
            raise ValueError(f"Completed artifact checksum mismatch: {path}")
    return saved


def _finalize(root, marker, identity, summary):
    artifacts = [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=file_hash(p))
                 for p in sorted(root.rglob("*")) if p.is_file() and p != marker and not p.name.startswith(".")]
    result = dict(summary, identity=identity, artifacts=artifacts)
    atomic_json(marker, result)
    return result


def run_compression(config, output_root, device="cpu", resume=False):
    from .dataset import list_jobs, load_panel
    from .benchmark import get_representations
    config_section = config.get("compression", config.get("h1", {}).get("compression", {}))
    options = dict(DEFAULTS, **config_section)
    root = Path(output_root)
    if (root / "provenance.json").exists():
        from .artifacts import require_compatible
        require_compatible(root, config)
    destination = root / "compression"
    jobs = list_jobs(root)
    manifests = [load_panel(root, job)["manifest"] for job in jobs]
    identity = _identity(dict(config=config, options=options), manifests)
    marker = destination / "summary.json"
    done = _verify_completed(destination, marker, identity, resume)
    if done is not None:
        return done
    guard = destination / "analysis_identity.json"
    if guard.exists() and json.loads(guard.read_text()).get("identity") != identity:
        raise ValueError("Incompatible partial compression output; use a fresh root")
    if guard.exists() and not resume:
        raise FileExistsError("Partial compression output exists; use --resume")
    atomic_json(guard, dict(identity=identity))
    start = time.monotonic()
    controls = exact_controls(destination / "controls")
    all_rows = []
    for job in jobs:
        panel = load_panel(root, job)
        split_counts = validate_splits(panel)
        job_dir = destination / job
        banks = get_representations(config, job_dir, panel, device=device, resume=resume)
        tasks = _task_arrays(panel)
        direct = {s: np.asarray(panel[s]["X"], float) for s in ("train", "development", "evaluation")}
        for task, (kind, key) in tasks.items():
            for readout in options["readouts"]:
                for budget in options["label_budgets"]:
                    path = job_dir / "direct_history" / task / readout / f"labels_{budget}"
                    row = _evaluate_readout(panel, direct, task, kind, key, readout, budget, options, path)
                    all_rows.append(dict(row, job_id=job, representation="direct_history", code_bits=None,
                                         dimension=direct["train"].shape[1], target_fingerprint=panel["manifest"]["target_fingerprint"]))
        for name in options["representations"]:
            if name not in banks:
                raise ValueError(f"Requested representation not produced: {name}")
            bank = banks[name]
            values = {s: np.asarray(bank[s if s in bank else "score"], float) for s in ("train", "development", "evaluation")}
            if any(len(values[s]) != len(panel[s]["X"]) for s in values):
                raise ValueError("Representation/context row alignment mismatch")
            for dimension in options["dimensions"]:
                for bits in options["code_bits"]:
                    point = job_dir / name / f"dim_{dimension}_bits_{bits}"
                    try:
                        allocation = allocate_bits(int(dimension), int(bits), options["max_scalar_bits"])
                        projection = fit_projection(values["train"], int(dimension))
                    except ValueError as exc:
                        all_rows.append(dict(status="unsupported", job_id=job, representation=name, dimension=dimension,
                                             code_bits=bits, reason=str(exc)))
                        continue
                    latent = {s: apply_projection(values[s], projection) for s in values}
                    codebook = fit_quantizer(latent["train"], int(bits), options["quantile_range"], options["max_scalar_bits"])
                    decoded, reports, code_arrays = {}, {}, {}
                    for split in values:
                        integer, decoded[split], reports[split] = quantize(latent[split], codebook)
                        packed = pack_codes(integer, allocation)
                        if packed.shape[1] * 8 != bits or not np.array_equal(unpack_codes(packed, allocation), integer):
                            raise AssertionError("Fixed-rate packing round trip failed")
                        code_arrays.update({split + "_packed": packed, split + "_decoded": decoded[split],
                                            split + "_context_id": panel[split]["context_id"]})
                    _npz(point / "codes.npz", **code_arrays)
                    _npz(point / "codebook.npz", **{k: v for k, v in codebook.items() if isinstance(v, np.ndarray)},
                         projection_center=projection["center"], projection_components=projection["components"])
                    atomic_json(point / "codebook.json", dict(codebook, clipping=reports, projection=projection,
                                                               representation=bank.get("metadata", {}),
                                                               exact_packed_bits_per_window=int(bits)))
                    for task, (kind, key) in tasks.items():
                        for readout in options["readouts"]:
                            for budget in options["label_budgets"]:
                                row = _evaluate_readout(panel, decoded, task, kind, key, readout, budget, options,
                                                        point / task / readout / f"labels_{budget}")
                                all_rows.append(dict(row, job_id=job, representation=name, dimension=dimension,
                                                     code_bits=bits, clipping=reports, split_circuits=split_counts,
                                                     target_fingerprint=panel["manifest"]["target_fingerprint"]))
    atomic_json(destination / "surface.json", dict(rows=all_rows))
    _plot_surface(all_rows, destination)
    return _finalize(destination, marker, identity, dict(schema_version=VERSION, status="complete" if jobs else "controls_only_no_panels",
                       executed_jobs=jobs, options=options, rows=all_rows, controls_passed=True,
                       elapsed_seconds=time.monotonic() - start,
                       estimands="held-out distortion/task accessibility at fixed operational code budgets; no certified MI loss; marginal task readouts do not certify joint task sufficiency",
                       uncertainty="nested train-circuit readout folds, raw per-context predictions, per-evaluation-circuit scores; few circuits cannot justify narrow population CIs"))


def _plot_surface(rows, destination):
    selected = [r for r in rows if r.get("status") == "complete" and r.get("code_bits") is not None]
    if not selected:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tasks = sorted({r["task"] for r in selected})
    fig, axes = plt.subplots(1, len(tasks), figsize=(max(5, 3 * len(tasks)), 3.5), squeeze=False, constrained_layout=True)
    for ax, task in zip(axes[0], tasks):
        subset = [r for r in selected if r["task"] == task]
        for name in sorted({r["representation"] for r in subset}):
            part = [r for r in subset if r["representation"] == name]
            ax.scatter([r["code_bits"] for r in part], [r["mean_context_loss"] for r in part], label=name, alpha=.65)
        ax.set(xlabel="Fixed bits/window", ylabel="Held-out loss", title=task)
        ax.set_xscale("log", base=2)
    axes[0, 0].legend(fontsize=7)
    for ext in ("png", "pdf"):
        fig.savefig(destination / f"rate_distortion_task_surface.{ext}", dpi=150)
    plt.close(fig)
