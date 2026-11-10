"""Declared label permissions; simulator labels are targets, never input tokens."""
from __future__ import annotations
import math


def _task(kind, target, factor, classes=0, context="causal EEG history", **extra):
    return dict(kind=kind, n_classes=classes, target=target, generating_factor=factor,
        input_window="forecast.history_s", visible_context=context,
        preprocessing="training-only channel scaling; causal antialiased 500-Hz EEG",
        parent="parent_trial_id", split="structure_id before windowing",
        allowed_labels="only explicitly named source labels during pretraining",
        class_balance="independent trial counts by label; missing labels excluded",
        metrics=["balanced_accuracy", "macro_f1"] if kind == "classification" else ["normalized_mse", "r2"],
        observability="pending; generator labels do not establish EEG recoverability", **extra)


TASKS = {
    "rhythm": _task("classification", "homogeneous versus rhythmic afferents", "modulation_depth > 0", 2),
    "carrier": _task("classification", "10 / 16 / 24 Hz among rhythmic trials", "carrier", 3),
    "baseline_alpha": _task("classification", "R0 versus 10-Hz rhythm", "carrier and modulation", 2),
    "baseline_beta_low": _task("classification", "R0 versus 16-Hz rhythm", "carrier and modulation", 2),
    "baseline_beta_high": _task("classification", "R0 versus 24-Hz rhythm", "carrier and modulation", 2),
    "nearby_carrier": _task("classification", "9 versus 11 Hz", "carrier", 2),
    "coherence": _task("classification", "D=0.5 versus D=2", "phase diffusion, not amplitude bursts", 2),
    "coherence_value": _task("regression", "exp(-D * 0.1 s)", "phase diffusion"),
    "burst": _task("classification", "tonic versus bounded Markov envelope", "envelope", 2),
    "burst_occupancy": _task("regression", "fraction of retained samples in high envelope state", "envelope"),
    "kinetics": _task("classification", "I-to-E decay x0.8 versus x1.2 with area preserved", "inhibitory kinetics", 2),
    "shared_fraction": _task("regression", "fraction of afferents participating in rate rhythm", "shared rate participation"),
    "envelope": _task("regression", "RMS of causal future block", "future signal amplitude"),
    "log_bandpower": _task("regression", "log 4-30 Hz Hann-periodogram power of future block (finite block resolution)", "future temporal power"),
    "waveform": _task("regression", "future EEG block beginning at declared lead", "stochastic future", target_units="training normalized V"),
    "masked_reconstruction": _task("regression", "masked raw EEG samples only", "recording", context="bidirectional visible samples within declared history"),
}


def labels_for(regime, privileged=None):
    privileged = privileged or {}
    labels = {k: (-1 if t["kind"] == "classification" else float("nan")) for k, t in TASKS.items()}
    m, f = regime["modulation_depth"], regime["frequency_hz"]
    labels["rhythm"] = int(m > 0)
    for task, freq in [("baseline_alpha", 10), ("baseline_beta_low", 16), ("baseline_beta_high", 24)]:
        if m == 0 or f == freq:
            labels[task] = int(m > 0)
    if m > 0:
        labels["carrier"] = {10: 0, 16: 1, 24: 2}.get(f, -1)
        labels["nearby_carrier"] = {9: 0, 11: 1}.get(f, -1)
        labels["coherence"] = {0.5: 0, 2.0: 1}.get(regime["diffusion"], -1)
        labels["coherence_value"] = math.exp(-regime["diffusion"] * .1)
        labels["burst"] = int(bool(regime["burst"]))
        labels["shared_fraction"] = regime["shared_fraction"]
        if "burst_occupancy" in privileged:
            labels["burst_occupancy"] = float(privileged["burst_occupancy"])
    labels["kinetics"] = {0.8: 0, 1.2: 1}.get(regime["kinetics_scale"], -1)
    return labels


def validate_tasks(config):
    for key in ["source_tasks", "heldout_tasks", "source_order"]:
        unknown = set(config["training"][key]) - TASKS.keys()
        if unknown:
            raise ValueError(f"Unknown tasks in {key}: {unknown}")
    if set(config["evaluation"]["tasks"]) - TASKS.keys():
        raise ValueError("Unknown evaluation task")
