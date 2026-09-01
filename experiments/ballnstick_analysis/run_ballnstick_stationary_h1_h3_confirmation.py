"""Disjoint confirmation of stationary BallAndStick hypotheses H1--H3.

H1 asks whether a mean-rate-matched, rhythmically driven toy circuit A is
distinguishable from a homogeneous-afferent reference B in ideal neural EEG.
H2 asks whether a weak uniform field moves A toward the population B target in
a frequency- and EEG-relative-phase-specific manner. H3 freezes the F0 rule
"detect 9/11 Hz from prestimulation EEG and apply that frequency at pi relative
phase" and compares it with sham, the F0-frozen best fixed action, and a uniform
random policy over the same four active actions.

Structure seed is the independent unit. Carrier, action, and postdecision
future are repeated measurements. This is a toy-model confirmation, not a
contextual bandit, a clinical experiment, or a disease model.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from decouple import config
from hydra.utils import to_absolute_path
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf
from scipy import signal, stats


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _action,
    _epoch_row,
    _plain,
    _run_condition,
    _sham,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    REFERENCE,
    SHAM,
    _action_specs,
    _context_specs,
    _crossover_summary,
    _epoch_signal,
    _episode_feature,
    _expected_action_map,
    _fit_reference_target,
    _frequency_token,
    _future_seed,
    _materialize_action,
    _metric_row,
    _reference_seeds,
    _screen_context,
    _target_distance,
    _validate_design as _validate_f0_design,
    _with_action_frequency,
    _with_hidden_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _process_eeg,
)


ROOT_NAME = "stationary_h1_h3_confirmation"
A_STATE = "A_elevated_alpha"
B_MATCHED = "B_matched_reference"
B_POPULATION = "B_population_reference"
TRANSVERSE = "matched_antiphase_transverse"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frozen_f0(cfg: DictConfig) -> dict[str, Any]:
    root = Path(to_absolute_path(str(cfg.analysis.frozen_f0.result_dir)))
    paths = {
        "conclusion": root / "experiment_conclusion.json",
        "provenance": root / "protocol_and_provenance.json",
        "metrics": root / "context_action_future_metrics.csv",
        "calibration": root / "reference_B_calibration.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("S1-C requires completed F0 files: " + ", ".join(missing))
    hashes = {name: _sha256(path) for name, path in paths.items()}
    expected = OmegaConf.to_container(
        cfg.analysis.frozen_f0.expected_sha256, resolve=True
    )
    for name, expected_hash in expected.items():
        if hashes[str(name)] != str(expected_hash):
            raise ValueError(f"Frozen F0 {name!r} changed after S1-C was specified.")

    conclusion = json.loads(paths["conclusion"].read_text())
    provenance = json.loads(paths["provenance"].read_text())
    if not bool(
        conclusion["conclusions"]["frequency_phase_contextual_feasibility_gate_passed"]
    ):
        raise ValueError("The source F0 directional gate did not pass.")
    frozen = cfg.analysis.frozen_f0
    if str(provenance["best_fixed_action_id"]) != str(
        frozen.expected_best_fixed_action_id
    ):
        raise ValueError("The F0-frozen best fixed action changed.")
    actions = provenance["actions"]
    active = [row for row in actions if str(row["id"]) != SHAM]
    if len(active) != 4 or not all(
        np.isclose(
            float(row["ac_amplitude_v_per_m"]),
            float(frozen.expected_amplitude_v_per_m),
        )
        for row in active
    ):
        raise ValueError("The F0-frozen active action grid changed.")
    rule = str(provenance["candidate_eeg_rule"])
    if "pi EEG-relative phase" not in rule:
        raise ValueError("The F0 rule no longer freezes pi EEG-relative phase.")

    screening = pd.read_csv(root / "prospective_screening.csv")
    metrics = pd.read_csv(paths["metrics"])
    calibration = pd.read_csv(paths["calibration"])
    source_seeds = set(calibration.seed.astype(int))
    for column in ("structure_seed", "drive_seed", "phase_seed", "trial_seed"):
        source_seeds.update(screening[column].astype(int))
    source_seeds.update(metrics.future_drive_seed.astype(int))
    return {
        "root": str(root),
        "sha256": hashes,
        "conclusion": conclusion,
        "provenance": provenance,
        "best_fixed_action_id": str(provenance["best_fixed_action_id"]),
        "preferred_relative_phase_rad": float(
            frozen.expected_preferred_relative_phase_rad
        ),
        "source_seeds": sorted(source_seeds),
    }


def _one_sample_t_power(*, n: int, effect_size: float, alpha: float) -> float:
    if n < 2 or effect_size <= 0.0:
        return 0.0
    critical = stats.t.ppf(1.0 - alpha, df=n - 1)
    return float(stats.nct.sf(critical, df=n - 1, nc=effect_size * np.sqrt(n)))


def _required_n(*, effect_size: float, alpha: float, target_power: float) -> int:
    for n in range(2, 1001):
        if _one_sample_t_power(n=n, effect_size=effect_size, alpha=alpha) >= target_power:
            return n
    raise ValueError("The requested power was not attained below 1001 structures.")


def _power_design(cfg: DictConfig) -> dict[str, Any]:
    block = cfg.analysis.power_design
    alpha = float(block.alpha_one_sided)
    target = float(block.target_power)
    result: dict[str, Any] = {
        "alpha_one_sided": alpha,
        "target_power": target,
        "planning_test": "one-sided paired t approximation",
        "inference_unit": "independent circuit structure",
        "repeated_axes": ["carrier frequency", "action", "postdecision future"],
    }
    for label, effect_key, planned_key in (
        ("H1", "H1_minimum_standardized_effect_dz", "H1_planned_independent_structures"),
        ("H2_H3", "H2_H3_minimum_standardized_effect_dz", "H2_H3_planned_independent_structures"),
    ):
        effect = float(block[effect_key])
        planned = int(block[planned_key])
        result[label] = {
            "minimum_standardized_effect_dz": effect,
            "planned_independent_structures": planned,
            "required_independent_structures": _required_n(
                effect_size=effect, alpha=alpha, target_power=target
            ),
            "a_priori_t_approximation_power": _one_sample_t_power(
                n=planned, effect_size=effect, alpha=alpha
            ),
        }
    return result


def _current_seed_set(cfg: DictConfig) -> set[int]:
    values = set(_reference_seeds(cfg))
    contexts = _context_specs(cfg)
    for context in contexts:
        for name in ("structure_seed", "drive_seed", "phase_seed", "trial_seed"):
            values.add(int(context[name]))
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            values.add(_future_seed(cfg, int(context["context_order"]), future_index))
    return values


def _validate_design(
    cfg: DictConfig, frozen_f0: dict[str, Any], power: dict[str, Any]
) -> None:
    _validate_f0_design(cfg)
    smoke = bool(cfg.analysis.smoke_test)
    if not np.isclose(
        float(cfg.analysis.tacs.amplitude_v_per_m),
        float(cfg.analysis.frozen_f0.expected_amplitude_v_per_m),
    ):
        raise ValueError("S1-C must retain the F0-frozen 0.4-V/m amplitude.")
    if int(cfg.analysis.timeline.baseline_steps) < (4 if smoke else 12):
        raise ValueError("S1-C needs 12 s of baseline, or 4 s in a smoke test.")
    if int(cfg.analysis.timeline.washout_steps) < 2:
        raise ValueError(
            "S1-C needs at least two seconds of washout for the 9/11-Hz "
            "field-removal audit."
        )
    stimulation_ms = (
        int(cfg.analysis.timeline.stimulation_steps)
        * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    )
    if stimulation_ms < (2000.0 if smoke else 4000.0):
        raise ValueError("S1-C must retain a four-second central stimulation endpoint.")
    overlap = _current_seed_set(cfg).intersection(frozen_f0["source_seeds"])
    if overlap:
        raise ValueError(
            "S1-C seed namespaces overlap F0: "
            + ", ".join(str(value) for value in sorted(overlap))
        )
    if smoke:
        return
    criteria = cfg.analysis.criteria
    if int(cfg.analysis.reference_calibration.n_seeds) < int(
        criteria.minimum_reference_seeds
    ):
        raise ValueError("S1-C has too few disjoint B calibration seeds.")
    if int(cfg.analysis.crossed_design.n_structure_seeds) != int(
        power["H1"]["planned_independent_structures"]
    ):
        raise ValueError("S1-C candidate structures must match the H1 power design.")
    if int(cfg.analysis.crossed_design.n_enrolled_structure_seeds) != int(
        power["H2_H3"]["planned_independent_structures"]
    ):
        raise ValueError("S1-C enrollment target must match the H2/H3 power design.")
    for label in ("H1", "H2_H3"):
        if int(power[label]["planned_independent_structures"]) < int(
            power[label]["required_independent_structures"]
        ) or float(power[label]["a_priori_t_approximation_power"]) < float(
            power["target_power"]
        ):
            raise ValueError(f"S1-C {label} power design is underpowered.")
    if int(cfg.analysis.crossed_design.n_future_continuations) < int(
        criteria.minimum_future_continuations
    ):
        raise ValueError("S1-C requires four independent futures per action.")


def _paper_psd(
    episode: dict[str, Any], *, epoch: str, cfg: DictConfig
) -> tuple[np.ndarray, np.ndarray]:
    raw, _ = _epoch_signal(episode, epoch, cfg)
    processed, fs_hz, _, _, _ = _process_eeg(
        raw, simulator_fs_hz=float(episode["simulator_fs_hz"]), cfg=cfg
    )
    requested = int(round(float(cfg.analysis.paper_psd.segment_seconds) * fs_hz))
    nperseg = min(processed.size, requested)
    if nperseg < 2:
        raise RuntimeError("The paper PSD has fewer than two samples.")
    noverlap = min(
        nperseg - 1,
        int(round(float(cfg.analysis.paper_psd.overlap_fraction) * nperseg)),
    )
    frequencies, psd = signal.welch(
        processed,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    keep = (
        (frequencies >= float(cfg.analysis.paper_psd.minimum_plot_hz))
        & (frequencies <= float(cfg.analysis.paper_psd.maximum_plot_hz))
    )
    return frequencies[keep], psd[keep]


def _psd_rows(
    episode: dict[str, Any], *, epoch: str, condition: str,
    action_id: str, context: dict[str, Any] | None, structure_seed: int,
    future_index: int, cfg: DictConfig,
) -> list[dict[str, Any]]:
    frequencies, psd = _paper_psd(episode, epoch=epoch, cfg=cfg)
    hidden = float("nan") if context is None else float(context["hidden_frequency_hz"])
    context_id = "population_B" if context is None else str(context["context_id"])
    eps = np.finfo(float).tiny
    return [{
        "condition": condition,
        "action_id": action_id,
        "epoch": epoch,
        "context_id": context_id,
        "structure_seed": int(structure_seed),
        "hidden_frequency_hz": hidden,
        "future_index": int(future_index),
        "frequency_hz": float(frequency),
        "psd_v2_per_hz": float(value),
        "log10_psd_v2_per_hz": float(np.log10(max(float(value), eps))),
    } for frequency, value in zip(frequencies, psd)]


def _summarize_psd(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    unit_keys = [
        "condition", "action_id", "hidden_frequency_hz", "structure_seed",
        "frequency_hz",
    ]
    units = frame.groupby(unit_keys, dropna=False, as_index=False).agg(
        mean_log10_psd=("log10_psd_v2_per_hz", "mean")
    )
    group_keys = ["condition", "action_id", "hidden_frequency_hz", "frequency_hz"]
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset) + 17
    )
    rows = []
    count = int(cfg.analysis.paper_psd.bootstrap_resamples)
    for key, group in units.groupby(group_keys, dropna=False, sort=True):
        values = group.mean_log10_psd.to_numpy(float)
        if values.size == 1:
            low = high = float(values[0])
        else:
            samples = rng.choice(values, size=(count, values.size), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
        mean = float(values.mean())
        rows.append({
            **dict(zip(group_keys, key)),
            "n_structures": int(values.size),
            "mean_log10_psd": mean,
            "ci_2p5_log10_psd": float(low),
            "ci_97p5_log10_psd": float(high),
            "geometric_mean_psd_v2_per_hz": float(10.0 ** mean),
            "ci_2p5_psd_v2_per_hz": float(10.0 ** low),
            "ci_97p5_psd_v2_per_hz": float(10.0 ** high),
        })
    return pd.DataFrame(rows)


def _exact_sign_flip(values: np.ndarray, cfg: DictConfig) -> tuple[float, str, int]:
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    n = values.size
    maximum = int(cfg.analysis.inference.exact_sign_flip_max_structures)
    if n <= maximum:
        signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=n)))
        null = (signs * values).mean(axis=1)
        return float(np.mean(null >= observed - 1.0e-15)), "exact", int(null.size)
    count = int(cfg.analysis.inference.monte_carlo_sign_flips)
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset)
    )
    signs = rng.choice((-1.0, 1.0), size=(count, n))
    null = (signs * values).mean(axis=1)
    return float((1 + np.count_nonzero(null >= observed)) / (count + 1)), "monte_carlo", count


def _bootstrap_ci(values: np.ndarray, cfg: DictConfig, offset: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset) + offset
    )
    samples = rng.choice(
        values,
        size=(int(cfg.analysis.inference.bootstrap_resamples), values.size),
        replace=True,
    ).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def _paired_inference(
    values: np.ndarray, *, metric: str, cfg: DictConfig, bootstrap_offset: int
) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / np.sqrt(n) if n > 1 else 0.0
    if n > 1 and sd > 0.0:
        statistic = mean / se
        t_p = float(stats.t.sf(statistic, df=n - 1))
        interval = stats.t.interval(0.95, df=n - 1, loc=mean, scale=se)
        try:
            wilcoxon = stats.wilcoxon(values, alternative="greater", method="auto")
            wilcoxon_stat = float(wilcoxon.statistic)
            wilcoxon_p = float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = wilcoxon_p = float("nan")
    else:
        statistic = t_p = wilcoxon_stat = wilcoxon_p = float("nan")
        interval = (mean, mean)
    sign_p, sign_method, sign_samples = _exact_sign_flip(values, cfg)
    bootstrap = _bootstrap_ci(values, cfg, bootstrap_offset)
    return {
        "metric": metric,
        "independent_structure_count": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "paired_standardized_effect_dz": mean / sd if sd > 0.0 else float("inf"),
        "positive_structure_count": int(np.count_nonzero(values > 0.0)),
        "positive_structure_fraction": float(np.mean(values > 0.0)),
        "paired_t_statistic": statistic,
        "paired_t_one_sided_p_value": t_p,
        "t_interval_95": [float(x) for x in interval],
        "structure_bootstrap_interval_95": [float(x) for x in bootstrap],
        "exact_sign_flip_one_sided_p_value": sign_p,
        "exact_sign_flip_method": sign_method,
        "exact_sign_flip_samples": sign_samples,
        "wilcoxon_signed_rank_statistic": wilcoxon_stat,
        "wilcoxon_one_sided_p_value": wilcoxon_p,
    }


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = np.minimum.accumulate(
        (ranked * len(values) / np.arange(1, len(values) + 1))[::-1]
    )[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _h1_tables(
    screening: pd.DataFrame, matched_b: pd.DataFrame, target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, float]]:
    paired = screening.merge(
        matched_b,
        on=["context_id", "structure_index", "structure_seed", "hidden_frequency_hz"],
        how="inner",
        validate="one_to_one",
    )
    paired["A_minus_B_log10_alpha_power"] = (
        paired.log10_alpha_power_8_12_hz - paired.B_log10_alpha_power_8_12_hz
    )
    paired["A_minus_B_target_distance"] = (
        paired.baseline_distance_to_B - paired.B_distance_to_B
    )
    paired["A_minus_B_E_rate_hz"] = (
        paired.baseline_E_firing_rate_hz - paired.B_E_firing_rate_hz
    )
    paired["A_minus_B_I_rate_hz"] = (
        paired.baseline_I_firing_rate_hz - paired.B_I_firing_rate_hz
    )
    hidden_excess = []
    for row in paired.itertuples():
        token = _frequency_token(float(row.hidden_frequency_hz))
        hidden_excess.append(
            float(getattr(row, f"log10_power_{token}hz"))
            - float(getattr(row, f"B_log10_power_{token}hz"))
        )
    paired["A_minus_B_hidden_band_log10_power"] = hidden_excess
    threshold = float(target["screening"]["reference_distance_threshold"])
    paired["A_classified_as_A"] = paired.baseline_distance_to_B > threshold
    paired["B_classified_as_B"] = paired.B_distance_to_B <= threshold
    paired["A_B_pair_rate_matched"] = (
        paired.A_minus_B_E_rate_hz.abs()
        <= float(cfg.analysis.rate_reference_tolerance_fraction)
        * paired.B_E_firing_rate_hz.abs().clip(lower=np.finfo(float).tiny)
    ) & (
        paired.A_minus_B_I_rate_hz.abs()
        <= float(cfg.analysis.rate_reference_tolerance_fraction)
        * paired.B_I_firing_rate_hz.abs().clip(lower=np.finfo(float).tiny)
    )
    structure = paired.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        context_count=("context_id", "nunique"),
        mean_A_minus_B_log10_alpha_power=("A_minus_B_log10_alpha_power", "mean"),
        mean_A_minus_B_hidden_band_log10_power=("A_minus_B_hidden_band_log10_power", "mean"),
        mean_A_minus_B_target_distance=("A_minus_B_target_distance", "mean"),
        all_rate_matched=("A_B_pair_rate_matched", "all"),
    )
    inference = _paired_inference(
        structure.mean_A_minus_B_log10_alpha_power.to_numpy(float),
        metric="structure-averaged A-minus-B log10 alpha power",
        cfg=cfg,
        bootstrap_offset=101,
    )
    sensitivity = float(paired.A_classified_as_A.mean())
    specificity = float(paired.B_classified_as_B.mean())
    classification = {
        "A_sensitivity": sensitivity,
        "B_specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2.0,
        "frequency_detection_accuracy": float(
            screening.frequency_detected_correctly.mean()
        ),
    }
    return paired, structure, inference, classification


def _enroll_structures(screening: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, list[int]]:
    required_contexts = len(cfg.analysis.states.frequencies_hz) * int(
        cfg.analysis.crossed_design.n_history_seeds
    )
    audit = screening.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        context_count=("context_id", "nunique"),
        eligible_context_count=("eligible", "sum"),
        all_contexts_eligible=("eligible", "all"),
    )
    audit["structure_screen_positive"] = (
        audit.context_count.eq(required_contexts) & audit.all_contexts_eligible
    )
    target = int(cfg.analysis.crossed_design.n_enrolled_structure_seeds)
    eligible = audit[audit.structure_screen_positive].sort_values("structure_index")
    enrolled = eligible.head(target).structure_seed.astype(int).tolist()
    audit["enrolled_for_H2_H3"] = audit.structure_seed.isin(enrolled)
    audit["selection_uses_stimulation_outcomes"] = False
    return audit, enrolled


def _confirmation_policy_comparison(
    expected: pd.DataFrame, screening: pd.DataFrame, *, frozen_fixed: str,
    preferred_phase: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = expected[expected.action_id.ne(SHAM)].copy()
    rows = []
    for context_id, group in expected.groupby("context_id"):
        screen = screening[screening.context_id.eq(context_id)].iloc[0]
        active_group = group[group.action_id.ne(SHAM)]
        detected = float(screen.detected_frequency_hz)
        policy = active_group[
            np.isclose(active_group.action_frequency_hz, detected)
            & np.isclose(active_group.relative_phase_offset_rad, preferred_phase)
        ].iloc[0]
        fixed = active_group[active_group.action_id.eq(frozen_fixed)].iloc[0]
        sham = group[group.action_id.eq(SHAM)].iloc[0]
        oracle = active_group.sort_values(["expected_distance_to_B", "action_id"]).iloc[0]
        random_distance = float(active_group.expected_distance_to_B.mean())
        rows.append({
            "context_id": str(context_id),
            "structure_index": int(screen.structure_index),
            "structure_seed": int(screen.structure_seed),
            "hidden_frequency_hz": float(screen.hidden_frequency_hz),
            "detected_frequency_hz": detected,
            "policy_action_id": str(policy.action_id),
            "frozen_best_fixed_action_id": frozen_fixed,
            "oracle_action_id": str(oracle.action_id),
            "policy_distance_to_B": float(policy.expected_distance_to_B),
            "fixed_distance_to_B": float(fixed.expected_distance_to_B),
            "random_policy_expected_distance_to_B": random_distance,
            "sham_distance_to_B": float(sham.expected_distance_to_B),
            "oracle_distance_to_B": float(oracle.expected_distance_to_B),
            "policy_advantage_over_fixed": float(
                fixed.expected_distance_to_B - policy.expected_distance_to_B
            ),
            "policy_advantage_over_random": random_distance - float(policy.expected_distance_to_B),
            "policy_advantage_over_sham": float(
                sham.expected_distance_to_B - policy.expected_distance_to_B
            ),
            "policy_oracle_regret": float(
                policy.expected_distance_to_B - oracle.expected_distance_to_B
            ),
            "random_policy_is_uniform_over_four_active_actions": True,
            "policy_uses_only_predecision_EEG": True,
            "policy_uses_hidden_state_or_spikes": False,
        })
    comparison = pd.DataFrame(rows)
    structure = comparison.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        context_count=("context_id", "nunique"),
        mean_policy_advantage_over_fixed=("policy_advantage_over_fixed", "mean"),
        mean_policy_advantage_over_random=("policy_advantage_over_random", "mean"),
        mean_policy_advantage_over_sham=("policy_advantage_over_sham", "mean"),
        mean_policy_oracle_regret=("policy_oracle_regret", "mean"),
    )
    return comparison, structure


def _structure_preserving_shuffle(
    expected: pd.DataFrame, comparison: pd.DataFrame, *, frozen_fixed: str,
    preferred_phase: float,
) -> tuple[pd.DataFrame, float]:
    active = expected[expected.action_id.ne(SHAM)]
    observed = float(
        comparison.groupby("structure_seed").policy_advantage_over_fixed.mean().mean()
    )
    structures = sorted(comparison.structure_seed.unique())
    values = []
    for permutation, bits in enumerate(itertools.product((0, 1), repeat=len(structures))):
        structure_effects = []
        for structure_seed, swap in zip(structures, bits):
            contexts = comparison[comparison.structure_seed.eq(structure_seed)].sort_values(
                "hidden_frequency_hz"
            )
            selected = contexts.detected_frequency_hz.to_numpy(float)
            if swap:
                selected = selected[::-1]
            effects = []
            for (_, row), selected_frequency in zip(contexts.iterrows(), selected):
                group = active[active.context_id.eq(row.context_id)]
                policy = group[
                    np.isclose(group.action_frequency_hz, selected_frequency)
                    & np.isclose(group.relative_phase_offset_rad, preferred_phase)
                ].iloc[0]
                fixed = group[group.action_id.eq(frozen_fixed)].iloc[0]
                effects.append(float(fixed.expected_distance_to_B - policy.expected_distance_to_B))
            structure_effects.append(float(np.mean(effects)))
        values.append({
            "permutation": permutation,
            "swapped_structure_count": int(sum(bits)),
            "shuffled_policy_advantage": float(np.mean(structure_effects)),
        })
    null = pd.DataFrame(values)
    p_value = float(np.mean(null.shuffled_policy_advantage >= observed - 1.0e-15))
    return null, p_value


def _h2_structure_table(
    crossover: pd.DataFrame, orientation: pd.DataFrame
) -> pd.DataFrame:
    result = crossover.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        context_count=("context_id", "nunique"),
        mean_matched_antiphase_improvement_vs_sham=(
            "matched_antiphase_improvement_vs_sham", "mean"
        ),
        mean_frequency_crossover_advantage=("frequency_crossover_advantage", "mean"),
        mean_phase_specific_advantage=("phase_specific_advantage", "mean"),
        mean_hidden_E_ppc_reduction=("matched_antiphase_hidden_E_ppc_reduction", "mean"),
    )
    if not orientation.empty:
        orient = orientation.groupby("structure_seed", as_index=False).agg(
            mean_axial_advantage_over_transverse=("axial_advantage_over_transverse", "mean")
        )
        result = result.merge(orient, on="structure_seed", how="left", validate="one_to_one")
    else:
        result["mean_axial_advantage_over_transverse"] = float("nan")
    return result


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> None:
    figure.tight_layout()
    figure.savefig(root / f"{stem}.png", dpi=300)
    figure.savefig(root / f"{stem}.pdf")
    plt.close(figure)


def _plot_psd(
    summary: pd.DataFrame, *, root: Path, stem: str, title: str,
    zoom: bool, stimulation: bool,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    colors = {
        B_MATCHED: "#222222", B_POPULATION: "#222222", SHAM: "#7f7f7f",
        "inphase": "#e17c05", "antiphase": "#1f77b4", "mismatch": "#b55ca5",
        A_STATE: "#c44e52",
    }
    for axis, hidden in zip(axes, (9.0, 11.0)):
        if stimulation:
            groups: list[tuple[str, pd.DataFrame]] = []
            reference = summary[summary.condition.eq(B_POPULATION)]
            groups.append(("B population reference", reference))
            current = summary[np.isclose(summary.hidden_frequency_hz, hidden)]
            action_ids = {
                "A sham": SHAM,
                "matched 0°": f"f{_frequency_token(hidden)}_inphase",
                "matched 180°": f"f{_frequency_token(hidden)}_antiphase",
                "mismatched 180°": f"f{_frequency_token(20.0-hidden)}_antiphase",
            }
            for label, action_id in action_ids.items():
                groups.append((label, current[current.action_id.eq(action_id)]))
        else:
            current = summary[np.isclose(summary.hidden_frequency_hz, hidden)]
            groups = [
                ("B homogeneous", current[current.condition.eq(B_MATCHED)]),
                (f"A {hidden:g}-Hz carrier", current[current.condition.eq(A_STATE)]),
            ]
        for label, group in groups:
            if group.empty:
                continue
            group = group.sort_values("frequency_hz")
            if label.startswith("B"):
                color = colors[B_MATCHED]
            elif "sham" in label:
                color = colors[SHAM]
            elif "mismatched" in label:
                color = colors["mismatch"]
            elif "180°" in label:
                color = colors["antiphase"]
            elif "0°" in label:
                color = colors["inphase"]
            else:
                color = colors[A_STATE]
            axis.plot(
                group.frequency_hz, group.geometric_mean_psd_v2_per_hz,
                label=label, color=color, linewidth=1.8,
            )
            axis.fill_between(
                group.frequency_hz,
                group.ci_2p5_psd_v2_per_hz,
                group.ci_97p5_psd_v2_per_hz,
                color=color, alpha=0.15, linewidth=0.0,
            )
        axis.axvspan(8.0, 12.0, color="#d9d9d9", alpha=0.25)
        axis.axvline(hidden, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_title(f"A carrier {hidden:g} Hz")
        axis.set_xlabel("Frequency (Hz)")
        if zoom:
            axis.set_xlim(5.0, 15.0)
        else:
            axis.set_xlim(1.0, 30.0)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Ideal EEG PSD (V²/Hz)")
    figure.suptitle(title)
    _save_figure(figure, root, stem)


def _plot_results(
    *, root: Path, baseline_psd: pd.DataFrame, stimulation_psd: pd.DataFrame,
    h1_paired: pd.DataFrame, h1_structure: pd.DataFrame,
    h2_structure: pd.DataFrame, h3_comparison: pd.DataFrame,
    h3_structure: pd.DataFrame, expected: pd.DataFrame,
    metrics: pd.DataFrame, orientation: pd.DataFrame,
) -> None:
    _plot_psd(
        baseline_psd, root=root, stem="figure_01_H1_baseline_PSD_1_30_Hz",
        title="H1: stationary elevated-alpha phenotype", zoom=False,
        stimulation=False,
    )
    _plot_psd(
        baseline_psd, root=root, stem="figure_02_H1_baseline_PSD_alpha_zoom",
        title="H1: alpha-region spectral phenotype", zoom=True,
        stimulation=False,
    )

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for _, row in h1_paired.iterrows():
        axes[0].plot(
            [0, 1],
            [row.B_log10_alpha_power_8_12_hz, row.log10_alpha_power_8_12_hz],
            color="0.75", linewidth=0.8,
        )
    axes[0].scatter(
        np.zeros(len(h1_paired)), h1_paired.B_log10_alpha_power_8_12_hz,
        color="#222222", s=20, label="B",
    )
    axes[0].scatter(
        np.ones(len(h1_paired)), h1_paired.log10_alpha_power_8_12_hz,
        color="#c44e52", s=20, label="A",
    )
    axes[0].set_xticks([0, 1], ["B homogeneous", "A elevated alpha"])
    axes[0].set_ylabel("Baseline log10 alpha power")
    axes[0].legend(fontsize=8)
    h1_positions = np.arange(len(h1_structure))
    axes[1].bar(
        h1_positions,
        h1_structure.mean_A_minus_B_log10_alpha_power,
        color=np.where(
            h1_structure.mean_A_minus_B_log10_alpha_power > 0.0,
            "#3a923a", "#c44e52",
        ),
    )
    axes[1].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1].set_xticks(
        h1_positions, h1_structure.structure_seed.astype(str),
        rotation=75, fontsize=7,
    )
    axes[1].set(
        xlabel="Independent structure seed",
        ylabel="Structure-mean A − B log10 alpha",
    )
    _save_figure(figure, root, "figure_03_H1_paired_phenotype_effects")

    _plot_psd(
        stimulation_psd, root=root, stem="figure_04_H2_stimulation_PSD_alpha_zoom",
        title="H2: frequency- and phase-specific neural EEG modulation",
        zoom=True, stimulation=True,
    )

    figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), sharey=False)
    columns = [
        ("mean_matched_antiphase_improvement_vs_sham", "Matched 180° vs sham"),
        ("mean_frequency_crossover_advantage", "Matched vs mismatched frequency"),
        ("mean_phase_specific_advantage", "180° vs 0° relative phase"),
    ]
    for axis, (column, title) in zip(axes, columns):
        values = h2_structure[column].to_numpy(float)
        axis.scatter(np.zeros_like(values), values, color="#1f77b4", s=35)
        axis.boxplot(values, positions=[0], widths=0.35, showfliers=False)
        axis.axhline(0.0, color="0.3", linewidth=0.8)
        axis.set_xticks([])
        axis.set_title(title, fontsize=9)
        axis.set_ylabel("Target-distance advantage")
    _save_figure(figure, root, "figure_05_H2_structure_level_causal_effects")

    figure, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    strategy = h3_comparison.groupby("structure_seed", as_index=False).agg(
        policy=("policy_distance_to_B", "mean"),
        fixed=("fixed_distance_to_B", "mean"),
        random=("random_policy_expected_distance_to_B", "mean"),
        sham=("sham_distance_to_B", "mean"),
    )
    means = strategy[["sham", "random", "fixed", "policy"]].mean()
    axes[0].bar(means.index, means.values, color=["0.5", "#b55ca5", "#e17c05", "#1f77b4"])
    axes[0].set_ylabel("Expected target distance (lower is better)")
    axes[0].set_title("Frozen strategy comparison")
    positions = np.arange(len(h3_structure))
    width = 0.36
    axes[1].bar(
        positions - width / 2, h3_structure.mean_policy_advantage_over_fixed,
        width, label="vs frozen best fixed",
    )
    axes[1].bar(
        positions + width / 2, h3_structure.mean_policy_advantage_over_random,
        width, label="vs uniform random",
    )
    axes[1].axhline(0.0, color="0.2", linewidth=0.8)
    axes[1].set_xticks(positions, h3_structure.structure_seed.astype(str), rotation=75, fontsize=7)
    axes[1].set_ylabel("EEG-rule advantage")
    axes[1].set_title("Independent-structure effects")
    axes[1].legend(fontsize=7)
    _save_figure(figure, root, "figure_06_H3_policy_comparison")

    active = expected[expected.action_id.ne(SHAM)]
    heat = active.pivot_table(
        index="hidden_frequency_hz", columns="action_id",
        values="expected_improvement_vs_sham", aggfunc="mean",
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    image = axes[0].imshow(heat.to_numpy(), cmap="RdBu_r", aspect="auto")
    axes[0].set_xticks(np.arange(len(heat.columns)), heat.columns, rotation=30, ha="right")
    axes[0].set_yticks(np.arange(len(heat.index)), [f"A {x:g} Hz" for x in heat.index])
    axes[0].set_title("Confirmed context–action map")
    figure.colorbar(image, ax=axes[0], label="Improvement vs sham")
    axes[1].scatter(
        metrics.E_firing_rate_change_hz,
        metrics.I_firing_rate_change_hz,
        c=np.where(metrics.action_id.eq(SHAM), "0.6", "#1f77b4"), s=16,
    )
    axes[1].axhline(0.0, color="0.4", linewidth=0.7)
    axes[1].axvline(0.0, color="0.4", linewidth=0.7)
    axes[1].set(
        xlabel="E firing-rate change (Hz)",
        ylabel="I firing-rate change (Hz)",
        title="Hidden rate-safety audit",
    )
    if not orientation.empty:
        axes[1].text(
            0.02, 0.98,
            f"Mean axial advantage over transverse: "
            f"{orientation.axial_advantage_over_transverse.mean():.3f}",
            transform=axes[1].transAxes, va="top", fontsize=7,
        )
    _save_figure(figure, root, "figure_07_action_map_and_safety")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    frozen_f0 = _load_frozen_f0(cfg)
    power = _power_design(cfg)
    _validate_design(cfg, frozen_f0, power)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### S1-C stationary H1-H3 confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
        print("\n### A priori power design")
        print(json.dumps(_plain(power), indent=2))
    comm.Barrier()
    started = time.perf_counter()

    calibration_rows: list[dict[str, Any]] = []
    calibration_psd_rows: list[dict[str, Any]] = []
    for seed in _reference_seeds(cfg):
        if rank == 0:
            print(f"B population calibration seed={seed}")
        b_cfg = _with_hidden_frequency(
            cfg, frequency_hz=9.0, phase_seed=seed, modulation_depth=0.0
        )
        episode = _run_condition(
            condition_id=REFERENCE,
            condition_cfg=b_cfg,
            action=_sham(b_cfg, REFERENCE),
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "calibration" / str(seed),
            comm=comm, size=size, rank=rank,
            structure_seed=seed, drive_seed=seed, phase_seed=seed,
        )
        if rank == 0:
            screen_feature, _, _ = _episode_feature(episode, "baseline", cfg)
            outcome_feature, _, _ = _episode_feature(episode, "stimulation", cfg)
            row = _epoch_row(episode)
            calibration_rows.append({
                "seed": seed,
                **outcome_feature,
                **{f"screen_{name}": value for name, value in screen_feature.items()},
                "E_firing_rate_hz": float(row.E_firing_rate_hz),
                "I_firing_rate_hz": float(row.I_firing_rate_hz),
            })
            calibration_psd_rows.extend(_psd_rows(
                episode, epoch="stimulation", condition=B_POPULATION,
                action_id=B_POPULATION, context=None, structure_seed=seed,
                future_index=-1, cfg=cfg,
            ))
    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        screening_calibration = pd.DataFrame({
            "seed": calibration.seed,
            **{
                f"log10_power_{_frequency_token(frequency)}hz": calibration[
                    f"screen_log10_power_{_frequency_token(frequency)}hz"
                ]
                for frequency in cfg.analysis.states.frequencies_hz
            },
        })
        target = {
            "screening": _fit_reference_target(screening_calibration, cfg),
            "outcome": _fit_reference_target(calibration, cfg),
            "duration_matching": (
                "screening target uses 12-s B baseline; outcome target uses the "
                "ramp-trimmed four-second B intervention interval"
            ),
        }
        calibration.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_B_spectral_target.json").write_text(
            json.dumps(_plain(target), indent=2)
        )
    else:
        calibration, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    matched_b_rows: list[dict[str, Any]] = []
    baseline_psd_rows: list[dict[str, Any]] = []
    for context in _context_specs(cfg):
        if rank == 0:
            print(
                f"H1 context={context['context_id']} structure={context['structure_seed']} "
                f"carrier={context['hidden_frequency_hz']:g} Hz"
            )
        a_cfg = _with_hidden_frequency(
            cfg,
            frequency_hz=float(context["hidden_frequency_hz"]),
            phase_seed=int(context["phase_seed"]),
            modulation_depth=float(cfg.analysis.states.modulation_depth),
        )
        first_future = _future_seed(cfg, int(context["context_order"]), 0)
        a_episode = _run_condition(
            condition_id=SHAM,
            condition_cfg=a_cfg,
            action=_materialize_action(a_cfg, _action_specs(cfg)[0]),
            stimulate=True,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "screening" / str(context["context_id"]) / A_STATE,
            comm=comm, size=size, rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
            future_drive_seed=first_future,
            phase_seed=int(context["phase_seed"]),
        )
        b_cfg = _with_hidden_frequency(
            cfg,
            frequency_hz=float(context["hidden_frequency_hz"]),
            phase_seed=int(context["phase_seed"]),
            modulation_depth=0.0,
        )
        b_episode = _run_condition(
            condition_id=B_MATCHED,
            condition_cfg=b_cfg,
            action=_sham(b_cfg, B_MATCHED),
            stimulate=False,
            seed=int(context["trial_seed"]),
            action_index=0,
            output_dir=root / "screening" / str(context["context_id"]) / B_MATCHED,
            comm=comm, size=size, rank=rank,
            structure_seed=int(context["structure_seed"]),
            drive_seed=int(context["drive_seed"]),
            phase_seed=int(context["phase_seed"]),
        )
        if rank == 0:
            screen = _screen_context(context, a_episode, target["screening"], cfg)
            screening_rows.append(screen)
            b_feature, _, _ = _episode_feature(b_episode, "baseline", cfg)
            b_epoch = _epoch_row(b_episode, "baseline")
            matched_b_rows.append({
                "context_id": str(context["context_id"]),
                "structure_index": int(context["structure_index"]),
                "structure_seed": int(context["structure_seed"]),
                "hidden_frequency_hz": float(context["hidden_frequency_hz"]),
                **{f"B_{name}": value for name, value in b_feature.items()},
                "B_distance_to_B": _target_distance(b_feature, target["screening"]),
                "B_E_firing_rate_hz": float(b_epoch.E_firing_rate_hz),
                "B_I_firing_rate_hz": float(b_epoch.I_firing_rate_hz),
                "matched_B_not_used_for_screening_or_policy": True,
            })
            baseline_psd_rows.extend(_psd_rows(
                a_episode, epoch="baseline", condition=A_STATE,
                action_id="unstimulated_A", context=context,
                structure_seed=int(context["structure_seed"]), future_index=-1,
                cfg=cfg,
            ))
            baseline_psd_rows.extend(_psd_rows(
                b_episode, epoch="baseline", condition=B_MATCHED,
                action_id="unstimulated_B", context=context,
                structure_seed=int(context["structure_seed"]), future_index=-1,
                cfg=cfg,
            ))
            print(
                f"screen: {'ELIGIBLE' if screen['eligible'] else 'EXCLUDED'}; "
                f"detected={screen['detected_frequency_hz']:g} Hz; "
                f"reason={screen['exclusion_reasons']}"
            )
        del a_episode, b_episode

    if rank == 0:
        screening = pd.DataFrame(screening_rows)
        matched_b = pd.DataFrame(matched_b_rows)
        enrollment, enrolled_structures = _enroll_structures(screening, cfg)
        screening["enrolled_for_H2_H3"] = screening.structure_seed.isin(enrolled_structures)
        h1_paired, h1_structure, h1_inference, h1_classification = _h1_tables(
            screening, matched_b, target, cfg
        )
        screening.to_csv(root / "prospective_screening.csv", index=False)
        enrollment.to_csv(root / "prospective_structure_enrollment.csv", index=False)
        matched_b.to_csv(root / "matched_B_phenotype_metrics.csv", index=False)
        h1_paired.to_csv(root / "H1_paired_context_phenotype.csv", index=False)
        h1_structure.to_csv(root / "H1_structure_level_phenotype.csv", index=False)
        (root / "H1_statistical_inference.json").write_text(json.dumps(_plain({
            "primary": h1_inference,
            "classification": h1_classification,
            "estimand": "unconditional stationary toy A-minus-B phenotype across candidate structures",
        }), indent=2))
        enough_enrolled = len(enrolled_structures) >= int(
            cfg.analysis.crossed_design.n_enrolled_structure_seeds
        )
    else:
        screening = matched_b = enrollment = h1_paired = h1_structure = None
        h1_inference = h1_classification = enrolled_structures = None
        enough_enrolled = None
    enrolled_structures = comm.bcast(enrolled_structures, root=0)
    enough_enrolled = bool(comm.bcast(enough_enrolled, root=0))

    if not enough_enrolled and not bool(cfg.analysis.smoke_test):
        if rank == 0:
            baseline_psd = _summarize_psd(pd.DataFrame(baseline_psd_rows), cfg)
            pd.DataFrame(baseline_psd_rows).to_csv(root / "baseline_PSD_long.csv", index=False)
            baseline_psd.to_csv(root / "baseline_PSD_summary.csv", index=False)
            conclusion = {
                "scope": "S1-C stationary ideal-neural-EEG confirmation",
                "checks": {
                    "H1_minimum_candidate_structures": True,
                    "H2_H3_minimum_prospectively_enrolled_structures": False,
                },
                "conclusions": {
                    "H1_observable_phenotype_confirmed": False,
                    "H2_causal_tacs_modulation_confirmed": False,
                    "H3_one_step_EEG_conditioned_control_confirmed": False,
                    "fixed_sequence_H1_H3_confirmation": False,
                },
                "H1_primary_inference": h1_inference,
                "H1_classification": h1_classification,
                "screening_yield": float(screening.eligible.mean()),
                "reason": "Fewer than twelve structures had both frequency contexts eligible before stimulation outcomes.",
            }
            (root / "experiment_conclusion.json").write_text(json.dumps(_plain(conclusion), indent=2))
            print("\nInsufficient prospective enrollment; H2/H3 not run.")
            print(f"Results saved to: {root}")
        return

    enrolled_contexts = [
        context for context in _context_specs(cfg)
        if int(context["structure_seed"]) in set(enrolled_structures)
    ]
    action_specs = _action_specs(cfg)
    metric_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    stimulation_psd_rows: list[dict[str, Any]] = list(calibration_psd_rows)
    screening_lookup = (
        screening.set_index("context_id").to_dict("index") if rank == 0 else None
    )
    for context in enrolled_contexts:
        if rank == 0:
            print(f"H2/H3 context={context['context_id']}")
            screen = screening_lookup[str(context["context_id"])]
        else:
            screen = None
        state_cfg = _with_hidden_frequency(
            cfg,
            frequency_hz=float(context["hidden_frequency_hz"]),
            phase_seed=int(context["phase_seed"]),
            modulation_depth=float(cfg.analysis.states.modulation_depth),
        )
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(cfg, int(context["context_order"]), future_index)
            sham_episode = _run_condition(
                condition_id=SHAM,
                condition_cfg=state_cfg,
                action=_materialize_action(state_cfg, action_specs[0]),
                stimulate=True,
                seed=int(context["trial_seed"]),
                action_index=0,
                output_dir=root / "episodes" / str(context["context_id"])
                / f"future_{future_index:02d}" / SHAM,
                comm=comm, size=size, rank=rank,
                structure_seed=int(context["structure_seed"]),
                drive_seed=int(context["drive_seed"]),
                future_drive_seed=future_seed,
                phase_seed=int(context["phase_seed"]),
            )
            for action_index, spec in enumerate(action_specs):
                if str(spec["id"]) == SHAM:
                    active_episode = sham_episode
                else:
                    action_cfg = _with_action_frequency(state_cfg, float(spec["frequency_hz"]))
                    active_episode = _run_condition(
                        condition_id=str(spec["id"]),
                        condition_cfg=action_cfg,
                        action=_materialize_action(action_cfg, spec),
                        stimulate=True,
                        seed=int(context["trial_seed"]),
                        action_index=action_index,
                        output_dir=root / "episodes" / str(context["context_id"])
                        / f"future_{future_index:02d}" / str(spec["id"]),
                        comm=comm, size=size, rank=rank,
                        structure_seed=int(context["structure_seed"]),
                        drive_seed=int(context["drive_seed"]),
                        future_drive_seed=future_seed,
                        phase_seed=int(context["phase_seed"]),
                    )
                if rank == 0:
                    metric_rows.append(_metric_row(
                        context=context, screening=screen, future_index=future_index,
                        spec=spec, sham_episode=sham_episode,
                        active_episode=active_episode, target=target["outcome"], cfg=cfg,
                    ))
                    stimulation_psd_rows.extend(_psd_rows(
                        active_episode, epoch="stimulation", condition=A_STATE,
                        action_id=str(spec["id"]), context=context,
                        structure_seed=int(context["structure_seed"]),
                        future_index=future_index, cfg=cfg,
                    ))
                if str(spec["id"]) != SHAM:
                    del active_episode

            if bool(cfg.analysis.orientation_control.enabled) and future_index < int(
                cfg.analysis.orientation_control.n_futures_per_context
            ):
                hidden = float(context["hidden_frequency_hz"])
                transverse_spec = {
                    "id": TRANSVERSE,
                    "role": "orientation_control",
                    "frequency_hz": hidden,
                    "relative_phase_offset_rad": float(frozen_f0["preferred_relative_phase_rad"]),
                    "ac_amplitude_v_per_m": float(cfg.analysis.tacs.amplitude_v_per_m),
                }
                transverse_cfg = _with_action_frequency(state_cfg, hidden)
                transverse_action = _action(
                    transverse_cfg,
                    identifier=TRANSVERSE,
                    role="orientation_control",
                    amplitude=float(cfg.analysis.tacs.amplitude_v_per_m),
                    montage=str(cfg.analysis.tacs.transverse_montage),
                    relative_offset=float(frozen_f0["preferred_relative_phase_rad"]),
                )
                transverse_episode = _run_condition(
                    condition_id=TRANSVERSE,
                    condition_cfg=transverse_cfg,
                    action=transverse_action,
                    stimulate=True,
                    seed=int(context["trial_seed"]),
                    action_index=len(action_specs),
                    output_dir=root / "episodes" / str(context["context_id"])
                    / f"future_{future_index:02d}" / TRANSVERSE,
                    comm=comm, size=size, rank=rank,
                    structure_seed=int(context["structure_seed"]),
                    drive_seed=int(context["drive_seed"]),
                    future_drive_seed=future_seed,
                    phase_seed=int(context["phase_seed"]),
                )
                if rank == 0:
                    transverse_metric = _metric_row(
                        context=context, screening=screen, future_index=future_index,
                        spec=transverse_spec, sham_episode=sham_episode,
                        active_episode=transverse_episode, target=target["outcome"], cfg=cfg,
                    )
                    axial_id = f"f{_frequency_token(hidden)}_antiphase"
                    axial = next(
                        row for row in reversed(metric_rows)
                        if row["context_id"] == context["context_id"]
                        and row["future_index"] == future_index
                        and row["action_id"] == axial_id
                    )
                    transverse_metric["axial_action_id"] = axial_id
                    transverse_metric["axial_distance_to_B"] = float(axial["active_distance_to_B"])
                    transverse_metric["axial_advantage_over_transverse"] = float(
                        transverse_metric["active_distance_to_B"] - axial["active_distance_to_B"]
                    )
                    orientation_rows.append(transverse_metric)
                del transverse_episode
            del sham_episode

    if rank != 0:
        return

    metrics = pd.DataFrame(metric_rows)
    orientation = pd.DataFrame(orientation_rows)
    expected = _expected_action_map(metrics)
    crossover = _crossover_summary(expected)
    h2_structure = _h2_structure_table(crossover, orientation)
    h2_primary = _paired_inference(
        h2_structure.mean_matched_antiphase_improvement_vs_sham.to_numpy(float),
        metric="matched-frequency 180-degree EEG-relative tACS improvement versus sham",
        cfg=cfg, bootstrap_offset=201,
    )
    h2_frequency = _paired_inference(
        h2_structure.mean_frequency_crossover_advantage.to_numpy(float),
        metric="matched-minus-mismatched frequency target-distance advantage",
        cfg=cfg, bootstrap_offset=202,
    )
    h2_phase = _paired_inference(
        h2_structure.mean_phase_specific_advantage.to_numpy(float),
        metric="180-degree-minus-0-degree relative-phase target-distance advantage",
        cfg=cfg, bootstrap_offset=203,
    )
    secondary_p = np.asarray([
        h2_frequency["exact_sign_flip_one_sided_p_value"],
        h2_phase["exact_sign_flip_one_sided_p_value"],
    ])
    secondary_q = _bh_fdr(secondary_p)
    h2_frequency["BH_FDR_q"] = float(secondary_q[0])
    h2_phase["BH_FDR_q"] = float(secondary_q[1])

    enrolled_screening = screening[screening.structure_seed.isin(enrolled_structures)]
    h3_comparison, h3_structure = _confirmation_policy_comparison(
        expected, enrolled_screening,
        frozen_fixed=str(frozen_f0["best_fixed_action_id"]),
        preferred_phase=float(frozen_f0["preferred_relative_phase_rad"]),
    )
    h3_fixed = _paired_inference(
        h3_structure.mean_policy_advantage_over_fixed.to_numpy(float),
        metric="frozen EEG rule advantage over F0-frozen best fixed action",
        cfg=cfg, bootstrap_offset=301,
    )
    h3_random = _paired_inference(
        h3_structure.mean_policy_advantage_over_random.to_numpy(float),
        metric="frozen EEG rule advantage over uniform random active policy",
        cfg=cfg, bootstrap_offset=302,
    )
    shuffle_null, shuffle_p = _structure_preserving_shuffle(
        expected, h3_comparison,
        frozen_fixed=str(frozen_f0["best_fixed_action_id"]),
        preferred_phase=float(frozen_f0["preferred_relative_phase_rad"]),
    )

    baseline_long = pd.DataFrame(baseline_psd_rows)
    stimulation_long = pd.DataFrame(stimulation_psd_rows)
    baseline_summary = _summarize_psd(baseline_long, cfg)
    stimulation_summary = _summarize_psd(stimulation_long, cfg)

    criteria = cfg.analysis.criteria
    h1_checks = {
        "minimum_candidate_structures": h1_structure.structure_seed.nunique()
        >= int(criteria.minimum_candidate_structures) or bool(cfg.analysis.smoke_test),
        "complete_matched_A_B_grid": len(h1_paired) == len(screening),
        "A_B_mean_afferent_rate_matched_by_construction": True,
        "matched_B_excluded_from_screening_and_policy": bool(
            matched_b.matched_B_not_used_for_screening_or_policy.all()
        ),
        "practically_elevated_alpha": float(h1_inference["mean"])
        >= float(criteria.minimum_H1_alpha_excess_log10),
        "primary_exact_structure_test_rejects_null": float(
            h1_inference["exact_sign_flip_one_sided_p_value"]
        ) <= float(criteria.maximum_primary_p_value),
        "phenotype_positive_across_structures": float(
            h1_inference["positive_structure_fraction"]
        ) >= float(criteria.minimum_H1_positive_structure_fraction),
        "held_reference_threshold_balanced_accuracy": float(
            h1_classification["balanced_accuracy"]
        ) >= float(criteria.minimum_H1_balanced_accuracy),
        "frequency_identified_from_predecision_EEG": float(
            h1_classification["frequency_detection_accuracy"]
        ) >= float(criteria.minimum_frequency_detection_accuracy),
        "A_B_firing_rates_matched": bool(h1_structure.all_rate_matched.all()),
    }
    h1_pass = bool(all(h1_checks.values()) and not bool(cfg.analysis.smoke_test))

    h2_checks = {
        "frozen_F0_passed_and_hash_locked": True,
        "confirmation_seeds_disjoint_from_F0": True,
        "minimum_prospectively_enrolled_structures": h2_structure.structure_seed.nunique()
        >= int(criteria.minimum_enrolled_structures) or bool(cfg.analysis.smoke_test),
        "multiple_independent_futures_per_action": int(expected.future_count.min())
        >= int(criteria.minimum_future_continuations) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_actions": float(
            metrics.baseline_relative_rms_error.max()
        ) <= float(criteria.maximum_baseline_relative_rms_error),
        "all_active_actions_use_fixed_0p4_V_per_m": bool(np.allclose(
            metrics.loc[metrics.action_id.ne(SHAM), "amplitude_v_per_m"], 0.4
        )),
        "action_phase_uses_only_preceding_EEG": float(
            metrics.action_phase_tracking_error_rad.max()
        ) <= float(criteria.maximum_phase_tracking_error_rad),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "field_removal_recovered": bool(metrics.field_removal_recovered.all()),
        "matched_antiphase_practically_improves_over_sham": float(h2_primary["mean"])
        >= float(criteria.minimum_H2_matched_antiphase_improvement),
        "primary_exact_structure_test_rejects_null": float(
            h2_primary["exact_sign_flip_one_sided_p_value"]
        ) <= float(criteria.maximum_primary_p_value),
        "primary_effect_positive_across_structures": float(
            h2_primary["positive_structure_fraction"]
        ) >= float(criteria.minimum_H2_positive_structure_fraction),
        "frequency_crossover_practical_and_FDR_significant": float(
            h2_frequency["mean"]
        ) >= float(criteria.minimum_H2_frequency_crossover) and float(
            h2_frequency["BH_FDR_q"]
        ) <= float(criteria.maximum_H2_secondary_FDR_q),
        "phase_crossover_practical_and_FDR_significant": float(h2_phase["mean"])
        >= float(criteria.minimum_H2_phase_crossover) and float(
            h2_phase["BH_FDR_q"]
        ) <= float(criteria.maximum_H2_secondary_FDR_q),
        "orientation_specific_directional_audit": bool(
            orientation.empty or orientation.axial_advantage_over_transverse.mean() > 0.0
        ),
    }
    h2_pass = bool(all(h2_checks.values()) and h1_pass)

    policy_actions = set(h3_comparison.policy_action_id)
    h3_checks = {
        "frozen_rule_loaded_without_refitting": True,
        "policy_uses_only_predecision_EEG": bool(
            h3_comparison.policy_uses_only_predecision_EEG.all()
        ),
        "hidden_generator_and_spikes_excluded_from_policy": bool(
            (~h3_comparison.policy_uses_hidden_state_or_spikes).all()
        ),
        "EEG_rule_uses_both_frequency_actions": len(policy_actions) == 2,
        "policy_advantage_over_fixed_is_practical": float(h3_fixed["mean"])
        >= float(criteria.minimum_H3_policy_advantage_over_fixed),
        "primary_exact_structure_test_rejects_null": float(
            h3_fixed["exact_sign_flip_one_sided_p_value"]
        ) <= float(criteria.maximum_primary_p_value),
        "policy_advantage_positive_across_structures": float(
            h3_fixed["positive_structure_fraction"]
        ) >= float(criteria.minimum_H3_positive_structure_fraction),
        "policy_beats_uniform_random_strategy": float(h3_random["mean"])
        >= float(criteria.minimum_H3_policy_advantage_over_random),
        "structure_preserving_context_shuffle_fails": float(shuffle_p)
        <= float(criteria.maximum_H3_context_shuffle_p_value),
    }
    h3_pass = bool(all(h3_checks.values()) and h2_pass)

    metrics.to_csv(root / "context_action_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_action_map.csv", index=False)
    crossover.to_csv(root / "H2_frequency_phase_crossover.csv", index=False)
    h2_structure.to_csv(root / "H2_structure_level_effects.csv", index=False)
    orientation.to_csv(root / "H2_orientation_control.csv", index=False)
    h3_comparison.to_csv(root / "H3_context_policy_comparison.csv", index=False)
    h3_structure.to_csv(root / "H3_structure_level_policy_effects.csv", index=False)
    shuffle_null.to_csv(root / "H3_structure_preserving_shuffle_null.csv", index=False)
    baseline_long.to_csv(root / "baseline_PSD_long.csv", index=False)
    baseline_summary.to_csv(root / "baseline_PSD_summary.csv", index=False)
    stimulation_long.to_csv(root / "stimulation_PSD_long.csv", index=False)
    stimulation_summary.to_csv(root / "stimulation_PSD_summary.csv", index=False)
    (root / "H2_statistical_inference.json").write_text(json.dumps(_plain({
        "primary_matched_antiphase_vs_sham": h2_primary,
        "secondary_frequency_crossover": h2_frequency,
        "secondary_phase_crossover": h2_phase,
        "multiplicity": "frequency and phase secondary exact tests use Benjamini-Hochberg FDR",
    }), indent=2))
    (root / "H3_statistical_inference.json").write_text(json.dumps(_plain({
        "primary_policy_vs_frozen_best_fixed": h3_fixed,
        "secondary_policy_vs_uniform_random": h3_random,
        "structure_preserving_context_shuffle_p_value": shuffle_p,
        "random_strategy": "uniform expectation over the four active counterfactual actions",
    }), indent=2))
    provenance = {
        "experiment": "S1-C stationary H1-H3 confirmation",
        "frozen_F0": {
            "root": frozen_f0["root"],
            "sha256": frozen_f0["sha256"],
            "best_fixed_action_id": frozen_f0["best_fixed_action_id"],
            "preferred_relative_phase_rad": frozen_f0["preferred_relative_phase_rad"],
        },
        "state_A": {
            "description": "mean-rate-matched stationary 9/11-Hz modulation of private Poisson afferents",
            "modulation_depth": float(cfg.analysis.states.modulation_depth),
        },
        "state_B": {
            "description": "homogeneous private Poisson afferents",
            "modulation_depth": 0.0,
        },
        "policy_observation": "12-s stimulation-free ideal neural EEG only",
        "actions": action_specs,
        "random_strategy": "uniform expectation over four active actions; no extra outcome-selected action",
        "matched_B_role": "H1 paired inference and plotting only; excluded from screening, target calibration, and policy",
        "statistical_unit": "independent circuit structure",
        "not_a_bandit": True,
        "not_a_disease_or_treatment_model": True,
        "concurrent_EEG_is_ideal_and_artifact_free": True,
        "power_design": power,
    }
    (root / "protocol_and_provenance.json").write_text(
        json.dumps(_plain(provenance), indent=2)
    )
    conclusion = {
        "scope": "S1-C stationary ideal-neural-EEG H1-H3 confirmation",
        "checks": {"H1": h1_checks, "H2": h2_checks, "H3": h3_checks},
        "conclusions": {
            "H1_observable_phenotype_confirmed": h1_pass,
            "H2_causal_tacs_modulation_confirmed": h2_pass,
            "H3_one_step_EEG_conditioned_control_confirmed": h3_pass,
            "fixed_sequence_H1_H3_confirmation": bool(h1_pass and h2_pass and h3_pass),
            "contextual_bandit_status": "NOT TRAINED OR TESTED",
        },
        "H1_primary_inference": h1_inference,
        "H1_classification": h1_classification,
        "H2_primary_inference": h2_primary,
        "H3_primary_inference": h3_fixed,
        "H3_uniform_random_inference": h3_random,
        "screening": {
            "candidate_structures": int(enrollment.structure_seed.nunique()),
            "screen_positive_structures": int(enrollment.structure_screen_positive.sum()),
            "enrolled_structures": len(enrolled_structures),
            "context_screening_yield": float(screening.eligible.mean()),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; frequency/action/future are repeats",
        "inference_boundary": "toy-model ideal-EEG confirmation; no disease, clinical, artifact-robust, or ML claim",
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_plain(conclusion), indent=2)
    )

    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            baseline_psd=baseline_summary,
            stimulation_psd=stimulation_summary,
            h1_paired=h1_paired,
            h1_structure=h1_structure,
            h2_structure=h2_structure,
            h3_comparison=h3_comparison,
            h3_structure=h3_structure,
            expected=expected,
            metrics=metrics,
            orientation=orientation,
        )

    print("\n### S1-C prospective screening")
    print(f"candidate structures: {int(enrollment.structure_seed.nunique())}")
    print(f"screen-positive structures: {int(enrollment.structure_screen_positive.sum())}")
    print(f"enrolled structures: {len(enrolled_structures)}")
    for hypothesis, checks in (("H1", h1_checks), ("H2", h2_checks), ("H3", h3_checks)):
        print(f"\n### {hypothesis} checks")
        for name, passed in checks.items():
            print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(f"\nH1 observable phenotype: {'CONFIRMED' if h1_pass else 'NOT CONFIRMED'}")
    print(f"H2 causal tACS modulation: {'CONFIRMED' if h2_pass else 'NOT CONFIRMED'}")
    print(f"H3 one-step EEG-conditioned control: {'CONFIRMED' if h3_pass else 'NOT CONFIRMED'}")
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
