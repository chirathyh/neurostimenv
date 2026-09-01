"""H4-C: disjoint confirmation of causal adaptive phase maintenance.

The hash-locked H4-BW2 discovery selected a 0.5-s EEG history refreshed every
125 ms.  H4-C freezes that controller, its 0.2-V/m axial field, EEG-selected
9/11-Hz carrier, pi-relative target, correction dynamics, eligibility target,
and eight-second B outcome target.  New circuit structures compare only sham,
one-time phase initialization, and the selected controller.

The independent inferential unit is circuit structure. Carrier frequency,
diffusion level, and future continuation are paired repeats. This is an ideal
neural-EEG toy-model confirmation, not a clinical, artifact-robust, disease,
contextual-bandit, or learned-prediction claim.
"""

from __future__ import annotations

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


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_action_map import (  # noqa: E402
    HIGH,
    LOW,
    _context_features,
    _future_seed,
    _run_context_specs,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    ONE_TIME,
    SHAM,
    _controller_modes,
    _metric_rows,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    FAST,
    _augment_metric_rows,
    _json_ready,
    _profile,
    _run_controller,
    _sha256,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (  # noqa: E402
    _augment_common_audit,
    _expected_map,
    _load_sources as _load_h4bw2_upstream,
)
from experiments.ballnstick_analysis.run_ballnstick_stationary_h1_h3_confirmation import (  # noqa: E402
    _one_sample_t_power,
    _paired_inference,
    _paper_psd,
    _required_n,
    _save_figure,
)


ROOT_NAME = "h4_confirmation"
SELECTED = FAST
EXPECTED_MODES = [SHAM, ONE_TIME, SELECTED]


def _load_frozen_h4bw2(cfg: DictConfig) -> dict[str, Any]:
    """Verify the complete source chain and load the frozen H4-C protocol."""
    sources = _load_h4bw2_upstream(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_h4bw2.result_dir)))
    files = {
        "conclusion": root / "experiment_conclusion.json",
        "candidate": root / "frozen_controller_candidate.json",
        "target": root / "frozen_duration_matched_B_target.json",
        "provenance": root / "protocol_and_provenance.json",
        "screening": root / "prospective_screening.csv",
        "metrics": root / "context_controller_future_metrics.csv",
        "updates": root / "causal_phase_updates.csv",
        "summary": root / "controller_selection_summary.csv",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen H4-BW2 sources: {missing}")
    observed = {name: _sha256(path) for name, path in files.items()}
    expected = {
        name: str(cfg.analysis.source_h4bw2.expected_sha256[name]) for name in files
    }
    if observed != expected:
        raise RuntimeError(
            f"H4-BW2 source hash mismatch: expected={expected}, observed={observed}"
        )
    conclusion = json.loads(files["conclusion"].read_text())
    candidate = json.loads(files["candidate"].read_text())
    target = json.loads(files["target"].read_text())
    provenance = json.loads(files["provenance"].read_text())
    if not all(bool(value) for value in conclusion["checks"].values()):
        raise RuntimeError("Frozen H4-BW2 did not pass every discovery check.")
    source_conclusion = conclusion["conclusions"]
    if (
        str(source_conclusion["selected_controller"]) != SELECTED
        or not bool(source_conclusion["cadence_candidate_found"])
        or not bool(source_conclusion["ready_for_disjoint_12_structure_H4_confirmation"])
    ):
        raise RuntimeError("Frozen H4-BW2 did not authorize H4 confirmation.")
    if str(candidate["selected_controller"]) != SELECTED:
        raise RuntimeError("H4-BW2 candidate JSON disagrees with its conclusion.")
    summary = pd.read_csv(files["summary"])
    selected = summary[summary.controller_mode.eq(SELECTED)]
    if len(selected) != 1 or not bool(selected.iloc[0].passes_cadence_gate):
        raise RuntimeError("The frozen selected controller did not pass H4-BW2.")
    for table_name in ("screening", "metrics"):
        table = pd.read_csv(files[table_name])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                sources["source_seed_union"].update(
                    table[column].dropna().astype(int).tolist()
                )
    sources["roots"]["h4bw2"] = str(root)
    sources["hashes"]["h4bw2"] = observed
    sources["candidate"] = candidate
    sources["target"] = target
    sources["H4BW2_conclusion"] = conclusion
    sources["H4BW2_provenance"] = provenance
    return sources


def _power_design(cfg: DictConfig) -> dict[str, Any]:
    block = cfg.analysis.power_design
    effect = float(block.minimum_standardized_effect_dz)
    alpha = float(block.alpha_one_sided)
    target = float(block.target_power)
    planned = int(block.planned_independent_structures)
    return {
        "alpha_one_sided": alpha,
        "target_power": target,
        "minimum_standardized_effect_dz": effect,
        "planned_independent_structures": planned,
        "required_independent_structures": _required_n(
            effect_size=effect, alpha=alpha, target_power=target
        ),
        "a_priori_t_approximation_power": _one_sample_t_power(
            n=planned, effect_size=effect, alpha=alpha
        ),
        "planning_test": "one-sided paired t approximation",
        "primary_inference": "one-sided exact structure-level sign flip",
        "statistical_unit": "independent circuit structure",
        "repeated_axes": ["carrier frequency", "diffusion level", "future"],
    }


def _candidate_structure_table(cfg: DictConfig) -> pd.DataFrame:
    contexts = pd.DataFrame(_run_context_specs(cfg))
    return contexts[["structure_seed"]].drop_duplicates().sort_values(
        "structure_seed"
    ).reset_index(drop=True)


def _validate_design(
    cfg: DictConfig, sources: dict[str, Any], power: dict[str, Any]
) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H4-C requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H4-C may not change recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H4-C freezes the 9/11-Hz state grid.")
    levels = [
        (str(x.label), float(x.diffusion_rad2_per_s))
        for x in cfg.analysis.states.phase_diffusion_levels
    ]
    if levels != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("H4-C freezes D={0.5, 2.0} rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H4-C freezes afferent modulation depth to 0.04.")
    if _controller_modes(cfg) != EXPECTED_MODES:
        raise ValueError(f"H4-C controller order must be {EXPECTED_MODES}.")

    candidate = sources["candidate"]
    expected_candidate = {
        "selected_controller": SELECTED,
        "selected_profile": {
            "adaptive": True, "history_ms": 500.0,
            "update_interval_ms": 125.0,
        },
        "initialization_history_ms": 1000.0,
        "common_audit_history_ms": 1000.0,
        "common_audit_interval_ms": 250.0,
        "correction_horizon_ms": 250.0,
        "amplitude_v_per_m": 0.2,
        "relative_phase_offset_rad": np.pi,
        "montage": "axial",
    }
    if str(candidate["selected_controller"]) != expected_candidate["selected_controller"]:
        raise ValueError("H4-C selected controller changed after discovery.")
    if candidate["selected_profile"] != expected_candidate["selected_profile"]:
        raise ValueError("H4-C selected profile changed after discovery.")
    for name in (
        "initialization_history_ms", "common_audit_history_ms",
        "common_audit_interval_ms", "correction_horizon_ms",
        "amplitude_v_per_m", "relative_phase_offset_rad",
    ):
        if not np.isclose(float(candidate[name]), float(expected_candidate[name])):
            raise ValueError(f"H4-C frozen candidate field changed: {name}.")
    if str(candidate["montage"]) != "axial":
        raise ValueError("H4-C freezes the axial montage.")
    profile = _profile(cfg, SELECTED)
    if profile != expected_candidate["selected_profile"]:
        raise ValueError("H4-C configured controller differs from frozen H4-BW2.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("H4-C freezes all active fields to 0.2 V/m.")
    for name, expected in (
        ("initialization_history_ms", 1000.0),
        ("common_audit_history_ms", 1000.0),
        ("common_audit_interval_ms", 250.0),
        ("correction_horizon_ms", 250.0),
        ("maximum_frequency_correction_hz", 2.0),
    ):
        if not np.isclose(float(cfg.analysis.tacs[name]), expected):
            raise ValueError(f"H4-C freezes {name}={expected:g}.")
    if not np.isclose(float(cfg.analysis.tacs.relative_phase_offset_rad), np.pi):
        raise ValueError("H4-C freezes the pi-relative phase target.")
    if not np.isclose(float(sources["target"]["outcome_duration_s"]), 8.0):
        raise ValueError("H4-C requires the frozen eight-second B outcome target.")

    timeline = cfg.analysis.timeline
    if int(timeline.baseline_steps) < (4 if smoke else 12):
        raise ValueError("H4-C requires a 12-s baseline, or 4 s in smoke mode.")
    if int(timeline.washout_steps) < (1 if smoke else 2):
        raise ValueError("H4-C requires a two-second washout.")
    endpoint_ms = (
        int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(timeline.stimulation_analysis_trim_ms)
    )
    if not smoke and not np.isclose(endpoint_ms, 8000.0):
        raise ValueError("Full H4-C freezes the eight-second primary endpoint.")
    if endpoint_ms <= 0.0 or not np.isclose(endpoint_ms / 1000.0, round(endpoint_ms / 1000.0)):
        raise ValueError("H4-C endpoint must contain complete one-second windows.")

    contexts = _run_context_specs(cfg)
    namespaces = [
        {int(x["structure_seed"]) for x in contexts},
        {int(x["history_seed"]) for x in contexts},
        {int(x["phase_seed"]) for x in contexts},
        {int(x["trial_seed"]) for x in contexts},
        {
            _future_seed(cfg, context, future_index)
            for context in contexts
            for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations))
        },
    ]
    if any(not values for values in namespaces):
        raise ValueError("Every H4-C seed namespace must be nonempty.")
    if any(
        namespaces[left].intersection(namespaces[right])
        for left in range(len(namespaces))
        for right in range(left + 1, len(namespaces))
    ):
        raise ValueError("H4-C seed namespaces overlap each other.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H4-C seeds overlap a frozen source experiment.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("An H4-C structure seed exceeds the uint32 mapping range.")
    if smoke:
        return
    if int(cfg.analysis.crossed_design.n_enrolled_structure_seeds) != int(
        power["planned_independent_structures"]
    ):
        raise ValueError("H4-C enrollment must match its a priori power design.")
    if int(power["planned_independent_structures"]) < int(
        power["required_independent_structures"]
    ) or float(power["a_priori_t_approximation_power"]) < float(
        power["target_power"]
    ):
        raise ValueError("H4-C a priori design is underpowered.")
    if int(cfg.analysis.crossed_design.n_structure_seeds) < int(
        cfg.analysis.crossed_design.n_enrolled_structure_seeds
    ):
        raise ValueError("H4-C needs at least as many candidates as enrolled structures.")
    if int(cfg.analysis.crossed_design.n_future_continuations) < int(
        cfg.analysis.criteria.minimum_future_continuations
    ):
        raise ValueError("H4-C requires six independent futures per controller.")


def _screen_structure_table(screening: pd.DataFrame) -> pd.DataFrame:
    expected_contexts = 4
    rows = []
    for structure_seed, group in screening.groupby("structure_seed", sort=True):
        complete = bool(
            len(group) == expected_contexts
            and group.hidden_frequency_hz.nunique() == 2
            and group.label.nunique() == 2
        )
        rows.append({
            "structure_seed": int(structure_seed),
            "context_count": int(len(group)),
            "complete_frequency_diffusion_grid": complete,
            "all_contexts_eligible": bool(complete and group.eligible.all()),
            "frequency_detection_accuracy": float(
                group.EEG_frequency_selection_correct.mean()
            ),
            "minimum_recent_resultant_to_rms": float(
                group.recent_resultant_to_rms.min()
            ),
            "minimum_alpha_excess_log10": float(
                group.context_alpha_excess_log10.min()
            ),
        })
    return pd.DataFrame(rows)


def _common_initialization(updates: pd.DataFrame) -> bool:
    initial = updates[
        updates.update_index.eq(0) & updates.controller_mode.ne(SHAM)
    ]
    if initial.empty:
        return False
    for _, group in initial.groupby(["context_id", "future_index"]):
        if set(group.controller_mode.astype(str)) != {ONE_TIME, SELECTED}:
            return False
        phases = group.desired_field_phase_rad.to_numpy(float)
        errors = np.angle(np.exp(1j * (phases - phases[0])))
        if np.max(np.abs(errors)) > 1.0e-10:
            return False
        if not np.allclose(group.phase_history_ms, 1000.0):
            return False
    return True


def _psd_rows(
    episode: dict[str, Any], *, context: dict[str, Any], future_index: int,
    controller_mode: str, cfg: DictConfig,
) -> list[dict[str, Any]]:
    frequencies, psd = _paper_psd(episode, epoch="stimulation", cfg=cfg)
    eps = np.finfo(float).tiny
    return [{
        "context_id": str(context["context_id"]),
        "structure_seed": int(context["structure_seed"]),
        "hidden_frequency_hz": float(context["hidden_frequency_hz"]),
        "label": str(context["label"]),
        "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        "future_index": int(future_index),
        "controller_mode": controller_mode,
        "frequency_hz": float(frequency),
        "psd_v2_per_hz": float(value),
        "log10_psd_v2_per_hz": float(np.log10(max(float(value), eps))),
    } for frequency, value in zip(frequencies, psd)]


def _summarize_psd(frame: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    units = frame.groupby([
        "controller_mode", "label", "hidden_frequency_hz", "structure_seed",
        "frequency_hz",
    ], as_index=False).agg(mean_log10_psd=("log10_psd_v2_per_hz", "mean"))
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.inference.random_seed_offset) + 41
    )
    count = int(cfg.analysis.paper_psd.bootstrap_resamples)
    rows = []
    keys = ["controller_mode", "label", "hidden_frequency_hz", "frequency_hz"]
    for key, group in units.groupby(keys, sort=True):
        values = group.mean_log10_psd.to_numpy(float)
        if values.size == 1:
            low = high = float(values[0])
        else:
            samples = rng.choice(values, size=(count, values.size), replace=True).mean(axis=1)
            low, high = np.quantile(samples, [0.025, 0.975])
        mean = float(values.mean())
        rows.append({
            **dict(zip(keys, key)),
            "n_structures": int(values.size),
            "mean_log10_psd": mean,
            "ci_2p5_log10_psd": float(low),
            "ci_97p5_log10_psd": float(high),
            "geometric_mean_psd_v2_per_hz": float(10.0 ** mean),
            "ci_2p5_psd_v2_per_hz": float(10.0 ** low),
            "ci_97p5_psd_v2_per_hz": float(10.0 ** high),
        })
    return pd.DataFrame(rows)


def _effect_tables(
    expected: pd.DataFrame, metrics: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = [
        "context_id", "structure_seed", "hidden_frequency_hz", "label",
        "diffusion_rad2_per_s", "context_C1",
    ]
    distance = expected.pivot(
        index=index, columns="controller_mode",
        values="expected_post_distance_to_B_log10",
    ).reset_index()
    phase = expected.pivot(
        index=index, columns="controller_mode",
        values="mean_abs_common_phase_error_rad",
    ).reset_index()
    distance["primary_refresh_advantage_over_one_time_log10"] = (
        distance[ONE_TIME] - distance[SELECTED]
    )
    distance["secondary_refresh_advantage_over_sham_log10"] = (
        distance[SHAM] - distance[SELECTED]
    )
    distance["one_time_advantage_over_sham_log10"] = (
        distance[SHAM] - distance[ONE_TIME]
    )
    distance["phase_error_reduction_vs_one_time_rad"] = (
        phase[ONE_TIME] - phase[SELECTED]
    )
    realized = metrics.pivot(
        index=["context_id", "future_index"], columns="controller_mode",
        values="post_distance_to_B_log10",
    ).reset_index()
    reliability = []
    for context_id, group in realized.groupby("context_id"):
        primary = group[ONE_TIME].to_numpy(float) - group[SELECTED].to_numpy(float)
        sham = group[SHAM].to_numpy(float) - group[SELECTED].to_numpy(float)
        reliability.append({
            "context_id": str(context_id),
            "future_count": int(len(group)),
            "refresh_win_fraction_over_one_time": float(np.mean(primary > 0.0)),
            "refresh_win_fraction_over_sham": float(np.mean(sham > 0.0)),
            "paired_primary_effect_sd_log10": float(np.std(primary, ddof=1)),
        })
    context = distance.merge(
        pd.DataFrame(reliability), on="context_id", how="left", validate="one_to_one"
    )
    structure = context.groupby("structure_seed", as_index=False).agg(
        context_count=("context_id", "nunique"),
        frequency_count=("hidden_frequency_hz", "nunique"),
        diffusion_count=("label", "nunique"),
        mean_primary_refresh_advantage_over_one_time_log10=(
            "primary_refresh_advantage_over_one_time_log10", "mean"
        ),
        mean_secondary_refresh_advantage_over_sham_log10=(
            "secondary_refresh_advantage_over_sham_log10", "mean"
        ),
        mean_one_time_advantage_over_sham_log10=(
            "one_time_advantage_over_sham_log10", "mean"
        ),
        mean_phase_error_reduction_vs_one_time_rad=(
            "phase_error_reduction_vs_one_time_rad", "mean"
        ),
        mean_refresh_win_fraction_over_one_time=(
            "refresh_win_fraction_over_one_time", "mean"
        ),
        mean_refresh_win_fraction_over_sham=(
            "refresh_win_fraction_over_sham", "mean"
        ),
        mean_paired_primary_effect_sd_log10=(
            "paired_primary_effect_sd_log10", "mean"
        ),
    )
    diffusion = context.groupby("label", as_index=False).agg(
        context_count=("context_id", "nunique"),
        mean_primary_advantage_log10=(
            "primary_refresh_advantage_over_one_time_log10", "mean"
        ),
        mean_secondary_advantage_vs_sham_log10=(
            "secondary_refresh_advantage_over_sham_log10", "mean"
        ),
        mean_phase_error_reduction_rad=(
            "phase_error_reduction_vs_one_time_rad", "mean"
        ),
        mean_future_win_fraction=("refresh_win_fraction_over_one_time", "mean"),
    )
    return context, structure, diffusion


def _temporal_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    return trajectories.groupby([
        "structure_seed", "controller_mode", "analysis_window_index",
    ], as_index=False).agg(
        mean_distance_to_B_log10=("distance_to_B_log10", "mean")
    )


def _phase_time_table(updates: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    first = updates.groupby([
        "context_id", "future_index", "controller_mode"
    ]).boundary_ms.transform("min")
    frame = updates.copy()
    frame["time_since_onset_ms"] = frame.boundary_ms - first
    interval = float(cfg.analysis.tacs.common_audit_interval_ms)
    frame = frame[np.isclose(
        np.mod(frame.time_since_onset_ms, interval), 0.0, atol=1.0e-8
    )]
    frame["abs_common_phase_error_rad"] = (
        frame.common_audit_phase_error_before_correction_rad.abs()
    )
    return frame.groupby([
        "structure_seed", "controller_mode", "time_since_onset_ms"
    ], as_index=False).agg(
        mean_abs_common_phase_error_rad=("abs_common_phase_error_rad", "mean")
    )


def _inference_table(inferences: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for contrast, result in inferences.items():
        rows.append({
            "contrast": contrast,
            "independent_structure_count": result["independent_structure_count"],
            "mean": result["mean"],
            "sd": result["sd"],
            "se": result["se"],
            "paired_standardized_effect_dz": result["paired_standardized_effect_dz"],
            "positive_structure_count": result["positive_structure_count"],
            "positive_structure_fraction": result["positive_structure_fraction"],
            "t_ci_95_low": result["t_interval_95"][0],
            "t_ci_95_high": result["t_interval_95"][1],
            "bootstrap_ci_95_low": result["structure_bootstrap_interval_95"][0],
            "bootstrap_ci_95_high": result["structure_bootstrap_interval_95"][1],
            "exact_sign_flip_one_sided_p_value": result[
                "exact_sign_flip_one_sided_p_value"
            ],
            "wilcoxon_one_sided_p_value": result["wilcoxon_one_sided_p_value"],
        })
    return pd.DataFrame(rows)


def _mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    critical = 2.201 if values.size == 12 else 1.96
    return mean, mean - critical * se, mean + critical * se


def _plot_results(
    *, root: Path, psd: pd.DataFrame, expected: pd.DataFrame,
    structure: pd.DataFrame, context: pd.DataFrame, temporal: pd.DataFrame,
    phase_time: pd.DataFrame, metrics: pd.DataFrame,
) -> None:
    colors = {SHAM: "#7f7f7f", ONE_TIME: "#e17c05", SELECTED: "#1f77b4"}
    labels = {SHAM: "Sham", ONE_TIME: "One-time phase", SELECTED: "0.5 s / 125 ms"}

    figure, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, sharey=True)
    for axis, (diffusion, carrier) in zip(
        axes.flat,
        [(LOW, 9.0), (LOW, 11.0), (HIGH, 9.0), (HIGH, 11.0)],
    ):
        for mode in EXPECTED_MODES:
            group = psd[
                psd.label.eq(diffusion)
                & psd.hidden_frequency_hz.eq(carrier)
                & psd.controller_mode.eq(mode)
            ]
            axis.plot(
                group.frequency_hz, group.geometric_mean_psd_v2_per_hz,
                color=colors[mode], label=labels[mode], linewidth=1.7,
            )
            axis.fill_between(
                group.frequency_hz, group.ci_2p5_psd_v2_per_hz,
                group.ci_97p5_psd_v2_per_hz, color=colors[mode], alpha=0.12,
                linewidth=0.0,
            )
        axis.axvspan(8.0, 12.0, color="0.85", alpha=0.25)
        axis.axvline(carrier, color="0.3", linestyle="--", linewidth=0.8)
        axis.set_yscale("log")
        axis.set_xlim(5.0, 15.0)
        axis.set_title(f"{diffusion.replace('_', ' ')}, carrier {carrier:g} Hz")
        axis.set_xlabel("Frequency (Hz)")
    axes[0, 0].set_ylabel("Ideal EEG PSD (V²/Hz)")
    axes[1, 0].set_ylabel("Ideal EEG PSD (V²/Hz)")
    axes[0, 0].legend(frameon=False, fontsize=8)
    figure.suptitle("H4: alpha-band EEG during the frozen controller comparison")
    _save_figure(figure, root, "figure_01_H4_stimulation_PSD_alpha_zoom")

    structure_arm = expected.groupby([
        "structure_seed", "label", "controller_mode"
    ], as_index=False).expected_post_distance_to_B_log10.mean()
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharey=True)
    for axis, diffusion in zip(axes, (LOW, HIGH)):
        subset = structure_arm[structure_arm.label.eq(diffusion)]
        for position, mode in enumerate(EXPECTED_MODES):
            values = subset[subset.controller_mode.eq(mode)].expected_post_distance_to_B_log10
            mean, low, high = _mean_ci(values.to_numpy(float))
            axis.bar(position, mean, color=colors[mode], width=0.65)
            axis.errorbar(position, mean, yerr=[[mean-low], [high-mean]], color="black", capsize=3)
        axis.set_xticks(range(3), [labels[x] for x in EXPECTED_MODES], rotation=18)
        axis.set_title(diffusion.replace("_", " "))
        axis.set_ylabel("Eight-second distance to B (log10)")
    figure.suptitle("Frozen H4 controller endpoint (structure-level 95% t intervals)")
    _save_figure(figure, root, "figure_02_H4_controller_endpoint")

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=False)
    contrasts = [
        ("mean_primary_refresh_advantage_over_one_time_log10", "Refreshed − one-time"),
        ("mean_secondary_refresh_advantage_over_sham_log10", "Refreshed − sham"),
    ]
    for axis, (column, title) in zip(axes, contrasts):
        values = structure[column].to_numpy(float)
        axis.scatter(np.zeros_like(values), values, color="#1f77b4", s=35, zorder=3)
        axis.boxplot(values, positions=[0], widths=0.28, showfliers=False)
        axis.axhline(0.0, color="0.25", linewidth=0.8)
        axis.axhline(0.01, color="0.4", linestyle="--", linewidth=0.8)
        axis.set_xticks([])
        axis.set_title(title)
        axis.set_ylabel("Target-distance advantage (log10)")
    figure.suptitle("H4 independent-structure causal effects")
    _save_figure(figure, root, "figure_03_H4_structure_level_effects")

    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    for mode in (ONE_TIME, SELECTED):
        group = phase_time[phase_time.controller_mode.eq(mode)]
        summary = group.groupby("time_since_onset_ms").mean_abs_common_phase_error_rad.agg(
            ["mean", "sem"]
        ).reset_index()
        axis.plot(summary.time_since_onset_ms / 1000.0, summary["mean"],
                  color=colors[mode], label=labels[mode], linewidth=1.8)
        axis.fill_between(
            summary.time_since_onset_ms / 1000.0,
            summary["mean"] - 1.96 * summary["sem"],
            summary["mean"] + 1.96 * summary["sem"],
            color=colors[mode], alpha=0.15, linewidth=0.0,
        )
    axis.set(
        xlabel="Time since tACS onset (s)", ylabel="Common-auditor |phase error| (rad)",
        title="Causal phase maintenance under phase diffusion",
    )
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_04_H4_phase_tracking")

    figure, axis = plt.subplots(figsize=(9.0, 4.4))
    for mode in EXPECTED_MODES:
        group = temporal[temporal.controller_mode.eq(mode)]
        summary = group.groupby("analysis_window_index").mean_distance_to_B_log10.agg(
            ["mean", "sem"]
        ).reset_index()
        axis.plot(summary.analysis_window_index, summary["mean"], marker="o",
                  color=colors[mode], label=labels[mode])
        axis.fill_between(
            summary.analysis_window_index,
            summary["mean"] - 1.96 * summary["sem"],
            summary["mean"] + 1.96 * summary["sem"],
            color=colors[mode], alpha=0.12, linewidth=0.0,
        )
    axis.set(
        xlabel="One-second endpoint window", ylabel="Distance to B (log10)",
        title="Evolution of the H4 EEG endpoint",
    )
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_05_H4_temporal_endpoint")

    state_order = [(LOW, 9.0), (LOW, 11.0), (HIGH, 9.0), (HIGH, 11.0)]
    heat = context.pivot_table(
        index="structure_seed", columns=["label", "hidden_frequency_hz"],
        values="primary_refresh_advantage_over_one_time_log10",
    ).reindex(columns=pd.MultiIndex.from_tuples(state_order))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 5.0))
    limit = float(np.nanmax(np.abs(heat.to_numpy())))
    image = axes[0].imshow(heat.to_numpy(), aspect="auto", cmap="RdBu_r",
                           vmin=-limit, vmax=limit)
    axes[0].set_xticks(range(4), ["low/9", "low/11", "high/9", "high/11"])
    axes[0].set_yticks(range(len(heat)), heat.index.astype(str), fontsize=7)
    axes[0].set(xlabel="Diffusion / carrier", ylabel="Structure seed",
                title="Primary advantage by crossed state")
    figure.colorbar(image, ax=axes[0], label="Refreshed advantage (log10)")
    active = metrics[metrics.controller_mode.ne(SHAM)]
    for mode in (ONE_TIME, SELECTED):
        group = active[active.controller_mode.eq(mode)]
        axes[1].scatter(group.post_E_firing_rate_hz, group.post_I_firing_rate_hz,
                        color=colors[mode], label=labels[mode], alpha=0.45, s=14)
    axes[1].set(xlabel="E firing rate (Hz)", ylabel="I firing rate (Hz)",
                title="Hidden firing-rate safety audit")
    axes[1].legend(frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_06_H4_crossed_effects_and_safety")


def _checks(
    *, cfg: DictConfig, sources: dict[str, Any], power: dict[str, Any],
    screening: pd.DataFrame, enrollment: pd.DataFrame, enrolled: list[int],
    metrics: pd.DataFrame, expected: pd.DataFrame, updates: pd.DataFrame,
    context: pd.DataFrame, structure: pd.DataFrame, diffusion: pd.DataFrame,
    inferences: dict[str, dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    primary = inferences["primary_refresh_vs_one_time"]
    secondary = inferences["secondary_refresh_vs_sham"]
    mechanism = inferences["phase_error_reduction_vs_one_time"]
    primary_pass = bool(
        float(primary["mean"]) >= float(criteria.practical_primary_advantage_log10)
        and float(primary["exact_sign_flip_one_sided_p_value"])
        <= float(criteria.maximum_primary_p_value)
        and float(primary["positive_structure_fraction"])
        >= float(criteria.minimum_positive_structure_fraction)
    )
    secondary_pass = bool(
        primary_pass
        and float(secondary["mean"])
        >= float(criteria.practical_secondary_advantage_vs_sham_log10)
        and float(secondary["exact_sign_flip_one_sided_p_value"])
        <= float(criteria.maximum_primary_p_value)
        and float(secondary["positive_structure_fraction"])
        >= float(criteria.minimum_positive_structure_fraction)
    )
    mechanism_pass = bool(
        float(mechanism["mean"]) > 0.0
        and float(mechanism["exact_sign_flip_one_sided_p_value"])
        <= float(criteria.maximum_primary_p_value)
        and float(mechanism["positive_structure_fraction"])
        >= float(criteria.minimum_positive_structure_fraction)
    )
    selected_metrics = metrics[metrics.controller_mode.eq(SELECTED)]
    active = metrics[metrics.controller_mode.ne(SHAM)]
    selected_updates = updates[updates.controller_mode.eq(SELECTED)]
    refreshed_updates = selected_updates[selected_updates.phase_refresh_applied.astype(bool)]
    rate_safe = bool(metrics.rate_safe.all())
    enrolled_screening = screening[screening.structure_seed.isin(enrolled)]
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    reference_rate_matched = all(
        abs(float(getattr(row, f"baseline_{population}_firing_rate_hz"))
            - float(sources["target"][f"reference_{population}_firing_rate_hz"]))
        <= tolerance * max(
            float(sources["target"][f"reference_{population}_firing_rate_hz"]),
            np.finfo(float).tiny,
        )
        for row in enrolled_screening.itertuples()
        for population in ("E", "I")
    )
    field_removed = bool(
        metrics.field_removal_recovered.all()
        and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
    )
    check = {
        "H4BW2_passed_hash_locked_and_selected_controller_frozen": True,
        "confirmation_seed_namespaces_disjoint_from_all_sources": True,
        "frozen_B_target_loaded_without_recalibration": bool(
            np.isclose(float(sources["target"]["outcome_duration_s"]), 8.0)
        ),
        "a_priori_structure_sample_size_powered": bool(
            power["planned_independent_structures"] >= power["required_independent_structures"]
            and power["a_priori_t_approximation_power"] >= power["target_power"]
        ),
        "screening_precedes_and_excludes_active_outcomes": bool(
            screening.screen_uses_only_predecision_ideal_EEG.all()
            and (~screening.screen_uses_action_outcome.astype(bool)).all()
        ),
        "afferent_mean_rate_matched_across_states_by_construction": True,
        "minimum_candidate_structures_screened": int(
            enrollment.structure_seed.nunique()
        ) >= int(criteria.minimum_candidate_structures) or bool(cfg.analysis.smoke_test),
        "complete_candidate_frequency_diffusion_screening_grid": bool(
            enrollment.complete_frequency_diffusion_grid.all()
        ),
        "minimum_prospectively_enrolled_structures": len(enrolled)
        >= int(criteria.minimum_enrolled_structures) or bool(cfg.analysis.smoke_test),
        "all_enrolled_structures_pass_frozen_EEG_screen": bool(
            enrollment.loc[enrollment.structure_seed.isin(enrolled), "all_contexts_eligible"].all()
        ),
        "complete_frequency_diffusion_grid_per_enrolled_structure": bool(
            len(structure)
            and (structure.context_count == 4).all()
            and (structure.frequency_count == 2).all()
            and (structure.diffusion_count == 2).all()
        ),
        "frequency_identified_from_predecision_EEG": bool(
            screening[screening.structure_seed.isin(enrolled)]
            .EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "six_independent_postdecision_futures": bool(
            len(expected)
            and expected.n_futures.min() >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_controllers_and_futures": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "all_active_arms_use_frozen_0p2_V_per_m": bool(
            len(active) and np.allclose(active.amplitude_v_per_m, 0.2)
        ),
        "all_active_arms_share_one_second_initialization": _common_initialization(updates),
        "selected_controller_refreshes_after_onset": bool(
            len(refreshed_updates) and refreshed_updates.update_index.min() > 0
        ),
        "phase_updates_use_only_preceding_EEG": bool(
            updates.estimate_is_strictly_causal.all()
            and (updates.estimate_stop_ms - updates.boundary_ms).max()
            <= float(criteria.maximum_causal_timing_error_ms)
        ),
        "phase_correction_is_frequency_bounded": bool(
            len(refreshed_updates)
            and refreshed_updates.frequency_correction_hz.abs().max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_updates": bool(
            active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "common_phase_estimates_actionable": bool(
            selected_metrics.common_phase_estimate_actionable_fraction.mean()
            >= float(criteria.minimum_common_phase_estimate_actionable_fraction)
        ),
        "primary_advantage_is_practically_meaningful": bool(
            float(primary["mean"]) >= float(criteria.practical_primary_advantage_log10)
        ),
        "primary_exact_structure_test_rejects_null": bool(
            float(primary["exact_sign_flip_one_sided_p_value"])
            <= float(criteria.maximum_primary_p_value)
        ),
        "primary_advantage_consistent_across_structures": bool(
            float(primary["positive_structure_fraction"])
            >= float(criteria.minimum_positive_structure_fraction)
        ),
        "primary_advantage_nonadverse_in_both_diffusion_levels": bool(
            len(diffusion) == 2 and (diffusion.mean_primary_advantage_log10 > 0.0).all()
        ),
        "paired_future_reliability_confirmed": bool(
            structure.mean_refresh_win_fraction_over_one_time.mean()
            >= float(criteria.minimum_realized_future_win_fraction)
        ),
        "fixed_sequence_refresh_beats_sham": secondary_pass,
        "causal_phase_maintenance_mechanism_confirmed": mechanism_pass,
        "all_actions_rate_safe": rate_safe,
        "reference_firing_rates_matched": bool(reference_rate_matched),
        "field_removal_recovered": field_removed,
        "policy_and_controller_exclude_hidden_state_and_spikes": bool(
            (~metrics.policy_uses_hidden_state_or_spikes.astype(bool)).all()
        ),
    }
    required = list(check)
    confirmed = bool(all(check[name] for name in required) and not bool(cfg.analysis.smoke_test))
    return check, {
        "H4_adaptive_phase_maintenance_confirmed": confirmed,
        "primary_fixed_sequence_passed": primary_pass,
        "secondary_vs_sham_tested_only_after_primary": primary_pass,
        "secondary_vs_sham_passed": secondary_pass,
        "phase_mechanism_passed": mechanism_pass,
        "selected_controller": SELECTED,
        "contextual_bandit_status": "NOT TRAINED OR TESTED",
        "claim_scope": (
            "frozen deterministic phase-refreshed feedback improves an ideal-neural-EEG "
            "target under toy shared phase diffusion"
        ),
    }


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    sources = _load_frozen_h4bw2(cfg)
    power = _power_design(cfg)
    _validate_design(cfg, sources, power)
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    if rank == 0:
        exists = bool(root.exists() and any(root.iterdir()))
    else:
        exists = None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H4-C frozen adaptive phase-maintenance confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
        print("\n### A priori power design")
        print(json.dumps(_json_ready(power), indent=2))
        (root / "a_priori_power_design.json").write_text(
            json.dumps(_json_ready(power), indent=2, allow_nan=False)
        )
    comm.Barrier()
    started = time.perf_counter()
    target = sources["target"]

    # Stage 1: prospectively screen every candidate before any active outcome.
    screening_rows: list[dict[str, Any]] = []
    screening_episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
    contexts = _run_context_specs(cfg)
    for context in contexts:
        if rank == 0:
            print(
                f"screen context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s"
            )
        state_cfg = _with_diffusion_state(cfg, {
            "frequency_hz": float(context["hidden_frequency_hz"]),
            "phase_seed": int(context["phase_seed"]),
            "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        })
        episode = _run_controller(
            condition_cfg=state_cfg,
            context=context,
            future_seed=_future_seed(cfg, context, 0),
            future_index=0,
            mode=SHAM,
            action_index=0,
            root=root / "screening",
            comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            screen = _context_features(episode, context, target, cfg)
            screening_rows.append(screen)
            screening_episodes[str(context["context_id"])] = episode
            print(
                f"screen: {'ELIGIBLE' if screen['eligible'] else 'EXCLUDED'}; "
                f"selected={screen['EEG_selected_frequency_hz']:g} Hz; "
                f"reason={screen['exclusion_reasons']}"
            )

    if rank == 0:
        screening = pd.DataFrame(screening_rows)
        enrollment = _screen_structure_table(screening)
        target_n = int(cfg.analysis.crossed_design.n_enrolled_structure_seeds)
        positive = enrollment[enrollment.all_contexts_eligible]
        enrolled = positive.structure_seed.astype(int).head(target_n).tolist()
        enrollment["enrolled"] = enrollment.structure_seed.isin(enrolled)
        screening["structure_enrolled"] = screening.structure_seed.isin(enrolled)
        screening.to_csv(root / "prospective_context_screening.csv", index=False)
        enrollment.to_csv(root / "prospective_structure_enrollment.csv", index=False)
    else:
        screening = enrollment = enrolled = None
    enrolled = comm.bcast(enrolled, root=0)
    target_enrollment = int(cfg.analysis.crossed_design.n_enrolled_structure_seeds)
    insufficient = len(enrolled) < target_enrollment and not bool(cfg.analysis.smoke_test)
    if not enrolled or insufficient:
        if rank == 0:
            conclusion = {
                "scope": "H4-C ideal-neural-EEG confirmation",
                "checks": {"minimum_prospectively_enrolled_structures": False},
                "conclusions": {"H4_adaptive_phase_maintenance_confirmed": False},
                "screening": {
                    "target_enrolled_structures": target_enrollment,
                    "actual_enrolled_structures": len(enrolled),
                    "stopped_before_active_outcomes": True,
                },
                "runtime_seconds": float(time.perf_counter() - started),
            }
            (root / "experiment_conclusion.json").write_text(
                json.dumps(conclusion, indent=2)
            )
        return

    # Stage 2: run only prospectively enrolled structures with the frozen arms.
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    screen_lookup = (
        {str(row.context_id): row._asdict() for row in screening.itertuples(index=False)}
        if rank == 0 else None
    )
    for context in contexts:
        if int(context["structure_seed"]) not in set(enrolled):
            continue
        context_id = str(context["context_id"])
        if rank == 0:
            screen = screen_lookup[context_id]
            baseline_reference = screening_episodes[context_id]
            selected_frequency = float(screen["EEG_selected_frequency_hz"])
            print(f"confirm context={context_id}; EEG carrier={selected_frequency:g} Hz")
        else:
            screen = baseline_reference = selected_frequency = None
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        state_cfg = _with_diffusion_state(cfg, {
            "frequency_hz": float(context["hidden_frequency_hz"]),
            "phase_seed": int(context["phase_seed"]),
            "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        })
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for action_index, mode in enumerate(_controller_modes(cfg)):
                if future_index == 0 and mode == SHAM:
                    episode = baseline_reference
                else:
                    episode = _run_controller(
                        condition_cfg=action_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        mode=mode,
                        action_index=action_index,
                        root=root / "confirmation",
                        comm=comm, size=size, rank=rank,
                    )
                if rank == 0:
                    episodes[mode] = episode
                    psd_rows.extend(_psd_rows(
                        episode, context=context, future_index=future_index,
                        controller_mode=mode, cfg=cfg,
                    ))
            if rank == 0:
                rows, trajectories, updates = _metric_rows(
                    context=context, screening=screen,
                    future_index=future_index, future_seed=future_seed,
                    episodes=episodes, baseline_reference=baseline_reference,
                    target=target, cfg=cfg,
                )
                _augment_metric_rows(rows, episodes, cfg)
                _augment_common_audit(rows, episodes, cfg)
                metric_rows.extend(rows)
                trajectory_rows.extend(trajectories)
                update_rows.extend(updates)

    if rank != 0:
        return
    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    expected = _expected_map(metrics)
    context_effects, structure_effects, diffusion_effects = _effect_tables(
        expected, metrics
    )
    temporal = _temporal_table(trajectories)
    phase_time = _phase_time_table(updates, cfg)
    psd_long = pd.DataFrame(psd_rows)
    psd_summary = _summarize_psd(psd_long, cfg)

    inferences = {
        "primary_refresh_vs_one_time": _paired_inference(
            structure_effects.mean_primary_refresh_advantage_over_one_time_log10.to_numpy(float),
            metric="frozen refreshed-controller target-distance advantage over one-time phase initialization",
            cfg=cfg, bootstrap_offset=401,
        ),
        "secondary_refresh_vs_sham": _paired_inference(
            structure_effects.mean_secondary_refresh_advantage_over_sham_log10.to_numpy(float),
            metric="frozen refreshed-controller target-distance advantage over sham",
            cfg=cfg, bootstrap_offset=402,
        ),
        "phase_error_reduction_vs_one_time": _paired_inference(
            structure_effects.mean_phase_error_reduction_vs_one_time_rad.to_numpy(float),
            metric="common-auditor absolute phase-error reduction versus one-time initialization",
            cfg=cfg, bootstrap_offset=403,
        ),
        "one_time_vs_sham_audit": _paired_inference(
            structure_effects.mean_one_time_advantage_over_sham_log10.to_numpy(float),
            metric="one-time phase initialization target-distance advantage over sham (audit)",
            cfg=cfg, bootstrap_offset=404,
        ),
    }
    inference_table = _inference_table(inferences)
    checks, conclusions = _checks(
        cfg=cfg, sources=sources, power=power, screening=screening,
        enrollment=enrollment, enrolled=enrolled, metrics=metrics,
        expected=expected, updates=updates, context=context_effects,
        structure=structure_effects, diffusion=diffusion_effects,
        inferences=inferences,
    )

    metrics.to_csv(root / "context_controller_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_controller_map.csv", index=False)
    context_effects.to_csv(root / "H4_context_level_effects.csv", index=False)
    structure_effects.to_csv(root / "H4_structure_level_effects.csv", index=False)
    diffusion_effects.to_csv(root / "H4_diffusion_level_effects.csv", index=False)
    trajectories.to_csv(root / "H4_one_second_EEG_trajectories.csv", index=False)
    temporal.to_csv(root / "H4_structure_temporal_endpoint.csv", index=False)
    updates.to_csv(root / "H4_causal_phase_updates.csv", index=False)
    phase_time.to_csv(root / "H4_structure_phase_tracking.csv", index=False)
    psd_long.to_csv(root / "H4_stimulation_PSD_long.csv", index=False)
    psd_summary.to_csv(root / "H4_stimulation_PSD_summary.csv", index=False)
    inference_table.to_csv(root / "H4_manuscript_statistical_table.csv", index=False)
    (root / "H4_statistical_inference.json").write_text(json.dumps(
        _json_ready({
            **inferences,
            "fixed_sequence": (
                "primary refreshed versus one-time; only after passing, "
                "secondary refreshed versus sham"
            ),
            "multiplicity": "one prespecified primary contrast; phase and one-time/sham are audits",
        }), indent=2, allow_nan=False,
    ))
    provenance = {
        "experiment": "H4-C disjoint adaptive phase-maintenance confirmation",
        "frozen_H4BW2": {
            "root": sources["roots"]["h4bw2"],
            "sha256": sources["hashes"]["h4bw2"],
            "candidate": sources["candidate"],
            "target": sources["target"],
        },
        "state_generator": {
            "equation": "dphi = 2*pi*f*dt + sqrt(2D)*dW",
            "frequency_hz": [9.0, 11.0],
            "diffusion_rad2_per_s": [0.5, 2.0],
            "modulation_depth": 0.04,
            "mean_afferent_rate_matched": True,
        },
        "controller": {
            "observation": "preceding 0.5 s of ideal neural EEG after one-second initialization",
            "refresh_interval_ms": 125.0,
            "correction_horizon_ms": 250.0,
            "amplitude_v_per_m": 0.2,
            "relative_phase_target_rad": np.pi,
            "carrier": "selected from 12-s predecision ideal EEG",
            "waveform_phase_continuous": True,
        },
        "arms": EXPECTED_MODES,
        "fixed_sequence": ["selected versus one-time", "selected versus sham"],
        "statistical_unit": "independent circuit structure",
        "power_design": power,
        "not_a_bandit_or_learned_predictor": True,
        "not_a_disease_or_clinical_treatment_model": True,
        "concurrent_EEG_is_ideal_and_artifact_free": True,
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False,
    ))
    conclusion = {
        "scope": "H4-C disjoint ideal-neural-EEG adaptive phase-maintenance confirmation",
        "checks": checks,
        "conclusions": conclusions,
        "primary_structure_level_inference": inferences["primary_refresh_vs_one_time"],
        "secondary_structure_level_inference": inferences["secondary_refresh_vs_sham"],
        "phase_mechanism_inference": inferences["phase_error_reduction_vs_one_time"],
        "screening": {
            "candidate_structures": int(enrollment.structure_seed.nunique()),
            "screen_positive_structures": int(enrollment.all_contexts_eligible.sum()),
            "enrolled_structures": len(enrolled),
            "context_screening_yield": float(screening.eligible.mean()),
        },
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; frequency/diffusion/future are repeats",
        "inference_boundary": (
            "toy-model ideal-neural-EEG adaptive-feedback confirmation; no general necessity, "
            "clinical, artifact-robust, disease, or machine-learning claim"
        ),
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False,
    ))

    if bool(cfg.experiment.plot):
        _plot_results(
            root=root, psd=psd_summary, expected=expected,
            structure=structure_effects, context=context_effects,
            temporal=temporal, phase_time=phase_time, metrics=metrics,
        )
    manifest = {
        "tables": sorted(path.name for path in root.glob("*.csv")),
        "statistics": sorted(path.name for path in root.glob("*.json")),
        "figures_png": sorted(path.name for path in root.glob("figure_*.png")),
        "figures_pdf": sorted(path.name for path in root.glob("figure_*.pdf")),
    }
    (root / "manuscript_artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print("\n### H4-C prospective screening")
    print(f"candidate structures: {int(enrollment.structure_seed.nunique())}")
    print(f"screen-positive structures: {int(enrollment.all_contexts_eligible.sum())}")
    print(f"enrolled structures: {len(enrolled)}")
    print("\n### H4-C confirmation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Primary structure-level inference")
    print(json.dumps(_json_ready(inferences["primary_refresh_vs_one_time"]), indent=2))
    print(
        "\nH4 adaptive phase maintenance: "
        f"{'CONFIRMED' if conclusions['H4_adaptive_phase_maintenance_confirmed'] else 'NOT CONFIRMED'}"
    )
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
