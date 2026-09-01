"""Confirm phase-invariant EEG reachability with a frozen tACS protocol.

This experiment follows the negative hierarchical phase result without
silently redefining that experiment.  The previously EEG-discovered 60-Hz
frequency is fixed before this run and absolute phase is not optimized.
Unstimulated A/B EEG on calibration seeds fits a one-dimensional,
phase-invariant log-band-power target.  A, B, axial tACS, and a transverse
field control are then evaluated on disjoint validation seeds.

The primary claim is limited to ideal neural-only simulated EEG.  A complex
observation-only sinusoid and a selected-frequency-excluded feature space are
retained as measurement audits, but are not allowed to redefine the primary
endpoint after results are observed.  Spikes and rates are hidden mechanism
and safety outcomes; they never enter target fitting or action selection.
"""

from __future__ import annotations

import json
import shutil
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

from experiments.ballnstick_analysis.run_ballnstick_entrainment_state import (  # noqa: E402
    _condition_config,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    EPOCHS,
    LOWER_ACTION,
    PRIMARY_ACTION,
    REFERENCE_CONDITIONS,
    SYNTHETIC_CONTROL,
    TRANSVERSE_CONTROL,
    _action,
    _active_action,
    _add_distances,
    _classification_accuracy,
    _episode_epoch_feature_row,
    _episode_window_feature_rows,
    _fit_centroid_model,
    _frequency_feature_names,
    _plain,
    _plot_psd,
    _plot_spike_timing,
    _representative_spike_rows,
    _simulate,
    _summary,
    _synthetic_feature_row,
    _validation_reachability,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


PHASE_INVARIANT_PREFIX = "phase_invariant"
EXCLUDED_PREFIX = "excluded"


def _selected_power_feature(frequency_hz: float) -> str:
    """Return the generic band-power feature name for a frozen frequency."""
    return _frequency_feature_names([float(frequency_hz)])[0]


def _audit_target_frequency(
    rows: pd.DataFrame,
    *,
    candidate_frequencies_hz: list[float],
    spectral_model: dict[str, Any],
) -> tuple[float, pd.DataFrame]:
    """Audit the largest A/B spectral shift without altering the protocol.

    Unlike a discovery selector, a confirmation audit must still return a
    table when every observed shift is non-positive.  That outcome should fail
    a predefined check, not terminate simulation before held-out validation.
    """
    stimulation = rows[rows.epoch.eq("stimulation")]
    a = stimulation[stimulation.condition_id.eq("A_async")].set_index("seed")
    b = stimulation[
        stimulation.condition_id.eq("B_rhythmic_reference")
    ].set_index("seed")
    common = a.index.intersection(b.index)
    scale = np.asarray(spectral_model["scale"], dtype=float)
    table_rows: list[dict[str, Any]] = []
    for index, (frequency, feature) in enumerate(
        zip(candidate_frequencies_hz, spectral_model["feature_names"])
    ):
        paired = (b.loc[common, feature] - a.loc[common, feature]).to_numpy(float)
        standardized = paired / float(scale[index])
        table_rows.append(
            {
                "frequency_hz": float(frequency),
                "feature_name": feature,
                "mean_log10_power_shift": float(paired.mean()),
                "mean_standardized_shift": float(standardized.mean()),
                "positive_seed_fraction": float(np.mean(paired > 0.0)),
            }
        )
    table = pd.DataFrame(table_rows).sort_values("frequency_hz")
    selected = float(
        table.sort_values(
            ["mean_standardized_shift", "frequency_hz"],
            ascending=[False, True],
        ).iloc[0].frequency_hz
    )
    table["largest_observed_shift"] = np.isclose(table.frequency_hz, selected)
    return selected, table


def _validation_conditions(cfg: DictConfig) -> tuple[str, ...]:
    conditions = [
        "A_async",
        "B_rhythmic_reference",
        PRIMARY_ACTION,
        TRANSVERSE_CONTROL,
    ]
    if bool(cfg.analysis.validation.include_lower_dose):
        conditions.insert(2, LOWER_ACTION)
    return tuple(conditions)


def _active_conditions(cfg: DictConfig) -> tuple[str, ...]:
    return tuple(
        condition
        for condition in _validation_conditions(cfg)
        if condition not in REFERENCE_CONDITIONS
    )


def _seeds(cfg: DictConfig, stage: str) -> list[int]:
    block = cfg.analysis[stage]
    first = int(cfg.experiment.seed) + int(block.seed_offset)
    return [first + index for index in range(int(block.n_seeds))]


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Phase-invariant tACS confirmation requires online mode.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("A, B, and every tACS arm require inhibition_scale=1.0.")

    candidates = [
        float(value) for value in cfg.analysis.candidate_frequencies_hz
    ]
    if len(candidates) < 2 or len(candidates) != len(set(candidates)):
        raise ValueError("Candidate EEG frequencies must be unique alternatives.")
    if any(value <= 0.0 for value in candidates):
        raise ValueError("Candidate EEG frequencies must be positive.")

    protocol = cfg.analysis.frozen_protocol
    frequency = float(protocol.frequency_hz)
    if not any(np.isclose(frequency, value) for value in candidates):
        raise ValueError("The frozen frequency must be one of the EEG candidates.")
    if not np.isclose(float(protocol.phase_rad), 0.0):
        raise ValueError(
            "Absolute phase is fixed to zero by convention, not optimized."
        )
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    primary = float(protocol.primary_amplitude_v_per_m)
    lower = float(protocol.lower_amplitude_v_per_m)
    if not (0.0 < lower < primary <= maximum):
        raise ValueError("Require 0 < lower < primary <= maximum field.")

    calibration = set(_seeds(cfg, "calibration"))
    validation = set(_seeds(cfg, "validation"))
    if len(calibration) < 2:
        raise ValueError("At least two matched calibration seeds are required.")
    if not validation or calibration.intersection(validation):
        raise ValueError("Calibration and validation seeds must be nonempty/disjoint.")
    # The inherited simulator maps a circuit seed to seed * 10,000 before
    # calling NumPy's legacy 32-bit seeding API.
    maximum_seed = max(calibration.union(validation))
    if maximum_seed * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError(
            "Circuit seeds are too large for the simulator's seed * 10,000 "
            "mapping; reduce experiment.seed or the stage offsets."
        )


def _actions(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    protocol = cfg.analysis.frozen_protocol
    frequency = float(protocol.frequency_hz)
    phase = float(protocol.phase_rad)
    axial = str(cfg.analysis.tacs.axial_montage)
    transverse = str(cfg.analysis.tacs.transverse_montage)
    actions = {
        "A_async": _action(
            identifier="A_async",
            role="sham_policy_action",
            amplitude_v_per_m=0.0,
            frequency_hz=frequency,
            phase_rad=phase,
            montage=axial,
        ),
        "B_rhythmic_reference": _action(
            identifier="B_rhythmic_reference",
            role="unstimulated_target",
            amplitude_v_per_m=0.0,
            frequency_hz=frequency,
            phase_rad=phase,
            montage=axial,
        ),
        PRIMARY_ACTION: _active_action(
            cfg,
            identifier=PRIMARY_ACTION,
            role="frozen_primary_action",
            amplitude_v_per_m=float(protocol.primary_amplitude_v_per_m),
            frequency_hz=frequency,
            phase_rad=phase,
        ),
        TRANSVERSE_CONTROL: _active_action(
            cfg,
            identifier=TRANSVERSE_CONTROL,
            role="orientation_control",
            amplitude_v_per_m=float(protocol.primary_amplitude_v_per_m),
            frequency_hz=frequency,
            phase_rad=phase,
            montage=transverse,
        ),
    }
    if bool(cfg.analysis.validation.include_lower_dose):
        actions[LOWER_ACTION] = _active_action(
            cfg,
            identifier=LOWER_ACTION,
            role="secondary_lower_dose",
            amplitude_v_per_m=float(protocol.lower_amplitude_v_per_m),
            frequency_hz=frequency,
            phase_rad=phase,
        )
    return actions


def _hidden_validation_rows(
    epoch_rows: pd.DataFrame,
    episodes: dict[int, dict[str, dict[str, Any]]],
    *,
    cfg: DictConfig,
) -> pd.DataFrame:
    indexed = {
        condition: epoch_rows[epoch_rows.condition_id.eq(condition)].set_index(
            ["seed", "epoch"]
        )
        for condition in _validation_conditions(cfg)
    }
    a = indexed["A_async"]
    residual_fraction = float(
        cfg.analysis.criteria.maximum_washout_residual_fraction
    )
    rows: list[dict[str, Any]] = []
    for seed, seed_episodes in episodes.items():
        for condition in _active_conditions(cfg):
            active = indexed[condition]
            ppc_gain = float(
                (active.loc[(seed, "stimulation"), "E_ppc"]
                 - active.loc[(seed, "baseline"), "E_ppc"])
                - (a.loc[(seed, "stimulation"), "E_ppc"]
                   - a.loc[(seed, "baseline"), "E_ppc"])
            )
            washout_gain = float(
                (active.loc[(seed, "washout"), "E_ppc"]
                 - active.loc[(seed, "baseline"), "E_ppc"])
                - (a.loc[(seed, "washout"), "E_ppc"]
                   - a.loc[(seed, "baseline"), "E_ppc"])
            )
            active_stim = active.loc[(seed, "stimulation")]
            a_stim = a.loc[(seed, "stimulation")]
            rows.append(
                {
                    "seed": int(seed),
                    "condition_id": condition,
                    "E_ppc_gain_difference_in_differences": ppc_gain,
                    "E_rate_change_vs_A_hz": float(
                        active_stim.E_firing_rate_hz
                        - a_stim.E_firing_rate_hz
                    ),
                    "I_rate_change_vs_A_hz": float(
                        active_stim.I_firing_rate_hz
                        - a_stim.I_firing_rate_hz
                    ),
                    "rate_safe": bool(
                        _relative_rate_safe(active_stim, a_stim, cfg)
                    ),
                    "washout_recovered": bool(
                        ppc_gain > 0.0
                        and abs(washout_gain)
                        <= residual_fraction
                        * max(abs(ppc_gain), np.finfo(float).eps)
                    ),
                    "baseline_relative_rms_error_vs_A": float(
                        _relative_rms_error(
                            np.asarray(
                                seed_episodes["A_async"]["raw_by_epoch"][
                                    "baseline"
                                ]
                            ),
                            np.asarray(
                                seed_episodes[condition]["raw_by_epoch"][
                                    "baseline"
                                ]
                            ),
                        )
                    ),
                }
            )
    return pd.DataFrame(rows)


def _paired_feature_effects(
    epoch_rows: pd.DataFrame,
    *,
    feature_names: list[str],
    cfg: DictConfig,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report direct shifts and paired closeness without changing the primary."""
    stimulation = epoch_rows[epoch_rows.epoch.eq("stimulation")]
    a = stimulation[stimulation.condition_id.eq("A_async")].set_index("seed")
    b = stimulation[
        stimulation.condition_id.eq("B_rhythmic_reference")
    ].set_index("seed")
    seed_rows: list[dict[str, Any]] = []
    for condition in (
        "B_rhythmic_reference",
        LOWER_ACTION,
        PRIMARY_ACTION,
        TRANSVERSE_CONTROL,
        SYNTHETIC_CONTROL,
    ):
        candidate = stimulation[stimulation.condition_id.eq(condition)].set_index(
            "seed"
        )
        for seed in a.index.intersection(candidate.index).intersection(b.index):
            for feature in feature_names:
                a_value = float(a.loc[seed, feature])
                b_value = float(b.loc[seed, feature])
                candidate_value = float(candidate.loc[seed, feature])
                seed_rows.append(
                    {
                        "seed": int(seed),
                        "condition_id": condition,
                        "feature": feature,
                        "A_value": a_value,
                        "B_value": b_value,
                        "candidate_value": candidate_value,
                        "candidate_minus_A": candidate_value - a_value,
                        "paired_B_closeness_improvement": (
                            abs(a_value - b_value)
                            - abs(candidate_value - b_value)
                        ),
                    }
                )
    seed_frame = pd.DataFrame(seed_rows)
    summaries: list[dict[str, Any]] = []
    for (condition, feature), group in seed_frame.groupby(
        ["condition_id", "feature"], sort=False
    ):
        shift = _summary(
            group.candidate_minus_A.to_numpy(float), cfg=cfg, rng=rng
        )
        closeness = _summary(
            group.paired_B_closeness_improvement.to_numpy(float),
            cfg=cfg,
            rng=rng,
        )
        summaries.append(
            {
                "condition_id": condition,
                "feature": feature,
                **{f"shift_{key}": value for key, value in shift.items()},
                **{
                    f"closeness_{key}": value
                    for key, value in closeness.items()
                },
            }
        )
    return seed_frame, pd.DataFrame(summaries)


def _plot_phase_invariant_validation(
    epoch_rows: pd.DataFrame,
    reachability: pd.DataFrame,
    hidden: pd.DataFrame,
    *,
    feature_name: str,
    root: Path,
) -> None:
    stimulation = epoch_rows[epoch_rows.epoch.eq("stimulation")]
    conditions = [
        "A_async",
        "B_rhythmic_reference",
        PRIMARY_ACTION,
        TRANSVERSE_CONTROL,
    ]
    labels = ["A", "B reference", "A + axial tACS", "A + transverse"]
    colors = ["#2878B5", "#2A9D37", "#E67E22", "#777777"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    axis = axes[0]
    for seed, group in stimulation[
        stimulation.condition_id.isin(conditions)
    ].groupby("seed"):
        indexed = group.set_index("condition_id")
        if all(condition in indexed.index for condition in conditions):
            axis.plot(
                np.arange(len(conditions)),
                [float(indexed.loc[c, feature_name]) for c in conditions],
                color="#BDBDBD",
                linewidth=0.8,
                alpha=0.7,
            )
    for index, (condition, color) in enumerate(zip(conditions, colors)):
        values = stimulation[
            stimulation.condition_id.eq(condition)
        ][feature_name].to_numpy(float)
        axis.scatter(
            np.full(values.size, index), values, color=color, s=24, zorder=3
        )
    axis.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
    axis.set_ylabel("log10 EEG band power")
    axis.set_title("Phase-invariant EEG target")

    axis = axes[1]
    ordered = ["B_rhythmic_reference", PRIMARY_ACTION, TRANSVERSE_CONTROL]
    for index, (condition, color, label) in enumerate(
        zip(ordered, [colors[1], colors[2], colors[3]], labels[1:])
    ):
        values = reachability[
            reachability.condition_id.eq(condition)
        ].target_distance_improvement.to_numpy(float)
        axis.scatter(
            np.full(values.size, index), values, color=color, alpha=0.8, s=25
        )
        axis.errorbar(
            index,
            values.mean(),
            yerr=(values.std(ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0,
            color="black",
            marker="o",
            capsize=3,
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(np.arange(3), ["B", "Axial", "Transverse"])
    axis.set_ylabel("distance-to-B improvement")
    axis.set_title("Held-out EEG movement")

    axis = axes[2]
    primary = hidden[hidden.condition_id.eq(PRIMARY_ACTION)].sort_values("seed")
    transverse = hidden[
        hidden.condition_id.eq(TRANSVERSE_CONTROL)
    ].sort_values("seed")
    for _, left in primary.iterrows():
        right = transverse[transverse.seed.eq(left.seed)].iloc[0]
        axis.plot(
            [0, 1],
            [
                float(left.E_ppc_gain_difference_in_differences),
                float(right.E_ppc_gain_difference_in_differences),
            ],
            color="#BDBDBD",
            linewidth=0.8,
        )
    axis.scatter(
        np.zeros(len(primary)),
        primary.E_ppc_gain_difference_in_differences,
        color=colors[2],
        s=25,
    )
    axis.scatter(
        np.ones(len(transverse)),
        transverse.E_ppc_gain_difference_in_differences,
        color=colors[3],
        s=25,
    )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks([0, 1], ["Axial", "Transverse"])
    axis.set_ylabel("E-population PPC gain")
    axis.set_title("Hidden spike entrainment")
    figure.tight_layout()
    figure.savefig(root / "figure_02_phase_invariant_validation.png", dpi=250)
    plt.close(figure)


def _plot_online_trajectory(
    window_rows: pd.DataFrame,
    *,
    root: Path,
) -> None:
    conditions = ["A_async", "B_rhythmic_reference", PRIMARY_ACTION]
    colors = ["#2878B5", "#2A9D37", "#E67E22"]
    labels = ["A", "B reference", "A + tACS"]
    figure, axis = plt.subplots(figsize=(9.0, 4.6))
    n_stimulation_windows = int(
        window_rows[window_rows.epoch.eq("stimulation")]
        .epoch_window_index.max()
    ) + 1
    order = [
        ("baseline", 0),
        *[("stimulation", index) for index in range(n_stimulation_windows)],
        ("washout", 0),
    ]
    for condition, color, label in zip(conditions, colors, labels):
        means = []
        sems = []
        for epoch, index in order:
            values = window_rows[
                window_rows.condition_id.eq(condition)
                & window_rows.epoch.eq(epoch)
                & window_rows.epoch_window_index.eq(index)
            ].phase_invariant_distance_to_B.to_numpy(float)
            means.append(float(values.mean()))
            sems.append(
                float(values.std(ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
        axis.errorbar(
            np.arange(len(order)), means, yerr=sems, marker="o",
            capsize=3, color=color, label=label
        )
    axis.axvspan(
        0.5,
        n_stimulation_windows + 0.5,
        color="#E67E22",
        alpha=0.07,
        label="tACS block",
    )
    tick_labels = [
        "Base",
        *[f"Stim {index + 1}" for index in range(n_stimulation_windows)],
        "Wash",
    ]
    axis.set_xticks(np.arange(len(order)), tick_labels)
    axis.set_ylabel("phase-invariant EEG distance to frozen B")
    axis.set_title("Online one-second state trajectory")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(root / "figure_04_online_state_trajectory.png", dpi=250)
    plt.close(figure)


def _plot_observation_audit(
    primary: pd.DataFrame,
    synthetic: pd.DataFrame,
    primary_excluded: pd.DataFrame,
    synthetic_excluded: pd.DataFrame,
    *,
    root: Path,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))
    for axis, real, control, title in (
        (axes[0], primary, synthetic, "Primary 60-Hz EEG endpoint"),
        (axes[1], primary_excluded, synthetic_excluded, "60-Hz bins removed"),
    ):
        real = real.sort_values("seed")
        control = control.sort_values("seed")
        for left, right in zip(control.itertuples(), real.itertuples()):
            axis.plot(
                [0, 1],
                [left.target_distance_improvement, right.target_distance_improvement],
                color="#BDBDBD",
                linewidth=0.8,
            )
        axis.scatter(
            np.zeros(len(control)), control.target_distance_improvement,
            color="#777777", s=25
        )
        axis.scatter(
            np.ones(len(real)), real.target_distance_improvement,
            color="#E67E22", s=25
        )
        axis.axhline(0.0, color="black", linewidth=1)
        axis.set_xticks([0, 1], ["Observation sine", "Real tACS"])
        axis.set_ylabel("distance-to-B improvement")
        axis.set_title(title)
    figure.tight_layout()
    figure.savefig(root / "figure_05_observation_audit.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_invariant_tacs"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### Phase-invariant BallAndStick tACS confirmation")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    rng = np.random.default_rng(int(cfg.experiment.seed) + 1_500_007)

    candidates = [
        float(value) for value in cfg.analysis.candidate_frequencies_hz
    ]
    selected_frequency_hz = float(cfg.analysis.frozen_protocol.frequency_hz)
    selected_phase_rad = float(cfg.analysis.frozen_protocol.phase_rad)
    selected_feature = _selected_power_feature(selected_frequency_hz)
    asynchronous_cfg = _condition_config(cfg, modulation_depth=0.0)
    reference_cfg = _condition_config(
        cfg, modulation_depth=float(cfg.analysis.reference.modulation_depth)
    )
    actions = _actions(cfg)

    # Calibration sees only unstimulated A/B and cannot alter the frozen action.
    calibration_rows: list[dict[str, Any]] = []
    for seed in _seeds(cfg, "calibration"):
        for action_index, condition in enumerate(REFERENCE_CONDITIONS):
            if rank == 0:
                print(f"calibration seed={seed}, condition={condition}")
            episode = _simulate(
                cfg=cfg,
                condition_cfg=(
                    asynchronous_cfg if condition == "A_async" else reference_cfg
                ),
                condition_id=condition,
                action=actions[condition],
                stimulate=False,
                seed=seed,
                action_index=action_index,
                output_dir=root / "calibration" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                row, _, _, _ = _episode_epoch_feature_row(
                    episode,
                    condition_id=condition,
                    epoch="stimulation",
                    selected_frequency_hz=selected_frequency_hz,
                    candidate_frequencies_hz=candidates,
                    cfg=cfg,
                )
                calibration_rows.append(row)

    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        spectral_model = _fit_centroid_model(
            calibration,
            feature_names=_frequency_feature_names(candidates),
        )
        replicated_frequency_hz, frequency_table = _audit_target_frequency(
            calibration,
            candidate_frequencies_hz=candidates,
            spectral_model=spectral_model,
        )
        primary_model = _fit_centroid_model(
            calibration, feature_names=[selected_feature]
        )
        excluded_model = _fit_centroid_model(
            calibration,
            feature_names=[
                "log10_total_power_excluding_selected",
                "relative_30_80_power_excluding_selected",
            ],
        )
        calibration = _add_distances(
            calibration, primary_model, prefix=PHASE_INVARIANT_PREFIX
        )
        calibration = _add_distances(
            calibration, excluded_model, prefix=EXCLUDED_PREFIX
        )
        calibration.to_csv(root / "calibration_eeg_metrics.csv", index=False)
        frequency_table.to_csv(
            root / "target_frequency_replication.csv", index=False
        )
    else:
        primary_model = excluded_model = None
        replicated_frequency_hz = None
        frequency_table = None
    primary_model = comm.bcast(primary_model, root=0)
    excluded_model = comm.bcast(excluded_model, root=0)
    replicated_frequency_hz = float(comm.bcast(replicated_frequency_hz, root=0))

    frozen = {
        "frequency_hz": selected_frequency_hz,
        "phase_rad": selected_phase_rad,
        "phase_policy": "fixed convention; not an action and not optimized",
        "primary_amplitude_v_per_m": float(
            cfg.analysis.frozen_protocol.primary_amplitude_v_per_m
        ),
        "lower_amplitude_v_per_m": float(
            cfg.analysis.frozen_protocol.lower_amplitude_v_per_m
        ),
        "lower_dose_enabled": bool(
            cfg.analysis.validation.include_lower_dose
        ),
        "montage": str(cfg.analysis.tacs.axial_montage),
        "primary_eeg_feature": selected_feature,
        "feature_is_phase_invariant": True,
        "selection_provenance": str(cfg.analysis.frozen_protocol.provenance),
        "calibration_frequency_replication_hz": replicated_frequency_hz,
    }
    if rank == 0:
        with (root / "frozen_phase_invariant_protocol.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    **frozen,
                    "phase_invariant_target_model": primary_model,
                    "excluded_target_model": excluded_model,
                },
                handle,
                indent=2,
            )

    # No protocol or feature selection is allowed below this point.
    epoch_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    psd_rows: list[dict[str, Any]] = []
    validation_episodes: dict[int, dict[str, dict[str, Any]]] = {}
    representative_spikes = None
    validation_conditions = _validation_conditions(cfg)
    validation_seeds = _seeds(cfg, "validation")
    for seed in validation_seeds:
        seed_episodes: dict[str, dict[str, Any]] = {}
        for action_index, condition in enumerate(validation_conditions):
            if rank == 0:
                print(f"validation seed={seed}, condition={condition}")
            episode = _simulate(
                cfg=cfg,
                condition_cfg=(
                    reference_cfg
                    if condition == "B_rhythmic_reference"
                    else asynchronous_cfg
                ),
                condition_id=condition,
                action=actions[condition],
                stimulate=condition not in REFERENCE_CONDITIONS,
                seed=seed,
                action_index=action_index,
                output_dir=root / "validation" / condition / f"seed_{seed}",
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                seed_episodes[condition] = episode
                for epoch in EPOCHS:
                    row, _, frequencies_hz, psd = _episode_epoch_feature_row(
                        episode,
                        condition_id=condition,
                        epoch=epoch,
                        selected_frequency_hz=selected_frequency_hz,
                        candidate_frequencies_hz=candidates,
                        cfg=cfg,
                    )
                    epoch_rows.append(row)
                    if epoch == "stimulation" and condition in (
                        "A_async", "B_rhythmic_reference", PRIMARY_ACTION
                    ):
                        psd_rows.extend(
                            {
                                "seed": seed,
                                "condition_id": condition,
                                "frequency_hz": float(frequency),
                                "psd_v2_hz": float(value),
                            }
                            for frequency, value in zip(frequencies_hz, psd)
                            if 1.0 <= frequency <= 100.0
                        )
                window_rows.extend(
                    _episode_window_feature_rows(
                        episode,
                        condition_id=condition,
                        selected_frequency_hz=selected_frequency_hz,
                        candidate_frequencies_hz=candidates,
                        cfg=cfg,
                    )
                )
        if rank == 0:
            synthetic, _, frequencies_hz, psd = _synthetic_feature_row(
                seed_episodes["A_async"],
                seed_episodes[PRIMARY_ACTION],
                selected_frequency_hz=selected_frequency_hz,
                candidate_frequencies_hz=candidates,
                cfg=cfg,
            )
            epoch_rows.append(synthetic)
            psd_rows.extend(
                {
                    "seed": seed,
                    "condition_id": SYNTHETIC_CONTROL,
                    "frequency_hz": float(frequency),
                    "psd_v2_hz": float(value),
                }
                for frequency, value in zip(frequencies_hz, psd)
                if 1.0 <= frequency <= 100.0
            )
            validation_episodes[seed] = seed_episodes
            if representative_spikes is None:
                representative_spikes = _representative_spike_rows(
                    seed_episodes,
                    selected_frequency_hz=selected_frequency_hz,
                )

    if rank != 0:
        return

    epoch_frame = pd.DataFrame(epoch_rows)
    window_frame = pd.DataFrame(window_rows)
    epoch_frame = _add_distances(
        epoch_frame, primary_model, prefix=PHASE_INVARIANT_PREFIX
    )
    epoch_frame = _add_distances(
        epoch_frame, excluded_model, prefix=EXCLUDED_PREFIX
    )
    window_frame = _add_distances(
        window_frame, primary_model, prefix=PHASE_INVARIANT_PREFIX
    )
    window_frame = _add_distances(
        window_frame, excluded_model, prefix=EXCLUDED_PREFIX
    )
    epoch_frame.to_csv(root / "validation_epoch_eeg_metrics.csv", index=False)
    window_frame.to_csv(root / "validation_window_eeg_metrics.csv", index=False)
    pd.DataFrame(psd_rows).to_csv(root / "validation_psd_long.csv", index=False)
    representative_spikes.to_csv(
        root / "representative_E_spikes.csv", index=False
    )

    primary_seed, primary_summary = _validation_reachability(
        epoch_frame, prefix=PHASE_INVARIANT_PREFIX, cfg=cfg, rng=rng
    )
    excluded_seed, excluded_summary = _validation_reachability(
        epoch_frame, prefix=EXCLUDED_PREFIX, cfg=cfg, rng=rng
    )
    reachability = pd.concat((primary_seed, excluded_seed), ignore_index=True)
    reachability_summary = pd.concat(
        (primary_summary, excluded_summary), ignore_index=True
    )
    reachability.to_csv(root / "validation_reachability.csv", index=False)
    reachability_summary.to_csv(
        root / "validation_reachability_summary.csv", index=False
    )

    feature_names = [
        selected_feature,
        "relative_selected_band_power",
        "selected_eeg_resultant_v",
    ]
    feature_seed, feature_summary = _paired_feature_effects(
        epoch_frame,
        feature_names=feature_names,
        cfg=cfg,
        rng=rng,
    )
    feature_seed.to_csv(
        root / "validation_phase_invariant_feature_effects.csv", index=False
    )
    feature_summary.to_csv(
        root / "validation_phase_invariant_feature_summary.csv", index=False
    )

    hidden = _hidden_validation_rows(
        epoch_frame[epoch_frame.condition_id.isin(validation_conditions)],
        validation_episodes,
        cfg=cfg,
    )
    hidden.to_csv(root / "validation_hidden_mechanism.csv", index=False)

    def reach_rows(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
        return frame[frame.condition_id.eq(condition)].sort_values("seed")

    def reach_summary(
        frame: pd.DataFrame, condition: str
    ) -> dict[str, Any]:
        return frame[frame.condition_id.eq(condition)].iloc[0].to_dict()

    reference_primary = reach_rows(primary_seed, "B_rhythmic_reference")
    active_primary = reach_rows(primary_seed, PRIMARY_ACTION)
    transverse_primary = reach_rows(primary_seed, TRANSVERSE_CONTROL)
    synthetic_primary = reach_rows(primary_seed, SYNTHETIC_CONTROL)
    reference_excluded = reach_rows(excluded_seed, "B_rhythmic_reference")
    active_excluded = reach_rows(excluded_seed, PRIMARY_ACTION)
    synthetic_excluded = reach_rows(excluded_seed, SYNTHETIC_CONTROL)

    reference_primary_summary = reach_summary(
        primary_summary, "B_rhythmic_reference"
    )
    active_primary_summary = reach_summary(primary_summary, PRIMARY_ACTION)
    orientation_advantage = _summary(
        active_primary.target_distance_improvement.to_numpy(float)
        - transverse_primary.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    beyond_synthetic = _summary(
        active_primary.target_distance_improvement.to_numpy(float)
        - synthetic_primary.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    excluded_reference_summary = _summary(
        reference_excluded.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    beyond_synthetic_excluded = _summary(
        active_excluded.target_distance_improvement.to_numpy(float)
        - synthetic_excluded.target_distance_improvement.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )

    stimulation = epoch_frame[epoch_frame.epoch.eq("stimulation")]
    a_stim = stimulation[stimulation.condition_id.eq("A_async")].set_index("seed")
    b_stim = stimulation[
        stimulation.condition_id.eq("B_rhythmic_reference")
    ].set_index("seed")
    active_stim = stimulation[
        stimulation.condition_id.eq(PRIMARY_ACTION)
    ].set_index("seed")
    reference_shift = _summary(
        (b_stim[selected_feature] - a_stim[selected_feature]).to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    active_shift = _summary(
        (active_stim[selected_feature] - a_stim[selected_feature]).to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    reference_rate_matched = float(
        np.mean(
            [
                _relative_rate_safe(b_stim.loc[seed], a_stim.loc[seed], cfg)
                for seed in validation_seeds
            ]
        )
    )

    primary_hidden = hidden[hidden.condition_id.eq(PRIMARY_ACTION)]
    ppc_summary = _summary(
        primary_hidden.E_ppc_gain_difference_in_differences.to_numpy(float),
        cfg=cfg,
        rng=rng,
    )
    seed_accuracy = _classification_accuracy(
        epoch_frame, prefix=PHASE_INVARIANT_PREFIX
    )
    window_accuracy = _classification_accuracy(
        window_frame, prefix=PHASE_INVARIANT_PREFIX
    )
    excluded_accuracy = _classification_accuracy(
        epoch_frame, prefix=EXCLUDED_PREFIX
    )

    criteria = cfg.analysis.criteria
    minimum_positive = float(criteria.minimum_positive_seed_fraction)
    selected_frequency_row = frequency_table[
        np.isclose(frequency_table.frequency_hz, selected_frequency_hz)
    ].iloc[0]
    confirmation_checks = {
        "minimum_calibration_seeds": len(_seeds(cfg, "calibration"))
        >= int(criteria.minimum_calibration_seeds),
        "minimum_validation_seeds": len(validation_seeds)
        >= int(criteria.minimum_validation_seeds),
        "frozen_frequency_replicated_from_unstimulated_eeg": (
            np.isclose(replicated_frequency_hz, selected_frequency_hz)
            and float(selected_frequency_row.positive_seed_fraction)
            >= minimum_positive
        ),
        "heldout_reference_phase_invariant_eeg_distinct": (
            float(reference_primary_summary["ci_2.5"]) > 0.0
            and float(reference_primary_summary["positive_seed_fraction"])
            >= minimum_positive
            and reference_shift["ci_2.5"] > 0.0
        ),
        "heldout_reference_classification": seed_accuracy
        >= float(criteria.minimum_reference_classification_accuracy),
        "one_second_eeg_observable": window_accuracy
        >= float(criteria.minimum_window_classification_accuracy),
        "primary_tacs_increases_target_band_power": (
            active_shift["ci_2.5"] > 0.0
            and active_shift["positive_seed_fraction"] >= minimum_positive
        ),
        "primary_tacs_moves_eeg_toward_B": (
            float(active_primary_summary["ci_2.5"]) > 0.0
            and float(active_primary_summary["positive_seed_fraction"])
            >= minimum_positive
        ),
        "orientation_specific_eeg_movement": (
            orientation_advantage["ci_2.5"] > 0.0
        ),
        "hidden_spike_timing_modulated": (
            ppc_summary["ci_2.5"] > 0.0
            and ppc_summary["positive_seed_fraction"] >= minimum_positive
        ),
        "reference_rate_matched": reference_rate_matched
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "tacs_rate_safe": float(primary_hidden.rate_safe.mean())
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "washout_reversible": float(primary_hidden.washout_recovered.mean())
        >= float(criteria.minimum_washout_recovery_seed_fraction),
        "baseline_causality": float(
            primary_hidden.baseline_relative_rms_error_vs_A.max()
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
    }
    measurement_audits = {
        "complex_observation_distinguishable_on_primary_endpoint": (
            beyond_synthetic["ci_2.5"] > 0.0
        ),
        "reference_observable_after_target_band_exclusion": (
            excluded_reference_summary["ci_2.5"] > 0.0
            and excluded_accuracy
            >= float(criteria.minimum_reference_classification_accuracy)
        ),
        "real_tacs_beyond_complex_observation_after_exclusion": (
            beyond_synthetic_excluded["ci_2.5"] > 0.0
        ),
    }
    directional_gate_checks = {
        "frozen_frequency_has_positive_calibration_shift": (
            float(selected_frequency_row.mean_standardized_shift) > 0.0
            and float(selected_frequency_row.positive_seed_fraction)
            >= minimum_positive
        ),
        "primary_tacs_increases_target_band_power_directionally": (
            active_shift["mean"] > 0.0
            and active_shift["positive_seed_fraction"] >= minimum_positive
        ),
        "primary_tacs_moves_toward_B_directionally": (
            float(active_primary_summary["mean"]) > 0.0
            and float(active_primary_summary["positive_seed_fraction"])
            >= minimum_positive
        ),
        "axial_advantage_directionally_positive": (
            orientation_advantage["mean"] > 0.0
        ),
        "spike_timing_not_directionally_adverse": ppc_summary["mean"] >= 0.0,
        "all_rates_safe": float(primary_hidden.rate_safe.mean())
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "baseline_causality": float(
            primary_hidden.baseline_relative_rms_error_vs_A.max()
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
    }

    # Export but do not fit a policy. With one fixed target, this first tests a
    # best-arm problem; context is justified only after a context/action
    # interaction is independently replicated.
    transition_rows: list[dict[str, Any]] = []
    policy_conditions = ["A_async"]
    if bool(cfg.analysis.validation.include_lower_dose):
        policy_conditions.append(LOWER_ACTION)
    policy_conditions.append(PRIMARY_ACTION)
    baseline_context = epoch_frame[
        epoch_frame.condition_id.eq("A_async")
        & epoch_frame.epoch.eq("baseline")
    ].set_index("seed")
    for seed in validation_seeds:
        for condition in policy_conditions:
            row = stimulation[
                stimulation.seed.eq(seed)
                & stimulation.condition_id.eq(condition)
            ].iloc[0]
            amplitude = (
                0.0
                if condition == "A_async"
                else float(actions[condition]["ac_amplitude_v_per_m"])
            )
            transition_rows.append(
                {
                    "seed": seed,
                    "action_id": condition,
                    "amplitude_v_per_m": amplitude,
                    "frequency_hz": selected_frequency_hz,
                    "phase_rad": selected_phase_rad,
                    "context_distance_to_B": float(
                        baseline_context.loc[
                            seed, "phase_invariant_distance_to_B"
                        ]
                    ),
                    "outcome_distance_to_B": float(
                        row.phase_invariant_distance_to_B
                    ),
                    "reward": -float(row.phase_invariant_distance_to_B),
                }
            )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(root / "future_bandit_transition_table.csv", index=False)
    rewards = transitions.groupby("action_id").reward.mean()
    oracle = transitions.loc[transitions.groupby("seed").reward.idxmax()]
    policy_diagnostic = {
        "best_fixed_action": str(rewards.idxmax()),
        "mean_reward_by_action": _plain(rewards.to_dict()),
        "oracle_best_action_counts": _plain(
            oracle.action_id.value_counts().to_dict()
        ),
        "oracle_advantage_over_best_fixed": float(
            oracle.reward.mean() - rewards.max()
        ),
        "ready_for_bandit": False,
        "interpretation": (
            "Descriptive export only. Confirm the EEG endpoint first; a "
            "contextual policy additionally requires a replicated "
            "context-by-action interaction."
        ),
    }

    psd_frame = pd.DataFrame(psd_rows)
    if bool(cfg.experiment.plot):
        _plot_psd(
            psd_frame,
            selected_frequency_hz=selected_frequency_hz,
            root=root,
        )
        source = root / "figure_01_validation_psd.png"
        if source.exists():
            source.rename(root / "figure_01_phase_invariant_psd.png")
        _plot_phase_invariant_validation(
            epoch_frame,
            primary_seed,
            hidden,
            feature_name=selected_feature,
            root=root,
        )
        # Reuse the established spike raster/phase plot, but keep the phase as
        # a hidden mechanism relative to the fixed stimulus, not an EEG state.
        _plot_spike_timing(
            representative_spikes,
            epoch_frame,
            representative_seed=validation_seeds[0],
            selected_frequency_hz=selected_frequency_hz,
            selected_phase_rad=selected_phase_rad,
            window_ms=float(cfg.env.simulation.obs_win_len),
            root=root,
        )
        spike_source = root / "figure_04_representative_spike_timing.png"
        if spike_source.exists():
            spike_source.rename(root / "figure_03_spike_timing.png")
        _plot_online_trajectory(window_frame, root=root)
        _plot_observation_audit(
            active_primary,
            synthetic_primary,
            active_excluded,
            synthetic_excluded,
            root=root,
        )

    conclusion = {
        "scope": (
            "Phase-invariant reachability in ideal neural-only EEG from a "
            "40-cell toy circuit; not simultaneous clinical tACS-EEG."
        ),
        "scientific_change_from_hierarchical_pilot": (
            "Absolute Fourier phase is no longer part of the A/B state and "
            "is not optimized. Frequency was frozen before this run."
        ),
        "hidden_reference_generator": {
            "frequency_hz": float(cfg.analysis.reference.frequency_hz),
            "modulation_depth": float(cfg.analysis.reference.modulation_depth),
            "used_to_change_frozen_protocol": False,
        },
        "frozen_protocol": frozen,
        "frequency_replication_table": _plain(
            frequency_table.to_dict("records")
        ),
        "reference_target_shift": reference_shift,
        "primary_tacs_target_shift": active_shift,
        "reference_phase_invariant_reachability": reference_primary_summary,
        "primary_phase_invariant_reachability": active_primary_summary,
        "orientation_advantage": orientation_advantage,
        "hidden_primary_E_ppc_gain": ppc_summary,
        "classification": {
            "phase_invariant_seed_accuracy": seed_accuracy,
            "phase_invariant_one_second_window_accuracy": window_accuracy,
            "excluded_seed_accuracy": excluded_accuracy,
        },
        "measurement_audit_statistics": {
            "advantage_beyond_complex_observation": beyond_synthetic,
            "excluded_reference": excluded_reference_summary,
            "advantage_beyond_complex_observation_after_exclusion": (
                beyond_synthetic_excluded
            ),
        },
        "confirmation_checks": confirmation_checks,
        "directional_gate_checks": directional_gate_checks,
        "directional_gate_passed": bool(all(directional_gate_checks.values())),
        "ideal_phase_invariant_reachability_passed": bool(
            all(confirmation_checks.values())
        ),
        "measurement_audits": measurement_audits,
        "artifact_robust_reachability_passed": bool(
            all(confirmation_checks.values())
            and all(measurement_audits.values())
        ),
        "future_policy_diagnostic": policy_diagnostic,
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (root / "experiment_conclusion.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(_plain(conclusion), handle, indent=2)

    print("\n### Frozen phase-invariant protocol")
    print(json.dumps(frozen, indent=2))
    print("\n### Phase-invariant confirmation checks")
    for name, passed in confirmation_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### Low-cost directional gate")
    for name, passed in directional_gate_checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "Directional gate: "
        f"{'PASSED' if conclusion['directional_gate_passed'] else 'NOT PASSED'}"
    )
    print("\n### Measurement audits (not primary success criteria)")
    for name, passed in measurement_audits.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nIdeal phase-invariant A -> B-like reachability: "
        + (
            "PASSED"
            if conclusion["ideal_phase_invariant_reachability_passed"]
            else "NOT PASSED"
        )
    )
    print(
        "Artifact-robust reachability: "
        + (
            "PASSED"
            if conclusion["artifact_robust_reachability_passed"]
            else "NOT PASSED"
        )
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
