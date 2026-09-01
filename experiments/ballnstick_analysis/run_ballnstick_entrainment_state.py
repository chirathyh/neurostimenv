"""Test whether tACS moves an asynchronous circuit toward an entrained state.

The reference state and the actuator use deliberately different mechanisms:

* A is the unchanged BallAndStick network driven by independent homogeneous
  Poisson synaptic events.
* B has identical cells, recurrence, mean afferent rate, and synaptic weights,
  but each independent afferent train has a sinusoidally modulated event rate.
* A+tACS retains A's asynchronous afferent statistics and receives a spatially
  uniform extracellular AC field only during the intervention block.

The primary state variable is E-population PPC at the predeclared reference
frequency.  Calibration selects only the modulation depth used to instantiate
an attainable B reference.  It never selects or changes the tACS action.  The
selected B setting and fixed tACS protocol are then evaluated on disjoint
matched circuit seeds.

This is an acute entrainment-state reachability benchmark.  It is not a model
of depression, treatment, synaptic repair, or a lasting stimulation effect.
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
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _analyze_episode,
    _bootstrap_ci,
    _plain_copy,
    _relative_rate_safe,
    _sign_flip_p,
    _simulate_episode,
)


PRIMARY_FEATURE = "E_ppc"
EPOCH = "stimulation"
CONDITION_ORDER = (
    "A_async",
    "B_rhythmic_reference",
    "A_tacs_axial",
    "A_tacs_transverse",
)


def _validate_design(cfg: DictConfig) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("Entrainment-state reachability requires online mode.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError(
            "This experiment keeps inhibition_scale=1.0 in A, B, and A+tACS."
        )

    frequency_hz = float(cfg.analysis.reference.frequency_hz)
    action_frequency_hz = float(cfg.analysis.tacs.frequency_hz)
    if frequency_hz <= 0.0:
        raise ValueError("The reference frequency must be positive.")
    if not np.isclose(frequency_hz, action_frequency_hz):
        raise ValueError(
            "The minimal experiment requires reference and tACS frequencies "
            "to match. Frequency controls belong in the follow-up design."
        )

    depths = [
        float(value)
        for value in cfg.analysis.calibration.modulation_depths
    ]
    if not depths or any(value <= 0.0 or value > 1.0 for value in depths):
        raise ValueError("Calibration modulation depths must be in (0, 1].")
    if len(depths) != len(set(depths)):
        raise ValueError("Calibration modulation depths must be unique.")
    envelope = float(
        cfg.analysis.reference.thinning_envelope_modulation_depth
    )
    if envelope < max(depths) or envelope > 1.0:
        raise ValueError(
            "The common thinning envelope must contain every calibration depth."
        )

    amplitude = float(cfg.analysis.tacs.amplitude_v_per_m)
    maximum = float(cfg.analysis.maximum_field_v_per_m)
    if amplitude <= 0.0 or amplitude > maximum:
        raise ValueError("tACS amplitude must be in (0, maximum_field_v_per_m].")

    populations = [str(value) for value in cfg.analysis.reference.populations]
    if not populations or not set(populations).issubset({"E", "I"}):
        raise ValueError("reference.populations must contain only E and/or I.")

    # The afferent rhythm uses absolute simulation time while the action phase
    # is initialized at the intervention boundary.  Requiring an integer number
    # of reference cycles before that boundary makes phase zero comparable.
    timeline = cfg.analysis.timeline
    block_start_ms = (
        int(timeline.burn_in_steps) + int(timeline.baseline_steps)
    ) * float(cfg.env.simulation.obs_win_len)
    cycles = frequency_hz * block_start_ms / 1000.0
    if not np.isclose(cycles, round(cycles), atol=1e-10):
        raise ValueError(
            "The intervention boundary must fall on an integer reference cycle."
        )


def _condition_config(
    base_cfg: DictConfig,
    *,
    modulation_depth: float,
) -> DictConfig:
    """Return a copied config with matched stochastic afferent statistics."""
    run_cfg = _plain_copy(base_cfg)
    OmegaConf.set_struct(run_cfg, False)
    depth = float(modulation_depth)
    populations = {
        str(value) for value in run_cfg.analysis.reference.populations
    }
    frequency_hz = float(run_cfg.analysis.reference.frequency_hz)
    phase_rad = float(run_cfg.analysis.reference.phase_rad)
    envelope = float(
        run_cfg.analysis.reference.thinning_envelope_modulation_depth
    )
    with open_dict(run_cfg):
        for population_name in ("E", "I"):
            rhythm = run_cfg.env.network.background[population_name].rhythm
            rhythm.enabled = population_name in populations
            rhythm.modulation_depth = depth if population_name in populations else 0.0
            rhythm.frequency_hz = frequency_hz
            rhythm.phase_rad = phase_rad
            rhythm.thinning_envelope_modulation_depth = (
                envelope if population_name in populations else 0.0
            )
    return run_cfg


def _action(
    cfg: DictConfig,
    *,
    identifier: str,
    role: str,
    montage: str,
    amplitude_v_per_m: float,
) -> dict[str, Any]:
    return {
        "id": identifier,
        "role": role,
        "montage": montage,
        "dc_offset_v_per_m": 0.0,
        "ac_amplitude_v_per_m": float(amplitude_v_per_m),
        "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
        "phase_rad": float(cfg.analysis.tacs.phase_rad),
    }


def _actions(cfg: DictConfig) -> dict[str, dict[str, Any]]:
    axial = str(cfg.analysis.tacs.axial_montage)
    transverse = str(cfg.analysis.tacs.transverse_montage)
    amplitude = float(cfg.analysis.tacs.amplitude_v_per_m)
    return {
        "A_async": _action(
            cfg,
            identifier="A_async",
            role="asynchronous_state",
            montage=axial,
            amplitude_v_per_m=0.0,
        ),
        "B_rhythmic_reference": _action(
            cfg,
            identifier="B_rhythmic_reference",
            role="synaptic_reference_state",
            montage=axial,
            amplitude_v_per_m=0.0,
        ),
        "A_tacs_axial": _action(
            cfg,
            identifier="A_tacs_axial",
            role="tacs_intervention",
            montage=axial,
            amplitude_v_per_m=amplitude,
        ),
        "A_tacs_transverse": _action(
            cfg,
            identifier="A_tacs_transverse",
            role="orientation_control",
            montage=transverse,
            amplitude_v_per_m=amplitude,
        ),
    }


def _epoch_map(rows: list[dict[str, Any]]) -> dict[str, pd.Series]:
    return {str(row["epoch"]): pd.Series(row) for row in rows}


def _rate_matched(candidate: pd.Series, reference: pd.Series, cfg: DictConfig) -> bool:
    return _relative_rate_safe(candidate, reference, cfg)


def _calibration_seed_row(
    *,
    seed: int,
    modulation_depth: float,
    asynchronous_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
    cfg: DictConfig,
) -> dict[str, Any]:
    asynchronous = _epoch_map(asynchronous_rows)[EPOCH]
    reference = _epoch_map(reference_rows)[EPOCH]
    target = float(cfg.analysis.reference.target_E_ppc)
    return {
        "seed": int(seed),
        "modulation_depth": float(modulation_depth),
        "A_E_ppc": float(asynchronous[PRIMARY_FEATURE]),
        "B_E_ppc": float(reference[PRIMARY_FEATURE]),
        "B_minus_A_E_ppc": float(
            reference[PRIMARY_FEATURE] - asynchronous[PRIMARY_FEATURE]
        ),
        "absolute_target_error": abs(float(reference[PRIMARY_FEATURE]) - target),
        "B_E_rate_hz": float(reference["E_firing_rate_hz"]),
        "A_E_rate_hz": float(asynchronous["E_firing_rate_hz"]),
        "B_I_rate_hz": float(reference["I_firing_rate_hz"]),
        "A_I_rate_hz": float(asynchronous["I_firing_rate_hz"]),
        "B_rate_matched_to_A": _rate_matched(reference, asynchronous, cfg),
        "B_E_plv_above_uniform_null": bool(
            reference["E_plv_above_uniform_null"]
        ),
    }


def _calibration_summary(rows: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    result = (
        rows.groupby("modulation_depth", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_A_E_ppc=("A_E_ppc", "mean"),
            mean_B_E_ppc=("B_E_ppc", "mean"),
            mean_B_minus_A_E_ppc=("B_minus_A_E_ppc", "mean"),
            positive_reference_seed_fraction=(
                "B_minus_A_E_ppc",
                lambda values: float(np.mean(np.asarray(values) > 0.0)),
            ),
            rate_matched_seed_fraction=("B_rate_matched_to_A", "mean"),
            above_null_seed_fraction=(
                "B_E_plv_above_uniform_null",
                "mean",
            ),
        )
        .sort_values("modulation_depth")
        .reset_index(drop=True)
    )
    result["absolute_target_error"] = abs(
        result["mean_B_E_ppc"] - float(cfg.analysis.reference.target_E_ppc)
    )
    return result


def _select_reference_depth(
    summary: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[float, bool]:
    minimum_safe = float(
        cfg.analysis.calibration.minimum_rate_matched_fraction
    )
    safe = summary[summary["rate_matched_seed_fraction"] >= minimum_safe]
    used_safe_subset = not safe.empty
    candidates = safe if used_safe_subset else summary
    selected = candidates.sort_values(
        ["absolute_target_error", "modulation_depth"],
        ascending=[True, True],
    ).iloc[0]
    return float(selected["modulation_depth"]), used_safe_subset


def _simulate_and_analyze(
    *,
    condition_id: str,
    condition_cfg: DictConfig,
    action: dict[str, Any],
    stimulate: bool,
    seed: int,
    action_index: int,
    output_dir: Path,
    comm,
    size: int,
    rank: int,
) -> tuple[list[dict[str, Any]] | None, dict[str, np.ndarray] | None]:
    simulation = _simulate_episode(
        condition_cfg,
        seed=seed,
        action=action,
        stimulate=stimulate,
        output_dir=output_dir,
        comm=comm,
        size=size,
        rank=rank,
    )
    if rank != 0:
        return None, None
    rows, raw = _analyze_episode(
        simulation,
        action=action,
        action_index=action_index,
        arm=condition_id,
        cfg=condition_cfg,
        output_dir=output_dir / "analysis",
    )
    for row in rows:
        row["condition_id"] = condition_id
    return rows, raw


def _validation_seed_row(
    *,
    seed: int,
    rows_by_condition: dict[str, list[dict[str, Any]]],
    raw_by_condition: dict[str, dict[str, np.ndarray]],
    cfg: DictConfig,
) -> dict[str, Any]:
    epochs = {
        name: _epoch_map(rows) for name, rows in rows_by_condition.items()
    }
    a = epochs["A_async"]
    b = epochs["B_rhythmic_reference"]
    tacs = epochs["A_tacs_axial"]
    transverse = epochs["A_tacs_transverse"]

    a_ppc = float(a[EPOCH][PRIMARY_FEATURE])
    b_ppc = float(b[EPOCH][PRIMARY_FEATURE])
    tacs_ppc = float(tacs[EPOCH][PRIMARY_FEATURE])
    transverse_ppc = float(transverse[EPOCH][PRIMARY_FEATURE])
    baseline_distance = abs(b_ppc - a_ppc)
    tacs_distance = abs(b_ppc - tacs_ppc)
    transverse_distance = abs(b_ppc - transverse_ppc)
    epsilon = np.finfo(float).eps

    tacs_gain = (
        (tacs_ppc - float(tacs["baseline"][PRIMARY_FEATURE]))
        - (a_ppc - float(a["baseline"][PRIMARY_FEATURE]))
    )
    transverse_gain = (
        (transverse_ppc - float(transverse["baseline"][PRIMARY_FEATURE]))
        - (a_ppc - float(a["baseline"][PRIMARY_FEATURE]))
    )
    washout_gain = (
        (
            float(tacs["washout"][PRIMARY_FEATURE])
            - float(tacs["baseline"][PRIMARY_FEATURE])
        )
        - (
            float(a["washout"][PRIMARY_FEATURE])
            - float(a["baseline"][PRIMARY_FEATURE])
        )
    )
    maximum_washout_fraction = float(
        cfg.analysis.criteria.maximum_washout_residual_fraction
    )

    return {
        "seed": int(seed),
        "A_E_ppc": a_ppc,
        "B_E_ppc": b_ppc,
        "A_tacs_E_ppc": tacs_ppc,
        "A_transverse_E_ppc": transverse_ppc,
        "reference_shift_E_ppc": b_ppc - a_ppc,
        "tacs_E_ppc_gain_difference_in_differences": tacs_gain,
        "transverse_E_ppc_gain_difference_in_differences": transverse_gain,
        "baseline_target_distance_E_ppc": baseline_distance,
        "tacs_target_distance_E_ppc": tacs_distance,
        "transverse_target_distance_E_ppc": transverse_distance,
        "target_distance_improvement_E_ppc": baseline_distance - tacs_distance,
        "fractional_target_distance_improvement": (
            (baseline_distance - tacs_distance) / max(baseline_distance, epsilon)
        ),
        "orientation_advantage_E_ppc_distance": (
            transverse_distance - tacs_distance
        ),
        "reference_direction_aligned": bool(
            (b_ppc - a_ppc) * tacs_gain > 0.0
        ),
        "B_E_plv_above_uniform_null": bool(
            b[EPOCH]["E_plv_above_uniform_null"]
        ),
        "A_tacs_E_plv_above_uniform_null": bool(
            tacs[EPOCH]["E_plv_above_uniform_null"]
        ),
        "B_rate_matched_to_A": _rate_matched(b[EPOCH], a[EPOCH], cfg),
        "A_tacs_rate_safe": _rate_matched(tacs[EPOCH], a[EPOCH], cfg),
        "A_E_rate_hz": float(a[EPOCH]["E_firing_rate_hz"]),
        "B_E_rate_hz": float(b[EPOCH]["E_firing_rate_hz"]),
        "A_tacs_E_rate_hz": float(tacs[EPOCH]["E_firing_rate_hz"]),
        "A_I_rate_hz": float(a[EPOCH]["I_firing_rate_hz"]),
        "B_I_rate_hz": float(b[EPOCH]["I_firing_rate_hz"]),
        "A_tacs_I_rate_hz": float(tacs[EPOCH]["I_firing_rate_hz"]),
        "tacs_washout_E_ppc_gain_difference_in_differences": washout_gain,
        "washout_recovered": bool(
            tacs_gain > 0.0
            and abs(washout_gain)
            <= maximum_washout_fraction * max(abs(tacs_gain), epsilon)
        ),
        "baseline_relative_rms_error_A_tacs_vs_A": _relative_rms_error(
            raw_by_condition["A_async"]["baseline"],
            raw_by_condition["A_tacs_axial"]["baseline"],
        ),
        "baseline_relative_rms_error_transverse_vs_A": _relative_rms_error(
            raw_by_condition["A_async"]["baseline"],
            raw_by_condition["A_tacs_transverse"]["baseline"],
        ),
        "A_tacs_minus_A_log10_total_power_1_80_excluding_stimulus": float(
            tacs[EPOCH]["log10_total_power_1_80_excluding_stimulus"]
            - a[EPOCH]["log10_total_power_1_80_excluding_stimulus"]
        ),
        "B_minus_A_log10_total_power_1_80_excluding_stimulus": float(
            b[EPOCH]["log10_total_power_1_80_excluding_stimulus"]
            - a[EPOCH]["log10_total_power_1_80_excluding_stimulus"]
        ),
    }


def _summary(
    rows: pd.DataFrame,
    *,
    column: str,
    cfg: DictConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    values = rows[column].to_numpy(dtype=float)
    ci_low, ci_high = _bootstrap_ci(
        values,
        rng=rng,
        n_bootstrap=int(cfg.analysis.n_bootstrap),
    )
    return {
        "mean": float(np.mean(values)),
        "ci_2.5": ci_low,
        "ci_97.5": ci_high,
        "positive_seed_fraction": float(np.mean(values > 0.0)),
        "paired_sign_flip_p": _sign_flip_p(
            values,
            rng=rng,
            n_permutations=int(cfg.analysis.n_permutations),
        ),
    }


def _plot_validation(epoch_rows: pd.DataFrame, root: Path) -> None:
    stimulation = epoch_rows[epoch_rows["epoch"] == EPOCH]
    values = {
        name: stimulation[stimulation["condition_id"] == name]
        .sort_values("seed")[PRIMARY_FEATURE]
        .to_numpy(dtype=float)
        for name in CONDITION_ORDER
    }
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(CONDITION_ORDER), dtype=float)
    for seed_index in range(min(value.size for value in values.values())):
        axis.plot(
            x,
            [values[name][seed_index] for name in CONDITION_ORDER],
            color="0.75",
            linewidth=0.8,
            alpha=0.8,
        )
    axis.scatter(
        x,
        [np.mean(values[name]) for name in CONDITION_ORDER],
        color=["black", "tab:green", "tab:blue", "tab:orange"],
        zorder=3,
    )
    axis.set_xticks(
        x,
        labels=["A async", "B synaptic", "A + axial tACS", "A + transverse"],
        rotation=12,
    )
    axis.set_ylabel("E-population PPC at reference frequency")
    axis.set_title("Matched-seed entrainment-state reachability")
    figure.tight_layout()
    figure.savefig(root / "validation_E_ppc_reachability.png", dpi=250)
    plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _validate_design(cfg)
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "entrainment_state"
    if rank == 0:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        print("\n### BallAndStick entrainment-state reachability")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    actions = _actions(cfg)
    base_seed = int(cfg.experiment.seed)
    calibration_seeds = [
        base_seed + int(cfg.analysis.calibration.seed_offset) + index
        for index in range(int(cfg.analysis.calibration.n_seeds))
    ]
    validation_seeds = [
        base_seed + int(cfg.analysis.validation.seed_offset) + index
        for index in range(int(cfg.analysis.validation.n_seeds))
    ]
    if set(calibration_seeds).intersection(validation_seeds):
        raise ValueError("Calibration and validation circuit seeds must be disjoint.")

    asynchronous_cfg = _condition_config(cfg, modulation_depth=0.0)
    calibration_epoch_rows: list[dict[str, Any]] = []
    calibration_seed_rows: list[dict[str, Any]] = []
    depths = sorted(
        float(value) for value in cfg.analysis.calibration.modulation_depths
    )
    for seed in calibration_seeds:
        if rank == 0:
            print(f"calibration seed={seed}, condition=A_async")
        a_rows, _ = _simulate_and_analyze(
            condition_id="A_async",
            condition_cfg=asynchronous_cfg,
            action=actions["A_async"],
            stimulate=False,
            seed=seed,
            action_index=0,
            output_dir=root / "calibration" / "A_async" / f"seed_{seed}",
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            calibration_epoch_rows.extend(a_rows)
        for depth_index, depth in enumerate(depths, start=1):
            if rank == 0:
                print(
                    f"calibration seed={seed}, B modulation_depth={depth:g}"
                )
            b_cfg = _condition_config(cfg, modulation_depth=depth)
            b_rows, _ = _simulate_and_analyze(
                condition_id=f"B_depth_{depth:g}",
                condition_cfg=b_cfg,
                action=actions["B_rhythmic_reference"],
                stimulate=False,
                seed=seed,
                action_index=depth_index,
                output_dir=(
                    root
                    / "calibration"
                    / f"B_depth_{depth:g}"
                    / f"seed_{seed}"
                ),
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                calibration_epoch_rows.extend(b_rows)
                calibration_seed_rows.append(
                    _calibration_seed_row(
                        seed=seed,
                        modulation_depth=depth,
                        asynchronous_rows=a_rows,
                        reference_rows=b_rows,
                        cfg=cfg,
                    )
                )

    if rank == 0:
        calibration_seed_frame = pd.DataFrame(calibration_seed_rows)
        calibration_summary = _calibration_summary(calibration_seed_frame, cfg)
        selected_depth, selected_from_safe_subset = _select_reference_depth(
            calibration_summary, cfg
        )
        pd.DataFrame(calibration_epoch_rows).to_csv(
            root / "calibration_epoch_metrics.csv", index=False
        )
        calibration_seed_frame.to_csv(
            root / "calibration_seed_metrics.csv", index=False
        )
        calibration_summary.to_csv(
            root / "calibration_reference_summary.csv", index=False
        )
    else:
        selected_depth = selected_from_safe_subset = None
    selected_depth = comm.bcast(selected_depth, root=0)
    selected_from_safe_subset = comm.bcast(selected_from_safe_subset, root=0)

    selected_reference = {
        "shared_circuit_settings": {
            "population_sizes": {
                "E": int(cfg.env.network.population.sizes.E),
                "I": int(cfg.env.network.population.sizes.I),
            },
            "inhibition_scale": float(cfg.analysis.inhibition_scale),
            "connection_probability": OmegaConf.to_container(
                cfg.env.network.connection_probability,
                resolve=True,
            ),
            "recurrent_weights_uS": OmegaConf.to_container(
                cfg.env.network.recurrent_weights,
                resolve=True,
            ),
            "afferent_drive": {
                population_name: {
                    "n_synapses_per_cell": int(
                        cfg.env.network.background[population_name].n_synapses
                    ),
                    "synaptic_weight_uS": float(
                        cfg.env.network.background[population_name].weight
                    ),
                    "mean_interval_ms": float(
                        cfg.env.network.background[population_name].interval_ms
                    ),
                    "mean_rate_hz_per_synapse": float(
                        1000.0
                        / cfg.env.network.background[population_name].interval_ms
                    ),
                }
                for population_name in ("E", "I")
            },
        },
        "condition_A": {
            "afferent_process": "homogeneous independent Poisson",
            "modulation_depth": 0.0,
            "rhythmic_rate_modulation": False,
        },
        "condition_B": {
            "afferent_process": (
                "independent Poisson with shared sinusoidal rate modulation"
            ),
            "modulation_depth": float(selected_depth),
            "frequency_hz": float(cfg.analysis.reference.frequency_hz),
            "phase_rad": float(cfg.analysis.reference.phase_rad),
            "populations": [
                str(value) for value in cfg.analysis.reference.populations
            ],
            "mean_afferent_rates_changed_from_A": False,
        },
        "identical_between_A_and_B": [
            "cell morphology and HH mechanisms",
            "E/I population sizes",
            "recurrent connectivity and weights",
            "mean afferent event rate and synaptic weight",
            "inhibition_scale=1.0",
        ],
        "tacs_action_applied_only_to_A": {
            "montage": str(cfg.analysis.tacs.axial_montage),
            "amplitude_v_per_m": float(cfg.analysis.tacs.amplitude_v_per_m),
            "frequency_hz": float(cfg.analysis.tacs.frequency_hz),
            "phase_rad": float(cfg.analysis.tacs.phase_rad),
        },
    }
    if rank == 0:
        with (root / "selected_reference.json").open("w", encoding="utf-8") as handle:
            json.dump(selected_reference, handle, indent=2)

    reference_cfg = _condition_config(
        cfg, modulation_depth=float(selected_depth)
    )
    validation_epoch_rows: list[dict[str, Any]] = []
    validation_seed_rows: list[dict[str, Any]] = []
    for seed in validation_seeds:
        rows_by_condition: dict[str, list[dict[str, Any]]] = {}
        raw_by_condition: dict[str, dict[str, np.ndarray]] = {}
        specifications = [
            ("A_async", asynchronous_cfg, False),
            ("B_rhythmic_reference", reference_cfg, False),
            ("A_tacs_axial", asynchronous_cfg, True),
            ("A_tacs_transverse", asynchronous_cfg, True),
        ]
        for action_index, (condition_id, condition_cfg, stimulate) in enumerate(
            specifications
        ):
            if rank == 0:
                print(f"validation seed={seed}, condition={condition_id}")
            condition_rows, condition_raw = _simulate_and_analyze(
                condition_id=condition_id,
                condition_cfg=condition_cfg,
                action=actions[condition_id],
                stimulate=stimulate,
                seed=seed,
                action_index=action_index,
                output_dir=(
                    root / "validation" / condition_id / f"seed_{seed}"
                ),
                comm=comm,
                size=size,
                rank=rank,
            )
            if rank == 0:
                rows_by_condition[condition_id] = condition_rows
                raw_by_condition[condition_id] = condition_raw
                validation_epoch_rows.extend(condition_rows)
        if rank == 0:
            validation_seed_rows.append(
                _validation_seed_row(
                    seed=seed,
                    rows_by_condition=rows_by_condition,
                    raw_by_condition=raw_by_condition,
                    cfg=cfg,
                )
            )

    if rank != 0:
        return

    epoch_frame = pd.DataFrame(validation_epoch_rows)
    seed_frame = pd.DataFrame(validation_seed_rows)
    epoch_frame.to_csv(root / "validation_epoch_metrics.csv", index=False)
    seed_frame.to_csv(root / "validation_seed_metrics.csv", index=False)
    if bool(cfg.experiment.plot):
        _plot_validation(epoch_frame, root)

    rng = np.random.default_rng(base_seed + 1_200_001)
    reference_shift = _summary(
        seed_frame,
        column="reference_shift_E_ppc",
        cfg=cfg,
        rng=rng,
    )
    tacs_gain = _summary(
        seed_frame,
        column="tacs_E_ppc_gain_difference_in_differences",
        cfg=cfg,
        rng=rng,
    )
    target_improvement = _summary(
        seed_frame,
        column="target_distance_improvement_E_ppc",
        cfg=cfg,
        rng=rng,
    )
    fractional_improvement = _summary(
        seed_frame,
        column="fractional_target_distance_improvement",
        cfg=cfg,
        rng=rng,
    )
    orientation = _summary(
        seed_frame,
        column="orientation_advantage_E_ppc_distance",
        cfg=cfg,
        rng=rng,
    )
    criteria = cfg.analysis.criteria
    checks = {
        "minimum_validation_seeds": int(seed_frame["seed"].nunique())
        >= int(criteria.minimum_validation_seeds),
        "reference_state_distinct": (
            reference_shift["mean"] >= float(criteria.minimum_reference_E_ppc_shift)
            and reference_shift["ci_2.5"] > 0.0
            and reference_shift["positive_seed_fraction"]
            >= float(criteria.minimum_positive_seed_fraction)
        ),
        "reference_rate_matched": float(
            seed_frame["B_rate_matched_to_A"].mean()
        )
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "reference_above_uniform_phase_null": float(
            seed_frame["B_E_plv_above_uniform_null"].mean()
        )
        >= float(criteria.minimum_above_null_seed_fraction),
        "positive_tacs_modulation": (
            tacs_gain["mean"] >= float(criteria.minimum_tacs_E_ppc_gain)
            and tacs_gain["ci_2.5"] > 0.0
            and tacs_gain["positive_seed_fraction"]
            >= float(criteria.minimum_positive_seed_fraction)
        ),
        "tacs_moves_A_toward_B": (
            target_improvement["mean"]
            >= float(criteria.minimum_target_distance_improvement_E_ppc)
            and target_improvement["ci_2.5"] > 0.0
            and target_improvement["positive_seed_fraction"]
            >= float(criteria.minimum_positive_seed_fraction)
        ),
        "reference_direction_alignment": float(
            seed_frame["reference_direction_aligned"].mean()
        )
        >= float(criteria.minimum_positive_seed_fraction),
        "tacs_above_uniform_phase_null": float(
            seed_frame["A_tacs_E_plv_above_uniform_null"].mean()
        )
        >= float(criteria.minimum_above_null_seed_fraction),
        "orientation_specific": orientation["ci_2.5"] > 0.0,
        "tacs_rate_safe": float(seed_frame["A_tacs_rate_safe"].mean())
        >= float(criteria.minimum_rate_safe_seed_fraction),
        "washout_reversible": float(seed_frame["washout_recovered"].mean())
        >= float(criteria.minimum_washout_recovery_seed_fraction),
        "baseline_causality": max(
            float(seed_frame["baseline_relative_rms_error_A_tacs_vs_A"].max()),
            float(seed_frame["baseline_relative_rms_error_transverse_vs_A"].max()),
        )
        <= float(criteria.maximum_baseline_relative_rms_error),
        "reference_selected_from_rate_matched_calibration": bool(
            selected_from_safe_subset
        ),
    }
    conclusion = {
        "scientific_scope": (
            "Acute movement from an asynchronous stochastic-drive state toward "
            "a separately generated rhythmically modulated synaptic reference. "
            "Not structural circuit conversion, depression, treatment, or a "
            "lasting after-effect."
        ),
        "selected_reference": selected_reference,
        "calibration_seeds": calibration_seeds,
        "validation_seeds": validation_seeds,
        "primary_feature": "E-population PPC at the reference frequency",
        "reference_shift": reference_shift,
        "tacs_modulation": tacs_gain,
        "target_distance_improvement": target_improvement,
        "fractional_target_distance_improvement": fractional_improvement,
        "orientation_advantage": orientation,
        "mean_rates_hz": {
            "A_E": float(seed_frame["A_E_rate_hz"].mean()),
            "B_E": float(seed_frame["B_E_rate_hz"].mean()),
            "A_tacs_E": float(seed_frame["A_tacs_E_rate_hz"].mean()),
            "A_I": float(seed_frame["A_I_rate_hz"].mean()),
            "B_I": float(seed_frame["B_I_rate_hz"].mean()),
            "A_tacs_I": float(seed_frame["A_tacs_I_rate_hz"].mean()),
        },
        "checks": checks,
        "minimal_entrainment_state_reachability_passed": bool(
            all(checks.values())
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    with (root / "experiment_conclusion.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(conclusion, handle, indent=2)

    print("\n### Selected B reference")
    print(json.dumps(selected_reference, indent=2))
    print("\n### Minimal entrainment-state checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nAcute A -> B-like entrainment: "
        f"{'PASSED' if conclusion['minimal_entrainment_state_reachability_passed'] else 'NOT PASSED'}"
    )
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
