"""H5-Dose-P0: bounded EEG-conditioned dose-opportunity mapping.

This exploratory system-identification experiment follows the negative H5-P2B
controller-profile map.  It freezes the H4-confirmed fast causal phase
controller and crosses the strength of the shared rhythmic afferent component
with sham, 0.1, 0.2, and 0.4 V/m.  It asks whether different constant doses are
preferable in different *observable* prestimulation EEG contexts.

The full-information dose oracle is a diagnostic, not a deployable policy.  A
prospective future split and leave-one-structure-out (LOSO) one-feature
threshold audit are included to avoid selecting and evaluating an action on
the same stochastic continuations.  Efficacy is evaluated on ideal neural EEG;
spikes, rates, dipoles, and representative membrane variables are mechanistic
or safety audits only.
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
from omegaconf import DictConfig, OmegaConf, open_dict
from scipy import signal


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _epoch_raw,
    _epoch_row,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _complex_response_decomposition,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    FULL,
    PARTIAL,
    RESPONSIVE,
    _augment_observation_rows,
    _future_seed,
    _run_controller,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_phase_tracker_response_mapping import (  # noqa: E402
    _load_sources as _load_p2b_upstream,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (  # noqa: E402
    _frozen_carrier_screen,
    _hash_locked_files,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    SHAM,
    _metric_rows,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    _augment_metric_rows,
    _json_ready,
    _profile,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_cadence_discovery import (  # noqa: E402
    _augment_common_audit,
)


ROOT_NAME = "h5_dose_opportunity"
EXPECTED_DOSES = [0.0, 0.1, 0.2, 0.4]


def _dose_id(dose: float) -> str:
    if np.isclose(dose, 0.0):
        return SHAM
    return f"dose_{dose:.1f}".replace(".", "p")


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Hash-lock the negative P2B result and all of its frozen ancestry."""
    sources = _load_p2b_upstream(cfg)
    root = Path(to_absolute_path(str(cfg.analysis.source_h5p2b.result_dir)))
    names = {
        "conclusion": "experiment_conclusion.json",
        "audit": "H5_P2B_response_mapping_audit.json",
        "screening": "prospective_screening.csv",
        "metrics": "context_controller_future_metrics.csv",
        "response_map": "controller_profile_response_map.csv",
        "associations": "EEG_feature_response_associations.csv",
        "provenance": "protocol_and_provenance.json",
    }
    files, hashes = _hash_locked_files(
        root, names, cfg.analysis.source_h5p2b.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    if (
        conclusion["conclusions"]["H5_P2B_active_response_mapping"] != "NOT PASSED"
        or bool(conclusion["conclusions"]["ready_for_H5_P2C_policy_development"])
    ):
        raise RuntimeError("H5-Dose-P0 requires the exact negative H5-P2B result.")
    source_seeds = set(sources["source_seed_union"])
    for key in ("screening", "metrics"):
        table = pd.read_csv(files[key])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed", "history_noise_seed", "future_noise_seed",
        ):
            if column in table:
                source_seeds.update(table[column].dropna().astype(int).tolist())
    sources.update({
        "roots": {**sources["roots"], "h5p2b": str(root)},
        "hashes": {**sources["hashes"], "h5p2b": hashes},
        "source_seed_union": source_seeds,
        "H5P2B_negative_preserved": True,
    })
    return sources


def _contexts(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    diffusion = cfg.analysis.states.phase_diffusion_levels[0]
    rows: list[dict[str, Any]] = []
    future_group = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        for history_index in range(int(block.n_history_seeds)):
            history_seed = (
                base + int(block.history_seed_offset)
                + 10 * structure_index + history_index
            )
            for frequency_index, frequency in enumerate(
                cfg.analysis.states.frequencies_hz
            ):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 10 * structure_index + frequency_index
                )
                paired_id = (
                    f"s{structure_index:02d}_h{history_index:02d}_"
                    f"f{int(round(float(frequency))):02d}"
                )
                trial_seed = base + int(block.trial_seed_offset) + future_group
                for shared_index, shared in enumerate(
                    cfg.analysis.states.shared_drive_levels
                ):
                    rows.append({
                        "context_order": len(rows),
                        "future_group_index": future_group,
                        "context_id": (
                            f"{paired_id}_q{shared_index:02d}_{shared.label}"
                        ),
                        "paired_shared_drive_context_id": paired_id,
                        "structure_index": structure_index,
                        "structure_seed": structure_seed,
                        "history_index": history_index,
                        "history_seed": history_seed,
                        "phase_seed": phase_seed,
                        "trial_seed": trial_seed,
                        "hidden_frequency_hz": float(frequency),
                        "label": str(diffusion.label),
                        "diffusion_rad2_per_s": float(
                            diffusion.diffusion_rad2_per_s
                        ),
                        "shared_drive_label": str(shared.label),
                        "shared_modulated_fraction": float(
                            shared.shared_modulated_fraction
                        ),
                    })
                future_group += 1
    if bool(cfg.analysis.smoke_test) and int(cfg.analysis.smoke_context_limit) > 0:
        return rows[:int(cfg.analysis.smoke_context_limit)]
    return rows


def _all_contexts(cfg: DictConfig) -> list[dict[str, Any]]:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(result):
        result.analysis.smoke_test = False
    return _contexts(result)


def _with_dose(cfg: DictConfig, dose: float, montage: str | None = None) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(result):
        result.analysis.actions.amplitude_v_per_m = float(dose)
        result.analysis.actions.controller_modes = [SHAM, RESPONSIVE]
        if montage is not None:
            result.analysis.tacs.axial_montage = str(montage)
    return result


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-Dose-P0 requires persistent online simulation.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-Dose-P0 may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-Dose-P0 freezes the 9/11-Hz carrier grid.")
    diffusions = list(cfg.analysis.states.phase_diffusion_levels)
    if len(diffusions) != 1 or not np.isclose(
        float(diffusions[0].diffusion_rad2_per_s), 0.5
    ):
        raise ValueError("H5-Dose-P0 fixes D=0.5 rad^2/s.")
    shared = [
        (str(x.label), float(x.shared_modulated_fraction))
        for x in cfg.analysis.states.shared_drive_levels
    ]
    if shared != [(PARTIAL, 0.5), (FULL, 1.0)]:
        raise ValueError("H5-Dose-P0 requires q={0.5,1.0}.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-Dose-P0 freezes modulation depth 0.04.")
    doses = [float(x) for x in cfg.analysis.actions.dose_actions_v_per_m]
    if doses != EXPECTED_DOSES:
        raise ValueError("H5-Dose-P0 action set must be sham/0.1/0.2/0.4 V/m.")
    if _profile(cfg, RESPONSIVE) != {
        "adaptive": True, "history_ms": 500.0, "update_interval_ms": 125.0,
    }:
        raise ValueError("The H4-confirmed fast phase controller changed.")
    if not bool(cfg.analysis.observation_noise.enabled) or not np.isclose(
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
        0.0,
    ):
        raise ValueError("This first mechanism map requires ideal observed EEG.")
    if float(cfg.env.simulation.obs_win_len) != 1000.0:
        raise ValueError("H5-Dose-P0 requires one-second outer windows.")
    endpoint_ms = (
        float(cfg.analysis.timeline.stimulation_steps)
        * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    )
    if not smoke and (
        int(cfg.analysis.timeline.baseline_steps) != 30
        or int(cfg.analysis.timeline.stimulation_steps) != 9
        or int(cfg.analysis.timeline.washout_steps) != 2
        or not np.isclose(endpoint_ms, 8000.0)
        or int(cfg.analysis.crossed_design.n_structure_seeds) != 3
        or int(cfg.analysis.crossed_design.n_history_seeds) != 1
        or int(cfg.analysis.crossed_design.n_future_continuations) != 4
    ):
        raise ValueError("Full H5-Dose-P0 requires the frozen 30/9/2-s 3x1x4 design.")
    if not np.isclose(float(sources["target"]["outcome_duration_s"]), 8.0):
        raise ValueError("Frozen population-B target is not the 8-s endpoint.")
    contexts = _all_contexts(cfg)
    expected = int(cfg.analysis.crossed_design.n_structure_seeds) * 2 * 2
    if len(contexts) != expected:
        raise ValueError("Crossed structure/frequency/shared-drive grid is incomplete.")
    namespaces = [
        {int(row[name]) for row in contexts}
        for name in ("structure_seed", "history_seed", "phase_seed", "trial_seed")
    ]
    namespaces.append({
        _future_seed(cfg, row, future)
        for row in contexts
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
    })
    if any(
        namespaces[i].intersection(namespaces[j])
        for i in range(len(namespaces)) for j in range(i + 1, len(namespaces))
    ):
        raise ValueError("H5-Dose-P0 seed namespaces overlap.")
    if set().union(*namespaces).intersection(sources["source_seed_union"]):
        raise ValueError("H5-Dose-P0 seeds overlap an upstream experiment.")
    if max(namespaces[0]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("Structure seed exceeds the uint32 connectivity mapping.")


def _screen(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    row, spectrum, temporal = _frozen_carrier_screen(
        episode, context, target, cfg
    )
    row.update({
        "paired_shared_drive_context_id": str(
            context["paired_shared_drive_context_id"]
        ),
        "observation_is_ideal_neural_EEG": True,
    })
    if bool(cfg.analysis.smoke_test) and bool(cfg.analysis.smoke_force_eligible):
        row.update({
            "eligible_before_smoke_override": bool(row["eligible"]),
            "eligible": True,
            "carrier_identified": True,
            "EEG_selected_frequency_hz": float(context["hidden_frequency_hz"]),
            "smoke_only_eligibility_override": True,
        })
    else:
        row["smoke_only_eligibility_override"] = False
    return row, spectrum, temporal


def _run_dose(
    *, condition_cfg: DictConfig, context: dict[str, Any], future_seed: int,
    future_index: int, dose: float, root: Path, comm: Any, size: int, rank: int,
    montage: str | None = None,
) -> dict[str, Any] | None:
    mode = SHAM if np.isclose(dose, 0.0) else RESPONSIVE
    dose_cfg = _with_dose(condition_cfg, dose, montage)
    episode = _run_controller(
        condition_cfg=dose_cfg,
        context=context,
        future_seed=future_seed,
        future_index=future_index,
        mode=mode,
        action_index=EXPECTED_DOSES.index(float(dose)),
        root=root / ("orientation" if montage else "dose_runs") / _dose_id(dose),
        comm=comm,
        size=size,
        rank=rank,
    )
    if episode is not None:
        episode["simulation"]["action"].update({
            "role": "H5_Dose_P0_fast_controller_dose_map",
            "dose_v_per_m": float(dose),
            "audit_montage": montage,
        })
    return episode


def _trimmed_signal(
    episode: dict[str, Any], epoch: str, cfg: DictConfig,
) -> tuple[np.ndarray, float, float]:
    raw = _epoch_raw(episode, epoch).astype(float)
    outputs = episode["simulation"]["outputs_by_epoch"][epoch]
    start_ms = float(outputs[0]["t_start_ms"])
    if epoch == "stimulation":
        trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
        samples = int(round(trim_ms / float(cfg.env.network.dt)))
        if samples:
            raw = raw[samples:-samples]
            start_ms += trim_ms
    return raw, start_ms, float(episode["simulator_fs_hz"])


def _eeg_coefficients(
    episode: dict[str, Any], frequency_hz: float, cfg: DictConfig,
) -> tuple[float, float]:
    raw, start_ms, fs_hz = _trimmed_signal(episode, "stimulation", cfg)
    processed, processed_fs, _, _, _ = _process_eeg(
        raw, simulator_fs_hz=fs_hz, cfg=cfg
    )
    return _fourier_coefficients(
        processed, fs_hz=processed_fs, start_ms=start_ms,
        frequency_hz=frequency_hz,
    )


def _dipole_coefficients(
    episode: dict[str, Any], frequency_hz: float, cfg: DictConfig,
) -> dict[str, float]:
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    chunks = []
    for output in outputs:
        probes = output.get("probe_data")
        if probes is None or len(probes) < 2:
            raise RuntimeError("Current-dipole probe missing from online output.")
        value = np.asarray(probes[1], dtype=float)
        if value.shape[0] != 3:
            raise RuntimeError(f"Expected (3,n) current dipole, got {value.shape}.")
        chunks.append(value)
    raw = np.concatenate(chunks, axis=1)
    trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    trim_samples = int(round(trim_ms / float(cfg.env.network.dt)))
    if trim_samples:
        raw = raw[:, trim_samples:-trim_samples]
    start_ms = float(outputs[0]["t_start_ms"]) + trim_ms
    fs_hz = 1000.0 / float(cfg.env.network.dt)
    coefficients = [
        _fourier_coefficients(
            raw[axis], fs_hz=fs_hz, start_ms=start_ms,
            frequency_hz=frequency_hz,
        )
        for axis in range(3)
    ]
    resultants = [float(np.hypot(*value)) for value in coefficients]
    return {
        "dipole_x_resultant_nA_um": resultants[0],
        "dipole_y_resultant_nA_um": resultants[1],
        "dipole_z_resultant_nA_um": resultants[2],
        "dipole_vector_resultant_nA_um": float(np.linalg.norm(resultants)),
    }


def _representative_state(
    episode: dict[str, Any], dose: float, context: dict[str, Any],
    future_index: int, frequency_hz: float, cfg: DictConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    representative_interval_ms = float(
        outputs[0]["representative_state_interval_ms"]
    )
    trim_samples = int(round(trim_ms / representative_interval_ms))
    start_ms = float(outputs[0]["t_start_ms"]) + trim_ms
    sites: dict[str, dict[str, list[np.ndarray]]] = {}
    for output in outputs:
        for site_id, values in output.get("representative_state", {}).items():
            target = sites.setdefault(site_id, {})
            for name, data in values.items():
                target.setdefault(name, []).append(np.asarray(data, dtype=float))
    summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    fs_hz = 1000.0 / representative_interval_ms
    representative = (
        int(context["structure_index"]) == 0
        and np.isclose(float(context["hidden_frequency_hz"]),
                       float(cfg.analysis.mechanism_audit.representative_frequency_hz))
        and (
            str(context["shared_drive_label"]) == FULL
            or bool(cfg.analysis.smoke_test)
        )
        and future_index == 0
    )
    stride = max(1, int(round(
        fs_hz / float(cfg.analysis.mechanism_audit.trace_resample_hz)
    )))
    for site_id, values in sites.items():
        joined = {name: np.concatenate(chunks) for name, chunks in values.items()}
        if trim_samples:
            joined = {name: value[trim_samples:-trim_samples]
                      for name, value in joined.items()}
        joined["apic_minus_soma_voltage_mV"] = (
            joined["apic_distal_voltage_mV"] - joined["soma_voltage_mV"]
        )
        for name, value in joined.items():
            cosine, sine = _fourier_coefficients(
                value, fs_hz=fs_hz, start_ms=start_ms,
                frequency_hz=frequency_hz,
            )
            summaries.append({
                "context_id": str(context["context_id"]),
                "structure_seed": int(context["structure_seed"]),
                "shared_drive_label": str(context["shared_drive_label"]),
                "future_index": int(future_index + 1),
                "dose_v_per_m": float(dose),
                "site_id": site_id,
                "signal": name,
                "mean": float(np.mean(value)),
                "sd": float(np.std(value)),
                "peak_to_peak": float(np.ptp(value)),
                "carrier_resultant": float(np.hypot(cosine, sine)),
            })
            if representative:
                times = (
                    start_ms + representative_interval_ms
                    + 1000.0 * np.arange(value.size) / fs_hz
                )
                for t_ms, sample in zip(times[::stride], value[::stride]):
                    traces.append({
                        "dose_v_per_m": float(dose), "site_id": site_id,
                        "signal": name, "time_ms": float(t_ms),
                        "value": float(sample),
                    })
    return summaries, traces


def _spectral_rows(
    episode: dict[str, Any], dose: float, context: dict[str, Any],
    future_index: int, cfg: DictConfig,
) -> list[dict[str, Any]]:
    if int(context["structure_index"]) != 0 or future_index != 0:
        return []
    raw, _, fs_hz = _trimmed_signal(episode, "stimulation", cfg)
    processed, fs_hz, _, _, _ = _process_eeg(
        raw, simulator_fs_hz=fs_hz, cfg=cfg
    )
    nperseg = min(len(processed), int(round(4.0 * fs_hz)))
    frequencies, psd = signal.welch(
        processed, fs=fs_hz, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density",
    )
    return [{
        "context_id": str(context["context_id"]),
        "hidden_frequency_hz": float(context["hidden_frequency_hz"]),
        "shared_drive_label": str(context["shared_drive_label"]),
        "dose_v_per_m": float(dose), "frequency_hz": float(frequency),
        "PSD_v2_per_hz": float(power),
    } for frequency, power in zip(frequencies, psd)]


def _dose_metric_rows(
    *, context: dict[str, Any], screening: dict[str, Any], future_index: int,
    future_seed: int, sham: dict[str, Any], active: dict[str, Any], dose: float,
    baseline_reference: dict[str, Any], target: dict[str, Any], cfg: DictConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair = {SHAM: sham, RESPONSIVE: active}
    pair_cfg = _with_dose(cfg, dose)
    rows, trajectories, updates = _metric_rows(
        context=context, screening=screening, future_index=future_index,
        future_seed=future_seed, episodes=pair,
        baseline_reference=baseline_reference, target=target, cfg=pair_cfg,
    )
    _augment_metric_rows(rows, pair, pair_cfg)
    _augment_common_audit(rows, pair, pair_cfg)
    _augment_observation_rows(rows, pair)
    label = _dose_id(dose)
    retained_mode = SHAM if np.isclose(dose, 0.0) else RESPONSIVE
    retained_rows = [
        row for row in rows if row["controller_mode"] == retained_mode
    ]
    retained_trajectories = [
        row for row in trajectories if row["controller_mode"] == retained_mode
    ]
    retained_updates = [
        row for row in updates if row["controller_mode"] == retained_mode
    ]
    for collection in (retained_rows, retained_trajectories, retained_updates):
        for row in collection:
            row["controller_profile"] = (
                SHAM if np.isclose(dose, 0.0) else RESPONSIVE
            )
            row["controller_mode"] = label
            row["dose_v_per_m"] = float(dose)
            row["shared_drive_label"] = str(context["shared_drive_label"])
            row["shared_modulated_fraction"] = float(
                context["shared_modulated_fraction"]
            )
            row["paired_shared_drive_context_id"] = str(
                context["paired_shared_drive_context_id"]
            )
            row["carrier_maximum_residual_evidence_db"] = float(
                screening["carrier_maximum_residual_evidence_db"]
            )
    sham_epoch = _epoch_row(sham, "stimulation")
    active_epoch = _epoch_row(active, "stimulation")
    for row in retained_rows:
        row.update({
            "post_E_ppc": float(active_epoch.E_ppc),
            "post_I_ppc": float(active_epoch.I_ppc),
            "hidden_E_ppc_reduction_vs_sham": float(
                sham_epoch.E_ppc - active_epoch.E_ppc
            ),
            "hidden_I_ppc_reduction_vs_sham": float(
                sham_epoch.I_ppc - active_epoch.I_ppc
            ),
            "spike_PPC_is_posthoc_not_policy_input": True,
        })
    return retained_rows, retained_trajectories, retained_updates


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "label", "diffusion_rad2_per_s",
        "shared_drive_label", "shared_modulated_fraction",
        "EEG_selected_frequency_hz", "carrier_maximum_residual_evidence_db",
        "dose_v_per_m", "controller_mode",
    ]
    return (
        metrics.groupby(group, as_index=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_post_distance_to_B_log10=("post_distance_to_B_log10", "mean"),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
            ),
            future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
            all_rate_safe=("rate_safe", "all"),
            all_physiological_washout_recovered=("field_removal_recovered", "all"),
            mean_phase_actionable_fraction=(
                "common_phase_estimate_actionable_fraction", "mean"
            ),
            maximum_field_boundary_discontinuity_v_per_m=(
                "maximum_field_boundary_discontinuity_v_per_m", "max"
            ),
            maximum_coherent_decomposition_error_v2=(
                "coherent_decomposition_error_v2", "max"
            ),
        )
        .sort_values(group).reset_index(drop=True)
    )


def _opportunity(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    active = expected[expected.dose_v_per_m > 0].copy()
    fixed = (
        active.groupby("dose_v_per_m", as_index=False)
        .expected_post_distance_to_B_log10.mean()
        .sort_values(["expected_post_distance_to_B_log10", "dose_v_per_m"])
    )
    best_fixed_dose = float(fixed.iloc[0].dose_v_per_m)
    best_fixed_distance = float(fixed.iloc[0].expected_post_distance_to_B_log10)
    best = (
        active.sort_values(["context_id", "expected_post_distance_to_B_log10",
                            "dose_v_per_m"])
        .groupby("context_id", as_index=False).first()
        .rename(columns={
            "dose_v_per_m": "expected_optimal_dose_v_per_m",
            "expected_post_distance_to_B_log10": "oracle_expected_distance_log10",
        })
    )
    fixed_by_context = active[np.isclose(active.dose_v_per_m, best_fixed_dose)][
        ["context_id", "expected_post_distance_to_B_log10"]
    ].rename(columns={
        "expected_post_distance_to_B_log10": "best_fixed_expected_distance_log10"
    })
    columns = [
        "context_id", "paired_shared_drive_context_id", "structure_seed",
        "hidden_frequency_hz", "shared_drive_label", "shared_modulated_fraction",
        "carrier_maximum_residual_evidence_db", "expected_optimal_dose_v_per_m",
        "oracle_expected_distance_log10",
    ]
    context_map = best[columns].merge(fixed_by_context, on="context_id")
    context_map["oracle_advantage_over_best_fixed_log10"] = (
        context_map.best_fixed_expected_distance_log10
        - context_map.oracle_expected_distance_log10
    )
    margin = float(cfg.analysis.criteria.practical_context_margin_log10)
    context_map["practical_alternative"] = (
        (~np.isclose(context_map.expected_optimal_dose_v_per_m, best_fixed_dose))
        & (context_map.oracle_advantage_over_best_fixed_log10 >= margin)
    )
    structure = (
        context_map.groupby("structure_seed", as_index=False)
        .agg(
            mean_oracle_advantage_log10=(
                "oracle_advantage_over_best_fixed_log10", "mean"
            ),
            practical_alternative_contexts=("practical_alternative", "sum"),
            optimal_dose_count=("expected_optimal_dose_v_per_m", "nunique"),
        )
    )

    selection = {int(x) for x in cfg.analysis.response_mapping.future_selection_indices}
    evaluation = {int(x) for x in cfg.analysis.response_mapping.future_evaluation_indices}
    available_futures = set(metrics.future_index.astype(int).unique())
    if bool(cfg.analysis.smoke_test) and not (
        selection.issubset(available_futures) and evaluation.issubset(available_futures)
    ):
        # A smoke exercises the analysis path with one continuation. Its
        # same-future diagnostic is explicitly noninferential; the full design
        # always retains the prespecified disjoint {1,2}/{3,4} split.
        selection = set(available_futures)
        evaluation = set(available_futures)
    split_rows = []
    for direction, learn, test in (
        ("first_half_to_second_half", selection, evaluation),
        ("second_half_to_first_half", evaluation, selection),
    ):
        learned = metrics[
            metrics.future_index.isin(learn) & (metrics.dose_v_per_m > 0)
        ].groupby(["context_id", "dose_v_per_m"], as_index=False).agg(
            distance=("post_distance_to_B_log10", "mean")
        )
        learned_choice = (
            learned.sort_values(["context_id", "distance", "dose_v_per_m"])
            .groupby("context_id", as_index=False).first()
            .rename(columns={"dose_v_per_m": "selected_dose_v_per_m"})
        )
        fixed_learn = (
            learned.groupby("dose_v_per_m", as_index=False).distance.mean()
            .sort_values(["distance", "dose_v_per_m"]).iloc[0]
        )
        fixed_dose = float(fixed_learn.dose_v_per_m)
        tested = metrics[
            metrics.future_index.isin(test) & (metrics.dose_v_per_m > 0)
        ]
        lookup = tested.groupby(
            ["context_id", "structure_seed", "dose_v_per_m"], as_index=False
        ).post_distance_to_B_log10.mean()
        for choice in learned_choice.itertuples():
            rows = lookup[lookup.context_id.eq(choice.context_id)]
            chosen = float(rows[np.isclose(
                rows.dose_v_per_m, choice.selected_dose_v_per_m
            )].post_distance_to_B_log10.iloc[0])
            fixed_value = float(rows[np.isclose(
                rows.dose_v_per_m, fixed_dose
            )].post_distance_to_B_log10.iloc[0])
            split_rows.append({
                "split_direction": direction,
                "context_id": choice.context_id,
                "structure_seed": int(rows.structure_seed.iloc[0]),
                "selected_dose_v_per_m": float(choice.selected_dose_v_per_m),
                "selection_best_fixed_dose_v_per_m": fixed_dose,
                "evaluation_selected_distance_log10": chosen,
                "evaluation_fixed_distance_log10": fixed_value,
                "evaluation_advantage_log10": fixed_value - chosen,
            })
    split = pd.DataFrame(split_rows)
    split_structure = split.groupby(
        ["split_direction", "structure_seed"], as_index=False
    ).evaluation_advantage_log10.mean()
    audit = {
        "best_fixed_active_dose_v_per_m": best_fixed_dose,
        "best_fixed_expected_distance_log10": best_fixed_distance,
        "oracle_expected_distance_log10": float(
            context_map.oracle_expected_distance_log10.mean()
        ),
        "mean_oracle_advantage_over_best_fixed_log10": float(
            structure.mean_oracle_advantage_log10.mean()
        ),
        "positive_structure_oracle_fraction": float(
            np.mean(structure.mean_oracle_advantage_log10 > 0)
        ),
        "optimal_dose_context_count": {
            str(float(key)): int(value) for key, value in
            context_map.expected_optimal_dose_v_per_m.value_counts().items()
        },
        "practical_alternative_context_count": int(
            context_map.practical_alternative.sum()
        ),
        "practical_alternative_structure_count": int(
            context_map.loc[context_map.practical_alternative,
                            "structure_seed"].nunique()
        ),
        "practical_alternative_shared_drive_count": int(
            context_map.loc[context_map.practical_alternative,
                            "shared_drive_label"].nunique()
        ),
        "future_split_mean_advantage_log10": float(
            split_structure.evaluation_advantage_log10.mean()
        ),
        "future_split_positive_structure_fraction": float(
            np.mean(split_structure.evaluation_advantage_log10 > 0)
        ),
        "future_split_selected_dose_count": int(
            split.selected_dose_v_per_m.nunique()
        ),
        "oracle_is_post_hoc_full_information": True,
    }
    return context_map, split, audit


def _fit_threshold(training: pd.DataFrame) -> dict[str, Any]:
    feature = "carrier_maximum_residual_evidence_db"
    doses = sorted(training.expected_optimal_dose_v_per_m.unique())
    # Retain the two doses with the greatest support. This is an exploratory
    # audit, and the retained pair must be frozen before later development.
    support = training.expected_optimal_dose_v_per_m.value_counts()
    retained = sorted(float(x) for x in support.index[:2])
    if len(retained) < 2:
        return {"available": False, "retained_doses": retained}
    values = np.sort(training[feature].unique())
    thresholds = [-np.inf, *list((values[:-1] + values[1:]) / 2.0), np.inf]
    best: dict[str, Any] | None = None
    for low_below in (True, False):
        for threshold in thresholds:
            chosen = np.where(
                (training[feature].to_numpy(float) <= threshold) == low_below,
                retained[0], retained[1],
            )
            # Use the full expected table supplied through frame attrs.
            lookup = training.attrs["dose_lookup"]
            loss = np.mean([
                float(lookup[(row.context_id, float(dose))])
                for row, dose in zip(training.itertuples(), chosen)
            ])
            candidate = {
                "available": True, "threshold": float(threshold),
                "low_dose_below_threshold": bool(low_below),
                "retained_doses": retained, "training_loss": float(loss),
            }
            if best is None or (loss, threshold, not low_below) < (
                best["training_loss"], best["threshold"],
                not best["low_dose_below_threshold"],
            ):
                best = candidate
    return best


def _loso_threshold(
    context_map: pd.DataFrame, expected: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    active = expected[expected.dose_v_per_m > 0]
    lookup = {
        (str(row.context_id), float(row.dose_v_per_m)):
            float(row.expected_post_distance_to_B_log10)
        for row in active.itertuples()
    }
    rows = []
    for held in sorted(context_map.structure_seed.unique()):
        train = context_map[context_map.structure_seed.ne(held)].copy()
        train.attrs["dose_lookup"] = lookup
        fitted = _fit_threshold(train)
        test = context_map[context_map.structure_seed.eq(held)]
        if not fitted["available"]:
            continue
        retained = fitted["retained_doses"]
        train_fixed = (
            active[active.structure_seed.ne(held) & active.dose_v_per_m.isin(retained)]
            .groupby("dose_v_per_m").expected_post_distance_to_B_log10.mean()
            .sort_values().index[0]
        )
        for sample in test.itertuples():
            below = sample.carrier_maximum_residual_evidence_db <= fitted["threshold"]
            selected = retained[0] if below == fitted["low_dose_below_threshold"] else retained[1]
            selected_loss = lookup[(sample.context_id, float(selected))]
            fixed_loss = lookup[(sample.context_id, float(train_fixed))]
            rows.append({
                "heldout_structure_seed": int(held),
                "context_id": str(sample.context_id),
                "feature_value": float(sample.carrier_maximum_residual_evidence_db),
                "threshold": float(fitted["threshold"]),
                "low_dose_below_threshold": bool(fitted["low_dose_below_threshold"]),
                "selected_dose_v_per_m": float(selected),
                "training_best_fixed_dose_v_per_m": float(train_fixed),
                "selected_distance_log10": float(selected_loss),
                "fixed_distance_log10": float(fixed_loss),
                "advantage_log10": float(fixed_loss - selected_loss),
            })
    result = pd.DataFrame(rows)
    if result.empty:
        return result, {"available": False}
    structure = result.groupby(
        "heldout_structure_seed", as_index=False
    ).advantage_log10.mean()
    return result, {
        "available": True,
        "mean_advantage_log10": float(structure.advantage_log10.mean()),
        "positive_structure_fraction": float(np.mean(structure.advantage_log10 > 0)),
        "selected_dose_count": int(result.selected_dose_v_per_m.nunique()),
        "feature": "carrier_maximum_residual_evidence_db",
        "exploratory_not_a_policy_confirmation": True,
    }


def _checks(
    *, screening: pd.DataFrame, metrics: pd.DataFrame, expected: pd.DataFrame,
    context_map: pd.DataFrame, audit: dict[str, Any],
    loso: dict[str, Any], sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.dose_v_per_m > 0]
    action_counts = context_map.expected_optimal_dose_v_per_m.value_counts()
    physiological_recovery = float(active.field_removal_recovered.mean())
    exact_removal = bool(np.all(active.final_extracellular_residual_mV == 0.0))
    accepted = screening[screening.carrier_identified]
    checks = {
        "source_H5P2B_negative_hash_locked": bool(sources["H5P2B_negative_preserved"]),
        "dose_map_seeds_disjoint_from_H1_H5P2B": True,
        "state_generator_distinct_from_tacs_action": True,
        "afferent_mean_rate_matched_across_shared_drive_by_construction": True,
        "complete_frequency_shared_drive_screening_grid": len(screening) == len(_contexts(cfg)),
        "screening_uses_only_predecision_ideal_EEG": bool(
            screening.observation_is_ideal_neural_EEG.all()
        ),
        "carrier_identification_coverage": float(
            screening.carrier_identified.mean()
        ) >= float(criteria.minimum_carrier_identification_coverage),
        "accepted_carrier_accuracy": bool(len(accepted)) and float(
            accepted.EEG_frequency_selection_correct.mean()
        ) >= float(criteria.minimum_accepted_carrier_accuracy),
        "minimum_eligible_contexts": len(eligible) >= int(criteria.minimum_eligible_contexts)
        or bool(cfg.analysis.smoke_test),
        "minimum_independent_structures": eligible.structure_seed.nunique()
        >= int(criteria.minimum_structure_seeds) or bool(cfg.analysis.smoke_test),
        "both_frequencies_and_shared_drive_levels_enrolled": bool(
            eligible.hidden_frequency_hz.nunique() == 2
            and eligible.shared_drive_label.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "multiple_independent_paired_postdecision_futures": bool(
            metrics.groupby(["context_id", "dose_v_per_m"]).future_index.nunique().min()
            >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_doses_and_futures": bool(
            metrics.groupby("context_id").observed_baseline_sha256.nunique().max() == 1
        ),
        "paired_baseline_numeric_identity": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "all_active_arms_use_frozen_fast_phase_controller": bool(
            active.controller_profile.eq(RESPONSIVE).all()
        ),
        "single_constant_dose_per_intervention": True,
        "phase_updates_use_only_preceding_EEG": bool(active.all_phase_estimates_causal.all()),
        "phase_correction_is_frequency_bounded": bool(
            active.maximum_abs_frequency_correction_hz.max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_updates": bool(
            active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "common_phase_estimates_actionable": bool(
            active.common_phase_estimate_actionable_fraction.mean()
            >= float(criteria.minimum_common_phase_estimate_actionable_fraction)
        ),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "exact_field_removal_confirmed": exact_removal,
        "physiological_washout_recovery_audit": physiological_recovery
        >= float(criteria.minimum_physiological_washout_recovery_fraction),
        "expected_oracle_uses_multiple_active_doses": int(len(action_counts))
        >= int(criteria.minimum_expected_active_dose_count),
        "practical_alternative_contexts_present": int(
            audit["practical_alternative_context_count"]
        ) >= int(criteria.minimum_practical_alternative_contexts),
        "practical_alternatives_cross_structures": int(
            audit["practical_alternative_structure_count"]
        ) >= int(criteria.minimum_practical_alternative_structures),
        "practical_alternatives_cross_shared_drive_levels": int(
            audit["practical_alternative_shared_drive_count"]
        ) >= int(criteria.minimum_practical_alternative_shared_drive_levels),
        "oracle_has_practical_advantage_over_best_fixed": float(
            audit["mean_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_expected_oracle_advantage_log10),
        "preferred_oracle_headroom_reached": float(
            audit["mean_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.preferred_oracle_headroom_log10),
        "independent_future_split_has_practical_advantage": float(
            audit["future_split_mean_advantage_log10"]
        ) >= float(criteria.minimum_future_split_advantage_log10),
        "future_split_advantage_cross_structure": float(
            audit["future_split_positive_structure_fraction"]
        ) >= float(criteria.minimum_future_split_positive_structure_fraction),
        "exploratory_EEG_threshold_uses_multiple_doses": bool(
            loso.get("selected_dose_count", 0)
            >= int(criteria.minimum_loso_selected_dose_count)
        ),
        "exploratory_EEG_threshold_beats_fixed_directionally": float(
            loso.get("mean_advantage_log10", -np.inf)
        ) > float(criteria.minimum_loso_threshold_advantage_log10),
        "coherent_energy_decomposition_exact": bool(
            expected.maximum_coherent_decomposition_error_v2.abs().max()
            <= float(criteria.maximum_coherent_decomposition_error_v2)
        ),
        "efficacy_uses_neural_EEG_and_policy_inputs_exclude_spikes": True,
    }
    mandatory = [
        "expected_oracle_uses_multiple_active_doses",
        "practical_alternative_contexts_present",
        "practical_alternatives_cross_structures",
        "practical_alternatives_cross_shared_drive_levels",
        "oracle_has_practical_advantage_over_best_fixed",
        "independent_future_split_has_practical_advantage",
        "future_split_advantage_cross_structure",
        "exploratory_EEG_threshold_uses_multiple_doses",
        "exploratory_EEG_threshold_beats_fixed_directionally",
    ]
    passed = all(checks[name] for name in checks if name != "preferred_oracle_headroom_reached")
    conclusions = {
        "H5_Dose_P0_contextual_dose_opportunity": "PASSED" if passed else "NOT PASSED",
        "ready_for_H5_dose_policy_development": bool(passed),
        "machine_learning_policy_status": "NOT TRAINED OR TESTED",
        "failed_opportunity_checks": [name for name in mandatory if not checks[name]],
        "preferred_headroom_is_advisory": True,
    }
    return checks, conclusions


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plots(
    *, root: Path, spectra: pd.DataFrame, expected: pd.DataFrame,
    context_map: pd.DataFrame, split: pd.DataFrame, metrics: pd.DataFrame,
    decomposition: pd.DataFrame, states: pd.DataFrame,
) -> None:
    tiny = np.finfo(float).tiny
    if not spectra.empty:
        figure, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for axis, (label, group) in zip(axes, spectra.groupby("shared_drive_label")):
            for dose, arm in group.groupby("dose_v_per_m"):
                summary = arm.groupby("frequency_hz").PSD_v2_per_hz.mean()
                keep = (summary.index >= 5) & (summary.index <= 15)
                axis.plot(summary.index[keep], 10 * np.log10(np.maximum(
                    summary.to_numpy()[keep], tiny)), label=f"{dose:g} V/m")
            axis.set(title=label, xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)")
            axis.legend(frameon=False, fontsize=8)
        _save_figure(figure, root, "figure_01_representative_stimulation_PSD")

    figure, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, (label, group) in zip(axes, expected.groupby("shared_drive_label")):
        active = group[group.dose_v_per_m > 0]
        for structure, values in active.groupby("structure_seed"):
            summary = values.groupby("dose_v_per_m").expected_post_distance_to_B_log10.mean()
            axis.plot(summary.index, summary, marker="o", alpha=0.7, label=str(structure))
        axis.set(title=label, xlabel="Dose (V/m)", ylabel="Distance to B (log10)")
    _save_figure(figure, root, "figure_02_contextual_dose_response")

    figure, axis = plt.subplots(figsize=(7, 4))
    for structure, group in context_map.groupby("structure_seed"):
        axis.scatter(group.carrier_maximum_residual_evidence_db,
                     group.expected_optimal_dose_v_per_m, label=str(structure))
    axis.set(xlabel="Prestimulation carrier residual evidence (dB)",
             ylabel="Expected optimal dose (V/m)",
             title="EEG context and full-information dose preference")
    axis.legend(title="Structure", frameon=False, fontsize=8)
    _save_figure(figure, root, "figure_03_EEG_context_dose_preference")

    figure, axis = plt.subplots(figsize=(7, 4))
    values = split.groupby("split_direction").evaluation_advantage_log10.mean()
    axis.bar(values.index, values.values, color=["#4c78a8", "#f58518"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="Held-out-future advantage over learned fixed dose (log10)",
             title="Independent-future response replication")
    axis.tick_params(axis="x", rotation=12)
    _save_figure(figure, root, "figure_04_future_split_validation")

    if not decomposition.empty:
        summary = decomposition.groupby("dose_v_per_m")[
            ["coherent_interference_cross_term_fraction",
             "coherent_induced_component_fraction",
             "coherent_net_change_fraction"]
        ].mean()
        figure, axis = plt.subplots(figsize=(7, 4))
        summary.plot(kind="bar", ax=axis)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(xlabel="Dose (V/m)", ylabel="Fraction of sham coherent energy",
                 title="Neural-EEG coherent energy decomposition")
        axis.legend(["interference", "induced", "net"], frameon=False)
        _save_figure(figure, root, "figure_05_coherent_energy_mechanism")

    if not states.empty:
        chosen = states[states.signal.eq("apic_minus_soma_voltage_mV")]
        if not chosen.empty:
            summary = chosen.groupby("dose_v_per_m").carrier_resultant.mean()
            figure, axis = plt.subplots(figsize=(7, 4))
            axis.plot(summary.index, summary.values, marker="o")
            axis.set(xlabel="Dose (V/m)", ylabel="Carrier voltage resultant (mV)",
                     title="Representative somatodendritic polarization")
            _save_figure(figure, root, "figure_06_cellular_polarization_audit")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    rates = metrics.groupby("dose_v_per_m")[[
        "post_E_firing_rate_hz", "post_I_firing_rate_hz"
    ]].mean()
    rates.plot(marker="o", ax=axes[0])
    axes[0].set(xlabel="Dose (V/m)", ylabel="Firing rate (Hz)",
                title="Rate safety")
    axes[0].legend(["E", "I"], frameon=False)
    ppc = metrics.groupby("dose_v_per_m")[[
        "hidden_E_ppc_reduction_vs_sham", "hidden_I_ppc_reduction_vs_sham"
    ]].mean()
    ppc.plot(marker="o", ax=axes[1])
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(xlabel="Dose (V/m)", ylabel="PPC reduction vs sham",
                title="Hidden spike-timing audit")
    axes[1].legend(["E", "I"], frameon=False)
    _save_figure(figure, root, "figure_07_rate_and_spike_timing")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    representative_state_requested = bool(
        cfg.analysis.mechanism_audit.record_representative_state
    )
    with open_dict(cfg):
        # The total dipole is recorded for every episode. High-rate membrane
        # variables are intentionally limited to one prespecified mechanism
        # context so this audit does not dominate the bounded experiment.
        cfg.env.online.record_representative_state = False
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / ROOT_NAME
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-Dose-P0 bounded contextual dose opportunity")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()
    target = sources["target"]

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    pre_spectra: list[pd.DataFrame] = []
    temporal_rows: list[pd.DataFrame] = []
    decomposition_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    orientation_rows: list[dict[str, Any]] = []
    contexts = _contexts(cfg)

    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"q={context['shared_modulated_fraction']:g}"
            )
        state_cfg = _with_context_state(cfg, context)
        record_this_context = bool(
            representative_state_requested
            and int(context["structure_index"]) == 0
            and np.isclose(
                float(context["hidden_frequency_hz"]),
                float(cfg.analysis.mechanism_audit.representative_frequency_hz),
            )
            and (
                str(context["shared_drive_label"]) == FULL
                or bool(cfg.analysis.smoke_test)
            )
        )
        with open_dict(state_cfg):
            state_cfg.env.online.record_representative_state = record_this_context
        first_future = _future_seed(state_cfg, context, 0)
        baseline_reference = _run_dose(
            condition_cfg=state_cfg, context=context, future_seed=first_future,
            future_index=0, dose=0.0, root=root, comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            screening, spectrum, temporal = _screen(
                baseline_reference, context, target, state_cfg
            )
            screening_rows.append(screening)
            if int(context["structure_index"]) == 0:
                pre_spectra.append(spectrum)
                temporal_rows.append(temporal)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'SHAM FALLBACK'}; "
                f"selected={selected_frequency:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            del baseline_reference
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(action_cfg, context, future_index)
            if future_index == 0:
                sham = baseline_reference
            else:
                sham = _run_dose(
                    condition_cfg=action_cfg, context=context,
                    future_seed=future_seed, future_index=future_index, dose=0.0,
                    root=root, comm=comm, size=size, rank=rank,
                )
            episodes: dict[float, dict[str, Any]] | None = ({0.0: sham} if rank == 0 else None)
            for dose in EXPECTED_DOSES[1:]:
                episode = _run_dose(
                    condition_cfg=action_cfg, context=context,
                    future_seed=future_seed, future_index=future_index, dose=dose,
                    root=root, comm=comm, size=size, rank=rank,
                )
                if rank == 0:
                    episodes[dose] = episode
            if rank == 0:
                sham_cos, sham_sin = _eeg_coefficients(
                    sham, selected_frequency, action_cfg
                )
                for dose in EXPECTED_DOSES:
                    episode = episodes[dose]
                    rows, trajectories, updates = _dose_metric_rows(
                        context=context, screening=screening,
                        future_index=future_index, future_seed=future_seed,
                        sham=sham, active=episode, dose=dose,
                        baseline_reference=baseline_reference, target=target,
                        cfg=action_cfg,
                    )
                    active_cos, active_sin = _eeg_coefficients(
                        episode, selected_frequency, action_cfg
                    )
                    decomposition = _complex_response_decomposition(
                        sham_cosine=sham_cos, sham_sine=sham_sin,
                        active_cosine=active_cos, active_sine=active_sin,
                    )
                    dipole = _dipole_coefficients(
                        episode, selected_frequency, action_cfg
                    )
                    for row in rows:
                        row.update(decomposition)
                        row.update(dipole)
                        row["exact_field_removed"] = bool(
                            row["final_extracellular_residual_mV"] == 0.0
                        )
                    metric_rows.extend(rows)
                    trajectory_rows.extend(trajectories)
                    update_rows.extend(updates)
                    decomposition_rows.append({
                        "context_id": context["context_id"],
                        "structure_seed": context["structure_seed"],
                        "shared_drive_label": context["shared_drive_label"],
                        "future_index": future_index + 1,
                        "dose_v_per_m": dose, **decomposition, **dipole,
                    })
                    summaries, traces = _representative_state(
                        episode, dose, context, future_index,
                        selected_frequency, action_cfg,
                    )
                    state_rows.extend(summaries)
                    trace_rows.extend(traces)
                    spectrum_rows.extend(_spectral_rows(
                        episode, dose, context, future_index, action_cfg
                    ))

                if (
                    bool(cfg.analysis.mechanism_audit.orientation_control_enabled)
                    and int(context["structure_index"]) == 0
                    and np.isclose(context["hidden_frequency_hz"],
                                   cfg.analysis.mechanism_audit.representative_frequency_hz)
                    and (
                        context["shared_drive_label"] == FULL
                        or bool(cfg.analysis.smoke_test)
                    )
                    and future_index == 0
                ):
                    orientation_dose = float(
                        cfg.analysis.mechanism_audit.orientation_control_dose_v_per_m
                    )
                    # Executed below on all ranks; retain only its root metric.
                else:
                    orientation_dose = None
            else:
                orientation_dose = None
            orientation_dose = comm.bcast(orientation_dose, root=0)
            if orientation_dose is not None:
                transverse = _run_dose(
                    condition_cfg=action_cfg, context=context,
                    future_seed=future_seed, future_index=future_index,
                    dose=orientation_dose, root=root, comm=comm, size=size,
                    rank=rank,
                    montage=str(cfg.analysis.mechanism_audit.orientation_control_montage),
                )
                if rank == 0:
                    axial = episodes[orientation_dose]
                    for montage, episode in (
                        ("axial", axial),
                        (str(cfg.analysis.mechanism_audit.orientation_control_montage),
                         transverse),
                    ):
                        row = _epoch_row(episode, "stimulation")
                        orientation_rows.append({
                            "montage": montage,
                            "dose_v_per_m": orientation_dose,
                            "log10_alpha_power": float(row.log10_alpha_power_8_12_hz),
                            "E_firing_rate_hz": float(row.E_firing_rate_hz),
                            "I_firing_rate_hz": float(row.I_firing_rate_hz),
                        })
        del baseline_reference

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    if pre_spectra:
        pd.concat(pre_spectra, ignore_index=True).to_csv(
            root / "representative_predecision_spectra.csv", index=False
        )
    if temporal_rows:
        pd.concat(temporal_rows, ignore_index=True).to_csv(
            root / "representative_predecision_temporal_evidence.csv", index=False
        )
    if not metric_rows:
        conclusion = {
            "scope": "H5-Dose-P0 bounded dose opportunity",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_Dose_P0_contextual_dose_opportunity": "NOT PASSED",
                "ready_for_H5_dose_policy_development": False,
                "machine_learning_policy_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(conclusion, indent=2))
        print("No eligible contexts; stopped after prospective screening.")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    trajectories = pd.DataFrame(trajectory_rows)
    updates = pd.DataFrame(update_rows)
    spectra = pd.DataFrame(spectrum_rows)
    decomposition = pd.DataFrame(decomposition_rows)
    states = pd.DataFrame(state_rows)
    traces = pd.DataFrame(trace_rows)
    orientation = pd.DataFrame(orientation_rows)
    expected = _expected_map(metrics)
    context_map, split, audit = _opportunity(expected, metrics, cfg)
    threshold_rows, threshold_audit = _loso_threshold(context_map, expected)
    checks, conclusions = _checks(
        screening=screening, metrics=metrics, expected=expected,
        context_map=context_map, audit=audit,
        loso=threshold_audit, sources=sources, cfg=cfg,
    )

    metrics.to_csv(root / "context_dose_future_metrics.csv", index=False)
    trajectories.to_csv(root / "one_second_EEG_trajectories.csv", index=False)
    updates.to_csv(root / "causal_phase_updates.csv", index=False)
    spectra.to_csv(root / "representative_stimulation_spectra.csv", index=False)
    decomposition.to_csv(root / "current_dipole_mechanism_audit.csv", index=False)
    states.to_csv(root / "representative_membrane_state_summary.csv", index=False)
    traces.to_csv(root / "representative_membrane_state_traces.csv", index=False)
    orientation.to_csv(root / "orientation_control.csv", index=False)
    expected.to_csv(root / "expected_context_dose_map.csv", index=False)
    context_map.to_csv(root / "dose_response_opportunity.csv", index=False)
    split.to_csv(root / "independent_future_split_validation.csv", index=False)
    threshold_rows.to_csv(root / "exploratory_EEG_threshold_LOSO.csv", index=False)
    audit_payload = {
        "dose_opportunity": audit,
        "exploratory_EEG_threshold": threshold_audit,
        "orientation_control": orientation.to_dict("records"),
    }
    (root / "H5_Dose_P0_opportunity_audit.json").write_text(json.dumps(
        _json_ready(audit_payload), indent=2, allow_nan=False
    ))
    provenance = {
        "experiment": "H5_Dose_P0_bounded_contextual_dose_opportunity",
        "frozen_sources": {"roots": sources["roots"], "hashes": sources["hashes"]},
        "frozen_population_B_target": target,
        "state_generator": {
            "carrier_hz": [9.0, 11.0], "phase_diffusion_rad2_per_s": 0.5,
            "modulation_depth": 0.04,
            "shared_modulated_afferent_fraction": [0.5, 1.0],
            "mean_afferent_rate_matched": True,
            "private_Poisson_streams_independent": True,
        },
        "causal_protocol": {
            "burn_in_s": 1, "predecision_EEG_s": 30,
            "stimulation_s": 9, "central_endpoint_s": 8, "washout_s": 2,
            "ramp_s_each_edge": 0.5,
            "doses_v_per_m": EXPECTED_DOSES,
            "carrier_estimator": str(cfg.analysis.response_mapping.frozen_estimator),
            "controller_profile": _profile(cfg, RESPONSIVE),
            "relative_phase_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
            "montage": str(cfg.analysis.tacs.axial_montage),
            "controller_observation": "ideal_neural_EEG",
            "efficacy_observation": "ideal_neural_EEG",
        },
        "design": {
            "independent_structures": int(screening.structure_seed.nunique()),
            "screened_contexts": int(len(screening)),
            "eligible_contexts": int(screening.eligible.sum()),
            "paired_futures": int(cfg.analysis.crossed_design.n_future_continuations),
            "main_action_future_outcomes_if_all_eligible": int(len(screening) * 4 * 4),
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Exploratory full-information mechanism and response mapping only; "
            "no ML policy is trained or confirmed. Field dose is a tissue-level "
            "simulator setting, not a clinical safety prescription."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-Dose-P0 bounded contextual dose opportunity",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure",
        "inference_boundary": provenance["inference_boundary"],
    }
    (root / "experiment_conclusion.json").write_text(json.dumps(
        _json_ready(conclusion), indent=2, allow_nan=False
    ))
    if bool(cfg.experiment.plot):
        _plots(
            root=root, spectra=spectra, expected=expected,
            context_map=context_map, split=split, metrics=metrics,
            decomposition=decomposition, states=states,
        )

    print("\n### H5-Dose-P0 screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H5-Dose-P0 feasibility checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### H5-Dose-P0 opportunity summary")
    print(json.dumps(_json_ready(audit_payload), indent=2, allow_nan=False))
    print(
        "\nContextual dose opportunity: "
        f"{conclusions['H5_Dose_P0_contextual_dose_opportunity']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
