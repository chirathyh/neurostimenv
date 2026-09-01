"""D1 full-information tACS action map for EEG-visible phase diffusion.

D0b confirmed on disjoint circuit structures that the one-step coherence of
successive, carrier-demodulated EEG phase increments (C1) distinguishes the
frozen low- and high-diffusion afferent states.  D1 asks the next causal
question: does this stimulation-free EEG context predict which *single*
constant action from sham, 0.2, or 0.4 V/m best moves the ideal neural EEG
toward a homogeneous-Poisson reference?

Frequency is selected from the predecision EEG and every active arm uses the
same frozen EEG-relative antiphase convention.  The full twelve-second history
estimates C1, while only the latest one-second EEG initializes stimulation
phase.  Counterfactual actions share an identical history and are averaged
over independent postdecision Poisson and phase-diffusion futures.

This is exploratory system identification.  It does not train a bandit, test a
frozen policy, model a disorder, or provide evidence about human tACS.
"""

from __future__ import annotations

import hashlib
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


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _action,
    _condition_for_seed,
    _epoch_raw,
    _epoch_row,
    _estimate_relative_field_phase,
    _phase_estimation_outputs,
    _plain,
    _run_condition,
    _sham,
    _wrap_phase,
)
from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression_dose_audit import (  # noqa: E402
    _field_removal_status,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _episode_feature,
    _frequency_token,
    _with_action_frequency,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_diffusion_validation import (  # noqa: E402
    _periodogram_metrics,
    _with_diffusion_state,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_increment_confirmation import (  # noqa: E402
    _window_phase_metrics,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)


LOW = "low_diffusion"
HIGH = "high_diffusion"
SHAM = 0.0


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    OmegaConf.set_struct(result, False)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    """Convert scientific outputs to strict, portable JSON values."""
    value = _plain(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _dose_id(dose: float) -> str:
    if np.isclose(float(dose), 0.0):
        return "sham"
    token = f"{float(dose):.3f}".rstrip("0").rstrip(".").replace(".", "p")
    return f"matched_antiphase_{token}_vpm"


def _doses(cfg: DictConfig) -> list[float]:
    return [
        float(cfg.analysis.actions.sham_dose_v_per_m),
        *[float(value) for value in cfg.analysis.actions.active_doses_v_per_m],
    ]


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    d0_root = Path(to_absolute_path(str(cfg.analysis.source_d0b.result_dir)))
    f0_root = Path(to_absolute_path(str(cfg.analysis.source_f0.result_dir)))
    files = {
        "d0b": {
            "conclusion": d0_root / "experiment_conclusion.json",
            "provenance": d0_root / "frozen_endpoint_provenance.json",
            "metrics": d0_root / "confirmation_eeg_metrics.csv",
        },
        "f0": {
            "conclusion": f0_root / "experiment_conclusion.json",
            "provenance": f0_root / "protocol_and_provenance.json",
            "metrics": f0_root / "context_action_future_metrics.csv",
            "calibration": f0_root / "reference_B_calibration.csv",
        },
    }
    missing = [str(path) for group in files.values() for path in group.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen D1 sources: {missing}")

    hashes: dict[str, dict[str, str]] = {}
    for name, group in files.items():
        observed = {key: _sha256(path) for key, path in group.items()}
        expected_block = cfg.analysis[f"source_{name}"].expected_sha256
        expected = {key: str(expected_block[key]) for key in group}
        if observed != expected:
            raise RuntimeError(
                f"{name.upper()} source hash mismatch: expected={expected}, observed={observed}"
            )
        hashes[name] = observed

    d0_conclusion = json.loads(files["d0b"]["conclusion"].read_text())
    d0_provenance = json.loads(files["d0b"]["provenance"].read_text())
    if not bool(d0_conclusion["summary"]["D0b_phase_increment_observability_confirmed"]):
        raise RuntimeError("D1 requires the frozen D0b EEG observability confirmation.")
    if not bool(d0_conclusion["summary"]["ready_for_D1_system_identification"]):
        raise RuntimeError("The frozen D0b source did not authorize D1 system identification.")
    if any(not bool(value) for value in d0_conclusion["checks"].values()):
        raise RuntimeError("A frozen D0b confirmation check is no longer positive.")
    endpoint = d0_provenance["frozen_endpoint"]
    generator = d0_provenance["frozen_generator"]
    if (
        str(endpoint["name"]) != "eeg_lag1_phase_increment_coherence_real"
        or not np.isclose(float(endpoint["phase_window_s"]), 1.0)
        or not np.isclose(float(endpoint["classification_threshold"]), 0.5180938906862573)
        or [float(x) for x in generator["frequencies_hz"]] != [9.0, 11.0]
        or not np.isclose(float(generator["modulation_depth"]), 0.04)
    ):
        raise RuntimeError("The frozen D0b endpoint or generator changed.")

    f0_conclusion = json.loads(files["f0"]["conclusion"].read_text())
    f0_provenance = json.loads(files["f0"]["provenance"].read_text())
    if not bool(
        f0_conclusion["conclusions"]["frequency_phase_contextual_feasibility_gate_passed"]
    ):
        raise RuntimeError("D1 requires the positive frozen F0 frequency/phase map.")
    required_f0 = {
        "matched_antiphase_improves_over_sham",
        "frequency_specific_crossover",
        "phase_specific_crossover",
        "frequency_identified_from_predecision_eeg",
        "action_phase_tracks_predecision_eeg",
        "all_actions_rate_safe",
    }
    if any(not bool(f0_conclusion["checks"].get(name, False)) for name in required_f0):
        raise RuntimeError("A required frozen F0 mechanism check is not positive.")
    if "pi EEG-relative phase" not in str(f0_provenance["candidate_eeg_rule"]):
        raise RuntimeError("The frozen F0 EEG-relative antiphase rule changed.")

    d0_metrics = pd.read_csv(files["d0b"]["metrics"])
    f0_metrics = pd.read_csv(files["f0"]["metrics"])
    f0_calibration = pd.read_csv(files["f0"]["calibration"])
    source_seeds: set[int] = set()
    for table in (d0_metrics, f0_metrics):
        for column in (
            "structure_seed", "drive_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                source_seeds.update(table[column].dropna().astype(int).tolist())
    if "seed" in f0_calibration:
        source_seeds.update(f0_calibration.seed.dropna().astype(int).tolist())
    return {
        "roots": {"d0b": str(d0_root), "f0": str(f0_root)},
        "hashes": hashes,
        "source_seed_union": source_seeds,
        "frozen_C1_threshold": float(endpoint["classification_threshold"]),
        "D0b_confirmed": True,
        "F0_passed": True,
    }


def _reference_seeds(cfg: DictConfig) -> list[int]:
    block = cfg.analysis.reference_calibration
    first = int(cfg.experiment.seed) + int(block.seed_offset)
    return [first + index for index in range(int(block.n_seeds))]


def _levels(cfg: DictConfig) -> list[dict[str, Any]]:
    return [
        {
            "label": str(level.label),
            "diffusion_rad2_per_s": float(level.diffusion_rad2_per_s),
        }
        for level in cfg.analysis.states.phase_diffusion_levels
    ]


def _context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    rows: list[dict[str, Any]] = []
    pair_order = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        for history_index in range(int(block.n_history_seeds)):
            history_seed = (
                base + int(block.history_seed_offset)
                + 10 * structure_index + history_index
            )
            for frequency_index, frequency_hz in enumerate(cfg.analysis.states.frequencies_hz):
                phase_seed = (
                    base + int(block.phase_seed_offset)
                    + 20 * structure_index + 2 * history_index + frequency_index
                )
                paired_id = (
                    f"s{structure_index:02d}_h{history_index:02d}_"
                    f"f{_frequency_token(float(frequency_hz))}"
                )
                for level in _levels(cfg):
                    order = len(rows)
                    rows.append({
                        "context_order": order,
                        "future_group_index": pair_order,
                        "context_id": f"{paired_id}_{level['label']}",
                        "paired_diffusion_context_id": paired_id,
                        "structure_index": structure_index,
                        "structure_seed": structure_seed,
                        "history_index": history_index,
                        "history_seed": history_seed,
                        "phase_seed": phase_seed,
                        "trial_seed": base + int(block.trial_seed_offset) + order,
                        "hidden_frequency_hz": float(frequency_hz),
                        **level,
                    })
                pair_order += 1
    return rows


def _run_context_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    rows = _context_specs(cfg)
    limit = int(cfg.analysis.smoke_context_limit)
    if bool(cfg.analysis.smoke_test) and limit > 0:
        return rows[:limit]
    return rows


def _future_seed(cfg: DictConfig, context: dict[str, Any], future_index: int) -> int:
    return (
        int(cfg.experiment.seed)
        + int(cfg.analysis.crossed_design.future_seed_offset)
        + 100 * int(context["future_group_index"])
        + int(future_index)
    )


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("D1 requires the persistent online simulator.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("D1 may not change recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("D1 freezes the carrier grid to 9 and 11 Hz.")
    observed_levels = [(x["label"], x["diffusion_rad2_per_s"]) for x in _levels(cfg)]
    if observed_levels != [(LOW, 0.5), (HIGH, 2.0)]:
        raise ValueError("D1 freezes D to 0.5 and 2.0 rad^2/s.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("D1 freezes afferent modulation depth to 0.04.")
    if not np.isclose(_wrap_phase(float(cfg.analysis.tacs.relative_phase_offset_rad)), np.pi):
        raise ValueError("D1 freezes the EEG-relative action phase to pi.")
    if int(cfg.analysis.tacs.phase_estimation_steps) != 1:
        raise ValueError("D1 freezes recent phase initialization to one online window.")
    doses = _doses(cfg)
    if doses != [0.0, 0.2, 0.4]:
        raise ValueError("D1 freezes the action set to sham, 0.2, and 0.4 V/m.")
    if max(doses) > float(cfg.analysis.maximum_field_v_per_m):
        raise ValueError("A D1 action exceeds the field limit.")

    timeline = cfg.analysis.timeline
    minimum_baseline = 4 if bool(cfg.analysis.smoke_test) else 12
    if int(timeline.baseline_steps) < minimum_baseline:
        raise ValueError(f"D1 requires at least {minimum_baseline} baseline seconds in this mode.")
    stimulation_ms = int(timeline.stimulation_steps) * float(cfg.env.simulation.obs_win_len)
    trim_ms = float(timeline.stimulation_analysis_trim_ms)
    if trim_ms < float(timeline.block_ramp_ms) or 2.0 * trim_ms >= stimulation_ms:
        raise ValueError("D1 stimulation trimming must contain both ramps and leave EEG.")
    if int(timeline.baseline_steps) < int(cfg.analysis.tacs.phase_estimation_steps):
        raise ValueError("The phase-estimation tail exceeds the baseline.")

    if not bool(cfg.analysis.smoke_test):
        if int(cfg.analysis.reference_calibration.n_seeds) < 6:
            raise ValueError("Full D1 requires six independent B references.")
        if int(cfg.analysis.crossed_design.n_structure_seeds) < 3:
            raise ValueError("Full D1 requires at least three independent structures.")
        if int(cfg.analysis.crossed_design.n_history_seeds) < 1:
            raise ValueError("Full D1 requires a history per structure.")
        if int(cfg.analysis.crossed_design.n_future_continuations) < 2:
            raise ValueError("Full D1 requires two independent postdecision futures.")

    contexts = _context_specs(cfg)
    references = set(_reference_seeds(cfg))
    namespaces = [
        references,
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
        raise ValueError("Every D1 seed namespace must be nonempty.")
    if any(
        namespaces[i].intersection(namespaces[j])
        for i in range(len(namespaces))
        for j in range(i + 1, len(namespaces))
    ):
        raise ValueError("D1 reference, structure, history, phase, trial, and future namespaces overlap.")
    d1_union = set().union(*namespaces)
    if d1_union.intersection(sources["source_seed_union"]):
        raise ValueError("D1 seeds overlap D0b or F0 source seeds.")
    if max(namespaces[1] | references) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("A D1 structure seed exceeds the uint32 mapping range.")


def _reference_cfg(cfg: DictConfig, seed: int) -> DictConfig:
    result = _condition_for_seed(cfg, seed=int(seed), modulation_depth=0.0)
    with open_dict(result):
        for population in ("E", "I"):
            rhythm = result.env.network.background[population].rhythm
            rhythm.phase_diffusion_rad2_per_s = 0.0
            rhythm.phase_diffusion_integration_dt_ms = float(
                cfg.analysis.states.phase_diffusion_integration_dt_ms
            )
    return result


def _fit_B_target(calibration: pd.DataFrame, cfg: DictConfig) -> dict[str, Any]:
    quantile = float(cfg.analysis.reference_calibration.upper_quantile)
    minimum_scale = float(cfg.analysis.reference_calibration.minimum_scale_log10)
    screen = calibration.baseline_log10_alpha_power.to_numpy(float)
    outcome = calibration.outcome_log10_alpha_power.to_numpy(float)
    screen_mean = float(np.mean(screen))
    outcome_mean = float(np.mean(outcome))
    return {
        "screening": {
            "mean_log10_alpha": screen_mean,
            "sd_log10_alpha": max(
                float(np.std(screen, ddof=1)) if len(screen) > 1 else 0.0,
                minimum_scale,
            ),
            "upper_quantile": quantile,
            "upper_quantile_log10_alpha": float(np.quantile(screen, quantile)),
            "minimum_elevated_threshold_log10_alpha": float(
                screen_mean + float(cfg.analysis.screening.minimum_alpha_excess_log10)
            ),
        },
        "outcome": {
            "mean_log10_alpha": outcome_mean,
            "sd_log10_alpha": max(
                float(np.std(outcome, ddof=1)) if len(outcome) > 1 else 0.0,
                minimum_scale,
            ),
        },
        "reference_E_firing_rate_hz": float(calibration.E_firing_rate_hz.mean()),
        "reference_I_firing_rate_hz": float(calibration.I_firing_rate_hz.mean()),
        "target_is_population_reference_not_seed_specific": True,
        "screening_and_outcome_durations_matched_separately": True,
    }


def _selected_carrier(
    processed: np.ndarray, *, fs_hz: float, cfg: DictConfig
) -> tuple[float, dict[str, float]]:
    # Peak search is over the complete alpha band and therefore does not use
    # the hidden generator frequency. The selected carrier is then snapped to
    # the frozen F0/D0b 9/11-Hz action grid.
    peak, _, _ = _periodogram_metrics(
        processed, fs_hz=fs_hz, frequency_hz=10.0, cfg=cfg
    )
    candidates = np.asarray(cfg.analysis.states.frequencies_hz, dtype=float)
    selected = float(candidates[np.argmin(
        np.abs(candidates - float(peak["detected_peak_frequency_hz"]))
    )])
    spectral, _, _ = _periodogram_metrics(
        processed, fs_hz=fs_hz, frequency_hz=selected, cfg=cfg
    )
    return selected, spectral


def _context_features(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> dict[str, Any]:
    raw = _epoch_raw(episode, "baseline")
    processed, fs_hz, _, _, generic = _process_eeg(
        raw, simulator_fs_hz=float(episode["simulator_fs_hz"]), cfg=cfg
    )
    selected, spectral = _selected_carrier(processed, fs_hz=fs_hz, cfg=cfg)
    outputs = episode["simulation"]["outputs_by_epoch"]["baseline"]
    start_ms = float(outputs[0]["t_start_ms"])
    phase = _window_phase_metrics(
        processed,
        fs_hz=fs_hz,
        start_ms=start_ms,
        frequency_hz=selected,
        phase_window_s=float(cfg.analysis.measurement.phase_window_s),
        temporal_chunk_s=float(cfg.analysis.measurement.temporal_chunk_s),
    )
    feature, _, _ = _episode_feature(episode, "baseline", cfg)
    phase_cfg = _with_action_frequency(cfg, selected)
    phase_outputs = _phase_estimation_outputs(outputs, phase_cfg)
    recent = _estimate_relative_field_phase(
        phase_outputs,
        simulator_fs_hz=float(episode["simulator_fs_hz"]),
        block_start_ms=float(episode["simulation"]["block_start_ms"]),
        relative_offset_rad=0.0,
        cfg=phase_cfg,
    )
    recent_ratio = float(recent["baseline_eeg_10hz_resultant_v"]) / max(
        float(recent["baseline_eeg_rms_v"]), np.finfo(float).tiny
    )
    baseline = _epoch_row(episode, "baseline")
    screen_target = target["screening"]
    elevated_threshold = max(
        float(screen_target["upper_quantile_log10_alpha"]),
        float(screen_target["minimum_elevated_threshold_log10_alpha"]),
    )
    alpha = float(feature["log10_alpha_power_8_12_hz"])
    limits = cfg.analysis.rate_guardrails_hz
    rate_safe = bool(
        float(limits.E_min) <= float(baseline.E_firing_rate_hz) <= float(limits.E_max)
        and float(limits.I_min) <= float(baseline.I_firing_rate_hz) <= float(limits.I_max)
    )
    alpha_present = bool(alpha >= elevated_threshold)
    phase_actionable = bool(
        recent_ratio >= float(cfg.analysis.screening.minimum_recent_resultant_to_rms)
    )
    reasons: list[str] = []
    if not alpha_present:
        reasons.append("elevated_alpha_phenotype_absent")
    if not phase_actionable:
        reasons.append("recent_phase_not_measurable")
    if not rate_safe:
        reasons.append("baseline_rate_guardrail_failed")
    return {
        **context,
        "context_C1": float(phase["phase_increment_coherence_real"]),
        "context_C1_abs": float(phase["phase_increment_coherence_abs"]),
        "context_C1_imag": float(phase["phase_increment_coherence_imag"]),
        "context_C1_temporal_sd": float(phase["temporal_chunk_C1_sd"]),
        "context_C1_window_count": int(phase["n_phase_windows"]),
        "context_spectral_concentration": float(spectral["spectral_concentration"]),
        "context_spectral_rms_width_hz": float(spectral["spectral_rms_width_hz"]),
        "context_log10_alpha_power": alpha,
        "context_alpha_excess_log10": float(
            alpha - float(screen_target["mean_log10_alpha"])
        ),
        "context_distance_to_B_log10": abs(
            alpha - float(screen_target["mean_log10_alpha"])
        ),
        "context_rms_v": float(generic["rms_v"]),
        "EEG_selected_frequency_hz": selected,
        "EEG_frequency_selection_correct": bool(
            np.isclose(selected, float(context["hidden_frequency_hz"]))
        ),
        "D0b_threshold_predicted_label": (
            LOW if float(phase["phase_increment_coherence_real"])
            >= float(target["frozen_C1_threshold"]) else HIGH
        ),
        "D0b_threshold_classification_correct": bool(
            (float(phase["phase_increment_coherence_real"])
             >= float(target["frozen_C1_threshold"]))
            == (str(context["label"]) == LOW)
        ),
        "recent_eeg_phase_at_block_rad": float(
            recent["baseline_eeg_phase_at_block_rad"]
        ),
        "recent_resultant_to_rms": recent_ratio,
        "recent_phase_estimation_steps": int(len(phase_outputs)),
        "baseline_E_firing_rate_hz": float(baseline.E_firing_rate_hz),
        "baseline_I_firing_rate_hz": float(baseline.I_firing_rate_hz),
        "alpha_phenotype_present": alpha_present,
        "recent_phase_actionable": phase_actionable,
        "baseline_rates_safe": rate_safe,
        "eligible": bool(not reasons),
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "screen_uses_only_predecision_ideal_EEG": True,
        "screen_uses_hidden_diffusion_or_frequency": False,
        "screen_uses_action_outcome": False,
        "phase_endpoint_uses_hidden_frequency": False,
    }


def _single_action(cfg: DictConfig, dose: float) -> dict[str, Any]:
    return _action(
        cfg,
        identifier=_dose_id(dose),
        role="sham_or_abstention" if np.isclose(dose, SHAM) else "single_constant_action",
        amplitude=float(dose),
        montage=str(cfg.analysis.tacs.axial_montage),
        relative_offset=float(cfg.analysis.tacs.relative_phase_offset_rad),
    )


def _run_replay(
    *, condition_cfg: DictConfig, context: dict[str, Any], future_seed: int,
    future_index: int, dose: float, action_index: int, root: Path,
    comm: Any, size: int, rank: int,
) -> dict[str, Any] | None:
    return _run_condition(
        condition_id=_dose_id(dose),
        condition_cfg=condition_cfg,
        action=_single_action(condition_cfg, dose),
        stimulate=True,
        seed=int(context["trial_seed"]),
        action_index=int(action_index),
        output_dir=(
            root / "episodes" / str(context["context_id"])
            / f"future_{future_index + 1:02d}" / _dose_id(dose)
        ),
        comm=comm,
        size=size,
        rank=rank,
        structure_seed=int(context["structure_seed"]),
        drive_seed=int(context["history_seed"]),
        future_drive_seed=int(future_seed),
        phase_seed=int(context["phase_seed"]),
    )


def _phase_tracking_error(
    episode: dict[str, Any], screening: dict[str, Any], cfg: DictConfig
) -> float:
    expected = _wrap_phase(
        float(screening["recent_eeg_phase_at_block_rad"])
        + np.pi / 2.0
        + float(cfg.analysis.tacs.relative_phase_offset_rad)
    )
    realized = float(episode["simulation"]["action"]["phase_rad"])
    return abs(float(np.angle(np.exp(1j * (realized - expected)))))


def _metric_rows(
    *, context: dict[str, Any], screening: dict[str, Any], future_index: int,
    future_seed: int, episodes: dict[float, dict[str, Any]],
    baseline_reference: dict[str, Any], target: dict[str, Any], cfg: DictConfig,
) -> list[dict[str, Any]]:
    sham_episode = episodes[SHAM]
    sham_feature, _, _ = _episode_feature(sham_episode, "stimulation", cfg)
    sham_row = _epoch_row(sham_episode, "stimulation")
    sham_baseline = _epoch_row(sham_episode, "baseline")
    sham_washout = _epoch_row(sham_episode, "washout")
    target_alpha = float(target["outcome"]["mean_log10_alpha"])
    sham_distance = abs(float(sham_feature["log10_alpha_power_8_12_hz"]) - target_alpha)
    token = _frequency_token(float(screening["EEG_selected_frequency_hz"]))
    rows: list[dict[str, Any]] = []
    for dose in _doses(cfg):
        episode = episodes[dose]
        feature, _, _ = _episode_feature(episode, "stimulation", cfg)
        outcome = _epoch_row(episode, "stimulation")
        baseline = _epoch_row(episode, "baseline")
        washout = _epoch_row(episode, "washout")
        post_alpha = float(feature["log10_alpha_power_8_12_hz"])
        post_distance = abs(post_alpha - target_alpha)
        baseline_error = _relative_rms_error(
            _epoch_raw(baseline_reference, "baseline"),
            _epoch_raw(episode, "baseline"),
        )
        if np.isclose(dose, SHAM):
            residual, tolerance, recovered = 0.0, 0.0, True
            rate_safe = _relative_rate_safe(sham_row, sham_row, cfg)
            phase_error = 0.0
        else:
            residual = float(
                (sham_washout.log10_alpha_power_8_12_hz
                 - sham_baseline.log10_alpha_power_8_12_hz)
                - (washout.log10_alpha_power_8_12_hz
                   - baseline.log10_alpha_power_8_12_hz)
            )
            recovered, tolerance = _field_removal_status(
                effect_log10=float(
                    sham_feature["log10_alpha_power_8_12_hz"] - post_alpha
                ),
                residual_log10=residual,
                cfg=cfg,
            )
            rate_safe = _relative_rate_safe(outcome, sham_row, cfg)
            phase_error = _phase_tracking_error(episode, screening, cfg)
        realized = episode["simulation"]["action"]
        rows.append({
            **context,
            **{
                name: screening[name]
                for name in (
                    "context_C1", "context_C1_abs", "context_C1_imag",
                    "context_C1_temporal_sd", "context_spectral_concentration",
                    "context_spectral_rms_width_hz", "context_log10_alpha_power",
                    "context_alpha_excess_log10", "context_distance_to_B_log10",
                    "EEG_selected_frequency_hz", "recent_resultant_to_rms",
                )
            },
            "future_index": int(future_index + 1),
            "future_drive_seed": int(future_seed),
            "dose_v_per_m": float(dose),
            "action_id": _dose_id(dose),
            "action_frequency_hz": float(realized["frequency_hz"]),
            "action_relative_phase_offset_rad": float(
                cfg.analysis.tacs.relative_phase_offset_rad
            ),
            "realized_amplitude_v_per_m": float(realized["ac_amplitude_v_per_m"]),
            "one_action_for_complete_intervention": bool(
                np.isclose(float(realized["ac_amplitude_v_per_m"]), dose)
            ),
            "action_frequency_matches_EEG_selection": bool(
                np.isclose(
                    float(realized["frequency_hz"]),
                    float(screening["EEG_selected_frequency_hz"]),
                )
            ),
            "action_phase_tracking_error_rad": float(phase_error),
            "phase_estimation_steps_used": int(
                episode["simulation"].get("phase_estimation_steps_used", 0)
            ),
            "frozen_B_outcome_mean_log10_alpha": target_alpha,
            "sham_post_distance_to_B_log10": float(sham_distance),
            "post_log10_alpha_power": post_alpha,
            "post_distance_to_B_log10": float(post_distance),
            "reward_negative_distance": float(-post_distance),
            "causal_distance_improvement_vs_sham_log10": float(
                sham_distance - post_distance
            ),
            "causal_alpha_suppression_vs_sham_log10": float(
                sham_feature["log10_alpha_power_8_12_hz"] - post_alpha
            ),
            "coherent_carrier_suppression_vs_sham_v": float(
                sham_feature[f"eeg_{token}hz_resultant_v"]
                - feature[f"eeg_{token}hz_resultant_v"]
            ),
            "carrier_band_suppression_vs_sham_log10": float(
                sham_feature[f"log10_power_{token}hz"]
                - feature[f"log10_power_{token}hz"]
            ),
            "hidden_E_ppc_reduction_vs_sham": float(sham_row.E_ppc - outcome.E_ppc),
            "hidden_I_ppc_reduction_vs_sham": float(sham_row.I_ppc - outcome.I_ppc),
            "sham_E_firing_rate_hz": float(sham_row.E_firing_rate_hz),
            "sham_I_firing_rate_hz": float(sham_row.I_firing_rate_hz),
            "post_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
            "post_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
            "E_rate_change_vs_sham_hz": float(
                outcome.E_firing_rate_hz - sham_row.E_firing_rate_hz
            ),
            "I_rate_change_vs_sham_hz": float(
                outcome.I_firing_rate_hz - sham_row.I_firing_rate_hz
            ),
            "rate_safe": bool(rate_safe),
            "field_removal_residual_log10": float(residual),
            "field_removal_tolerance_log10": float(tolerance),
            "field_removal_recovered": bool(recovered),
            "final_extracellular_residual_mV": float(
                episode["simulation"]["final_residual_mV"]
            ),
            "baseline_relative_rms_error": float(baseline_error),
            "policy_uses_hidden_state_or_spikes": False,
        })
    return rows


def _expected_action_map(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "context_id", "paired_diffusion_context_id", "context_order",
        "future_group_index", "structure_index", "structure_seed",
        "history_index", "history_seed", "phase_seed", "hidden_frequency_hz",
        "label", "diffusion_rad2_per_s", "context_C1", "context_C1_abs",
        "context_C1_imag", "context_C1_temporal_sd",
        "context_spectral_concentration", "context_spectral_rms_width_hz",
        "context_log10_alpha_power", "context_alpha_excess_log10",
        "context_distance_to_B_log10", "EEG_selected_frequency_hz",
        "recent_resultant_to_rms", "dose_v_per_m", "action_id",
    ]
    return metrics.groupby(keys, as_index=False, dropna=False).agg(
        n_future_continuations=("future_index", "nunique"),
        expected_post_distance_to_B_log10=("post_distance_to_B_log10", "mean"),
        future_sd_post_distance_log10=("post_distance_to_B_log10", "std"),
        expected_causal_distance_improvement_vs_sham_log10=(
            "causal_distance_improvement_vs_sham_log10", "mean"
        ),
        expected_alpha_suppression_vs_sham_log10=(
            "causal_alpha_suppression_vs_sham_log10", "mean"
        ),
        expected_coherent_carrier_suppression_v=(
            "coherent_carrier_suppression_vs_sham_v", "mean"
        ),
        expected_carrier_band_suppression_log10=(
            "carrier_band_suppression_vs_sham_log10", "mean"
        ),
        expected_hidden_E_ppc_reduction=("hidden_E_ppc_reduction_vs_sham", "mean"),
        all_rate_safe=("rate_safe", "all"),
        all_field_removal_recovered=("field_removal_recovered", "all"),
        maximum_baseline_relative_rms_error=("baseline_relative_rms_error", "max"),
        maximum_action_phase_tracking_error_rad=("action_phase_tracking_error_rad", "max"),
    )


def _best_fixed_action(expected: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    # Equal weighting is given to independent structures before selecting the
    # population-best fixed comparator. Histories/frequencies/diffusion states
    # are repeated measurements rather than independent samples.
    by_structure = expected.groupby(
        ["structure_seed", "dose_v_per_m"], as_index=False
    ).expected_post_distance_to_B_log10.mean()
    fixed = by_structure.groupby("dose_v_per_m", as_index=False).agg(
        mean_structure_distance_to_B_log10=(
            "expected_post_distance_to_B_log10", "mean"
        )
    ).sort_values(["mean_structure_distance_to_B_log10", "dose_v_per_m"])
    return float(fixed.iloc[0].dose_v_per_m), fixed


def _context_action_summary(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    fixed_dose, fixed_table = _best_fixed_action(expected)
    practical = float(cfg.analysis.criteria.practical_action_advantage_log10)
    rows: list[dict[str, Any]] = []
    for context_id, group in expected.groupby("context_id", sort=False):
        oracle = group.sort_values(
            ["expected_post_distance_to_B_log10", "dose_v_per_m"]
        ).iloc[0]
        fixed = group[np.isclose(group.dose_v_per_m, fixed_dose)].iloc[0]
        realized = metrics[metrics.context_id.eq(context_id)]
        winners: list[float] = []
        for _, future in realized.groupby("future_index", sort=True):
            winner = future.sort_values(
                ["post_distance_to_B_log10", "dose_v_per_m"]
            ).iloc[0]
            winners.append(float(winner.dose_v_per_m))
        advantage = float(
            fixed.expected_post_distance_to_B_log10
            - oracle.expected_post_distance_to_B_log10
        )
        source = group.iloc[0]
        rows.append({
            "context_id": str(context_id),
            "paired_diffusion_context_id": str(source.paired_diffusion_context_id),
            "structure_index": int(source.structure_index),
            "structure_seed": int(source.structure_seed),
            "history_index": int(source.history_index),
            "history_seed": int(source.history_seed),
            "hidden_frequency_hz": float(source.hidden_frequency_hz),
            "label": str(source.label),
            "diffusion_rad2_per_s": float(source.diffusion_rad2_per_s),
            "context_C1": float(source.context_C1),
            "context_spectral_concentration": float(source.context_spectral_concentration),
            "context_alpha_excess_log10": float(source.context_alpha_excess_log10),
            "EEG_selected_frequency_hz": float(source.EEG_selected_frequency_hz),
            "expected_optimal_dose_v_per_m": float(oracle.dose_v_per_m),
            "expected_optimal_distance_to_B_log10": float(
                oracle.expected_post_distance_to_B_log10
            ),
            "best_fixed_dose_v_per_m": fixed_dose,
            "best_fixed_distance_to_B_log10": float(
                fixed.expected_post_distance_to_B_log10
            ),
            "expected_oracle_advantage_over_best_fixed_log10": advantage,
            "practical_nonfixed_opportunity": bool(
                not np.isclose(float(oracle.dose_v_per_m), fixed_dose)
                and advantage >= practical
            ),
            "realized_winner_agreement_fraction": float(np.mean(
                np.isclose(winners, float(oracle.dose_v_per_m))
            )),
            "realized_winner_doses_v_per_m": ";".join(f"{x:g}" for x in winners),
        })
    summary = pd.DataFrame(rows)
    structure = summary.groupby(
        ["structure_index", "structure_seed"], as_index=False
    ).agg(
        context_count=("context_id", "nunique"),
        mean_oracle_advantage_over_best_fixed_log10=(
            "expected_oracle_advantage_over_best_fixed_log10", "mean"
        ),
        practical_nonfixed_context_count=("practical_nonfixed_opportunity", "sum"),
        mean_realized_winner_agreement_fraction=(
            "realized_winner_agreement_fraction", "mean"
        ),
    )
    audit = {
        "best_fixed_dose_v_per_m": fixed_dose,
        "expected_optimal_doses_v_per_m": sorted(
            summary.expected_optimal_dose_v_per_m.unique().tolist()
        ),
        "practical_nonfixed_context_count": int(
            summary.practical_nonfixed_opportunity.sum()
        ),
        "practical_nonfixed_structure_count": int(
            summary.loc[summary.practical_nonfixed_opportunity, "structure_seed"].nunique()
        ),
        "mean_oracle_advantage_over_best_fixed_log10": float(
            summary.expected_oracle_advantage_over_best_fixed_log10.mean()
        ),
        "mean_realized_winner_agreement_fraction": float(
            summary.realized_winner_agreement_fraction.mean()
        ),
        "fixed_action_table": fixed_table.to_dict(orient="records"),
    }
    return summary, structure, audit


def _diffusion_dose_interaction(
    expected: pd.DataFrame, cfg: DictConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    low_dose, high_dose = [
        float(x) for x in cfg.analysis.actions.active_doses_v_per_m
    ]
    pivot = expected.pivot_table(
        index=[
            "context_id", "paired_diffusion_context_id", "structure_seed",
            "hidden_frequency_hz", "label", "diffusion_rad2_per_s", "context_C1",
        ],
        columns="dose_v_per_m",
        values="expected_post_distance_to_B_log10",
    ).reset_index()
    pivot["high_minus_low_dose_advantage_log10"] = (
        pivot[low_dose] - pivot[high_dose]
    )
    paired_rows: list[dict[str, Any]] = []
    for paired_id, group in pivot.groupby("paired_diffusion_context_id"):
        if set(group.label) != {LOW, HIGH}:
            continue
        low = group[group.label.eq(LOW)].iloc[0]
        high = group[group.label.eq(HIGH)].iloc[0]
        interaction = float(
            low.high_minus_low_dose_advantage_log10
            - high.high_minus_low_dose_advantage_log10
        )
        paired_rows.append({
            "paired_diffusion_context_id": str(paired_id),
            "structure_seed": int(low.structure_seed),
            "hidden_frequency_hz": float(low.hidden_frequency_hz),
            "low_C1": float(low.context_C1),
            "high_C1": float(high.context_C1),
            "low_high_minus_low_dose_advantage_log10": float(
                low.high_minus_low_dose_advantage_log10
            ),
            "high_high_minus_low_dose_advantage_log10": float(
                high.high_minus_low_dose_advantage_log10
            ),
            "diffusion_by_dose_interaction_log10": interaction,
            "absolute_diffusion_by_dose_interaction_log10": abs(interaction),
        })
    paired = pd.DataFrame(paired_rows)
    x = pivot.context_C1.to_numpy(float)
    y = pivot.high_minus_low_dose_advantage_log10.to_numpy(float)
    if len(x) >= 2 and float(np.ptp(x)) > 0.0:
        slope, intercept = np.polyfit(x, y, 1)
        correlation = float(np.corrcoef(x, y)[0, 1])
    else:
        slope, intercept, correlation = float("nan"), float("nan"), float("nan")
    audit = {
        "low_dose_v_per_m": low_dose,
        "high_dose_v_per_m": high_dose,
        "C1_dose_contrast_slope": float(slope),
        "C1_dose_contrast_intercept": float(intercept),
        "C1_dose_contrast_correlation": correlation,
        "mean_absolute_diffusion_by_dose_interaction_log10": (
            float(paired.absolute_diffusion_by_dose_interaction_log10.mean())
            if not paired.empty else float("nan")
        ),
        "mean_signed_diffusion_by_dose_interaction_log10": (
            float(paired.diffusion_by_dose_interaction_log10.mean())
            if not paired.empty else float("nan")
        ),
    }
    return paired, audit


def _fit_arm_models(
    training: pd.DataFrame, *, doses: list[float], features: list[str], ridge: float,
) -> dict[str, Any]:
    raw = training[features].to_numpy(float)
    mean = raw.mean(axis=0)
    scale = raw.std(axis=0, ddof=0)
    scale[scale <= np.finfo(float).eps] = 1.0
    x = np.column_stack((np.ones(len(raw)), (raw - mean) / scale))
    penalty = np.diag([0.0, *([1.0] * len(features))]) * float(ridge)
    coefficients: dict[float, np.ndarray] = {}
    for dose in doses:
        y = training[f"reward_dose_{dose:g}"].to_numpy(float)
        coefficients[dose] = np.linalg.pinv(x.T @ x + penalty) @ x.T @ y
    return {"mean": mean, "scale": scale, "coefficients": coefficients}


def _policy_table(
    expected: pd.DataFrame, summary: pd.DataFrame, cfg: DictConfig,
    *, feature_override: pd.Series | None = None,
) -> pd.DataFrame:
    features = [str(x) for x in cfg.analysis.context.policy_features]
    if features != ["context_C1"]:
        raise ValueError("D1 freezes the learnable policy context to C1 only.")
    table = summary.copy()
    if feature_override is not None:
        table["context_C1"] = table.context_id.map(feature_override)
    doses = _doses(cfg)
    for dose in doses:
        values = expected[np.isclose(expected.dose_v_per_m, dose)].set_index(
            "context_id"
        ).expected_post_distance_to_B_log10
        table[f"reward_dose_{dose:g}"] = -table.context_id.map(values)
    fixed_dose = float(summary.best_fixed_dose_v_per_m.iloc[0])
    structures = sorted(table.structure_seed.unique())
    if len(structures) < 2:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for heldout in structures:
        train = table[table.structure_seed.ne(heldout)]
        test = table[table.structure_seed.eq(heldout)]
        model = _fit_arm_models(
            train,
            doses=doses,
            features=features,
            ridge=float(cfg.analysis.context.ridge_penalty),
        )
        raw = test[features].to_numpy(float)
        x = np.column_stack((
            np.ones(len(test)), (raw - model["mean"]) / model["scale"]
        ))
        predictions = np.column_stack([
            x @ model["coefficients"][dose] for dose in doses
        ])
        for index, (_, source) in enumerate(test.iterrows()):
            selected = doses[int(np.argmax(predictions[index]))]
            selected_distance = -float(source[f"reward_dose_{selected:g}"])
            fixed_distance = -float(source[f"reward_dose_{fixed_dose:g}"])
            rows.append({
                "context_id": str(source.context_id),
                "heldout_structure_seed": int(heldout),
                "label": str(source.label),
                "hidden_frequency_hz": float(source.hidden_frequency_hz),
                "observed_context_C1": float(source.context_C1),
                "selected_dose_v_per_m": float(selected),
                "selected_expected_distance_to_B_log10": selected_distance,
                "best_fixed_dose_v_per_m": fixed_dose,
                "best_fixed_expected_distance_to_B_log10": fixed_distance,
                "contextual_advantage_over_best_fixed_log10": float(
                    fixed_distance - selected_distance
                ),
                "matches_expected_oracle": bool(np.isclose(
                    selected, float(source.expected_optimal_dose_v_per_m)
                )),
                "policy_uses_only_predecision_C1": True,
            })
    return pd.DataFrame(rows)


def _shuffle_null(
    expected: pd.DataFrame, summary: pd.DataFrame, observed_policy: pd.DataFrame,
    cfg: DictConfig,
) -> tuple[pd.DataFrame, float]:
    if observed_policy.empty:
        return pd.DataFrame(), 1.0
    observed = float(
        observed_policy.groupby("heldout_structure_seed")
        .contextual_advantage_over_best_fixed_log10.mean().mean()
    )
    rng = np.random.default_rng(
        int(cfg.experiment.seed) + int(cfg.analysis.context_shuffle.random_seed_offset)
    )
    values: list[float] = []
    for _ in range(int(cfg.analysis.context_shuffle.n_permutations)):
        shuffled = summary[["context_id", "structure_seed", "context_C1"]].copy()
        shuffled["context_C1"] = shuffled.groupby("structure_seed")["context_C1"].transform(
            lambda values: rng.permutation(values.to_numpy(float))
        )
        override = shuffled.set_index("context_id").context_C1
        policy = _policy_table(expected, summary, cfg, feature_override=override)
        value = float(
            policy.groupby("heldout_structure_seed")
            .contextual_advantage_over_best_fixed_log10.mean().mean()
        )
        values.append(value)
    null = pd.DataFrame({
        "permutation": np.arange(len(values)),
        "shuffled_C1_policy_advantage_log10": values,
    })
    p_value = float(
        (1 + np.count_nonzero(np.asarray(values) >= observed)) / (len(values) + 1)
    )
    return null, p_value


def _checks_and_conclusions(
    *, calibration: pd.DataFrame, screening: pd.DataFrame, metrics: pd.DataFrame,
    expected: pd.DataFrame, summary: pd.DataFrame, structure: pd.DataFrame,
    policy: pd.DataFrame, interaction: pd.DataFrame,
    interaction_audit: dict[str, Any], opportunity_audit: dict[str, Any],
    shuffle_p: float, sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.dose_v_per_m.gt(0.0)]
    reference_rates = {
        "E": float(calibration.E_firing_rate_hz.mean()),
        "I": float(calibration.I_firing_rate_hz.mean()),
    }
    tolerance = float(cfg.analysis.rate_reference_tolerance_fraction)
    rate_matched = []
    for row in eligible.itertuples():
        rate_matched.append(all(
            abs(float(getattr(row, f"baseline_{population}_firing_rate_hz"))
                - reference_rates[population])
            <= tolerance * max(reference_rates[population], np.finfo(float).tiny)
            for population in ("E", "I")
        ))
    selected_policy_actions = (
        sorted(policy.selected_dose_v_per_m.unique().tolist())
        if not policy.empty else []
    )
    policy_structure = (
        policy.groupby("heldout_structure_seed")
        .contextual_advantage_over_best_fixed_log10.mean()
        if not policy.empty else pd.Series(dtype=float)
    )
    mean_policy_advantage = (
        float(policy_structure.mean()) if len(policy_structure) else float("nan")
    )
    positive_policy_fraction = (
        float(np.mean(policy_structure > 0.0)) if len(policy_structure) else 0.0
    )
    optimal_actions = [
        float(x) for x in opportunity_audit["expected_optimal_doses_v_per_m"]
    ]
    expected_active_actions = [x for x in optimal_actions if x > 0.0]
    interaction_mean = float(
        interaction_audit["mean_absolute_diffusion_by_dose_interaction_log10"]
    )
    checks = {
        "source_D0b_hash_locked_and_confirmed": bool(sources["D0b_confirmed"]),
        "source_F0_hash_locked_and_passed": bool(sources["F0_passed"]),
        "D1_seeds_disjoint_from_D0b_and_F0": True,
        "reference_target_calibrated_on_disjoint_B_seeds": bool(
            len(calibration) >= int(criteria.minimum_reference_seeds)
            or bool(cfg.analysis.smoke_test)
        ),
        "complete_crossed_screening_grid": len(screening) == len(_run_context_specs(cfg)),
        "screening_uses_only_predecision_ideal_EEG": bool(
            screening.screen_uses_only_predecision_ideal_EEG.all()
        ),
        "screening_excludes_hidden_state_and_action_outcomes": bool(
            (~screening.screen_uses_hidden_diffusion_or_frequency.astype(bool)).all()
            and (~screening.screen_uses_action_outcome.astype(bool)).all()
        ),
        "state_generator_is_distinct_from_tacs_action": True,
        "afferent_mean_rate_matched_across_states_by_construction": True,
        "minimum_eligible_contexts": len(eligible) >= int(criteria.minimum_eligible_contexts),
        "minimum_independent_structures": eligible.structure_seed.nunique()
        >= int(criteria.minimum_structure_seeds),
        "both_diffusion_levels_enrolled": eligible.label.nunique()
        >= int(criteria.minimum_diffusion_levels_enrolled),
        "both_frequencies_enrolled": eligible.hidden_frequency_hz.nunique()
        >= int(criteria.minimum_frequencies_enrolled),
        "frequency_identified_from_predecision_EEG": bool(
            len(eligible) > 0
            and eligible.EEG_frequency_selection_correct.mean()
            >= float(criteria.minimum_frequency_detection_accuracy)
        ),
        "all_enrolled_recent_phase_measurable": bool(
            len(eligible) > 0 and eligible.recent_phase_actionable.all()
        ),
        "multiple_independent_postdecision_futures": bool(
            len(expected) > 0
            and expected.n_future_continuations.min()
            >= int(criteria.minimum_future_continuations)
        ),
        "identical_predecision_EEG_across_actions_and_futures": bool(
            len(metrics) > 0
            and metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "single_constant_action_per_intervention": bool(
            len(metrics) > 0 and metrics.one_action_for_complete_intervention.all()
        ),
        "action_frequency_is_EEG_matched": bool(
            len(active) > 0 and active.action_frequency_matches_EEG_selection.all()
        ),
        "action_phase_uses_recent_one_second_EEG": bool(
            len(active) > 0
            and (active.phase_estimation_steps_used == 1).all()
            and active.action_phase_tracking_error_rad.max()
            <= float(criteria.maximum_phase_tracking_error_rad)
        ),
        "all_actions_rate_safe": bool(len(metrics) > 0 and metrics.rate_safe.all()),
        "reference_rate_matched": bool(rate_matched and all(rate_matched)),
        "field_removal_recovered": bool(
            len(metrics) > 0
            and metrics.field_removal_recovered.all()
            and np.allclose(metrics.final_extracellular_residual_mV, 0.0)
        ),
        "expected_oracle_uses_multiple_actions": len(optimal_actions)
        >= int(criteria.minimum_expected_action_count),
        "expected_oracle_retains_an_active_action": len(expected_active_actions)
        >= int(criteria.minimum_active_action_count),
        "practical_nonfixed_opportunity_in_multiple_contexts": int(
            opportunity_audit["practical_nonfixed_context_count"]
        ) >= int(criteria.minimum_nonfixed_contexts),
        "practical_nonfixed_opportunity_across_structures": int(
            opportunity_audit["practical_nonfixed_structure_count"]
        ) >= int(criteria.minimum_nonfixed_structures),
        "expected_oracle_has_mean_practical_advantage": float(
            opportunity_audit["mean_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_mean_oracle_advantage_log10),
        "realized_optimum_reproducible_across_futures": float(
            opportunity_audit["mean_realized_winner_agreement_fraction"]
        ) >= float(criteria.minimum_realized_winner_agreement_fraction),
        "diffusion_context_changes_relative_dose_response": bool(
            len(interaction) > 0
            and np.isfinite(interaction_mean)
            and interaction_mean
            >= float(criteria.minimum_mean_absolute_diffusion_dose_interaction_log10)
        ),
        "exploratory_C1_rule_uses_multiple_actions": len(selected_policy_actions) >= 2,
        "exploratory_C1_rule_beats_best_fixed_directionally": bool(
            np.isfinite(mean_policy_advantage)
            and mean_policy_advantage
            > float(criteria.minimum_policy_advantage_over_best_fixed_log10)
        ),
        "C1_policy_advantage_positive_across_structures": positive_policy_fraction
        >= float(criteria.minimum_positive_structure_fraction),
        "C1_context_beats_structure_preserving_shuffle": shuffle_p
        <= float(criteria.maximum_context_shuffle_p_value),
        "policy_uses_only_predecision_C1": bool(
            len(policy) > 0 and policy.policy_uses_only_predecision_C1.all()
        ),
        "latent_diffusion_frequency_and_hidden_spikes_excluded_from_policy": True,
    }
    mapping_gate = [
        "source_D0b_hash_locked_and_confirmed",
        "source_F0_hash_locked_and_passed",
        "D1_seeds_disjoint_from_D0b_and_F0",
        "reference_target_calibrated_on_disjoint_B_seeds",
        "complete_crossed_screening_grid",
        "screening_uses_only_predecision_ideal_EEG",
        "screening_excludes_hidden_state_and_action_outcomes",
        "minimum_eligible_contexts",
        "minimum_independent_structures",
        "both_diffusion_levels_enrolled",
        "both_frequencies_enrolled",
        "frequency_identified_from_predecision_EEG",
        "all_enrolled_recent_phase_measurable",
        "multiple_independent_postdecision_futures",
        "identical_predecision_EEG_across_actions_and_futures",
        "single_constant_action_per_intervention",
        "action_frequency_is_EEG_matched",
        "action_phase_uses_recent_one_second_EEG",
        "all_actions_rate_safe",
        "reference_rate_matched",
        "field_removal_recovered",
        "expected_oracle_uses_multiple_actions",
        "expected_oracle_retains_an_active_action",
        "practical_nonfixed_opportunity_in_multiple_contexts",
        "practical_nonfixed_opportunity_across_structures",
        "expected_oracle_has_mean_practical_advantage",
        "realized_optimum_reproducible_across_futures",
        "diffusion_context_changes_relative_dose_response",
    ]
    policy_gate = [
        *mapping_gate,
        "exploratory_C1_rule_uses_multiple_actions",
        "exploratory_C1_rule_beats_best_fixed_directionally",
        "C1_policy_advantage_positive_across_structures",
        "C1_context_beats_structure_preserving_shuffle",
        "policy_uses_only_predecision_C1",
        "latent_diffusion_frequency_and_hidden_spikes_excluded_from_policy",
    ]
    mapping_passed = bool(all(checks[name] for name in mapping_gate))
    ready = bool(
        all(checks[name] for name in policy_gate)
        and not bool(cfg.analysis.smoke_test)
    )
    conclusions = {
        "D1_full_information_action_map_feasible": mapping_passed,
        "ready_for_disjoint_contextual_policy_confirmation": ready,
        "screened_context_count": int(len(screening)),
        "eligible_context_count": int(len(eligible)),
        "screening_yield": float(len(eligible) / max(len(screening), 1)),
        "frequency_detection_accuracy": (
            float(eligible.EEG_frequency_selection_correct.mean())
            if len(eligible) else float("nan")
        ),
        "selected_policy_actions_v_per_m": selected_policy_actions,
        "mean_C1_policy_advantage_over_best_fixed_log10": mean_policy_advantage,
        "positive_policy_structure_fraction": positive_policy_fraction,
        "structure_preserving_context_shuffle_p_value": float(shuffle_p),
        **opportunity_audit,
        **interaction_audit,
        "contextual_bandit_status": "NOT TRAINED OR TESTED",
        "claim_scope": "exploratory ideal-neural-EEG full-information system identification",
        "smoke_test": bool(cfg.analysis.smoke_test),
    }
    return checks, conclusions


def _plot_results(
    *, root: Path, screening: pd.DataFrame, expected: pd.DataFrame,
    summary: pd.DataFrame, structure: pd.DataFrame, policy: pd.DataFrame,
    sources: dict[str, Any],
) -> None:
    eligible = screening[screening.eligible]
    figure, axis = plt.subplots(figsize=(7.0, 4.8))
    for label, group in eligible.groupby("label"):
        axis.scatter(
            group.context_C1,
            group.context_alpha_excess_log10,
            label=label,
            s=70,
            alpha=0.85,
        )
    axis.axvline(
        float(sources["frozen_C1_threshold"]), color="black", linestyle="--",
        label="frozen D0b threshold",
    )
    axis.set(
        xlabel="Predecision EEG phase-increment coherence C1",
        ylabel="Predecision alpha excess over B (log10)",
        title="D1 stimulation-free EEG contexts",
    )
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(root / "figure_01_predecision_C1_contexts.png", dpi=250)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)
    for axis, (label, group) in zip(axes, expected.groupby("label")):
        for dose, dose_group in group.groupby("dose_v_per_m"):
            axis.scatter(
                dose_group.context_C1,
                dose_group.expected_post_distance_to_B_log10,
                label=f"{dose:g} V/m",
                s=55,
            )
        axis.set(title=label, xlabel="Predecision C1")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Expected post-action distance to B (log10)")
    figure.suptitle("Full-information context–action map")
    figure.tight_layout()
    figure.savefig(root / "figure_02_C1_dose_response.png", dpi=250)
    plt.close(figure)

    ordered = summary.sort_values("context_C1")
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    scatter = axis.scatter(
        ordered.context_C1,
        ordered.expected_optimal_dose_v_per_m,
        c=ordered.diffusion_rad2_per_s,
        cmap="viridis",
        s=80,
        edgecolor="black",
        linewidth=0.5,
    )
    axis.set(
        xlabel="Predecision EEG C1",
        ylabel="Expected-optimal single action (V/m)",
        title="Observed context and expected action",
    )
    figure.colorbar(scatter, ax=axis, label="Hidden D (audit only, rad²/s)")
    figure.tight_layout()
    figure.savefig(root / "figure_03_expected_action_map.png", dpi=250)
    plt.close(figure)

    if not policy.empty:
        structure_policy = policy.groupby("heldout_structure_seed", as_index=False).agg(
            advantage=("contextual_advantage_over_best_fixed_log10", "mean")
        )
        figure, axis = plt.subplots(figsize=(6.4, 4.2))
        axis.bar(
            structure_policy.heldout_structure_seed.astype(str),
            structure_policy.advantage,
            color=np.where(structure_policy.advantage > 0.0, "#2CA02C", "#D62728"),
        )
        axis.axhline(0.0, color="0.25", linewidth=0.9)
        axis.set(
            xlabel="Held-out circuit structure",
            ylabel="LOSO C1-rule advantage over best fixed (log10)",
            title="Exploratory EEG-only policy audit",
        )
        figure.tight_layout()
        figure.savefig(root / "figure_04_exploratory_policy.png", dpi=250)
        plt.close(figure)


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / "phase_diffusion_action_map"
    if rank == 0:
        output_exists = bool(root.exists() and any(root.iterdir()))
    else:
        output_exists = None
    output_exists = bool(comm.bcast(output_exists, root=0))
    if output_exists:
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### D1 phase-diffusion full-information action map")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    calibration_rows: list[dict[str, Any]] = []
    for seed in _reference_seeds(cfg):
        if rank == 0:
            print(f"B reference calibration seed={seed}")
        reference_cfg = _reference_cfg(cfg, seed)
        episode = _run_condition(
            condition_id="B_homogeneous_reference",
            condition_cfg=reference_cfg,
            action=_sham(reference_cfg, "B_homogeneous_reference"),
            stimulate=False,
            seed=int(seed),
            action_index=0,
            output_dir=root / "reference_B" / str(seed),
            comm=comm,
            size=size,
            rank=rank,
            structure_seed=int(seed),
            drive_seed=int(seed),
            phase_seed=int(seed),
        )
        if rank == 0:
            baseline, _, _ = _episode_feature(episode, "baseline", cfg)
            outcome, _, _ = _episode_feature(episode, "stimulation", cfg)
            row = _epoch_row(episode, "stimulation")
            calibration_rows.append({
                "seed": int(seed),
                "baseline_log10_alpha_power": float(
                    baseline["log10_alpha_power_8_12_hz"]
                ),
                "outcome_log10_alpha_power": float(
                    outcome["log10_alpha_power_8_12_hz"]
                ),
                "E_firing_rate_hz": float(row.E_firing_rate_hz),
                "I_firing_rate_hz": float(row.I_firing_rate_hz),
            })
    if rank == 0:
        calibration = pd.DataFrame(calibration_rows)
        target = _fit_B_target(calibration, cfg)
        target["frozen_C1_threshold"] = float(sources["frozen_C1_threshold"])
        calibration.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_B_target.json").write_text(
            json.dumps(_json_ready(target), indent=2, allow_nan=False)
        )
    else:
        calibration, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    doses = _doses(cfg)
    contexts = _run_context_specs(cfg)
    for context in contexts:
        if rank == 0:
            print(
                f"context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"D={context['diffusion_rad2_per_s']:g} rad^2/s"
            )
        state_cfg = _with_diffusion_state(cfg, {
            "frequency_hz": float(context["hidden_frequency_hz"]),
            "phase_seed": int(context["phase_seed"]),
            "diffusion_rad2_per_s": float(context["diffusion_rad2_per_s"]),
        })
        first_future = _future_seed(cfg, context, 0)
        baseline_reference = _run_replay(
            condition_cfg=state_cfg,
            context=context,
            future_seed=first_future,
            future_index=0,
            dose=SHAM,
            action_index=0,
            root=root,
            comm=comm,
            size=size,
            rank=rank,
        )
        if rank == 0:
            screening = _context_features(
                baseline_reference, context, target, cfg
            )
            screening_rows.append(screening)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'EXCLUDED'}; "
                f"C1={screening['context_C1']:.3f}; "
                f"selected={selected_frequency:g} Hz; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency = None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        if not eligible:
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        n_futures = int(cfg.analysis.crossed_design.n_future_continuations)
        for future_index in range(n_futures):
            future_seed = _future_seed(cfg, context, future_index)
            episodes: dict[float, dict[str, Any]] | None = {} if rank == 0 else None
            for dose_index, dose in enumerate(doses):
                if future_index == 0 and np.isclose(dose, SHAM):
                    episode = baseline_reference
                else:
                    episode = _run_replay(
                        condition_cfg=action_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        dose=dose,
                        action_index=future_index * len(doses) + dose_index,
                        root=root,
                        comm=comm,
                        size=size,
                        rank=rank,
                    )
                if rank == 0:
                    episodes[dose] = episode
            if rank == 0:
                metric_rows.extend(_metric_rows(
                    context=context,
                    screening=screening,
                    future_index=future_index,
                    future_seed=future_seed,
                    episodes=episodes,
                    baseline_reference=baseline_reference,
                    target=target,
                    cfg=cfg,
                ))
        del baseline_reference

    if rank != 0:
        return

    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "D1 exploratory full-information ideal-EEG system identification",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "D1_full_information_action_map_feasible": False,
                "ready_for_disjoint_contextual_policy_confirmation": False,
                "contextual_bandit_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
            "interpretation": "No context passed the prospective EEG screen.",
        }
        (root / "experiment_conclusion.json").write_text(
            json.dumps(_json_ready(conclusion), indent=2, allow_nan=False)
        )
        print("\nNo eligible contexts; D1 feasibility gate: NOT PASSED")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    expected = _expected_action_map(metrics)
    summary, structure, opportunity_audit = _context_action_summary(
        expected, metrics, cfg
    )
    interaction, interaction_audit = _diffusion_dose_interaction(expected, cfg)
    policy = _policy_table(expected, summary, cfg)
    shuffle, shuffle_p = _shuffle_null(expected, summary, policy, cfg)
    checks, conclusions = _checks_and_conclusions(
        calibration=calibration,
        screening=screening,
        metrics=metrics,
        expected=expected,
        summary=summary,
        structure=structure,
        policy=policy,
        interaction=interaction,
        interaction_audit=interaction_audit,
        opportunity_audit=opportunity_audit,
        shuffle_p=shuffle_p,
        sources=sources,
        cfg=cfg,
    )

    metrics.to_csv(root / "context_action_future_metrics.csv", index=False)
    expected.to_csv(root / "expected_context_action_map.csv", index=False)
    summary.to_csv(root / "context_optimal_action_summary.csv", index=False)
    structure.to_csv(root / "structure_level_opportunity.csv", index=False)
    interaction.to_csv(root / "paired_diffusion_dose_interaction.csv", index=False)
    policy.to_csv(root / "exploratory_loso_C1_policy.csv", index=False)
    shuffle.to_csv(root / "structure_preserving_C1_shuffle_null.csv", index=False)
    provenance = {
        "experiment": "D1_phase_diffusion_full_information_action_map",
        "frozen_sources": {
            "roots": sources["roots"],
            "hashes": sources["hashes"],
        },
        "state_generator": {
            "frequencies_hz": [9.0, 11.0],
            "phase_diffusion_rad2_per_s": [0.5, 2.0],
            "modulation_depth": 0.04,
            "shared_latent_phase_private_Poisson_events": True,
        },
        "B_reference": "homogeneous Poisson afferents calibrated on disjoint population seeds",
        "actions": [
            {
                "dose_v_per_m": dose,
                "frequency_policy": "EEG-selected 9/11-Hz carrier",
                "relative_phase_offset_rad": float(
                    cfg.analysis.tacs.relative_phase_offset_rad
                ),
                "phase_estimation_history_s": float(
                    cfg.analysis.tacs.phase_estimation_steps
                    * cfg.env.simulation.obs_win_len / 1000.0
                ),
                "one_constant_action_for_complete_intervention": True,
            }
            for dose in doses
        ],
        "learnable_policy_features": ["context_C1"],
        "frequency_and_recent_phase_are_deterministic_signal_processing": True,
        "hidden_variables_used_for_policy": False,
        "statistical_unit": "independent circuit structure",
        "histories_frequencies_diffusion_levels_and_futures_are_repeats": True,
        "not_a_bandit_or_confirmatory_experiment": True,
        "not_a_disease_or_human_treatment_model": True,
    }
    (root / "protocol_and_provenance.json").write_text(
        json.dumps(_json_ready(provenance), indent=2, allow_nan=False)
    )
    conclusion = {
        "scope": "D1 exploratory ideal-neural-EEG full-information system identification",
        "checks": checks,
        "conclusions": conclusions,
        "runtime_seconds": float(time.perf_counter() - started),
        "statistical_unit": "independent circuit structure; other axes are repeated measures",
        "inference_boundary": (
            "directional feasibility only; a positive map must be frozen and "
            "tested on new structures before policy or bandit claims"
        ),
    }
    (root / "experiment_conclusion.json").write_text(
        json.dumps(_json_ready(conclusion), indent=2, allow_nan=False)
    )
    if bool(cfg.experiment.plot):
        _plot_results(
            root=root,
            screening=screening,
            expected=expected,
            summary=summary,
            structure=structure,
            policy=policy,
            sources=sources,
        )

    print("\n### D1 screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### D1 full-information action-map checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print(
        "\nD1 full-information action map: "
        f"{'PASSED' if conclusions['D1_full_information_action_map_feasible'] else 'NOT PASSED'}"
    )
    print(
        "Ready for disjoint contextual policy confirmation: "
        f"{'YES' if conclusions['ready_for_disjoint_contextual_policy_confirmation'] else 'NO'}"
    )
    print("Contextual bandit status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
