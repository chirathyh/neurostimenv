"""H5-O0: EEG-observable orientation by montage-profile opportunity map.

This bounded exploratory study introduces one deliberately transparent source
of response heterogeneity: the common somatodendritic orientation of the toy
population.  Two equal-norm local tissue-field vectors represent two
precomputed tACS electrode-current/montage profiles.  The elevated-rhythm A
state, homogeneous mean-rate-matched B reference, recurrent circuit, carrier
estimator, field amplitude, and H4-confirmed causal phase tracker remain fixed.

The experiment first calibrates orientation-specific population-B EEG targets,
then prospectively screens A from stimulation-free multichannel ideal EEG.  It
maps sham and both montage profiles through paired stochastic futures.  Hidden
orientation is an audit label only.  No policy is fitted and H5 cannot be
claimed from this feasibility study.
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
from lfpykit.eegmegcalc import FourSphereVolumeConductor
from mpi4py import MPI
from omegaconf import DictConfig, OmegaConf, open_dict


MAIN_PATH = config("MAIN_PATH")
sys.path.insert(1, MAIN_PATH)

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (  # noqa: E402
    _epoch_row,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (  # noqa: E402
    FULL,
    RESPONSIVE,
    _future_seed,
    _run_controller,
    _with_context_state,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (  # noqa: E402
    MT_POOLED,
    OBSERVED,
    _estimate_multitaper_methods,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_response_mapping import (  # noqa: E402
    _hash_locked_files,
)
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (  # noqa: E402
    _fourier_coefficients,
    _process_eeg,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (  # noqa: E402
    SHAM,
    _field_removal_status,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (  # noqa: E402
    _json_ready,
    _profile,
)
from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (  # noqa: E402
    _relative_rms_error,
)
from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (  # noqa: E402
    _relative_rate_safe,
)
from experiments.ballnstick_analysis.run_ballnstick_frequency_phase_feasibility import (  # noqa: E402
    _with_action_frequency,
)


ROOT_NAME = "h5_montage_orientation_opportunity"
PROFILE_Z = "montage_profile_z"
PROFILE_60 = "montage_profile_60deg"
ACTIVE_PROFILES = [PROFILE_Z, PROFILE_60]
ALL_ACTIONS = [SHAM, *ACTIVE_PROFILES]
TOPOGRAPHY_FEATURES = [
    "topography_vertex",
    "topography_right_xz",
    "topography_left_xz",
]


def _load_sources(cfg: DictConfig) -> dict[str, Any]:
    """Hash-lock the final negative H5 kinetics/dose branch."""
    root = Path(to_absolute_path(str(cfg.analysis.source_h5k0.result_dir)))
    names = {
        "conclusion": "experiment_conclusion.json",
        "audit": "H5_K0_susceptibility_opportunity_audit.json",
        "screening": "prospective_screening.csv",
        "metrics": "context_dose_future_metrics.csv",
        "expected_map": "expected_context_dose_map.csv",
        "opportunity": "dose_response_opportunity.csv",
        "future_split": "independent_future_split_validation.csv",
        "provenance": "protocol_and_provenance.json",
    }
    files, hashes = _hash_locked_files(
        root, names, cfg.analysis.source_h5k0.expected_sha256
    )
    conclusion = json.loads(files["conclusion"].read_text())
    if (
        conclusion["conclusions"]["H5_K0_inhibitory_kinetics_dose_opportunity"]
        != "NOT PASSED"
        or bool(conclusion["conclusions"]["ready_for_H5_dose_policy_development"])
    ):
        raise RuntimeError("H5-O0 requires the exact frozen negative H5-K0 result.")
    provenance = json.loads(files["provenance"].read_text())
    if not bool(provenance["frozen_population_B_target"][
        "target_is_population_reference_not_seed_specific"
    ]):
        raise RuntimeError("H5-K0 provenance no longer contains a population target.")
    seeds: set[int] = set()
    for key in ("screening", "metrics"):
        table = pd.read_csv(files[key])
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "future_drive_seed",
        ):
            if column in table:
                seeds.update(table[column].dropna().astype(int).tolist())
    return {
        "root": str(root),
        "hashes": hashes,
        "source_seed_union": seeds,
        "H5K0_negative_preserved": True,
        "upstream_provenance": provenance["frozen_sources"],
    }


def _orientation_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "orientation_label": str(value.label),
        "rotation_y_rad": float(value.rotation_y_rad),
        "orientation_axis_x": float(value.axis[0]),
        "orientation_axis_y": float(value.axis[1]),
        "orientation_axis_z": float(value.axis[2]),
        "matched_profile": str(value.matched_profile),
    } for value in cfg.analysis.states.orientations]


def _profile_specs(cfg: DictConfig) -> list[dict[str, Any]]:
    return [{
        "montage_profile": str(value.id),
        "field_x": float(value.field_direction[0]),
        "field_y": float(value.field_direction[1]),
        "field_z": float(value.field_direction[2]),
    } for value in cfg.analysis.actions.montage_profiles]


def _contexts(cfg: DictConfig, *, apply_smoke_limit: bool = True) -> list[dict[str, Any]]:
    block = cfg.analysis.crossed_design
    base = int(cfg.experiment.seed)
    diffusion = cfg.analysis.states.phase_diffusion_levels[0]
    rows: list[dict[str, Any]] = []
    future_group = 0
    for structure_index in range(int(block.n_structure_seeds)):
        structure_seed = base + int(block.structure_seed_offset) + structure_index
        history_seed = base + int(block.history_seed_offset) + structure_index
        for frequency_index, frequency in enumerate(cfg.analysis.states.frequencies_hz):
            phase_seed = (
                base + int(block.phase_seed_offset)
                + 10 * structure_index + frequency_index
            )
            trial_seed = base + int(block.trial_seed_offset) + future_group
            pair_id = f"s{structure_index:02d}_f{int(round(float(frequency))):02d}"
            for orientation_index, orientation in enumerate(_orientation_specs(cfg)):
                rows.append({
                    "context_order": len(rows),
                    "future_group_index": future_group,
                    "context_id": (
                        f"{pair_id}_o{orientation_index:02d}_"
                        f"{orientation['orientation_label']}"
                    ),
                    "paired_orientation_context_id": pair_id,
                    "structure_index": structure_index,
                    "structure_seed": structure_seed,
                    "history_index": 0,
                    "history_seed": history_seed,
                    "phase_seed": phase_seed,
                    "trial_seed": trial_seed,
                    "hidden_frequency_hz": float(frequency),
                    "label": str(diffusion.label),
                    "diffusion_rad2_per_s": float(diffusion.diffusion_rad2_per_s),
                    "shared_drive_label": FULL,
                    "shared_modulated_fraction": 1.0,
                    **orientation,
                })
            future_group += 1
    if (
        apply_smoke_limit and bool(cfg.analysis.smoke_test)
        and int(cfg.analysis.smoke_context_limit) > 0
    ):
        return rows[:int(cfg.analysis.smoke_context_limit)]
    return rows


def _reference_contexts(cfg: DictConfig) -> list[dict[str, Any]]:
    block = cfg.analysis.reference_calibration
    base = int(cfg.experiment.seed)
    n_structures = 1 if bool(cfg.analysis.smoke_test) else int(block.n_structure_seeds)
    rows: list[dict[str, Any]] = []
    for structure_index in range(n_structures):
        for orientation_index, orientation in enumerate(_orientation_specs(cfg)):
            group = 1000 + 10 * structure_index + orientation_index
            rows.append({
                "context_order": len(rows),
                "future_group_index": group,
                "context_id": f"B_s{structure_index:02d}_{orientation['orientation_label']}",
                "paired_orientation_context_id": f"B_s{structure_index:02d}",
                "structure_index": structure_index,
                "structure_seed": base + int(block.structure_seed_offset) + structure_index,
                "history_index": 0,
                "history_seed": base + int(block.history_seed_offset) + structure_index,
                "phase_seed": base + int(block.phase_seed_offset) + structure_index,
                "trial_seed": base + int(block.trial_seed_offset) + group,
                "reference_future_seed": base + int(block.future_seed_offset) + group,
                "hidden_frequency_hz": 9.0,
                "label": "homogeneous_reference",
                "diffusion_rad2_per_s": 0.0,
                "shared_drive_label": "homogeneous_reference",
                "shared_modulated_fraction": 1.0,
                **orientation,
            })
    return rows


def _field_projection(context: dict[str, Any], profile: dict[str, Any]) -> float:
    axis = np.asarray([
        context["orientation_axis_x"], context["orientation_axis_y"],
        context["orientation_axis_z"],
    ], dtype=float)
    field = np.asarray([
        profile["field_x"], profile["field_y"], profile["field_z"],
    ], dtype=float)
    axis /= np.linalg.norm(axis)
    field /= np.linalg.norm(field)
    return float(abs(np.dot(axis, field)))


def _head_from_local_rotation(context: dict[str, Any]) -> np.ndarray:
    """Return the proper y-axis rotation from local column to head space."""
    theta = float(context["rotation_y_rad"])
    cosine, sine = np.cos(theta), np.sin(theta)
    return np.asarray([
        [cosine, 0.0, sine],
        [0.0, 1.0, 0.0],
        [-sine, 0.0, cosine],
    ], dtype=float)


def _validate_design(cfg: DictConfig, sources: dict[str, Any]) -> None:
    smoke = bool(cfg.analysis.smoke_test)
    if not bool(sources["H5K0_negative_preserved"]):
        raise ValueError("The negative H5-K0 source was not preserved.")
    if str(cfg.analysis.simulator) != "online":
        raise ValueError("H5-O0 requires persistent online simulation.")
    if not np.isclose(float(cfg.analysis.inhibition_scale), 1.0):
        raise ValueError("H5-O0 may not alter recurrent inhibition.")
    if [float(x) for x in cfg.analysis.states.frequencies_hz] != [9.0, 11.0]:
        raise ValueError("H5-O0 freezes the 9/11-Hz carrier grid.")
    diffusions = list(cfg.analysis.states.phase_diffusion_levels)
    if len(diffusions) != 1 or not np.isclose(
        float(diffusions[0].diffusion_rad2_per_s), 0.5
    ):
        raise ValueError("H5-O0 fixes D=0.5 rad^2/s.")
    shared = list(cfg.analysis.states.shared_drive_levels)
    if len(shared) != 1 or not np.isclose(
        float(shared[0].shared_modulated_fraction), 1.0
    ):
        raise ValueError("H5-O0 fixes the shared rhythmic-afferent fraction to one.")
    if not np.isclose(float(cfg.analysis.states.modulation_depth), 0.04):
        raise ValueError("H5-O0 freezes afferent modulation depth 0.04.")
    orientations = _orientation_specs(cfg)
    if [x["orientation_label"] for x in orientations] != [
        "orientation_0deg", "orientation_60deg"
    ]:
        raise ValueError("H5-O0 requires the predeclared 0/60-degree orientations.")
    profiles = _profile_specs(cfg)
    if [x["montage_profile"] for x in profiles] != ACTIVE_PROFILES:
        raise ValueError("H5-O0 requires exactly the z and 60-degree profiles.")
    for vector in [
        [x["orientation_axis_x"], x["orientation_axis_y"], x["orientation_axis_z"]]
        for x in orientations
    ] + [[x["field_x"], x["field_y"], x["field_z"]] for x in profiles]:
        if not np.isclose(np.linalg.norm(vector), 1.0, atol=1.0e-12):
            raise ValueError("Every orientation and field direction must be unit norm.")
    projection = {
        (context["orientation_label"], profile["montage_profile"]):
        _field_projection(context, profile)
        for context in orientations for profile in profiles
    }
    if not (
        np.isclose(projection[("orientation_0deg", PROFILE_Z)], 1.0)
        and np.isclose(projection[("orientation_0deg", PROFILE_60)], 0.5)
        and np.isclose(projection[("orientation_60deg", PROFILE_Z)], 0.5)
        and np.isclose(projection[("orientation_60deg", PROFILE_60)], 1.0)
    ):
        raise ValueError("H5-O0 field-projection matrix changed.")
    if not np.isclose(float(cfg.analysis.actions.amplitude_v_per_m), 0.2):
        raise ValueError("Both active montage profiles must use 0.2 V/m.")
    if _profile(cfg, RESPONSIVE) != {
        "adaptive": True, "history_ms": 500.0, "update_interval_ms": 125.0,
    }:
        raise ValueError("The H4-confirmed fast phase controller changed.")
    if not bool(cfg.analysis.observation_noise.enabled) or not np.isclose(
        float(cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg), 0.0
    ):
        raise ValueError("H5-O0 is an ideal-neural-EEG mechanism screen.")
    locations = np.asarray(cfg.analysis.eeg_array.locations_um, dtype=float)
    if locations.shape != (3, 3) or not np.allclose(
        np.linalg.norm(locations, axis=1), 90000.0, atol=1.0e-6
    ):
        raise ValueError("H5-O0 requires the frozen three-sensor scalp array.")
    if not smoke and (
        int(cfg.analysis.timeline.baseline_steps) != 30
        or int(cfg.analysis.timeline.stimulation_steps) != 9
        or int(cfg.analysis.timeline.washout_steps) != 2
        or int(cfg.analysis.reference_calibration.n_structure_seeds) != 3
        or int(cfg.analysis.crossed_design.n_structure_seeds) != 3
        or int(cfg.analysis.crossed_design.n_future_continuations) != 4
    ):
        raise ValueError("The full H5-O0 design is frozen to 30/9/2 s and 3x4.")
    endpoint_ms = (
        int(cfg.analysis.timeline.stimulation_steps)
        * float(cfg.env.simulation.obs_win_len)
        - 2.0 * float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
    )
    if endpoint_ms <= 0.0 or (not smoke and not np.isclose(endpoint_ms, 8000.0)):
        raise ValueError("H5-O0 requires the central eight-second endpoint.")
    contexts = _contexts(cfg, apply_smoke_limit=False)
    if len(contexts) != int(cfg.analysis.crossed_design.n_structure_seeds) * 4:
        raise ValueError("The crossed frequency-orientation grid is incomplete.")
    if not all(group.orientation_label.nunique() == 2 for _, group in
               pd.DataFrame(contexts).groupby("paired_orientation_context_id")):
        raise ValueError("Every biological history must be paired across orientation.")
    references = _reference_contexts(cfg)
    namespace_sets = [
        {int(row[column]) for row in contexts}
        for column in ("structure_seed", "history_seed", "phase_seed", "trial_seed")
    ]
    namespace_sets.append({
        _future_seed(cfg, row, future)
        for row in contexts
        for future in range(int(cfg.analysis.crossed_design.n_future_continuations))
    })
    namespace_sets.extend([
        {int(row[column]) for row in references}
        for column in (
            "structure_seed", "history_seed", "phase_seed", "trial_seed",
            "reference_future_seed",
        )
    ])
    if any(
        namespace_sets[i].intersection(namespace_sets[j])
        for i in range(len(namespace_sets)) for j in range(i + 1, len(namespace_sets))
    ):
        raise ValueError("H5-O0 seed namespaces overlap.")
    if set().union(*namespace_sets).intersection(sources["source_seed_union"]):
        raise ValueError("H5-O0 seeds overlap the frozen H5 lineage.")
    if max(namespace_sets[0] | namespace_sets[5]) * 10_000 > np.iinfo(np.uint32).max:
        raise ValueError("H5-O0 structure seed exceeds the uint32 mapping.")


def _add_profiles(cfg: DictConfig) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(result):
        for profile in cfg.analysis.actions.montage_profiles:
            result.env.online.stimulation.montages[str(profile.id)] = [
                float(x) for x in profile.field_direction
            ]
    return result


def _with_orientation_state(
    cfg: DictConfig, context: dict[str, Any], *, homogeneous_B: bool = False,
) -> DictConfig:
    result = _with_context_state(_add_profiles(cfg), context)
    with open_dict(result):
        # Keep morphology, background synapse locations, and recurrent
        # synapse locations in one canonical local-column frame.  Rotating the
        # LFPy cells here would change this model's z-dependent synapse-location
        # selection and would therefore confound anatomy with circuit state.
        result.env.network.population.rotation_x_rad = 0.0
        result.env.network.population.rotation_y_rad = 0.0
        # The online phase tracker must observe the same head-frame dipole as
        # the offline multichannel endpoint.  This optional transform is absent
        # from every legacy/H1--H4 configuration, whose EEG path is unchanged.
        result.env.eeg.dipole_rotation_matrix = _head_from_local_rotation(
            context
        ).tolist()
        if homogeneous_B:
            for population in ("E", "I"):
                rhythm = result.env.network.background[population].rhythm
                rhythm.enabled = True
                rhythm.modulation_depth = 0.0
                rhythm.phase_diffusion_rad2_per_s = 0.0
                # Preserve the same candidate-process envelope and mean rate.
                rhythm.thinning_envelope_modulation_depth = float(
                    cfg.analysis.reference.thinning_envelope_modulation_depth
                )
    return result


def _with_profile(
    cfg: DictConfig, context: dict[str, Any], profile: str, *,
    phase_sensor_index: int,
) -> DictConfig:
    result = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    locations = np.asarray(cfg.analysis.eeg_array.locations_um, dtype=float)
    specification = next(
        value for value in _profile_specs(cfg)
        if value["montage_profile"] == profile
    )
    head_field = np.asarray([
        specification["field_x"], specification["field_y"],
        specification["field_z"],
    ], dtype=float)
    # A head-frame montage field acts on a canonical local-column model as
    # E_local = R_local_to_head.T @ E_head.
    local_field = _head_from_local_rotation(context).T @ head_field
    with open_dict(result):
        result.analysis.actions.amplitude_v_per_m = float(
            cfg.analysis.actions.amplitude_v_per_m
        )
        result.analysis.actions.controller_modes = [SHAM, RESPONSIVE]
        result.analysis.tacs.axial_montage = str(profile)
        result.env.online.stimulation.montages[str(profile)] = local_field.tolist()
        result.env.eeg.locations = [locations[int(phase_sensor_index)].tolist()]
    return result


def _run_profile(
    *, condition_cfg: DictConfig, context: dict[str, Any], future_seed: int,
    future_index: int, profile: str, phase_sensor_index: int, root: Path,
    comm: Any, size: int, rank: int,
) -> dict[str, Any] | None:
    mode = SHAM if profile == SHAM else RESPONSIVE
    configured_profile = PROFILE_Z if profile == SHAM else profile
    run_cfg = _with_profile(
        condition_cfg, context, configured_profile,
        phase_sensor_index=phase_sensor_index
    )
    episode = _run_controller(
        condition_cfg=run_cfg,
        context=context,
        future_seed=future_seed,
        future_index=future_index,
        mode=mode,
        action_index=ALL_ACTIONS.index(profile),
        root=root / "episodes_by_profile" / profile,
        comm=comm,
        size=size,
        rank=rank,
    )
    if episode is not None:
        episode["simulation"]["action"].update({
            "id": profile,
            "role": "H5_O0_fixed_montage_profile",
            "montage_profile": profile,
            "phase_sensor_index": int(phase_sensor_index),
            "one_profile_for_complete_intervention": True,
        })
    return episode


def _dipole_by_epoch(episode: dict[str, Any], epoch: str) -> np.ndarray:
    chunks = []
    for output in episode["simulation"]["outputs_by_epoch"][epoch]:
        probes = output.get("probe_data")
        if probes is None or len(probes) < 2:
            raise RuntimeError("Current-dipole probe missing from online output.")
        value = np.asarray(probes[1], dtype=float)
        if value.ndim != 2 or value.shape[0] != 3:
            raise RuntimeError(f"Expected (3,n) current dipole, got {value.shape}.")
        chunks.append(value)
    return np.concatenate(chunks, axis=1)


def _multichannel_raw(
    episode: dict[str, Any], epoch: str, context: dict[str, Any],
    cfg: DictConfig, *, trim: bool = False,
) -> tuple[np.ndarray, float]:
    local_dipole = _dipole_by_epoch(episode, epoch)
    dipole = _head_from_local_rotation(context) @ local_dipole
    start_ms = float(
        episode["simulation"]["outputs_by_epoch"][epoch][0]["t_start_ms"]
    )
    if trim:
        trim_ms = float(cfg.analysis.timeline.stimulation_analysis_trim_ms)
        count = int(round(trim_ms / float(cfg.env.network.dt)))
        if count:
            dipole = dipole[:, count:-count]
            start_ms += trim_ms
    conductor = FourSphereVolumeConductor(
        np.asarray(cfg.analysis.eeg_array.locations_um, dtype=float),
        cfg.env.eeg.foursphereheadmodel["radii"],
        cfg.env.eeg.foursphereheadmodel["sigmas"],
    )
    eeg = np.asarray(conductor.get_dipole_potential(
        dipole, np.asarray(cfg.env.network.position, dtype=float)
    ), dtype=float) * 1.0e-3
    if eeg.shape != (len(cfg.analysis.eeg_array.labels), dipole.shape[1]):
        raise RuntimeError(f"Unexpected multichannel EEG shape {eeg.shape}.")
    if not np.all(np.isfinite(eeg)):
        raise RuntimeError("Multichannel EEG contains non-finite samples.")
    return eeg, start_ms


def _multichannel_features(
    eeg: np.ndarray, *, start_ms: float, carrier_hz: float, cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    powers: list[float] = []
    carrier_powers: list[float] = []
    spectra: list[pd.DataFrame] = []
    processed_channels: list[np.ndarray] = []
    processed_fs = None
    for index, (label, values) in enumerate(zip(cfg.analysis.eeg_array.labels, eeg)):
        processed, fs_hz, frequencies, psd, features = _process_eeg(
            values,
            simulator_fs_hz=1000.0 / float(cfg.env.network.dt),
            cfg=cfg,
        )
        if processed_fs is not None and not np.isclose(processed_fs, fs_hz):
            raise RuntimeError("EEG channels obtained different processed rates.")
        processed_fs = float(fs_hz)
        processed_channels.append(processed)
        powers.append(float(features["alpha_power"]))
        carrier_mask = np.abs(frequencies - float(carrier_hz)) <= 0.75
        carrier_indices = np.flatnonzero(carrier_mask)
        if carrier_indices.size >= 2:
            carrier_power = float(np.trapz(
                psd[carrier_indices], frequencies[carrier_indices]
            ))
        elif carrier_indices.size == 1:
            # A one-second smoke endpoint has 1-Hz Welch bins and therefore
            # only one bin inside the frozen +/-0.75-Hz carrier band.  Treat
            # its density as one-bin energy; full 8/30-s analyses integrate
            # multiple bins and never use this fallback.
            spacing = float(np.median(np.diff(frequencies)))
            carrier_power = float(psd[carrier_indices[0]] * spacing)
        else:
            raise RuntimeError(
                f"No PSD bin lies within 0.75 Hz of {carrier_hz:g} Hz."
            )
        carrier_powers.append(carrier_power)
        spectra.append(pd.DataFrame({
            "sensor_index": index,
            "sensor_label": str(label),
            "frequency_hz": frequencies,
            "PSD_v2_per_hz": psd,
        }))
    total_alpha = float(np.sum(powers))
    total_carrier = float(np.sum(carrier_powers))
    tiny = np.finfo(float).tiny
    normalized = np.asarray(carrier_powers) / max(total_carrier, tiny)
    phase_sensor = int(np.argmax(powers))
    recent_count = int(round(1.0 * float(processed_fs)))
    recent = processed_channels[phase_sensor][-recent_count:]
    cosine, sine = _fourier_coefficients(
        recent,
        fs_hz=float(processed_fs),
        start_ms=float(start_ms + eeg.shape[1] * cfg.env.network.dt - 1000.0),
        frequency_hz=float(carrier_hz),
    )
    recent_rms = float(np.sqrt(np.mean(recent ** 2)))
    result = {
        "global_alpha_power_v2": total_alpha,
        "global_log10_alpha_power": float(np.log10(max(total_alpha, tiny))),
        "global_carrier_power_v2": total_carrier,
        "phase_sensor_index": phase_sensor,
        "phase_sensor_label": str(cfg.analysis.eeg_array.labels[phase_sensor]),
        "recent_resultant_to_rms": float(
            np.hypot(cosine, sine) / max(recent_rms, tiny)
        ),
    }
    for name, value in zip(TOPOGRAPHY_FEATURES, normalized):
        result[name] = float(value)
    return result, pd.concat(spectra, ignore_index=True)


def _spike_hash(episode: dict[str, Any], epoch: str) -> str:
    digest = hashlib.sha256()
    for population in ("E", "I"):
        for output in episode["simulation"]["outputs_by_epoch"][epoch]:
            values = np.asarray(
                output["spikes"][population]["times_ms"], dtype="<f8"
            )
            digest.update(population.encode())
            digest.update(values.tobytes())
    return digest.hexdigest()


def _dipole_norm_rms(episode: dict[str, Any], epoch: str) -> float:
    dipole = _dipole_by_epoch(episode, epoch)
    return float(np.sqrt(np.mean(np.sum(dipole ** 2, axis=0))))


def _carrier_screen(
    episode: dict[str, Any], context: dict[str, Any], target: dict[str, Any],
    cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    eeg, start_ms = _multichannel_raw(episode, "baseline", context, cfg)
    preliminary, spectrum = _multichannel_features(
        eeg, start_ms=start_ms,
        carrier_hz=float(context["hidden_frequency_hz"]), cfg=cfg,
    )
    selected_sensor = int(preliminary["phase_sensor_index"])
    if bool(cfg.analysis.smoke_test) and bool(cfg.analysis.smoke_force_eligible):
        selected_frequency = float(context["hidden_frequency_hz"])
        identified = True
        evidence = float("nan")
        margin = float("nan")
    else:
        processed, fs_hz, _, _, _ = _process_eeg(
            eeg[selected_sensor],
            simulator_fs_hz=1000.0 / float(cfg.env.network.dt), cfg=cfg,
        )
        rows, _, _ = _estimate_multitaper_methods(
            processed,
            fs_hz=fs_hz,
            hidden_frequency_hz=float(context["hidden_frequency_hz"]),
            input_signal=OBSERVED,
            cfg=cfg,
        )
        selected = next(row for row in rows if row["estimator"] == MT_POOLED)
        selected_frequency = float(selected["selected_frequency_hz"])
        identified = bool(selected["identified"])
        evidence = float(selected["maximum_residual_evidence_db"])
        margin = float(selected["evidence_margin_db"])
    # Recompute the carrier-normalized topography at the EEG-selected carrier.
    features, spectrum = _multichannel_features(
        eeg, start_ms=start_ms, carrier_hz=selected_frequency, cfg=cfg
    )
    orientation_target = target[str(context["orientation_label"])]
    alpha_excess = (
        float(features["global_log10_alpha_power"])
        - float(orientation_target["screening_mean_log10_alpha"])
    )
    outcome = _epoch_row(episode, "baseline")
    criteria = cfg.analysis.criteria
    phenotype = alpha_excess >= float(criteria.minimum_A_minus_B_alpha_log10)
    phase_actionable = (
        float(features["recent_resultant_to_rms"])
        >= float(criteria.minimum_recent_resultant_to_rms)
    )
    eligible = bool(identified and phenotype and phase_actionable)
    if bool(cfg.analysis.smoke_test) and bool(cfg.analysis.smoke_force_eligible):
        eligible = True
    reasons = []
    if not identified:
        reasons.append("carrier_estimator_abstained")
    if not phenotype:
        reasons.append("elevated_alpha_phenotype_absent")
    if not phase_actionable:
        reasons.append("recent_phase_not_actionable")
    row = {
        **context,
        **features,
        "EEG_selected_frequency_hz": selected_frequency,
        "carrier_identified": identified,
        "EEG_frequency_selection_correct": bool(np.isclose(
            selected_frequency, float(context["hidden_frequency_hz"])
        )),
        "carrier_maximum_residual_evidence_db": evidence,
        "carrier_evidence_margin_db": margin,
        "alpha_excess_over_orientation_B_log10": alpha_excess,
        "alpha_phenotype_present": phenotype,
        "recent_phase_actionable": phase_actionable,
        "eligible": eligible,
        "exclusion_reasons": ";".join(reasons) if reasons else "none",
        "baseline_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
        "baseline_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
        "baseline_spike_sha256": _spike_hash(episode, "baseline"),
        "baseline_dipole_norm_rms_nA_um": _dipole_norm_rms(episode, "baseline"),
        "screening_uses_only_predecision_multichannel_ideal_EEG": True,
        "hidden_orientation_used_only_for_audit_and_target_stratification": True,
    }
    spectrum = spectrum.assign(
        context_id=str(context["context_id"]),
        structure_seed=int(context["structure_seed"]),
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        orientation_label=str(context["orientation_label"]),
        condition="A_predecision",
    )
    return row, spectrum


def _reference_target(
    rows: pd.DataFrame,
) -> dict[str, Any]:
    target: dict[str, Any] = {}
    for orientation, group in rows.groupby("orientation_label"):
        target[str(orientation)] = {
            "screening_mean_log10_alpha": float(
                group.screening_log10_alpha_power.mean()
            ),
            "screening_sd_log10_alpha": float(
                group.screening_log10_alpha_power.std(ddof=1)
            ) if len(group) > 1 else 0.0,
            "outcome_mean_log10_alpha": float(group.outcome_log10_alpha_power.mean()),
            "outcome_sd_log10_alpha": float(
                group.outcome_log10_alpha_power.std(ddof=1)
            ) if len(group) > 1 else 0.0,
            "n_reference_structures": int(group.structure_seed.nunique()),
        }
    return target


def _paired_orientation_baseline_audit(
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair_id, group in screening.groupby("paired_orientation_context_id"):
        if group.orientation_label.nunique() != 2:
            continue
        zero = group[group.orientation_label.eq("orientation_0deg")].iloc[0]
        tilted = group[group.orientation_label.eq("orientation_60deg")].iloc[0]
        denominator = max(
            abs(float(zero.baseline_dipole_norm_rms_nA_um)),
            np.finfo(float).tiny,
        )
        rows.append({
            "paired_orientation_context_id": str(pair_id),
            "structure_seed": int(zero.structure_seed),
            "hidden_frequency_hz": float(zero.hidden_frequency_hz),
            "spike_trains_identical": bool(
                zero.baseline_spike_sha256 == tilted.baseline_spike_sha256
            ),
            "absolute_E_rate_difference_hz": abs(
                float(zero.baseline_E_firing_rate_hz)
                - float(tilted.baseline_E_firing_rate_hz)
            ),
            "absolute_I_rate_difference_hz": abs(
                float(zero.baseline_I_firing_rate_hz)
                - float(tilted.baseline_I_firing_rate_hz)
            ),
            "dipole_norm_relative_error": abs(
                float(zero.baseline_dipole_norm_rms_nA_um)
                - float(tilted.baseline_dipole_norm_rms_nA_um)
            ) / denominator,
        })
    table = pd.DataFrame(rows)
    if table.empty:
        return table, {
            "all_spike_trains_identical": False,
            "maximum_rate_difference_hz": float("inf"),
            "maximum_dipole_norm_relative_error": float("inf"),
        }
    return table, {
        "all_spike_trains_identical": bool(table.spike_trains_identical.all()),
        "maximum_rate_difference_hz": float(max(
            table.absolute_E_rate_difference_hz.max(),
            table.absolute_I_rate_difference_hz.max(),
        )),
        "maximum_dipole_norm_relative_error": float(
            table.dipole_norm_relative_error.max()
        ),
    }


def _orientation_loso(
    screening: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = screening[screening.eligible.astype(bool)].copy()
    rows: list[dict[str, Any]] = []
    for structure in sorted(eligible.structure_seed.unique()):
        train = eligible[eligible.structure_seed.ne(structure)]
        test = eligible[eligible.structure_seed.eq(structure)]
        if train.empty or test.empty or train.orientation_label.nunique() < 2:
            continue
        center = train[TOPOGRAPHY_FEATURES].mean().to_numpy(float)
        scale = train[TOPOGRAPHY_FEATURES].std(ddof=0).to_numpy(float)
        scale[~np.isfinite(scale) | (scale <= np.finfo(float).tiny)] = 1.0
        centroids = {
            str(label): (
                group[TOPOGRAPHY_FEATURES].mean().to_numpy(float) - center
            ) / scale
            for label, group in train.groupby("orientation_label")
        }
        for sample in test.itertuples():
            vector = (
                np.asarray([getattr(sample, name) for name in TOPOGRAPHY_FEATURES])
                - center
            ) / scale
            distances = {
                label: float(np.linalg.norm(vector - centroid))
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=lambda x: (distances[x], x))
            rows.append({
                "context_id": str(sample.context_id),
                "structure_seed": int(structure),
                "true_orientation_label": str(sample.orientation_label),
                "predicted_orientation_label": str(predicted),
                "correct": predicted == str(sample.orientation_label),
                "classifier": "LOSO standardized nearest centroid",
                "features": ";".join(TOPOGRAPHY_FEATURES),
            })
    predictions = pd.DataFrame(rows)
    recalls = {
        str(label): float(group.correct.mean())
        for label, group in predictions.groupby("true_orientation_label")
    } if not predictions.empty else {}
    return predictions, {
        "LOSO_balanced_accuracy": (
            float(np.mean(list(recalls.values()))) if recalls else float("nan")
        ),
        "LOSO_recall": recalls,
        "features": TOPOGRAPHY_FEATURES,
        "orientation_label_used_only_for_scoring": True,
        "observability_audit_not_a_trained_policy": True,
    }


def _representative_voltage_coefficients(
    episode: dict[str, Any], frequency_hz: float, cfg: DictConfig,
) -> dict[str, complex]:
    values: dict[str, dict[str, list[np.ndarray]]] = {}
    outputs = episode["simulation"]["outputs_by_epoch"]["stimulation"]
    for output in outputs:
        for site, signals in output.get("representative_state", {}).items():
            destination = values.setdefault(site, {})
            for name, samples in signals.items():
                destination.setdefault(name, []).append(np.asarray(samples, dtype=float))
    result: dict[str, complex] = {}
    interval_ms = float(outputs[0]["representative_state_interval_ms"])
    trim_count = int(round(
        float(cfg.analysis.timeline.stimulation_analysis_trim_ms) / interval_ms
    ))
    start_ms = float(outputs[0]["t_start_ms"]) + float(
        cfg.analysis.timeline.stimulation_analysis_trim_ms
    )
    for site, signals in values.items():
        if "soma_voltage_mV" not in signals or "apic_distal_voltage_mV" not in signals:
            continue
        differential = (
            np.concatenate(signals["apic_distal_voltage_mV"])
            - np.concatenate(signals["soma_voltage_mV"])
        )
        if trim_count:
            differential = differential[trim_count:-trim_count]
        cosine, sine = _fourier_coefficients(
            differential,
            fs_hz=1000.0 / interval_ms,
            start_ms=start_ms,
            frequency_hz=frequency_hz,
        )
        result[site] = complex(cosine, sine)
    return result


def _metric_row(
    *, episode: dict[str, Any], sham: dict[str, Any], baseline: dict[str, Any],
    context: dict[str, Any], screening: dict[str, Any], target: dict[str, Any],
    profile: str, future_index: int, future_seed: int, cfg: DictConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    active_eeg, start_ms = _multichannel_raw(
        episode, "stimulation", context, cfg, trim=True
    )
    sham_eeg, sham_start_ms = _multichannel_raw(
        sham, "stimulation", context, cfg, trim=True
    )
    active_features, spectra = _multichannel_features(
        active_eeg, start_ms=start_ms,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    sham_features, _ = _multichannel_features(
        sham_eeg, start_ms=sham_start_ms,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    target_alpha = float(target[str(context["orientation_label"])][
        "outcome_mean_log10_alpha"
    ])
    distance = abs(float(active_features["global_log10_alpha_power"]) - target_alpha)
    sham_distance = abs(float(sham_features["global_log10_alpha_power"]) - target_alpha)
    outcome = _epoch_row(episode, "stimulation")
    sham_outcome = _epoch_row(sham, "stimulation")
    baseline_outcome = _epoch_row(episode, "baseline")
    sham_baseline_outcome = _epoch_row(sham, "baseline")
    washout_outcome = _epoch_row(episode, "washout")
    sham_washout_outcome = _epoch_row(sham, "washout")
    episode_baseline, baseline_start = _multichannel_raw(
        episode, "baseline", context, cfg
    )
    common_baseline, _ = _multichannel_raw(baseline, "baseline", context, cfg)
    active_washout, washout_start = _multichannel_raw(
        episode, "washout", context, cfg
    )
    sham_washout, sham_washout_start = _multichannel_raw(
        sham, "washout", context, cfg
    )
    washout_features, _ = _multichannel_features(
        active_washout, start_ms=washout_start,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    sham_washout_features, _ = _multichannel_features(
        sham_washout, start_ms=sham_washout_start,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    baseline_features, _ = _multichannel_features(
        episode_baseline, start_ms=baseline_start,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    common_features, _ = _multichannel_features(
        common_baseline, start_ms=baseline_start,
        carrier_hz=float(screening["EEG_selected_frequency_hz"]), cfg=cfg,
    )
    if profile == SHAM:
        residual, tolerance, recovered = 0.0, 0.0, True
        rate_safe = _relative_rate_safe(sham_outcome, sham_outcome, cfg)
    else:
        residual = float(
            (float(sham_washout_features["global_log10_alpha_power"])
             - float(common_features["global_log10_alpha_power"]))
            - (float(washout_features["global_log10_alpha_power"])
               - float(baseline_features["global_log10_alpha_power"]))
        )
        recovered, tolerance = _field_removal_status(
            effect_log10=(
                float(sham_features["global_log10_alpha_power"])
                - float(active_features["global_log10_alpha_power"])
            ),
            residual_log10=residual,
            cfg=cfg,
        )
        rate_safe = _relative_rate_safe(sham_outcome, outcome, cfg)
    updates = pd.DataFrame(episode["simulation"]["phase_updates"])
    active_updates = updates.iloc[1:] if len(updates) > 1 else updates
    profile_spec = next(
        value for value in _profile_specs(cfg)
        if value["montage_profile"] == (PROFILE_Z if profile == SHAM else profile)
    )
    projection = 0.0 if profile == SHAM else _field_projection(context, profile_spec)
    head_field = np.asarray([
        profile_spec["field_x"], profile_spec["field_y"], profile_spec["field_z"],
    ], dtype=float)
    local_field = _head_from_local_rotation(context).T @ head_field
    if profile == SHAM:
        local_field = np.zeros(3, dtype=float)
    row = {
        **context,
        **{name: screening[name] for name in TOPOGRAPHY_FEATURES},
        "EEG_selected_frequency_hz": float(screening["EEG_selected_frequency_hz"]),
        "phase_sensor_index": int(screening["phase_sensor_index"]),
        "phase_sensor_label": str(screening["phase_sensor_label"]),
        "future_index": int(future_index + 1),
        "future_drive_seed": int(future_seed),
        "montage_profile": profile,
        "amplitude_v_per_m": 0.0 if profile == SHAM else float(
            cfg.analysis.actions.amplitude_v_per_m
        ),
        "field_projection_fraction": projection,
        "applied_local_field_direction_x": float(local_field[0]),
        "applied_local_field_direction_y": float(local_field[1]),
        "applied_local_field_direction_z": float(local_field[2]),
        "effective_axial_amplitude_v_per_m": projection * (
            0.0 if profile == SHAM else float(cfg.analysis.actions.amplitude_v_per_m)
        ),
        "profile_matches_orientation": bool(profile == context["matched_profile"]),
        "post_global_log10_alpha_power": float(
            active_features["global_log10_alpha_power"]
        ),
        "post_distance_to_orientation_B_log10": distance,
        "causal_distance_improvement_vs_sham_log10": sham_distance - distance,
        "causal_alpha_suppression_vs_sham_log10": (
            float(sham_features["global_log10_alpha_power"])
            - float(active_features["global_log10_alpha_power"])
        ),
        "post_E_firing_rate_hz": float(outcome.E_firing_rate_hz),
        "post_I_firing_rate_hz": float(outcome.I_firing_rate_hz),
        "post_E_ppc": float(outcome.E_ppc),
        "post_I_ppc": float(outcome.I_ppc),
        "hidden_E_ppc_reduction_vs_sham": float(sham_outcome.E_ppc - outcome.E_ppc),
        "hidden_I_ppc_reduction_vs_sham": float(sham_outcome.I_ppc - outcome.I_ppc),
        "rate_safe": bool(rate_safe),
        "washout_residual_log10": residual,
        "washout_tolerance_log10": tolerance,
        "field_removal_recovered": bool(recovered),
        "final_extracellular_residual_mV": float(
            episode["simulation"]["final_residual_mV"]
        ),
        "baseline_relative_rms_error": float(_relative_rms_error(
            common_baseline.reshape(-1), episode_baseline.reshape(-1)
        )),
        "baseline_E_rate_difference_hz": float(
            baseline_outcome.E_firing_rate_hz - sham_baseline_outcome.E_firing_rate_hz
        ),
        "phase_update_count": int(len(updates)),
        "mean_abs_phase_error_before_correction_rad": float(
            active_updates.phase_error_before_correction_rad.abs().mean()
        ),
        "maximum_abs_frequency_correction_hz": float(
            active_updates.frequency_correction_hz.abs().max()
        ),
        "maximum_field_boundary_discontinuity_v_per_m": float(
            updates.field_boundary_discontinuity_v_per_m.max()
        ),
        "all_phase_estimates_causal": bool(updates.estimate_is_strictly_causal.all()),
        "common_phase_estimate_actionable_fraction": float(
            np.mean(updates.common_audit_resultant_to_rms.to_numpy(float)
                    >= float(cfg.analysis.criteria.minimum_recent_resultant_to_rms))
        ),
        "policy_uses_hidden_orientation_frequency_or_spikes": False,
        "efficacy_uses_multichannel_ideal_neural_EEG": True,
    }
    spectra = spectra.assign(
        context_id=str(context["context_id"]),
        structure_seed=int(context["structure_seed"]),
        orientation_label=str(context["orientation_label"]),
        hidden_frequency_hz=float(context["hidden_frequency_hz"]),
        future_index=int(future_index + 1),
        montage_profile=profile,
        condition="stimulation",
    )
    return row, spectra


def _expected_map(metrics: pd.DataFrame) -> pd.DataFrame:
    group = [
        "context_id", "paired_orientation_context_id", "structure_seed",
        "hidden_frequency_hz", "orientation_label", "rotation_y_rad",
        "matched_profile", *TOPOGRAPHY_FEATURES, "EEG_selected_frequency_hz",
        "phase_sensor_index", "phase_sensor_label", "montage_profile",
        "field_projection_fraction", "effective_axial_amplitude_v_per_m",
        "applied_local_field_direction_x", "applied_local_field_direction_y",
        "applied_local_field_direction_z",
        "profile_matches_orientation",
    ]
    return (
        metrics.groupby(group, as_index=False, dropna=False)
        .agg(
            n_futures=("future_index", "nunique"),
            expected_distance_to_B_log10=(
                "post_distance_to_orientation_B_log10", "mean"
            ),
            future_sd_distance_log10=(
                "post_distance_to_orientation_B_log10", "std"
            ),
            expected_improvement_vs_sham_log10=(
                "causal_distance_improvement_vs_sham_log10", "mean"
            ),
            expected_alpha_suppression_vs_sham_log10=(
                "causal_alpha_suppression_vs_sham_log10", "mean"
            ),
            all_rate_safe=("rate_safe", "all"),
            all_field_removal_recovered=("field_removal_recovered", "all"),
        )
        .sort_values(group).reset_index(drop=True)
    )


def _opportunity(
    expected: pd.DataFrame, metrics: pd.DataFrame, cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    active = expected[expected.montage_profile.isin(ACTIVE_PROFILES)].copy()
    fixed = (
        active.groupby("montage_profile", as_index=False)
        .expected_distance_to_B_log10.mean()
        .sort_values(["expected_distance_to_B_log10", "montage_profile"])
    )
    best_fixed = str(fixed.iloc[0].montage_profile)
    best_fixed_distance = float(fixed.iloc[0].expected_distance_to_B_log10)
    best = (
        active.sort_values([
            "context_id", "expected_distance_to_B_log10", "montage_profile"
        ]).groupby("context_id", as_index=False).first()
        .rename(columns={
            "montage_profile": "expected_optimal_profile",
            "expected_distance_to_B_log10": "oracle_expected_distance_log10",
        })
    )
    context_map = best[[
        "context_id", "structure_seed", "hidden_frequency_hz",
        "orientation_label", "matched_profile", *TOPOGRAPHY_FEATURES,
        "expected_optimal_profile", "oracle_expected_distance_log10",
    ]].copy()
    fixed_lookup = active[active.montage_profile.eq(best_fixed)].set_index(
        "context_id"
    ).expected_distance_to_B_log10
    context_map["best_fixed_profile"] = best_fixed
    context_map["best_fixed_expected_distance_log10"] = context_map.context_id.map(
        fixed_lookup
    )
    context_map["oracle_advantage_over_best_fixed_log10"] = (
        context_map.best_fixed_expected_distance_log10
        - context_map.oracle_expected_distance_log10
    )
    margin = float(cfg.analysis.criteria.practical_context_margin_log10)
    context_map["practical_nonfixed_opportunity"] = (
        context_map.oracle_advantage_over_best_fixed_log10 >= margin
    )
    context_map["oracle_matches_geometric_profile"] = (
        context_map.expected_optimal_profile == context_map.matched_profile
    )

    pivot = active.pivot(
        index=["context_id", "structure_seed", "hidden_frequency_hz",
               "orientation_label", "matched_profile"],
        columns="montage_profile", values="expected_distance_to_B_log10",
    ).reset_index()
    pivot["matched_distance_log10"] = np.where(
        pivot.matched_profile.eq(PROFILE_Z), pivot[PROFILE_Z], pivot[PROFILE_60]
    )
    pivot["mismatched_distance_log10"] = np.where(
        pivot.matched_profile.eq(PROFILE_Z), pivot[PROFILE_60], pivot[PROFILE_Z]
    )
    pivot["matched_advantage_log10"] = (
        pivot.mismatched_distance_log10 - pivot.matched_distance_log10
    )
    structure = (
        pivot.groupby(["structure_seed", "orientation_label"], as_index=False)
        .matched_advantage_log10.mean()
    )

    selection = [int(x) for x in cfg.analysis.response_mapping.future_selection_indices]
    evaluation = [int(x) for x in cfg.analysis.response_mapping.future_evaluation_indices]
    split_rows: list[dict[str, Any]] = []
    for split_name, train_indices, test_indices in (
        ("futures_1_2_to_3_4", selection, evaluation),
        ("futures_3_4_to_1_2", evaluation, selection),
    ):
        train = metrics[
            metrics.future_index.isin(train_indices)
            & metrics.montage_profile.isin(ACTIVE_PROFILES)
        ]
        test = metrics[
            metrics.future_index.isin(test_indices)
            & metrics.montage_profile.isin(ACTIVE_PROFILES)
        ]
        selected = (
            train.groupby(["context_id", "montage_profile"], as_index=False)
            .post_distance_to_orientation_B_log10.mean()
            .sort_values(["context_id", "post_distance_to_orientation_B_log10",
                          "montage_profile"])
            .groupby("context_id", as_index=False).first()
            .rename(columns={"montage_profile": "selected_profile"})
        )
        fixed_train = (
            train.groupby("montage_profile")
            .post_distance_to_orientation_B_log10.mean().sort_values()
        )
        trained_fixed = str(fixed_train.index[0])
        for context_id, group in test.groupby("context_id"):
            chosen = str(selected.set_index("context_id").loc[
                context_id, "selected_profile"
            ])
            chosen_loss = float(group.loc[
                group.montage_profile.eq(chosen),
                "post_distance_to_orientation_B_log10",
            ].mean())
            fixed_loss = float(group.loc[
                group.montage_profile.eq(trained_fixed),
                "post_distance_to_orientation_B_log10",
            ].mean())
            first = group.iloc[0]
            split_rows.append({
                "split_direction": split_name,
                "context_id": context_id,
                "structure_seed": int(first.structure_seed),
                "orientation_label": str(first.orientation_label),
                "selected_profile": chosen,
                "trained_best_fixed_profile": trained_fixed,
                "evaluation_advantage_log10": fixed_loss - chosen_loss,
            })
    split = pd.DataFrame(split_rows)
    split_structure = (
        split.groupby(["split_direction", "structure_seed"], as_index=False)
        .evaluation_advantage_log10.mean()
    )
    future_winners = (
        metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)]
        .sort_values(["context_id", "future_index",
                      "post_distance_to_orientation_B_log10", "montage_profile"])
        .groupby(["context_id", "future_index"], as_index=False).first()
    )
    agreement = future_winners.merge(
        context_map[["context_id", "expected_optimal_profile"]], on="context_id"
    )
    agreement_fraction = float(np.mean(
        agreement.montage_profile == agreement.expected_optimal_profile
    ))
    by_orientation = {
        str(label): float(group.matched_advantage_log10.mean())
        for label, group in pivot.groupby("orientation_label")
    }
    structure_oracle = (
        context_map.groupby("structure_seed", as_index=False)
        .oracle_advantage_over_best_fixed_log10.mean()
    )
    audit = {
        "best_fixed_profile": best_fixed,
        "fixed_profile_expected_distance_log10": {
            str(row.montage_profile): float(row.expected_distance_to_B_log10)
            for row in fixed.itertuples()
        },
        "oracle_expected_distance_log10": float(
            context_map.oracle_expected_distance_log10.mean()
        ),
        "mean_oracle_advantage_over_best_fixed_log10": float(
            context_map.oracle_advantage_over_best_fixed_log10.mean()
        ),
        "positive_structure_oracle_fraction": float(np.mean(
            structure_oracle.oracle_advantage_over_best_fixed_log10 > 0
        )),
        "optimal_profile_context_count": {
            str(key): int(value) for key, value in
            context_map.expected_optimal_profile.value_counts().items()
        },
        "matched_advantage_by_orientation_log10": by_orientation,
        "geometric_match_fraction": float(
            context_map.oracle_matches_geometric_profile.mean()
        ),
        "practical_nonfixed_context_count": int(
            context_map.practical_nonfixed_opportunity.sum()
        ),
        "mean_realized_winner_agreement_fraction": agreement_fraction,
        "future_split_mean_advantage_log10": float(
            split_structure.evaluation_advantage_log10.mean()
        ),
        "future_split_positive_structure_fraction": float(np.mean(
            split_structure.evaluation_advantage_log10 > 0
        )),
        "oracle_is_posthoc_full_information_and_not_deployable": True,
    }
    return context_map, pivot, split, audit


def _mechanism_audit(
    episodes: dict[str, dict[str, Any]], context: dict[str, Any],
    future_index: int, cfg: DictConfig,
) -> list[dict[str, Any]]:
    if not (
        int(context["structure_index"]) == 0
        and np.isclose(float(context["hidden_frequency_hz"]),
                       float(cfg.analysis.mechanism_audit.representative_frequency_hz))
        and future_index == 0
    ):
        return []
    sham = _representative_voltage_coefficients(
        episodes[SHAM], float(context["hidden_frequency_hz"]), cfg
    )
    rows: list[dict[str, Any]] = []
    for profile in ACTIVE_PROFILES:
        active = _representative_voltage_coefficients(
            episodes[profile], float(context["hidden_frequency_hz"]), cfg
        )
        profile_spec = next(
            x for x in _profile_specs(cfg) if x["montage_profile"] == profile
        )
        for site in sorted(set(sham).intersection(active)):
            rows.append({
                "context_id": str(context["context_id"]),
                "orientation_label": str(context["orientation_label"]),
                "montage_profile": profile,
                "profile_matches_orientation": profile == context["matched_profile"],
                "field_projection_fraction": _field_projection(context, profile_spec),
                "site_id": site,
                "induced_apic_minus_soma_carrier_resultant_mV": float(
                    abs(active[site] - sham[site])
                ),
            })
    return rows


def _checks(
    *, screening: pd.DataFrame, references: pd.DataFrame,
    baseline_audit: dict[str, Any], observability: dict[str, Any],
    metrics: pd.DataFrame, opportunity: dict[str, Any], mechanism: pd.DataFrame,
    sources: dict[str, Any], cfg: DictConfig,
) -> tuple[dict[str, bool], dict[str, Any]]:
    criteria = cfg.analysis.criteria
    eligible = screening[screening.eligible]
    active = metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)]
    accepted = screening[screening.carrier_identified]
    mechanism_ordered = False
    if not mechanism.empty:
        summary = mechanism.groupby([
            "orientation_label", "profile_matches_orientation"
        ]).induced_apic_minus_soma_carrier_resultant_mV.mean().unstack()
        mechanism_ordered = bool(
            True in summary.columns and False in summary.columns
            and (summary[True] > summary[False]).all()
        )
    projection_matrix_ordered = all(
        _field_projection(context, profile) > _field_projection(context, other)
        for context in _orientation_specs(cfg)
        for profile in _profile_specs(cfg)
        for other in _profile_specs(cfg)
        if profile["montage_profile"] == context["matched_profile"]
        and other["montage_profile"] != context["matched_profile"]
    )
    matched_by_orientation = opportunity["matched_advantage_by_orientation_log10"]
    checks = {
        "source_H5K0_negative_hash_locked": bool(sources["H5K0_negative_preserved"]),
        "H5O0_seeds_disjoint_from_frozen_H1_H5_lineage": True,
        "orientation_is_distinct_from_A_B_state_generator": True,
        "afferent_mean_rate_matched_between_A_and_B_by_construction": True,
        "orientation_specific_B_target_calibrated_before_active_outcomes": bool(
            references.orientation_label.nunique() == 2
            and references.structure_seed.nunique()
            >= (1 if bool(cfg.analysis.smoke_test)
                else int(criteria.minimum_reference_structures))
        ),
        "complete_frequency_orientation_screening_grid": bool(
            len(screening) == len(_contexts(cfg))
        ),
        "screening_uses_only_predecision_multichannel_ideal_EEG": bool(
            screening.screening_uses_only_predecision_multichannel_ideal_EEG.all()
        ),
        "matched_rotation_preserves_spike_trajectories": bool(
            baseline_audit["all_spike_trains_identical"]
        ),
        "matched_rotation_preserves_firing_rates": float(
            baseline_audit["maximum_rate_difference_hz"]
        ) <= float(criteria.maximum_paired_rate_difference_hz),
        "matched_rotation_preserves_dipole_norm": float(
            baseline_audit["maximum_dipole_norm_relative_error"]
        ) <= float(criteria.maximum_paired_dipole_norm_relative_error),
        "minimum_eligible_contexts": bool(
            len(eligible) >= int(criteria.minimum_eligible_contexts)
        ) or bool(cfg.analysis.smoke_test),
        "minimum_independent_structures": bool(
            eligible.structure_seed.nunique() >= int(criteria.minimum_structure_seeds)
        ) or bool(cfg.analysis.smoke_test),
        "both_frequencies_and_orientations_enrolled": bool(
            eligible.hidden_frequency_hz.nunique() == 2
            and eligible.orientation_label.nunique() == 2
        ) or bool(cfg.analysis.smoke_test),
        "carrier_identification_coverage": float(
            screening.carrier_identified.mean()
        ) >= float(criteria.minimum_carrier_identification_coverage),
        "accepted_carrier_accuracy": bool(len(accepted)) and float(
            accepted.EEG_frequency_selection_correct.mean()
        ) >= float(criteria.minimum_accepted_carrier_accuracy),
        "recent_phase_is_actionable": float(
            screening.recent_phase_actionable.mean()
        ) >= float(criteria.minimum_common_phase_estimate_actionable_fraction),
        "orientation_observable_from_phase_invariant_EEG_topography": bool(
            np.isfinite(observability["LOSO_balanced_accuracy"])
            and observability["LOSO_balanced_accuracy"]
            >= float(criteria.minimum_orientation_LOSO_balanced_accuracy)
        ) or bool(cfg.analysis.smoke_test),
        "complete_paired_action_future_grid": bool(
            metrics.groupby(["context_id", "montage_profile"]).future_index.nunique().min()
            >= int(criteria.minimum_future_continuations)
        ) or bool(cfg.analysis.smoke_test),
        "identical_predecision_EEG_across_actions_and_futures": bool(
            metrics.baseline_relative_rms_error.max()
            <= float(criteria.maximum_baseline_relative_rms_error)
        ),
        "both_active_profiles_use_identical_0p2_V_per_m": bool(
            np.isclose(active.amplitude_v_per_m, 0.2).all()
        ),
        "one_montage_profile_used_for_complete_intervention": True,
        "action_frequency_is_EEG_selected": True,
        "phase_updates_use_only_preceding_EEG": bool(
            active.all_phase_estimates_causal.all()
        ),
        "phase_correction_is_frequency_bounded": bool(
            active.maximum_abs_frequency_correction_hz.max()
            <= float(criteria.maximum_frequency_correction_hz)
        ),
        "field_waveform_continuous_across_updates": bool(
            active.maximum_field_boundary_discontinuity_v_per_m.max()
            <= float(criteria.maximum_field_boundary_discontinuity_v_per_m)
        ),
        "configured_field_projection_crossover_is_exact": projection_matrix_ordered,
        "representative_cellular_polarization_is_projection_ordered": mechanism_ordered
        or bool(cfg.analysis.smoke_test),
        "matched_profile_improves_both_orientations": bool(
            len(matched_by_orientation) == 2
            and all(value >= float(criteria.minimum_matched_advantage_each_orientation_log10)
                    for value in matched_by_orientation.values())
        ),
        "expected_oracle_uses_both_montage_profiles": bool(
            len(opportunity["optimal_profile_context_count"]) == 2
        ),
        "expected_oracle_has_practical_advantage_over_best_fixed": float(
            opportunity["mean_oracle_advantage_over_best_fixed_log10"]
        ) >= float(criteria.minimum_expected_oracle_advantage_log10),
        "independent_future_split_has_practical_advantage": float(
            opportunity["future_split_mean_advantage_log10"]
        ) >= float(criteria.minimum_future_split_advantage_log10),
        "future_split_advantage_cross_structure": float(
            opportunity["future_split_positive_structure_fraction"]
        ) >= float(criteria.minimum_future_split_positive_structure_fraction),
        "realized_optimal_profile_reproducible_across_futures": float(
            opportunity["mean_realized_winner_agreement_fraction"]
        ) >= float(criteria.minimum_realized_winner_agreement_fraction),
        "all_actions_rate_safe": bool(metrics.rate_safe.all()),
        "exact_field_removal_confirmed": bool(
            np.isclose(metrics.final_extracellular_residual_mV, 0.0).all()
        ),
        "physiological_washout_recovery_audit": float(
            active.field_removal_recovered.mean()
        ) >= float(criteria.minimum_physiological_washout_recovery_fraction),
        "policy_inputs_exclude_hidden_orientation_frequency_and_spikes": bool(
            (~metrics.policy_uses_hidden_orientation_frequency_or_spikes).all()
        ),
    }
    passed = all(checks.values())
    conclusions = {
        "H5_O0_montage_orientation_opportunity": "PASSED" if passed else "NOT PASSED",
        "ready_for_disjoint_EEG_policy_development": bool(passed),
        "machine_learning_policy_status": "NOT TRAINED OR TESTED",
        "failed_checks": [name for name, value in checks.items() if not value],
    }
    return checks, conclusions


def _save_figure(figure: plt.Figure, root: Path, stem: str) -> None:
    figure.tight_layout()
    for suffix in ("png", "pdf"):
        figure.savefig(root / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plots(
    *, root: Path, reference_spectra: pd.DataFrame,
    screening_spectra: pd.DataFrame, screening: pd.DataFrame,
    expected: pd.DataFrame, context_map: pd.DataFrame,
    split: pd.DataFrame, metrics: pd.DataFrame, mechanism: pd.DataFrame,
    cfg: DictConfig,
) -> None:
    tiny = np.finfo(float).tiny
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, orientation in zip(axes, ["orientation_0deg", "orientation_60deg"]):
        for label, table, color in (
            ("B homogeneous", reference_spectra, "#4c78a8"),
            ("A rhythmic", screening_spectra, "#e45756"),
        ):
            view = table[
                table.orientation_label.eq(orientation)
                & table.sensor_label.eq("vertex")
            ]
            summary = view.groupby("frequency_hz").PSD_v2_per_hz.mean()
            keep = (summary.index >= 5) & (summary.index <= 15)
            axis.plot(summary.index[keep], 10.0 * np.log10(np.maximum(
                summary.to_numpy()[keep], tiny
            )), label=label, color=color)
        axis.set(title=orientation.replace("orientation_", "Population "),
                 xlabel="Frequency (Hz)", ylabel="PSD (dB V²/Hz)")
        axis.legend(frameon=False)
    _save_figure(figure, root, "figure_01_A_B_multichannel_PSD")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    labels = [str(x) for x in cfg.analysis.eeg_array.labels]
    for axis, orientation in zip(axes, ["orientation_0deg", "orientation_60deg"]):
        view = screening[screening.orientation_label.eq(orientation)]
        means = view[TOPOGRAPHY_FEATURES].mean().to_numpy(float)
        errors = view[TOPOGRAPHY_FEATURES].std(ddof=1).fillna(0).to_numpy(float)
        axis.bar(labels, means, yerr=errors, color="#72b7b2")
        axis.set(title=orientation.replace("orientation_", "Population "),
                 ylabel="Fraction of carrier-band array power")
        axis.tick_params(axis="x", rotation=20)
    _save_figure(figure, root, "figure_02_predecision_EEG_topography")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, orientation in zip(axes, ["orientation_0deg", "orientation_60deg"]):
        view = expected[
            expected.orientation_label.eq(orientation)
            & expected.montage_profile.isin(ALL_ACTIONS)
        ]
        summary = view.groupby("montage_profile").expected_distance_to_B_log10.agg(
            ["mean", "std"]
        ).reindex(ALL_ACTIONS)
        axis.bar(summary.index, summary["mean"], yerr=summary["std"],
                 color=["#bab0ac", "#4c78a8", "#f58518"])
        axis.set(title=orientation.replace("orientation_", "Population "),
                 ylabel="Distance to orientation-specific B (log10)")
        axis.tick_params(axis="x", rotation=18)
    _save_figure(figure, root, "figure_03_montage_response_crossover")

    figure, axis = plt.subplots(figsize=(7, 4))
    for orientation, group in context_map.groupby("orientation_label"):
        axis.scatter(
            group.topography_right_xz - group.topography_left_xz,
            [ACTIVE_PROFILES.index(value) for value in group.expected_optimal_profile],
            label=orientation,
        )
    axis.set(
        xlabel="Right-minus-left carrier-power fraction",
        ylabel="Expected optimal profile (0=z, 1=60°)",
        title="EEG topography and full-information profile preference",
    )
    axis.legend(frameon=False)
    _save_figure(figure, root, "figure_04_EEG_context_profile_preference")

    figure, axis = plt.subplots(figsize=(7, 4))
    summary = split.groupby("split_direction").evaluation_advantage_log10.mean()
    axis.bar(summary.index, summary.values, color=["#4c78a8", "#f58518"])
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(ylabel="Held-out-future advantage over trained fixed profile (log10)",
             title="Independent-future response replication")
    axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, root, "figure_05_future_split_validation")

    if not mechanism.empty:
        figure, axis = plt.subplots(figsize=(8, 4))
        summary = mechanism.groupby([
            "orientation_label", "montage_profile"
        ]).induced_apic_minus_soma_carrier_resultant_mV.mean().unstack()
        summary.plot(kind="bar", ax=axis, color=["#f58518", "#4c78a8"])
        axis.set(ylabel="Induced apical-minus-somatic carrier response (mV)",
                 title="Representative-cell polarization audit")
        axis.tick_params(axis="x", rotation=15)
        _save_figure(figure, root, "figure_06_cellular_polarization")

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    rates = metrics.groupby("montage_profile")[[
        "post_E_firing_rate_hz", "post_I_firing_rate_hz"
    ]].mean().reindex(ALL_ACTIONS)
    rates.plot(kind="bar", ax=axes[0], color=["#4c78a8", "#e45756"])
    axes[0].set(ylabel="Firing rate (Hz)", title="Rate-safety audit")
    phase = metrics[metrics.montage_profile.isin(ACTIVE_PROFILES)].groupby(
        "montage_profile"
    ).mean_abs_phase_error_before_correction_rad.mean().reindex(ACTIVE_PROFILES)
    axes[1].bar(phase.index, phase.values, color=["#4c78a8", "#f58518"])
    axes[1].set(ylabel="Mean absolute phase error (rad)",
                title="Causal phase-maintenance audit")
    for axis in axes:
        axis.tick_params(axis="x", rotation=15)
    _save_figure(figure, root, "figure_07_safety_and_phase")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    sources = _load_sources(cfg)
    _validate_design(cfg, sources)
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    root = Path(to_absolute_path(str(cfg.experiment.dir))) / str(
        cfg.analysis.output_root_name
    )
    exists = bool(root.exists() and any(root.iterdir())) if rank == 0 else None
    if bool(comm.bcast(exists, root=0)):
        raise FileExistsError(f"Refusing to overwrite existing results: {root}")
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        print("\n### H5-O0 montage-orientation opportunity")
        print(OmegaConf.to_yaml(cfg.analysis, resolve=True))
    comm.Barrier()
    started = time.perf_counter()

    # Calibrate B completely before any active A outcome is simulated.
    reference_rows: list[dict[str, Any]] = []
    reference_spectra: list[pd.DataFrame] = []
    for context in _reference_contexts(cfg):
        if rank == 0:
            print(f"B target: {context['context_id']}")
        state_cfg = _with_orientation_state(cfg, context, homogeneous_B=True)
        with open_dict(state_cfg):
            state_cfg.env.online.record_representative_state = False
        episode = _run_profile(
            condition_cfg=state_cfg,
            context=context,
            future_seed=int(context["reference_future_seed"]),
            future_index=0,
            profile=SHAM,
            phase_sensor_index=0,
            root=root / "reference_calibration",
            comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            baseline_eeg, baseline_start = _multichannel_raw(
                episode, "baseline", context, state_cfg
            )
            outcome_eeg, outcome_start = _multichannel_raw(
                episode, "stimulation", context, state_cfg, trim=True
            )
            baseline_features, spectrum = _multichannel_features(
                baseline_eeg, start_ms=baseline_start, carrier_hz=9.0, cfg=state_cfg
            )
            outcome_features, _ = _multichannel_features(
                outcome_eeg, start_ms=outcome_start, carrier_hz=9.0, cfg=state_cfg
            )
            reference_rows.append({
                **context,
                "screening_log10_alpha_power": float(
                    baseline_features["global_log10_alpha_power"]
                ),
                "outcome_log10_alpha_power": float(
                    outcome_features["global_log10_alpha_power"]
                ),
            })
            reference_spectra.append(spectrum.assign(
                context_id=str(context["context_id"]),
                structure_seed=int(context["structure_seed"]),
                orientation_label=str(context["orientation_label"]),
                condition="B_homogeneous",
            ))
    if rank == 0:
        references = pd.DataFrame(reference_rows)
        target = _reference_target(references)
        references.to_csv(root / "reference_B_calibration.csv", index=False)
        (root / "frozen_orientation_specific_B_target.json").write_text(
            json.dumps(_json_ready(target), indent=2, allow_nan=False)
        )
    else:
        references, target = None, None
    target = comm.bcast(target, root=0)

    screening_rows: list[dict[str, Any]] = []
    screening_spectra: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    stimulation_spectra: list[pd.DataFrame] = []
    mechanism_rows: list[dict[str, Any]] = []
    contexts = _contexts(cfg)
    for context in contexts:
        if rank == 0:
            print(
                f"A context={context['context_id']} structure={context['structure_seed']} "
                f"f={context['hidden_frequency_hz']:g} Hz "
                f"orientation={context['orientation_label']}"
            )
        state_cfg = _with_orientation_state(cfg, context)
        record = bool(
            cfg.analysis.mechanism_audit.record_representative_state
            and int(context["structure_index"]) == 0
            and np.isclose(float(context["hidden_frequency_hz"]),
                           float(cfg.analysis.mechanism_audit.representative_frequency_hz))
        )
        with open_dict(state_cfg):
            state_cfg.env.online.record_representative_state = record
        first_future = _future_seed(state_cfg, context, 0)
        baseline = _run_profile(
            condition_cfg=state_cfg, context=context, future_seed=first_future,
            future_index=0, profile=SHAM, phase_sensor_index=0,
            root=root / "screening", comm=comm, size=size, rank=rank,
        )
        if rank == 0:
            screening, spectrum = _carrier_screen(
                baseline, context, target, state_cfg
            )
            screening_rows.append(screening)
            screening_spectra.append(spectrum)
            eligible = bool(screening["eligible"])
            selected_frequency = float(screening["EEG_selected_frequency_hz"])
            phase_sensor = int(screening["phase_sensor_index"])
            print(
                f"screen: {'ELIGIBLE' if eligible else 'SHAM FALLBACK'}; "
                f"selected={selected_frequency:g} Hz; sensor={phase_sensor}; "
                f"reason={screening['exclusion_reasons']}"
            )
        else:
            screening, eligible, selected_frequency, phase_sensor = None, None, None, None
        eligible = bool(comm.bcast(eligible, root=0))
        selected_frequency = float(comm.bcast(selected_frequency, root=0))
        phase_sensor = int(comm.bcast(phase_sensor, root=0))
        if not eligible:
            continue
        action_cfg = _with_action_frequency(state_cfg, selected_frequency)
        for future_index in range(int(cfg.analysis.crossed_design.n_future_continuations)):
            future_seed = _future_seed(action_cfg, context, future_index)
            episodes: dict[str, dict[str, Any]] | None = {} if rank == 0 else None
            for profile in ALL_ACTIONS:
                if profile == SHAM and future_index == 0 and phase_sensor == 0:
                    episode = baseline
                else:
                    episode = _run_profile(
                        condition_cfg=action_cfg,
                        context=context,
                        future_seed=future_seed,
                        future_index=future_index,
                        profile=profile,
                        phase_sensor_index=phase_sensor,
                        root=root / "active_mapping",
                        comm=comm, size=size, rank=rank,
                    )
                if rank == 0:
                    episodes[profile] = episode
            if rank == 0:
                sham = episodes[SHAM]
                for profile in ALL_ACTIONS:
                    row, spectrum = _metric_row(
                        episode=episodes[profile], sham=sham, baseline=sham,
                        context=context, screening=screening, target=target,
                        profile=profile, future_index=future_index,
                        future_seed=future_seed, cfg=action_cfg,
                    )
                    metric_rows.append(row)
                    if int(context["structure_index"]) == 0 and future_index == 0:
                        stimulation_spectra.append(spectrum)
                mechanism_rows.extend(_mechanism_audit(
                    episodes, context, future_index, action_cfg
                ))

    if rank != 0:
        return
    screening = pd.DataFrame(screening_rows)
    screening.to_csv(root / "prospective_screening.csv", index=False)
    reference_spectrum = pd.concat(reference_spectra, ignore_index=True)
    predecision_spectrum = pd.concat(screening_spectra, ignore_index=True)
    reference_spectrum.to_csv(root / "reference_B_multichannel_PSD.csv", index=False)
    predecision_spectrum.to_csv(root / "predecision_A_multichannel_PSD.csv", index=False)
    if not metric_rows:
        conclusion = {
            "scope": "H5-O0 montage-orientation opportunity",
            "checks": {"minimum_eligible_contexts": False},
            "conclusions": {
                "H5_O0_montage_orientation_opportunity": "NOT PASSED",
                "ready_for_disjoint_EEG_policy_development": False,
                "machine_learning_policy_status": "NOT TRAINED OR TESTED",
            },
            "runtime_seconds": float(time.perf_counter() - started),
        }
        (root / "experiment_conclusion.json").write_text(json.dumps(
            conclusion, indent=2
        ))
        print("No eligible contexts; stopped after prospective screening.")
        print(f"Results saved to: {root}")
        return

    metrics = pd.DataFrame(metric_rows)
    spectra = pd.concat(stimulation_spectra, ignore_index=True)
    mechanism = pd.DataFrame(mechanism_rows)
    expected = _expected_map(metrics)
    context_map, crossover, split, opportunity = _opportunity(expected, metrics, cfg)
    baseline_pairs, baseline_audit = _paired_orientation_baseline_audit(screening)
    predictions, observability = _orientation_loso(screening)
    checks, conclusions = _checks(
        screening=screening, references=references,
        baseline_audit=baseline_audit, observability=observability,
        metrics=metrics, opportunity=opportunity, mechanism=mechanism,
        sources=sources, cfg=cfg,
    )

    metrics.to_csv(root / "context_montage_future_metrics.csv", index=False)
    spectra.to_csv(root / "representative_stimulation_multichannel_PSD.csv", index=False)
    mechanism.to_csv(root / "representative_cellular_polarization.csv", index=False)
    expected.to_csv(root / "expected_context_montage_map.csv", index=False)
    context_map.to_csv(root / "montage_response_opportunity.csv", index=False)
    crossover.to_csv(root / "matched_mismatched_crossover.csv", index=False)
    split.to_csv(root / "independent_future_split_validation.csv", index=False)
    baseline_pairs.to_csv(root / "paired_orientation_baseline_audit.csv", index=False)
    predictions.to_csv(root / "orientation_EEG_observability_LOSO.csv", index=False)
    audit = {
        "orientation_EEG_observability": observability,
        "rotation_baseline_invariance": baseline_audit,
        "montage_profile_opportunity": opportunity,
    }
    (root / "H5_O0_montage_orientation_opportunity_audit.json").write_text(
        json.dumps(_json_ready(audit), indent=2, allow_nan=False)
    )
    provenance = {
        "experiment": "H5_O0_montage_orientation_opportunity",
        "frozen_negative_source": {
            "root": sources["root"], "hashes": sources["hashes"],
            "upstream_provenance": sources["upstream_provenance"],
        },
        "state_definition": {
            "A": "mean-rate-matched 9/11-Hz shared rhythmic afferent drive",
            "B": "homogeneous mean-rate-matched afferent drive",
            "orientation_is_not_A_or_B": True,
            "modulation_depth": 0.04,
            "phase_diffusion_rad2_per_s": 0.5,
            "shared_modulated_fraction": 1.0,
        },
        "orientation_context": _orientation_specs(cfg),
        "montage_profiles": _profile_specs(cfg),
        "montage_interpretation": (
            "Each action is a fixed precomputed electrode-current profile "
            "represented by its local unit tissue-field direction; no scalp "
            "current optimization is performed."
        ),
        "EEG_array": {
            "labels": list(cfg.analysis.eeg_array.labels),
            "locations_um": OmegaConf.to_container(
                cfg.analysis.eeg_array.locations_um, resolve=True
            ),
            "context_features": TOPOGRAPHY_FEATURES,
            "phase_sensor_selection": str(
                cfg.analysis.eeg_array.phase_sensor_selection
            ),
        },
        "causal_protocol": {
            "burn_in_s": int(cfg.analysis.timeline.burn_in_steps),
            "predecision_EEG_s": int(cfg.analysis.timeline.baseline_steps),
            "stimulation_s": int(cfg.analysis.timeline.stimulation_steps),
            "central_endpoint_s": 8.0,
            "washout_s": int(cfg.analysis.timeline.washout_steps),
            "ramp_s_each_edge": float(cfg.analysis.timeline.block_ramp_ms) / 1000.0,
            "amplitude_v_per_m": float(cfg.analysis.actions.amplitude_v_per_m),
            "carrier_estimator": MT_POOLED,
            "phase_controller": _profile(cfg, RESPONSIVE),
            "relative_phase_rad": float(cfg.analysis.tacs.relative_phase_offset_rad),
        },
        "design": {
            "reference_structures": int(references.structure_seed.nunique()),
            "independent_A_structures": int(screening.structure_seed.nunique()),
            "screened_contexts": int(len(screening)),
            "eligible_contexts": int(screening.eligible.sum()),
            "paired_futures": int(cfg.analysis.crossed_design.n_future_continuations),
            "action_future_outcomes": int(len(metrics)),
            "statistical_unit": "independent circuit structure",
        },
        "inference_boundary": (
            "Exploratory ideal-neural-EEG full-information positive-control map. "
            "No policy is fitted, local field profiles are not clinical scalp "
            "montages, and H5 is not established."
        ),
    }
    (root / "protocol_and_provenance.json").write_text(json.dumps(
        _json_ready(provenance), indent=2, allow_nan=False
    ))
    conclusion = {
        "scope": "H5-O0 montage-orientation opportunity",
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
            root=root, reference_spectra=reference_spectrum,
            screening_spectra=predecision_spectrum, screening=screening,
            expected=expected, context_map=context_map, split=split,
            metrics=metrics, mechanism=mechanism, cfg=cfg,
        )

    print("\n### H5-O0 screening")
    print(f"contexts screened: {len(screening)}")
    print(f"eligible contexts: {int(screening.eligible.sum())}")
    print(f"screening yield: {float(screening.eligible.mean()):.3f}")
    print("\n### H5-O0 montage-orientation checks")
    for name, passed in checks.items():
        print(f"{name}: {'PASSED' if passed else 'NOT PASSED'}")
    print("\n### H5-O0 opportunity summary")
    print(json.dumps(_json_ready(audit), indent=2, allow_nan=False))
    print(
        "\nMontage-orientation contextual opportunity: "
        f"{conclusions['H5_O0_montage_orientation_opportunity']}"
    )
    print("Machine-learning policy status: NOT TRAINED OR TESTED")
    print(f"Results saved to: {root}")


if __name__ == "__main__":
    main()
