"""Focused tests for H5-P2B active tracker-response mapping."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import open_dict

from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (
    CONSERVATIVE,
    RESPONSIVE,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_phase_tracker_response_mapping import (
    HIGH_NOISE,
    LOW_NOISE,
    P2B_CONTEXT_FEATURES,
    _context_specs,
    _feature_response_associations,
    _load_sources,
    _predecision_tracker_features,
    _response_map,
    _run_context_specs,
    _validate_design,
    _with_noise_fraction,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (
    _one_second_rows,
)


class H5PhaseTrackerResponseMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        GlobalHydra.instance().clear()
        with initialize_config_dir(
            version_base=None,
            config_dir=str((Path(__file__).parents[1] / "configs").resolve()),
        ):
            cls.cfg = compose(
                config_name="config",
                overrides=[
                    "experiment.seed=1",
                    "env=ballnstick",
                    "analysis=ballnstick_h5_phase_tracker_response_mapping",
                    "env.simulation.obs_win_len=1000",
                ],
            )

    def test_sources_hash_lock_and_crossed_design_is_disjoint(self) -> None:
        sources = _load_sources(self.cfg)
        _validate_design(self.cfg, sources)
        contexts = _context_specs(self.cfg)
        self.assertEqual(len(contexts), 48)
        self.assertEqual(len({row["structure_seed"] for row in contexts}), 6)
        self.assertEqual(
            {row["hidden_frequency_hz"] for row in contexts}, {9.0, 11.0}
        )
        self.assertEqual(
            {row["diffusion_rad2_per_s"] for row in contexts}, {0.5, 2.0}
        )
        self.assertEqual(
            {row["observation_noise_fraction"] for row in contexts}, {0.25, 0.5}
        )
        pairs = pd.DataFrame(contexts).groupby("paired_noise_context_id")
        self.assertTrue(pairs.size().eq(2).all())
        for column in ("structure_seed", "history_seed", "phase_seed", "trial_seed"):
            self.assertTrue(pairs[column].nunique().eq(1).all())
        new_seeds = {
            int(row[column])
            for row in contexts
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed"
            )
        }
        self.assertTrue(new_seeds.isdisjoint(sources["source_seed_union"]))

    def test_smoke_selection_contains_paired_noise_and_target_corners(self) -> None:
        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.analysis.smoke_test = True
            cfg.analysis.smoke_context_limit = 4
        selected = _run_context_specs(cfg)
        self.assertEqual(len(selected), 4)
        self.assertEqual({row["hidden_frequency_hz"] for row in selected}, {9.0})
        corners = {
            (row["label"], row["observation_noise_label"]) for row in selected
        }
        self.assertEqual(corners, {
            ("high_diffusion", LOW_NOISE),
            ("high_diffusion", HIGH_NOISE),
            ("low_diffusion", LOW_NOISE),
            ("low_diffusion", HIGH_NOISE),
        })
        self.assertTrue(
            pd.DataFrame(selected).groupby("paired_noise_context_id").size().eq(2).all()
        )

    def test_noise_override_changes_measurement_not_generator(self) -> None:
        modified = _with_noise_fraction(self.cfg, 0.5)
        self.assertAlmostEqual(
            float(modified.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
            0.5,
        )
        self.assertAlmostEqual(
            float(self.cfg.analysis.observation_noise.rms_fraction_of_baseline_neural_eeg),
            0.25,
        )
        self.assertEqual(
            OmegaLike(self.cfg.env.network.background.E.rhythm),
            OmegaLike(modified.env.network.background.E.rhythm),
        )

    def test_predecision_tracker_features_are_finite_and_causal(self) -> None:
        dt_ms = float(self.cfg.env.network.dt)
        times = np.arange(dt_ms, 10_000.0 + 0.5 * dt_ms, dt_ms)
        phase = 2.0 * np.pi * 9.0 * times / 1000.0
        values = np.sin(phase) + 0.05 * np.sin(2.0 * np.pi * 3.0 * times / 1000.0)
        episode = {
            "simulation": {
                "observed_outputs_by_epoch": {
                    "baseline": [{
                        "eeg_v": values,
                        "sample_times_ms": times,
                        "t_stop_ms": 10_000.0,
                    }]
                }
            },
            "simulator_fs_hz": 1000.0 / dt_ms,
        }
        features, diagnostics = _predecision_tracker_features(
            episode, 9.0, self.cfg
        )
        self.assertEqual(set(features), set(
            name for name in P2B_CONTEXT_FEATURES if name.startswith("pre_")
        ))
        self.assertTrue(np.isfinite(list(features.values())).all())
        self.assertEqual(set(diagnostics.tracker_profile), {
            CONSERVATIVE, RESPONSIVE,
        })
        self.assertTrue(diagnostics.uses_only_predecision_observed_EEG.all())
        self.assertEqual(len(diagnostics), 2 * 64)

    def test_response_map_recovers_prespecified_active_crossover(self) -> None:
        expected_rows, metric_rows = [], []
        for structure in range(6):
            for label, noise_label, noise_fraction, advantage in (
                ("high_diffusion", LOW_NOISE, 0.25, 0.04),
                ("low_diffusion", HIGH_NOISE, 0.50, -0.04),
                ("low_diffusion", LOW_NOISE, 0.25, -0.01),
                ("high_diffusion", HIGH_NOISE, 0.50, 0.01),
            ):
                context_id = f"s{structure}_{label}_{noise_label}"
                base = {
                    "context_id": context_id,
                    "paired_noise_context_id": f"s{structure}_{label}",
                    "structure_seed": 800000 + structure,
                    "hidden_frequency_hz": 9.0,
                    "label": label,
                    "diffusion_rad2_per_s": 2.0 if label == "high_diffusion" else 0.5,
                    "observation_noise_label": noise_label,
                    "observation_noise_fraction": noise_fraction,
                    "shared_drive_label": "full_shared_drive",
                    "shared_modulated_fraction": 1.0,
                    "EEG_selected_frequency_hz": 9.0,
                    **{feature: float(structure) for feature in P2B_CONTEXT_FEATURES},
                }
                slow_distance = 0.20
                fast_distance = slow_distance - advantage
                slow_phase = 0.80 if advantage > 0 else 0.50
                fast_phase = 0.50 if advantage > 0 else 0.80
                for mode, distance, phase_error in (
                    (CONSERVATIVE, slow_distance, slow_phase),
                    (RESPONSIVE, fast_distance, fast_phase),
                ):
                    expected_rows.append({
                        **base,
                        "controller_mode": mode,
                        "expected_post_distance_to_B_log10": distance,
                        "future_sd_post_distance_log10": 0.005,
                        "mean_abs_common_phase_error_rad": phase_error,
                    })
                    for future in range(4):
                        metric_rows.append({
                            "context_id": context_id,
                            "future_index": future + 1,
                            "controller_mode": mode,
                            "post_distance_to_B_log10": distance + 0.001 * future,
                        })
        action_map, structure, directions, opportunity = _response_map(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows), self.cfg
        )
        self.assertEqual(set(action_map.expected_optimal_profile), {
            CONSERVATIVE, RESPONSIVE,
        })
        self.assertAlmostEqual(
            opportunity["mean_fast_advantage_high_diffusion_low_noise_log10"],
            0.04,
        )
        self.assertAlmostEqual(
            opportunity["mean_slow_advantage_low_diffusion_high_noise_log10"],
            0.04,
        )
        self.assertEqual(len(structure), 6)
        self.assertTrue(directions.both_target_directions_positive.all())
        self.assertGreater(
            opportunity["phase_error_advantage_response_spearman_rho"], 0.85
        )

    def test_structure_preserving_association_detects_observed_feature(self) -> None:
        rows = []
        rng = np.random.default_rng(109)
        for structure in range(6):
            for context in range(8):
                driver = float(context - 3.5)
                row = {
                    "structure_seed": 900000 + structure,
                    "fast_advantage_over_slow_log10": 0.02 * driver,
                }
                for feature in P2B_CONTEXT_FEATURES:
                    row[feature] = (
                        driver if feature == "pre_fast_slow_phase_disagreement_rad"
                        else float(rng.normal())
                    )
                rows.append(row)
        cfg = self.cfg.copy()
        with open_dict(cfg):
            cfg.analysis.response_mapping.association_permutations = 999
        associations, audit = _feature_response_associations(
            pd.DataFrame(rows), cfg
        )
        selected = associations[
            associations.feature.eq("pre_fast_slow_phase_disagreement_rad")
        ].iloc[0]
        self.assertGreater(selected.structure_centered_spearman_rho, 0.95)
        self.assertTrue(bool(selected.passes_response_association_gate))
        self.assertEqual(
            audit["selected_candidate_response_feature"],
            "pre_fast_slow_phase_disagreement_rad",
        )

    def test_policy_features_exclude_hidden_and_configured_state(self) -> None:
        forbidden = ("hidden", "diffusion", "noise_fraction", "spike", "phase_rad")
        for feature in P2B_CONTEXT_FEATURES:
            self.assertFalse(any(token in feature for token in forbidden), feature)

    def test_zero_trim_retains_all_one_second_trajectory_windows(self) -> None:
        dt_ms = float(self.cfg.env.network.dt)
        fs_hz = 1000.0 / dt_ms
        time_s = np.arange(int(4.0 * fs_hz), dtype=float) / fs_hz
        raw = 1.0e-10 * np.sin(2.0 * np.pi * 9.0 * time_s)
        episode = {
            "raw_by_epoch": {"stimulation": raw},
            "simulation": {"block_start_ms": 31_000.0},
            "simulator_fs_hz": fs_hz,
        }
        context = {
            "context_id": "synthetic_context",
            "structure_seed": 1,
            "hidden_frequency_hz": 9.0,
            "label": "high_diffusion",
            "diffusion_rad2_per_s": 2.0,
        }
        rows = _one_second_rows(
            episode,
            context=context,
            screening={"context_C1": 0.5},
            future_index=0,
            mode="sham",
            target_alpha=-20.0,
            cfg=self.cfg,
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            [row["analysis_window_index"] for row in rows], [1, 2, 3, 4]
        )
        self.assertTrue(np.isfinite([
            row["distance_to_B_log10"] for row in rows
        ]).all())


def OmegaLike(value: object) -> str:
    """Stable lightweight equality representation for a nested config node."""
    return str(value)


if __name__ == "__main__":
    unittest.main()
