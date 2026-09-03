"""Focused design tests for H5-P0 controller-profile system identification."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_h5_controller_profile_feasibility import (
    CONSERVATIVE,
    CONTEXT_FEATURES,
    FULL,
    PARTIAL,
    RESPONSIVE,
    _ar1_path,
    _context_specs,
    _controller_action_map,
    _controller_modes,
    _future_seed,
    _noise_seeds,
    _profile,
    _with_context_state,
)


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "env=ballnstick",
                "analysis=ballnstick_h5_controller_profile_feasibility",
            ],
        )


class H5ControllerProfileFeasibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()

    def test_profiles_change_only_tracker_bandwidth(self):
        self.assertEqual(
            _controller_modes(self.cfg),
            ["sham", CONSERVATIVE, RESPONSIVE],
        )
        self.assertEqual(
            _profile(self.cfg, CONSERVATIVE),
            {"adaptive": True, "history_ms": 1000.0, "update_interval_ms": 250.0},
        )
        self.assertEqual(
            _profile(self.cfg, RESPONSIVE),
            {"adaptive": True, "history_ms": 500.0, "update_interval_ms": 125.0},
        )
        self.assertEqual(float(self.cfg.analysis.actions.amplitude_v_per_m), 0.2)

    def test_shared_drive_pair_has_nested_common_random_numbers(self):
        contexts = _context_specs(self.cfg)
        self.assertEqual(len(contexts), 24)
        for _, pair in pd.DataFrame(contexts).groupby(
            "paired_shared_drive_context_id"
        ):
            self.assertEqual(set(pair.shared_drive_label), {PARTIAL, FULL})
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed",
                "future_group_index",
            ):
                self.assertEqual(pair[column].nunique(), 1)
        partial = next(row for row in contexts if row["shared_drive_label"] == PARTIAL)
        state = _with_context_state(self.cfg, partial)
        for population in ("E", "I"):
            self.assertEqual(
                float(state.env.network.background[population].rhythm.shared_modulated_fraction),
                0.5,
            )

    def test_drive_and_observation_seed_namespaces_are_disjoint(self):
        contexts = _context_specs(self.cfg)
        future = {
            _future_seed(self.cfg, context, index)
            for context in contexts
            for index in range(
                int(self.cfg.analysis.crossed_design.n_future_continuations)
            )
        }
        noise = {
            seed
            for context in contexts
            for index in range(
                int(self.cfg.analysis.crossed_design.n_future_continuations)
            )
            for seed in _noise_seeds(self.cfg, context, index)
        }
        self.assertFalse(future.intersection(noise))

    def test_observation_noise_preserves_history_and_splits_future(self):
        common = dict(
            n_samples=1000,
            split_sample=600,
            history_seed=10,
            coefficient=0.95,
        )
        first = _ar1_path(**common, future_seed=20)
        second = _ar1_path(**common, future_seed=21)
        np.testing.assert_array_equal(first[:600], second[:600])
        self.assertFalse(np.array_equal(first[600:], second[600:]))

    def test_full_information_map_compares_expected_not_single_future(self):
        expected_rows = []
        metric_rows = []
        winners = [CONSERVATIVE, RESPONSIVE, CONSERVATIVE, RESPONSIVE]
        for index, winner in enumerate(winners):
            context = {
                "context_id": f"c{index}",
                "paired_shared_drive_context_id": f"p{index // 2}",
                "structure_seed": 100 + index // 2,
                "hidden_frequency_hz": 9.0 if index % 2 == 0 else 11.0,
                "label": "low_diffusion" if index < 2 else "high_diffusion",
                "diffusion_rad2_per_s": 0.5 if index < 2 else 2.0,
                "shared_drive_label": PARTIAL if index % 2 == 0 else FULL,
                "shared_modulated_fraction": 0.5 if index % 2 == 0 else 1.0,
                **{feature: 0.1 * (index + 1) for feature in CONTEXT_FEATURES},
            }
            for mode in (CONSERVATIVE, RESPONSIVE):
                distance = 0.1 if mode == winner else 0.2
                expected_rows.append({
                    **context,
                    "controller_mode": mode,
                    "expected_post_distance_to_B_log10": distance,
                    "future_sd_post_distance_log10": 0.01,
                    "mean_abs_common_phase_error_rad": 0.2,
                })
                for future in (1, 2):
                    metric_rows.append({
                        "context_id": context["context_id"],
                        "future_index": future,
                        "controller_mode": mode,
                        "post_distance_to_B_log10": distance,
                    })
        action_map, _, audit = _controller_action_map(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows)
        )
        self.assertEqual(set(action_map.expected_optimal_profile), set(winners))
        self.assertEqual(
            audit["optimal_profile_context_count"],
            {CONSERVATIVE: 2, RESPONSIVE: 2},
        )
        self.assertEqual(
            audit["mean_realized_optimal_profile_agreement_fraction"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
