"""Unit tests for the EEG-primary A-to-B reachability experiment."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_eeg_reachability import (
    LOWER_ACTION,
    PRIMARY_ACTION,
    SYNTHETIC_CONTROL,
    TRANSVERSE_CONTROL,
    _actions,
    _eeg_target_quadratures,
    _fit_eeg_axis,
    _score_eeg_rows,
    _state_metrics,
)


class EegReachabilityAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "analysis": {
                    "tacs": {
                        "frequency_hz": 60.0,
                        "phase_rad": 1.5 * np.pi,
                        "primary_amplitude_v_per_m": 0.8,
                        "lower_amplitude_v_per_m": 0.5,
                        "axial_montage": "axial",
                        "transverse_montage": "transverse_x",
                    },
                    "n_bootstrap": 20,
                    "n_permutations": 20,
                }
            }
        )

    def test_target_quadratures_recover_sine_phase(self):
        fs_hz = 500.0
        time_s = (np.arange(500, dtype=float) + 1.0) / fs_hz
        signal = np.sin(2.0 * np.pi * 60.0 * time_s)

        result = _eeg_target_quadratures(
            signal,
            fs_hz=fs_hz,
            start_ms=0.0,
            frequency_hz=60.0,
        )

        self.assertAlmostEqual(result["eeg_target_cosine"], 0.0, places=12)
        self.assertAlmostEqual(result["eeg_target_sine"], 1.0, places=12)
        self.assertAlmostEqual(result["eeg_target_resultant"], 1.0, places=12)

    def test_discovery_axis_is_oriented_from_A_to_B(self):
        rows = pd.DataFrame(
            {
                "seed": [1, 2, 1, 2],
                "epoch": ["stimulation"] * 4,
                "condition_id": [
                    "A_async",
                    "A_async",
                    "B_rhythmic_reference",
                    "B_rhythmic_reference",
                ],
                "f1": [-1.2, -0.8, 0.8, 1.2],
                "f2": [0.1, -0.1, 0.1, -0.1],
            }
        )

        axis = _fit_eeg_axis(rows, feature_names=["f1", "f2"])
        scores = _score_eeg_rows(rows, axis)

        self.assertLess(float(np.mean(scores[:2])), 0.0)
        self.assertGreater(float(np.mean(scores[2:])), 0.0)
        self.assertGreater(axis["B_centroid_score"], axis["A_centroid_score"])

    def test_action_set_is_small_and_separates_orientation_control(self):
        actions = _actions(self._config())

        self.assertEqual(actions["A_async"]["ac_amplitude_v_per_m"], 0.0)
        self.assertEqual(actions[LOWER_ACTION]["ac_amplitude_v_per_m"], 0.5)
        self.assertEqual(actions[PRIMARY_ACTION]["ac_amplitude_v_per_m"], 0.8)
        self.assertEqual(actions[PRIMARY_ACTION]["montage"], "axial")
        self.assertEqual(
            actions[TRANSVERSE_CONTROL]["montage"], "transverse_x"
        )

    def test_reachability_rows_include_sham_policy_action(self):
        condition_scores = {
            "A_async": -1.0,
            "B_rhythmic_reference": 1.0,
            PRIMARY_ACTION: 0.5,
            LOWER_ACTION: 0.0,
            TRANSVERSE_CONTROL: -0.8,
            SYNTHETIC_CONTROL: -0.5,
        }
        rows = pd.DataFrame(
            [
                {
                    "seed": 7,
                    "epoch": "stimulation",
                    "condition_id": condition,
                    "primary_eeg_score": score,
                }
                for condition, score in condition_scores.items()
            ]
        )

        seed_rows, _ = _state_metrics(
            rows,
            score_name="primary_eeg_score",
            target_score=1.0,
            cfg=self._config(),
            rng=np.random.default_rng(2),
        )
        indexed = seed_rows.set_index("condition_id")

        self.assertIn("A_async", indexed.index)
        self.assertAlmostEqual(
            indexed.loc["A_async", "target_distance_improvement"], 0.0
        )
        self.assertAlmostEqual(
            indexed.loc[PRIMARY_ACTION, "target_distance_improvement"], 1.5
        )


if __name__ == "__main__":
    unittest.main()
