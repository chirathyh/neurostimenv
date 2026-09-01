"""Focused tests for the frozen D0b EEG phase-increment endpoint."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_phase_increment_confirmation import (
    _context_specs,
    _exact_sign_flip,
    _phase_increment_from_phases,
    _window_phase_metrics,
)


class BallAndStickPhaseIncrementConfirmationTests(unittest.TestCase):
    def test_phase_increment_endpoint_has_expected_limits(self):
        stable = _phase_increment_from_phases(np.zeros(12))
        alternating = _phase_increment_from_phases(
            np.arange(12, dtype=float) * np.pi
        )
        self.assertAlmostEqual(
            stable["phase_increment_coherence_real"], 1.0, places=14
        )
        self.assertAlmostEqual(
            alternating["phase_increment_coherence_real"], -1.0, places=14
        )
        self.assertEqual(stable["phase_increment_count"], 11)

    def test_window_endpoint_demodulates_a_stationary_carrier(self):
        fs_hz = 500.0
        duration_s = 12.0
        times = (np.arange(int(fs_hz * duration_s)) + 1.0) / fs_hz
        eeg = np.cos(2.0 * np.pi * 9.0 * times + 0.4)
        metrics = _window_phase_metrics(
            eeg,
            fs_hz=fs_hz,
            start_ms=0.0,
            frequency_hz=9.0,
            phase_window_s=1.0,
            temporal_chunk_s=4.0,
        )
        self.assertAlmostEqual(
            metrics["phase_increment_coherence_real"], 1.0, places=12
        )
        self.assertEqual(metrics["phase_increment_count"], 11)
        self.assertEqual(len(metrics["temporal_chunk_C1_values"]), 3)

    def test_confirmation_seed_grid_is_paired_and_unique(self):
        cfg = OmegaConf.create({
            "experiment": {"seed": 1},
            "analysis": {
                "states": {
                    "frequencies_hz": [9.0, 11.0],
                    "phase_diffusion_levels": [
                        {"label": "low_diffusion", "diffusion_rad2_per_s": 0.5},
                        {"label": "high_diffusion", "diffusion_rad2_per_s": 2.0},
                    ],
                },
                "crossed_design": {
                    "n_structure_seeds": 6,
                    "n_history_seeds": 2,
                    "structure_seed_offset": 418000,
                    "drive_seed_offset": 436000,
                    "phase_seed_offset": 437000,
                    "trial_seed_offset": 438000,
                },
            },
        })
        specs = _context_specs(cfg)
        self.assertEqual(len(specs), 48)
        grouped = {}
        for row in specs:
            grouped.setdefault(row["context_id"], []).append(row)
        self.assertTrue(all(len(rows) == 2 for rows in grouped.values()))
        for rows in grouped.values():
            self.assertEqual(
                len({(x["structure_seed"], x["drive_seed"], x["phase_seed"]) for x in rows}),
                1,
            )
            self.assertEqual({x["label"] for x in rows}, {
                "low_diffusion", "high_diffusion"
            })
        namespaces = [
            {int(row[name]) for row in specs}
            for name in ("structure_seed", "drive_seed", "phase_seed", "trial_seed")
        ]
        self.assertTrue(all(
            not namespaces[left].intersection(namespaces[right])
            for left in range(len(namespaces))
            for right in range(left + 1, len(namespaces))
        ))

    def test_six_all_positive_structures_support_exact_test(self):
        cfg = OmegaConf.create({
            "experiment": {"seed": 1},
            "analysis": {
                "inference": {
                    "exact_sign_flip_max_structures": 20,
                    "monte_carlo_sign_flips": 1000,
                    "random_seed_offset": 1,
                }
            },
        })
        p_value, method, samples = _exact_sign_flip(
            np.asarray([0.2, 0.3, 0.25, 0.4, 0.22, 0.35]), cfg
        )
        self.assertEqual(method, "exact")
        self.assertEqual(samples, 64)
        self.assertAlmostEqual(p_value, 1.0 / 64.0)


if __name__ == "__main__":
    unittest.main()
