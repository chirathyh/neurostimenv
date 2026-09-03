"""Focused design and signal-processing tests for H5-I0."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_h5_iaf_measurement_validation import (
    GAUSSIAN,
    LEGACY,
    RAW_LONG,
    SMOOTHED,
    _context_specs,
    _estimate_iaf_methods,
    _select_discovery_estimator,
    _unit_ar1_noise,
)


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "env=ballnstick",
                "analysis=ballnstick_h5_iaf_measurement_validation",
            ],
        )


class H5IAFMeasurementValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()

    def test_crossed_design_is_disjoint_and_complete(self):
        rows = pd.DataFrame(_context_specs(self.cfg))
        self.assertEqual(len(rows), 72)
        self.assertEqual(len(rows[rows.split.eq("discovery")]), 24)
        self.assertEqual(len(rows[rows.split.eq("confirmation")]), 48)
        discovery = set(rows[rows.split.eq("discovery")].structure_seed)
        confirmation = set(rows[rows.split.eq("confirmation")].structure_seed)
        self.assertTrue(discovery.isdisjoint(confirmation))
        for _, structure in rows.groupby("structure_seed"):
            self.assertEqual(len(structure), 8)
            self.assertEqual(set(structure.hidden_frequency_hz), {9.0, 11.0})
            self.assertEqual(set(structure.diffusion_rad2_per_s), {0.5, 2.0})
            self.assertEqual(set(structure.shared_modulated_fraction), {0.5, 1.0})

    def test_shared_drive_pairs_use_nested_common_random_numbers(self):
        rows = pd.DataFrame(_context_specs(self.cfg))
        for _, pair in rows.groupby("paired_shared_drive_context_id"):
            self.assertEqual(len(pair), 2)
            self.assertEqual(set(pair.shared_modulated_fraction), {0.5, 1.0})
            for column in (
                "structure_seed", "history_seed", "phase_seed", "trial_seed",
                "noise_seed",
            ):
                self.assertEqual(pair[column].nunique(), 1)

    def test_ar1_noise_is_deterministic_and_unit_rms(self):
        first = _unit_ar1_noise(20_000, seed=123, coefficient=0.95)
        second = _unit_ar1_noise(20_000, seed=123, coefficient=0.95)
        np.testing.assert_array_equal(first, second)
        self.assertAlmostEqual(float(np.sqrt(np.mean(first**2))), 1.0, places=12)
        self.assertGreater(float(np.corrcoef(first[:-1], first[1:])[0, 1]), 0.93)

    def test_robust_estimators_recover_clean_9_and_11_hz_signals(self):
        fs_hz = 500.0
        duration_s = 30.0
        time_s = np.arange(int(fs_hz * duration_s)) / fs_hz
        rng = np.random.default_rng(1234)
        for hidden in (9.0, 11.0):
            eeg = (
                np.sin(2.0 * np.pi * hidden * time_s + 0.3)
                + 0.20 * rng.standard_normal(time_s.size)
            )
            rows, spectrum = _estimate_iaf_methods(
                eeg,
                fs_hz=fs_hz,
                hidden_frequency_hz=hidden,
                cfg=self.cfg,
            )
            self.assertFalse(spectrum.empty)
            by_name = {row["estimator"]: row for row in rows}
            self.assertEqual(set(by_name), {LEGACY, RAW_LONG, SMOOTHED, GAUSSIAN})
            for estimator in (SMOOTHED, GAUSSIAN):
                self.assertEqual(by_name[estimator]["selected_frequency_hz"], hidden)
                self.assertTrue(by_name[estimator]["frequency_detected_correctly"])
                self.assertTrue(by_name[estimator]["identified"])

    def test_legacy_benchmark_cannot_be_selected(self):
        rows = []
        for estimator, accuracy in (
            (LEGACY, 1.0),
            (SMOOTHED, 0.90),
            (GAUSSIAN, 0.85),
        ):
            rows.append({
                "estimator": estimator,
                "accuracy": accuracy,
                "accepted_fraction": 0.90,
                "accepted_accuracy": 0.95,
                "minimum_frequency_accuracy": 0.80,
                "minimum_structure_accuracy": 0.75,
                "mean_absolute_peak_error_hz": 0.1,
            })
        selected, selection = _select_discovery_estimator(
            pd.DataFrame(rows), self.cfg
        )
        self.assertEqual(selected, SMOOTHED)
        self.assertNotIn(LEGACY, set(selection.estimator))


if __name__ == "__main__":
    unittest.main()
