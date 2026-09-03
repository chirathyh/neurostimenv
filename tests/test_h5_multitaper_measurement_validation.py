"""Focused design and signal-processing tests for H5-I0b."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_h5_iaf_measurement_validation import (
    GAUSSIAN,
)
from experiments.ballnstick_analysis.run_ballnstick_h5_multitaper_measurement_validation import (
    MT_POOLED,
    MT_TEMPORAL,
    NEURAL,
    OBSERVED,
    _context_specs,
    _estimate_multitaper_methods,
    _multitaper_log_psd,
    _select_discovery_estimator,
    _soft_support,
)


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "env=ballnstick",
                "analysis=ballnstick_h5_multitaper_measurement_validation",
            ],
        )


class H5MultitaperMeasurementValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()

    def test_full_crossed_design_is_disjoint_and_complete(self):
        rows = pd.DataFrame(_context_specs(self.cfg))
        self.assertEqual(len(rows), 144)
        self.assertEqual(len(rows[rows.split.eq("discovery")]), 48)
        self.assertEqual(len(rows[rows.split.eq("confirmation")]), 96)
        discovery = set(rows[rows.split.eq("discovery")].structure_seed)
        confirmation = set(rows[rows.split.eq("confirmation")].structure_seed)
        self.assertTrue(discovery.isdisjoint(confirmation))
        for _, structure in rows.groupby("structure_seed"):
            self.assertEqual(len(structure), 8)
            self.assertEqual(set(structure.hidden_frequency_hz), {9.0, 11.0})
            self.assertEqual(set(structure.diffusion_rad2_per_s), {0.5, 2.0})
            self.assertEqual(set(structure.shared_modulated_fraction), {0.5, 1.0})

    def test_multitaper_psd_is_finite_and_recovers_clean_carriers(self):
        fs_hz = 500.0
        time_s = np.arange(int(30.0 * fs_hz)) / fs_hz
        for frequency in (9.0, 11.0):
            values = np.sin(2.0 * np.pi * frequency * time_s + 0.31)
            bins, log_psd = _multitaper_log_psd(
                values,
                fs_hz=fs_hz,
                time_bandwidth=3.0,
                number_of_tapers=5,
                zero_padding_s=32.0,
            )
            self.assertTrue(np.all(np.isfinite(log_psd)))
            alpha = bins[(bins >= 8.0) & (bins <= 12.0)]
            alpha_psd = log_psd[(bins >= 8.0) & (bins <= 12.0)]
            self.assertLess(abs(float(alpha[np.argmax(alpha_psd)]) - frequency), 0.15)

    def test_pooled_estimators_recover_noisy_9_and_11_hz(self):
        fs_hz = 500.0
        time_s = np.arange(int(30.0 * fs_hz)) / fs_hz
        rng = np.random.default_rng(510)
        for hidden in (9.0, 11.0):
            values = (
                np.sin(2.0 * np.pi * hidden * time_s + 0.2)
                + 0.35 * rng.standard_normal(time_s.size)
            )
            rows, spectrum, windows = _estimate_multitaper_methods(
                values,
                fs_hz=fs_hz,
                hidden_frequency_hz=hidden,
                input_signal=OBSERVED,
                cfg=self.cfg,
            )
            self.assertFalse(spectrum.empty)
            self.assertFalse(windows.empty)
            by_name = {row["estimator"]: row for row in rows}
            self.assertEqual(set(by_name), {GAUSSIAN, MT_POOLED, MT_TEMPORAL})
            for estimator in (MT_POOLED, MT_TEMPORAL):
                self.assertEqual(by_name[estimator]["selected_frequency_hz"], hidden)
                self.assertTrue(by_name[estimator]["frequency_detected_correctly"])
                self.assertTrue(by_name[estimator]["identified"])

    def test_soft_support_preserves_evidence_magnitude(self):
        # Four weak incorrect windows must not outweigh two strong correct ones.
        deltas = np.asarray([2.0, 1.5, -0.1, -0.1, -0.1, -0.1])
        self.assertGreater(_soft_support(deltas, 11.0), 0.85)
        self.assertLess(_soft_support(deltas, 9.0), 0.15)

    def test_frozen_gaussian_and_neural_audit_cannot_be_selected(self):
        rows = []
        for input_signal in (OBSERVED, NEURAL):
            for estimator, accuracy in (
                (GAUSSIAN, 1.0),
                (MT_POOLED, 0.90),
                (MT_TEMPORAL, 0.88),
            ):
                rows.append({
                    "input_signal": input_signal,
                    "estimator": estimator,
                    "accuracy": accuracy,
                    "accepted_fraction": 0.90,
                    "accepted_accuracy": 0.95,
                    "wrong_action_rate": 0.02,
                    "minimum_frequency_accuracy": 0.85,
                    "minimum_diffusion_accuracy": 0.85,
                    "minimum_shared_drive_accuracy": 0.85,
                    "minimum_structure_accuracy": 0.80,
                    "mean_absolute_peak_error_hz": 0.2,
                })
        selected, selection = _select_discovery_estimator(
            pd.DataFrame(rows), self.cfg
        )
        self.assertEqual(selected, MT_POOLED)
        self.assertEqual(set(selection.input_signal), {OBSERVED})
        self.assertNotIn(GAUSSIAN, set(selection.estimator))


if __name__ == "__main__":
    unittest.main()
