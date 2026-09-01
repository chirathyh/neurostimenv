"""Unit tests for EEG-relative alpha-suppression experiment logic."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_alpha_suppression import (
    _field_phase_from_eeg_coefficients,
    _select_reference_depth,
    _select_suppressive_offset,
)


class AlphaSuppressionAnalysisTests(unittest.TestCase):
    def test_zero_offset_aligns_field_waveform_with_extrapolated_eeg(self):
        frequency_hz = 10.0
        block_start_ms = 4125.0
        coefficient_phase = 0.7
        cosine = 2.0 * np.cos(coefficient_phase)
        sine = 2.0 * np.sin(coefficient_phase)

        field_phase, eeg_phase = _field_phase_from_eeg_coefficients(
            cosine,
            sine,
            block_start_ms=block_start_ms,
            frequency_hz=frequency_hz,
            relative_offset_rad=0.0,
        )

        self.assertAlmostEqual(np.sin(field_phase), np.cos(eeg_phase), places=12)
        self.assertAlmostEqual(np.cos(field_phase), -np.sin(eeg_phase), places=12)

    def test_opposite_offset_reverses_field_waveform(self):
        aligned, _ = _field_phase_from_eeg_coefficients(
            1.0, -0.5, block_start_ms=5000.0, frequency_hz=10.0,
            relative_offset_rad=0.0,
        )
        opposite, _ = _field_phase_from_eeg_coefficients(
            1.0, -0.5, block_start_ms=5000.0, frequency_hz=10.0,
            relative_offset_rad=np.pi,
        )
        self.assertAlmostEqual(np.sin(opposite), -np.sin(aligned), places=12)
        self.assertAlmostEqual(np.cos(opposite), -np.cos(aligned), places=12)

    def test_calibration_selects_smallest_qualified_rate_matched_depth(self):
        summary = pd.DataFrame({
            "modulation_depth": [0.04, 0.08, 0.12],
            "mean": [0.01, 0.04, 0.08],
            "positive_seed_count": [2, 2, 2],
            "rate_matched_fraction": [1.0, 1.0, 1.0],
        })
        cfg = OmegaConf.create({"analysis": {"criteria": {
            "minimum_reference_log10_alpha_shift": 0.02,
            "minimum_calibration_positive_seeds": 2,
        }}})
        selected, passed = _select_reference_depth(summary, cfg)
        self.assertTrue(passed)
        self.assertAlmostEqual(selected, 0.08)

    def test_phase_selection_uses_alpha_suppression_not_hidden_ppc(self):
        summary = pd.DataFrame({
            "relative_phase_offset_rad": [0.0, np.pi],
            "mean": [0.08, 0.02],
            "mean_target_distance_improvement": [0.03, 0.01],
            "mean_E_ppc_reduction": [-1.0, 10.0],
            "rate_safe_fraction": [1.0, 1.0],
        })
        selected, positive = _select_suppressive_offset(summary)
        self.assertTrue(positive)
        self.assertAlmostEqual(selected, 0.0)


if __name__ == "__main__":
    unittest.main()
