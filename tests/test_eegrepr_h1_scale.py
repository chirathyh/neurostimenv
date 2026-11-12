"""Unitary FFT, strict paired targets, actual timing and own-risk scale checks."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.eeg_representation.h1 import scale
from tests.test_eegrepr_h1_compression import panel_fixture


class FFTTests(unittest.TestCase):
    def test_complex_fft_risk_and_covariance_trace_with_unit_scaling(self):
        rng = np.random.default_rng(9)
        y = rng.normal(size=(40, 17)) @ rng.normal(size=(17, 11))
        pred = rng.normal(size=11)
        initial = scale.fft_invariance(y, pred)
        transformed = np.fft.fft(y, axis=-1, norm="ortho")
        centered = transformed - transformed.mean(axis=0)
        covariance = centered.conj().T @ centered / (len(y) - 1)
        self.assertAlmostEqual(initial["covariance_trace"], float(np.trace(covariance).real))
        for factor in (1e-6, 1000):
            result = scale.fft_invariance(y * factor, pred * factor)
            self.assertAlmostEqual(result["complex_fft_mse"] / factor ** 2, initial["real_mse"])
            self.assertAlmostEqual(result["complex_fft_covariance_trace"] / factor ** 2, initial["covariance_trace"])

    def test_magnitude_phase_loss_is_distinct_target(self):
        result = scale.fft_controls()
        self.assertGreater(result["phase_removing_counterexample"]["waveform_mse"], .9)
        self.assertLess(result["phase_removing_counterexample"]["max_power_spectrum_difference"], 1e-12)
        self.assertEqual(result["dft_bin_examples"][1]["bin_spacing_hz"], 7.8125)


class ScalePanelTests(unittest.TestCase):
    def test_same_physical_target_requires_ids_and_numeric_match(self):
        first, second = panel_fixture(), panel_fixture()
        second["manifest"]["target_fingerprint"] = "different_acquisition"
        second["manifest"]["input"]["acquisition_bits"] = 4
        self.assertTrue(scale.assert_fixed_physical_target(first, second))
        second["evaluation"]["Y_physical"][0, 0, 0] += 1e-12
        with self.assertRaisesRegex(ValueError, "Physical future targets differ"):
            scale.assert_fixed_physical_target(first, second)
        second = panel_fixture()
        second["evaluation"]["context_id"][0] = "unpaired"
        with self.assertRaisesRegex(ValueError, "Unpaired"):
            scale.assert_fixed_physical_target(first, second)

    def test_actual_registry_fft_pair_has_same_physical_loss(self):
        from experiments.eeg_representation.h1.targets import TargetSpec, operator_from_projected
        first, second = panel_fixture(), panel_fixture()
        wave = TargetSpec(cutoff_ms=1000., history_s=.016, lead_ms=10., duration_ms=8., output_grid_hz=500.)
        fft = TargetSpec(cutoff_ms=1000., history_s=.016, lead_ms=10., duration_ms=8., output_grid_hz=500., operator="full_complex_fft")
        first["manifest"]["target_spec"] = wave.to_dict()
        second["manifest"].update(operator="full_complex_fft", target_spec=fft.to_dict())
        for split in ("train", "development", "evaluation"):
            y = first[split]["Y_physical"]
            second[split]["Y_physical"] = np.asarray([[operator_from_projected(row, fft) for row in context] for context in y])
        self.assertTrue(scale.assert_fft_equivalent(first, second))
        second["evaluation"]["Y_physical"] *= .5
        with self.assertRaisesRegex(ValueError, "registry transform"):
            scale.assert_fft_equivalent(first, second)

    def test_summary_uses_executed_rate_own_r0_and_m1_unavailable(self):
        panel = panel_fixture()
        result, arrays = scale.summarize_panel(panel)
        self.assertEqual(result["input_sample_rate_hz"], 500)
        self.assertEqual(result["input_samples"], 8)
        self.assertEqual(result["output_dimensions"], 4)
        self.assertGreater(result["r0_standardized"], result["state_floor_standardized"])
        self.assertAlmostEqual(result["r0_physical"], result["r0_standardized"] * 1e-12)
        panel["evaluation"]["future_mask"][:, 1:] = False
        result, arrays = scale.summarize_panel(panel)
        self.assertIsNone(result["state_floor_standardized"])
        self.assertTrue(np.isnan(arrays["state_floor"]).all())

    def test_rate_availability_grid_mismatch_rejected(self):
        panel = panel_fixture()
        panel["manifest"]["input"]["sample_rate_hz"] = 100
        with self.assertRaisesRegex(ValueError, "recorded timestamps"):
            scale.summarize_panel(panel)
        panel = panel_fixture()
        panel["evaluation"]["input_time_ms"][-1] = 1
        with self.assertRaisesRegex(ValueError, "past cutoff"):
            scale.summarize_panel(panel)

    def test_small_scale_pipeline_and_actual_jobs_not_config_list(self):
        panel = panel_fixture()
        config = {"jobs": [{"id": "fixture"}, {"id": "never_executed"}]}
        with tempfile.TemporaryDirectory() as temp, patch("experiments.eeg_representation.h1.dataset.list_jobs", return_value=["fixture"]), \
                patch("experiments.eeg_representation.h1.dataset.load_panel", return_value=panel):
            result = scale.run_scale(config, temp)
            self.assertEqual(result["executed_jobs"], ["fixture"])
            self.assertEqual(result["actual_executed_axes"]["input_sample_rate_hz"], ["500.0"])
            self.assertTrue((Path(temp) / "scale/controls/fft_controls.json").exists())
            self.assertEqual(scale.run_scale(config, temp, resume=True)["identity"], result["identity"])


if __name__ == "__main__":
    unittest.main()
