"""Unit tests for hierarchical EEG-only tACS action identification."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from env.models.neuron.stimulation import apply_raised_cosine_block_envelope
from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (
    _fit_centroid_model,
    _fourier_coefficients,
    _match_complex_observation,
    _select_target_frequency,
)


class HierarchicalTacsAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "analysis": {
                    "target_fs_hz": 500,
                    "low_hz": 0.5,
                    "high_hz": 100.0,
                    "timeline": {"block_ramp_ms": 250.0},
                }
            }
        )

    def test_fourier_coefficients_recover_amplitude_and_phase(self):
        fs_hz = 500.0
        time_s = (np.arange(1500, dtype=float) + 1.0) / fs_hz
        signal = 2.0e-6 * np.cos(2.0 * np.pi * 60.0 * time_s)
        signal += -3.0e-6 * np.sin(2.0 * np.pi * 60.0 * time_s)

        cosine, sine = _fourier_coefficients(
            signal,
            fs_hz=fs_hz,
            start_ms=0.0,
            frequency_hz=60.0,
        )

        self.assertAlmostEqual(cosine, 2.0e-6, places=12)
        self.assertAlmostEqual(sine, -3.0e-6, places=12)

    def test_target_frequency_is_selected_from_eeg_power_not_metadata(self):
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
                "log10_band_power_40_hz": [0.0, 0.1, 0.1, 0.2],
                "log10_band_power_60_hz": [0.0, 0.1, 1.0, 1.1],
                "log10_band_power_80_hz": [0.0, 0.1, -0.1, 0.0],
            }
        )
        features = [
            "log10_band_power_40_hz",
            "log10_band_power_60_hz",
            "log10_band_power_80_hz",
        ]
        model = _fit_centroid_model(rows, feature_names=features)

        selected, table = _select_target_frequency(
            rows,
            candidate_frequencies_hz=[40.0, 60.0, 80.0],
            spectral_model=model,
        )

        self.assertEqual(selected, 60.0)
        self.assertTrue(
            table.loc[table.frequency_hz.eq(60.0), "selected_from_eeg"].item()
        )

    def test_complex_observation_control_matches_both_quadratures(self):
        cfg = self._config()
        fs_hz = 500.0
        start_ms = 1000.0
        stop_ms = 4000.0
        times_ms = start_ms + (np.arange(1500, dtype=float) + 1.0) * 2.0
        rng = np.random.default_rng(3)
        a_raw = rng.normal(0.0, 2e-8, size=times_ms.size)
        angle = 2.0 * np.pi * 60.0 * times_ms / 1000.0
        cosine = apply_raised_cosine_block_envelope(
            np.cos(angle),
            time_ms=times_ms,
            block_start_ms=start_ms,
            block_stop_ms=stop_ms,
            ramp_ms=250.0,
        )
        sine = apply_raised_cosine_block_envelope(
            np.sin(angle),
            time_ms=times_ms,
            block_start_ms=start_ms,
            block_stop_ms=stop_ms,
            ramp_ms=250.0,
        )
        active_raw = a_raw + 4e-8 * cosine - 2e-8 * sine
        outputs = [
            {
                "t_start_ms": start_ms + index * 1000.0,
                "t_stop_ms": start_ms + (index + 1) * 1000.0,
                "sample_times_ms": times_ms[index * 500 : (index + 1) * 500],
            }
            for index in range(3)
        ]

        def episode(raw):
            return {
                "raw_by_epoch": {"stimulation": raw},
                "simulator_fs_hz": fs_hz,
                "simulation": {"outputs_by_epoch": {"stimulation": outputs}},
            }

        synthetic, diagnostics = _match_complex_observation(
            episode(a_raw),
            episode(active_raw),
            selected_frequency_hz=60.0,
            cfg=cfg,
        )

        self.assertLess(abs(diagnostics["cosine_residual_v"]), 1e-15)
        self.assertLess(abs(diagnostics["sine_residual_v"]), 1e-15)
        self.assertTrue(np.all(np.isfinite(synthetic)))


if __name__ == "__main__":
    unittest.main()
