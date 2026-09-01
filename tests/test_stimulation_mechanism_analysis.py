"""Unit tests for the confirmatory stimulation-analysis primitives."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_stimulation_mechanism import (
    _analyze_eeg,
    _make_synthetic_epoch,
    _phase_metrics,
    _stimulus_excluded_features,
)


class StimulationMechanismAnalysisTests(unittest.TestCase):
    @staticmethod
    def _analysis_config():
        return OmegaConf.create(
            {
                "analysis": {
                    "target_fs_hz": 500,
                    "low_hz": 0.5,
                    "high_hz": 100.0,
                    "stimulus_exclusion_half_width_hz": 1.0,
                    "protocol": {
                        "frequency_hz": 10.0,
                        "block_ramp_ms": 250.0,
                    },
                    "synthetic_control": {
                        "match_driven_band_power": True,
                        "phase_rad": 0.0,
                    },
                }
            }
        )

    def test_phase_locking_is_one_for_cycle_aligned_spikes(self):
        metrics = _phase_metrics(
            np.arange(0.0, 1000.0, 100.0),
            frequency_hz=10.0,
            phase_origin_ms=0.0,
        )

        self.assertAlmostEqual(metrics["plv"], 1.0)
        self.assertAlmostEqual(metrics["rayleigh_z"], 10.0)

    def test_primary_features_remove_the_stimulation_fundamental(self):
        frequencies_hz = np.arange(0.0, 100.5, 0.5)
        psd = np.ones_like(frequencies_hz)
        psd[np.isclose(frequencies_hz, 10.0)] = 1000.0

        features = _stimulus_excluded_features(
            frequencies_hz,
            psd,
            stimulus_frequency_hz=10.0,
            half_width_hz=1.0,
        )

        raw_total = float(
            np.trapz(
                psd[(frequencies_hz >= 1.0) & (frequencies_hz <= 80.0)],
                frequencies_hz[
                    (frequencies_hz >= 1.0) & (frequencies_hz <= 80.0)
                ],
            )
        )
        self.assertLess(features["total_power_1_80_excluding_stimulus"], raw_total)
        self.assertGreater(features["stimulus_frequency_power"], 100.0)
        self.assertTrue(
            np.isfinite(features["relative_gamma_power_excluding_stimulus"])
        )

    def test_synthetic_control_matches_active_driven_band_power(self):
        cfg = self._analysis_config()
        fs_hz = 500.0
        time_s = np.arange(int(4.0 * fs_hz)) / fs_hz
        b_raw = 1e-10 * np.sin(2.0 * np.pi * 50.0 * time_s)
        active_raw = b_raw + 8e-11 * np.sin(2.0 * np.pi * 10.0 * time_s)

        synthetic, amplitude, achieved = _make_synthetic_epoch(
            b_raw,
            active_raw,
            simulator_fs_hz=fs_hz,
            cfg=cfg,
        )
        active_features, _, _, _ = _analyze_eeg(
            active_raw,
            simulator_fs_hz=fs_hz,
            cfg=cfg,
        )

        self.assertGreater(amplitude, 0.0)
        self.assertEqual(synthetic.shape, b_raw.shape)
        self.assertAlmostEqual(
            achieved / active_features["alpha_power"],
            1.0,
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
