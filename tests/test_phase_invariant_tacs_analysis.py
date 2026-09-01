"""Unit tests for the phase-invariant tACS confirmation analysis."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_hierarchical_tacs import (
    _add_distances,
    _fit_centroid_model,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_invariant_tacs import (
    _actions,
    _audit_target_frequency,
    _selected_power_feature,
    _validate_design,
)


class PhaseInvariantTacsAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "experiment": {"seed": 1},
                "analysis": {
                    "simulator": "online",
                    "inhibition_scale": 1.0,
                    "maximum_field_v_per_m": 0.8,
                    "candidate_frequencies_hz": [40.0, 60.0, 80.0],
                    "frozen_protocol": {
                        "frequency_hz": 60.0,
                        "phase_rad": 0.0,
                        "primary_amplitude_v_per_m": 0.8,
                        "lower_amplitude_v_per_m": 0.5,
                    },
                    "calibration": {"n_seeds": 2, "seed_offset": 100},
                    "validation": {
                        "n_seeds": 2,
                        "seed_offset": 200,
                        "include_lower_dose": False,
                    },
                    "tacs": {
                        "axial_montage": "axial",
                        "transverse_montage": "transverse_x",
                    },
                },
            }
        )

    def test_primary_feature_is_power_not_absolute_phase(self):
        feature = _selected_power_feature(60.0)
        self.assertEqual(feature, "log10_band_power_60_hz")
        self.assertNotIn("cosine", feature)
        self.assertNotIn("sine", feature)
        self.assertNotIn("phase", feature)

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
                feature: [0.0, 0.1, 1.0, 1.1],
            }
        )
        model = _fit_centroid_model(rows, feature_names=[feature])
        probes = pd.DataFrame(
            {
                feature: [0.8, 0.8],
                "selected_eeg_cosine_v": [1.0, -1.0],
                "selected_eeg_sine_v": [-2.0, 2.0],
            }
        )
        scored = _add_distances(probes, model, prefix="phase_invariant")
        self.assertAlmostEqual(
            scored.phase_invariant_distance_to_B.iloc[0],
            scored.phase_invariant_distance_to_B.iloc[1],
        )

    def test_action_phase_is_fixed_and_not_searched(self):
        cfg = self._config()
        _validate_design(cfg)
        actions = _actions(cfg)
        self.assertEqual(
            {float(action["phase_rad"]) for action in actions.values()}, {0.0}
        )
        self.assertEqual(float(actions["A_tacs_primary"]["frequency_hz"]), 60.0)
        self.assertEqual(
            float(actions["A_tacs_primary"]["ac_amplitude_v_per_m"]), 0.8
        )

    def test_frequency_audit_reports_an_all_negative_result(self):
        features = [
            "log10_band_power_40_hz",
            "log10_band_power_60_hz",
            "log10_band_power_80_hz",
        ]
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
                features[0]: [1.0, 1.1, 0.9, 1.0],
                features[1]: [1.0, 1.1, 0.5, 0.6],
                features[2]: [1.0, 1.1, 0.7, 0.8],
            }
        )
        model = _fit_centroid_model(rows, feature_names=features)
        selected, table = _audit_target_frequency(
            rows,
            candidate_frequencies_hz=[40.0, 60.0, 80.0],
            spectral_model=model,
        )
        self.assertEqual(selected, 40.0)
        self.assertLess(table.mean_standardized_shift.max(), 0.0)
        self.assertTrue(
            table.loc[table.frequency_hz.eq(40.0), "largest_observed_shift"].item()
        )

    def test_nonzero_absolute_phase_is_rejected(self):
        cfg = self._config()
        cfg.analysis.frozen_protocol.phase_rad = np.pi
        with self.assertRaisesRegex(ValueError, "fixed to zero"):
            _validate_design(cfg)

    def test_calibration_and_validation_seeds_must_be_disjoint(self):
        cfg = self._config()
        cfg.analysis.validation.seed_offset = 101
        with self.assertRaisesRegex(ValueError, "nonempty/disjoint"):
            _validate_design(cfg)

    def test_oversized_simulator_seed_is_rejected_before_execution(self):
        cfg = self._config()
        cfg.experiment.seed = 900_000
        with self.assertRaisesRegex(ValueError, "seeds are too large"):
            _validate_design(cfg)


if __name__ == "__main__":
    unittest.main()
