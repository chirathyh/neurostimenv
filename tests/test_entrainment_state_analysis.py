"""Unit tests for the minimal entrainment-state reachability experiment."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_entrainment_state import (
    _calibration_summary,
    _condition_config,
    _select_reference_depth,
    _validation_seed_row,
)


class EntrainmentStateAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "analysis": {
                    "reference": {
                        "populations": ["E", "I"],
                        "frequency_hz": 60.0,
                        "phase_rad": 0.0,
                        "target_E_ppc": 0.02,
                        "thinning_envelope_modulation_depth": 0.4,
                    },
                    "calibration": {
                        "minimum_rate_matched_fraction": 1.0,
                    },
                    "criteria": {
                        "maximum_washout_residual_fraction": 0.5,
                    },
                    "rate_reference_tolerance_fraction": 0.2,
                    "rate_guardrails_hz": {
                        "E_min": 0.1,
                        "E_max": 30.0,
                        "I_min": 0.1,
                        "I_max": 60.0,
                    },
                },
                "env": {
                    "network": {
                        "background": {
                            name: {
                                "rhythm": {
                                    "enabled": False,
                                    "modulation_depth": 0.0,
                                    "frequency_hz": 60.0,
                                    "phase_rad": 0.0,
                                    "thinning_envelope_modulation_depth": 0.0,
                                }
                            }
                            for name in ("E", "I")
                        }
                    }
                },
            }
        )

    def test_condition_config_changes_timing_only_on_a_copy(self):
        base = self._config()
        condition = _condition_config(base, modulation_depth=0.3)

        self.assertFalse(base.env.network.background.E.rhythm.enabled)
        self.assertEqual(
            base.env.network.background.E.rhythm.modulation_depth,
            0.0,
        )
        for population in ("E", "I"):
            rhythm = condition.env.network.background[population].rhythm
            self.assertTrue(rhythm.enabled)
            self.assertAlmostEqual(rhythm.modulation_depth, 0.3)
            self.assertAlmostEqual(
                rhythm.thinning_envelope_modulation_depth,
                0.4,
            )

    def test_calibration_selects_rate_matched_depth_nearest_target(self):
        rows = pd.DataFrame(
            {
                "seed": [1, 1, 1],
                "modulation_depth": [0.1, 0.2, 0.3],
                "A_E_ppc": [0.0, 0.0, 0.0],
                "B_E_ppc": [0.006, 0.019, 0.021],
                "B_minus_A_E_ppc": [0.006, 0.019, 0.021],
                "B_rate_matched_to_A": [True, True, False],
                "B_E_plv_above_uniform_null": [False, True, True],
            }
        )
        summary = _calibration_summary(rows, self._config())
        depth, used_safe = _select_reference_depth(summary, self._config())

        self.assertTrue(used_safe)
        self.assertAlmostEqual(depth, 0.2)

    @staticmethod
    def _condition_rows(*, baseline_ppc, stimulation_ppc, washout_ppc):
        return [
            {
                "epoch": epoch,
                "E_ppc": ppc,
                "E_firing_rate_hz": 3.4,
                "I_firing_rate_hz": 7.2,
                "E_plv_above_uniform_null": bool(ppc > 0.005),
                "log10_total_power_1_80_excluding_stimulus": -19.1,
            }
            for epoch, ppc in (
                ("baseline", baseline_ppc),
                ("stimulation", stimulation_ppc),
                ("washout", washout_ppc),
            )
        ]

    def test_validation_row_measures_A_to_B_distance_and_orientation(self):
        rows = {
            "A_async": self._condition_rows(
                baseline_ppc=0.0,
                stimulation_ppc=0.0,
                washout_ppc=0.0,
            ),
            "B_rhythmic_reference": self._condition_rows(
                baseline_ppc=0.02,
                stimulation_ppc=0.02,
                washout_ppc=0.02,
            ),
            "A_tacs_axial": self._condition_rows(
                baseline_ppc=0.0,
                stimulation_ppc=0.018,
                washout_ppc=0.001,
            ),
            "A_tacs_transverse": self._condition_rows(
                baseline_ppc=0.0,
                stimulation_ppc=0.0,
                washout_ppc=0.0,
            ),
        }
        raw = {
            name: {
                "baseline": np.asarray([0.0, 1.0, -1.0]),
                "stimulation": np.asarray([0.0]),
                "washout": np.asarray([0.0]),
            }
            for name in rows
        }
        result = _validation_seed_row(
            seed=1,
            rows_by_condition=rows,
            raw_by_condition=raw,
            cfg=self._config(),
        )

        self.assertAlmostEqual(result["baseline_target_distance_E_ppc"], 0.02)
        self.assertAlmostEqual(result["tacs_target_distance_E_ppc"], 0.002)
        self.assertAlmostEqual(
            result["target_distance_improvement_E_ppc"],
            0.018,
        )
        self.assertAlmostEqual(
            result["orientation_advantage_E_ppc_distance"],
            0.018,
        )
        self.assertTrue(result["reference_direction_aligned"])
        self.assertTrue(result["washout_recovered"])
        self.assertEqual(
            result["baseline_relative_rms_error_A_tacs_vs_A"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
