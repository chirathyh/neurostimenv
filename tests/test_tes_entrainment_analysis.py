"""Unit tests for the T1 reversible-entrainment analysis primitives."""

import unittest

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from experiments.ballnstick_analysis.run_ballnstick_tes_entrainment import (
    _build_discovery_actions,
    _difference_in_differences,
    _phase_locking_metrics,
    _validation_actions,
)


class TesEntrainmentAnalysisTests(unittest.TestCase):
    @staticmethod
    def _config():
        return OmegaConf.create(
            {
                "analysis": {
                    "maximum_field_v_per_m": 0.8,
                    "protocol": {
                        "phase_rad": 0.0,
                        "axial_montage": "axial",
                        "transverse_montage": "transverse_x",
                    },
                    "discovery": {
                        "amplitudes_v_per_m": [0.2, 0.8],
                        "frequencies_hz": [10.0, 20.0, 40.0],
                    },
                    "validation": {
                        "include_frequency_neighbors": True,
                        "include_dose_controls": True,
                    },
                }
            }
        )

    def test_cycle_aligned_spikes_have_unit_plv_and_ppc(self):
        metrics = _phase_locking_metrics(
            np.arange(0.0, 1000.0, 100.0),
            frequency_hz=10.0,
            phase_origin_ms=0.0,
            n_surrogates=200,
            rng=np.random.default_rng(1),
        )

        self.assertAlmostEqual(metrics["plv"], 1.0)
        self.assertAlmostEqual(metrics["ppc"], 1.0)
        self.assertEqual(metrics["plv_above_uniform_null"], 1.0)

    def test_evenly_spaced_phases_do_not_look_entrained(self):
        metrics = _phase_locking_metrics(
            np.arange(0.0, 100.0, 10.0),
            frequency_hz=10.0,
            phase_origin_ms=0.0,
            n_surrogates=200,
            rng=np.random.default_rng(2),
        )

        self.assertAlmostEqual(metrics["plv"], 0.0, places=12)
        self.assertLess(metrics["ppc"], 0.0)
        self.assertEqual(metrics["plv_above_uniform_null"], 0.0)

    def test_difference_in_differences_removes_shared_time_drift(self):
        value = _difference_in_differences(
            active_baseline=0.10,
            active_epoch=0.30,
            sham_baseline=0.10,
            sham_epoch=0.15,
        )

        self.assertAlmostEqual(value, 0.15)

    def test_discovery_grid_is_realistic_and_unique(self):
        actions = _build_discovery_actions(self._config())

        self.assertEqual(len(actions), 6)
        self.assertEqual(len({action["id"] for action in actions}), 6)
        self.assertTrue(
            all(action["ac_amplitude_v_per_m"] <= 0.8 for action in actions)
        )
        self.assertTrue(all(action["montage"] == "axial" for action in actions))

    def test_validation_freezes_action_and_adds_controls(self):
        cfg = self._config()
        discovery = _build_discovery_actions(cfg)
        selected = next(
            action
            for action in discovery
            if np.isclose(action["ac_amplitude_v_per_m"], 0.8)
            and np.isclose(action["frequency_hz"], 20.0)
        )

        actions = _validation_actions(selected, discovery, cfg)
        table = pd.DataFrame(actions).set_index("id")

        self.assertEqual(table.loc["selected_axial", "montage"], "axial")
        self.assertEqual(
            table.loc["selected_transverse", "montage"], "transverse_x"
        )
        self.assertEqual(
            table.loc["lower_frequency_control", "frequency_hz"], 10.0
        )
        self.assertEqual(
            table.loc["upper_frequency_control", "frequency_hz"], 40.0
        )
        self.assertEqual(
            table.loc["dose_control_a0p2", "ac_amplitude_v_per_m"], 0.2
        )
        non_dose = table[table["role"] != "dose_control"]
        self.assertTrue(np.allclose(non_dose["ac_amplitude_v_per_m"], 0.8))


if __name__ == "__main__":
    unittest.main()
