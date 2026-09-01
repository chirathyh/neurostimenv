"""Focused design and selection tests for H4-BW."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from hydra import compose, initialize_config_dir

from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (
    ONE_TIME,
)
from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_bandwidth_discovery import (
    CURRENT,
    FAST,
    SHORT,
    _common_initialization,
    _comparison_tables,
    _fixed_horizon_phase_slew,
)


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="config",
            overrides=[
                "env=ballnstick",
                "analysis=ballnstick_phase_refresh_bandwidth_discovery",
            ],
        )


class PhaseRefreshBandwidthDiscoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = _config()

    def test_fixed_horizon_does_not_change_gain_at_faster_observation_rate(self):
        # The selected profile's update interval is deliberately absent from
        # this helper: both 125- and 250-ms observers use the frozen 250-ms
        # correction horizon and therefore the same proportional command.
        result = _fixed_horizon_phase_slew(
            self.cfg,
            carrier_hz=9.0,
            target_phase_rad=np.pi / 2.0,
            oscillator_phase_rad=0.0,
        )
        self.assertAlmostEqual(result["frequency_correction_hz"], 1.0)
        self.assertAlmostEqual(result["command_frequency_hz"], 10.0)

    def test_selection_uses_expected_futures_and_prefers_slower_tied_candidate(self):
        expected_rows = []
        metric_rows = []
        values = {
            ONE_TIME: [0.31, 0.29, 0.32, 0.28],
            CURRENT: [0.23, 0.21, 0.24, 0.20],
            SHORT: [0.21, 0.19, 0.22, 0.18],
            FAST: [0.215, 0.195, 0.225, 0.185],
        }
        errors = {ONE_TIME: 1.5, CURRENT: 0.7, SHORT: 0.5, FAST: 0.45}
        for structure in (1, 2, 3):
            for label, diffusion in (("low_diffusion", 0.5), ("high_diffusion", 2.0)):
                context = f"s{structure}_{label}"
                for mode, futures in values.items():
                    expected_rows.append({
                        "context_id": context,
                        "structure_seed": structure,
                        "hidden_frequency_hz": 9.0,
                        "label": label,
                        "diffusion_rad2_per_s": diffusion,
                        "context_C1": 0.8 if diffusion == 0.5 else 0.2,
                        "controller_mode": mode,
                        "expected_post_distance_to_B_log10": float(np.mean(futures)),
                        "future_sd_post_distance_log10": float(np.std(futures, ddof=1)),
                        "mean_abs_phase_error_rad": errors[mode],
                        "phase_estimate_actionable_fraction": 1.0,
                        "correction_saturation_fraction": 0.0,
                    })
                    for future_index, distance in enumerate(futures, start=1):
                        metric_rows.append({
                            "context_id": context,
                            "future_index": future_index,
                            "controller_mode": mode,
                            "post_distance_to_B_log10": distance,
                        })
        _, _, summary, selection = _comparison_tables(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows), self.cfg
        )
        self.assertTrue(summary[summary.controller_mode.eq(SHORT)].iloc[0].passes_bandwidth_gate)
        self.assertTrue(summary[summary.controller_mode.eq(FAST)].iloc[0].passes_bandwidth_gate)
        # FAST is only 0.005 log10 worse/better within the frozen 0.01 tie
        # margin; the predeclared parsimony rule selects 250-ms updates.
        self.assertEqual(selection["selected_controller"], SHORT)

    def test_common_initialization_rejects_a_controller_specific_phase(self):
        rows = []
        modes = [ONE_TIME, CURRENT, SHORT, FAST]
        for mode in modes:
            rows.append({
                "context_id": "c1",
                "future_index": 1,
                "controller_mode": mode,
                "update_index": 0,
                "desired_field_phase_rad": 0.7,
                "phase_history_ms": 1000.0,
            })
        self.assertTrue(_common_initialization(pd.DataFrame(rows)))
        rows[-1]["desired_field_phase_rad"] = 0.8
        self.assertFalse(_common_initialization(pd.DataFrame(rows)))


if __name__ == "__main__":
    unittest.main()
