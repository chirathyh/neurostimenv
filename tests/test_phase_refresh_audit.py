"""Focused controller and repeated-future tests for D1-R."""

import unittest

import numpy as np
import pandas as pd

from experiments.ballnstick_analysis.run_ballnstick_phase_refresh_audit import (
    ONE_TIME,
    REFRESHED,
    _comparison_tables,
    _phase_slew_frequency,
    _signed_phase_error,
)
from experiments.ballnstick_analysis.plot_ballnstick_phase_refresh_example import (
    _controller_signals,
)


class PhaseRefreshAuditTests(unittest.TestCase):
    def test_signed_phase_error_uses_shortest_circular_direction(self):
        self.assertAlmostEqual(_signed_phase_error(0.1, 2.0 * np.pi - 0.1), 0.2)
        self.assertAlmostEqual(_signed_phase_error(2.0 * np.pi - 0.1, 0.1), -0.2)

    def test_frequency_slew_is_bounded_and_phase_continuous_by_construction(self):
        result = _phase_slew_frequency(
            carrier_hz=10.0,
            target_phase_rad=np.pi,
            oscillator_phase_rad=0.0,
            update_interval_ms=250.0,
            maximum_correction_hz=2.0,
        )
        self.assertAlmostEqual(abs(result["frequency_correction_hz"]), 2.0)
        self.assertAlmostEqual(result["command_frequency_hz"], 12.0)
        # The helper changes frequency, never the oscillator's starting phase.
        corrected_stop = 2.0 * np.pi * result["command_frequency_hz"] * 0.25
        carrier_target = np.pi + 2.0 * np.pi * 10.0 * 0.25
        self.assertAlmostEqual(
            _signed_phase_error(carrier_target, corrected_stop), 0.0, places=12
        )

    def test_expected_controller_is_estimated_from_all_futures(self):
        expected_rows = []
        metric_rows = []
        for structure in (1, 2, 3):
            for label, diffusion in (("low_diffusion", 0.5), ("high_diffusion", 2.0)):
                context = f"s{structure}_{label}"
                expected_distances = {ONE_TIME: 0.30, REFRESHED: 0.20}
                for mode, distance in expected_distances.items():
                    expected_rows.append({
                        "context_id": context,
                        "structure_seed": structure,
                        "hidden_frequency_hz": 9.0,
                        "label": label,
                        "diffusion_rad2_per_s": diffusion,
                        "context_C1": 0.8 if diffusion == 0.5 else 0.2,
                        "controller_mode": mode,
                        "expected_post_distance_to_B_log10": distance,
                        "future_sd_post_distance_log10": 0.01,
                        "mean_abs_phase_error_rad": (
                            0.8 if mode == ONE_TIME else 0.2
                        ),
                    })
                    for future, realized in enumerate(
                        ([0.31, 0.29, 0.32, 0.28] if mode == ONE_TIME
                         else [0.21, 0.19, 0.22, 0.18]),
                        start=1,
                    ):
                        metric_rows.append({
                            "context_id": context,
                            "future_index": future,
                            "controller_mode": mode,
                            "post_distance_to_B_log10": realized,
                        })
        comparison, structure, audit = _comparison_tables(
            pd.DataFrame(expected_rows), pd.DataFrame(metric_rows)
        )
        self.assertTrue((comparison.expected_active_winner == REFRESHED).all())
        self.assertTrue(np.allclose(
            comparison.refresh_advantage_over_one_time_log10, 0.1
        ))
        self.assertTrue(np.allclose(structure.mean_refresh_advantage_log10, 0.1))
        self.assertAlmostEqual(audit["mean_realized_winner_agreement_fraction"], 1.0)
        self.assertLess(
            audit["mean_refreshed_phase_error_rad"],
            audit["mean_one_time_phase_error_rad"],
        )

    def test_visualization_reconstructs_continuous_bounded_field(self):
        boundaries = np.arange(0.0, 1000.0, 250.0)
        command_frequencies = np.array([10.0, 11.0, 9.0, 10.0])
        oscillator_phases = [0.0]
        for frequency in command_frequencies[:-1]:
            oscillator_phases.append(
                (oscillator_phases[-1] + 2.0 * np.pi * frequency * 0.25)
                % (2.0 * np.pi)
            )
        rows = pd.DataFrame({
            "update_index": np.arange(4),
            "boundary_ms": boundaries,
            "carrier_frequency_hz": np.full(4, 10.0),
            "estimated_eeg_phase_at_boundary_rad": (
                2.0 * np.pi * 10.0 * boundaries / 1000.0
            ) % (2.0 * np.pi),
            "eeg_resultant_v": np.full(4, 1.0e-9),
            "oscillator_phase_before_update_rad": oscillator_phases,
            "command_frequency_hz": command_frequencies,
            "phase_error_before_correction_rad": np.array([0.0, 0.4, -0.3, 0.0]),
        })
        signals = _controller_signals(
            rows,
            amplitude_v_per_m=0.2,
            block_ramp_ms=0.0,
            samples_per_second=4000.0,
        )
        field = np.asarray(signals["field_v_per_m"])
        time_s = np.asarray(signals["time_s"])
        self.assertLessEqual(float(np.max(np.abs(field))), 0.2 + 1.0e-12)
        for boundary in (0.25, 0.5, 0.75):
            index = int(np.argmin(np.abs(time_s - boundary)))
            self.assertLess(abs(field[index] - field[index - 1]), 0.01)
        self.assertTrue(np.all(np.isfinite(signals["eeg_carrier_nV"])))


if __name__ == "__main__":
    unittest.main()
