"""Regression tests for causal online sinusoidal waveform boundaries."""

import unittest

import numpy as np

from env.models.neuron.stimulation import (
    apply_raised_cosine_block_envelope,
    make_sinusoidal_electric_field,
    make_sinusoidal_stimulation,
)
from env.models.neuron.extracellular_online import (
    OnlineExtracellularController,
)


DT_MS = 0.0625


def _waveform(action, start_ms, duration_ms, phase_rad):
    return make_sinusoidal_stimulation(
        amplitude_mA=action[0],
        frequency_hz=action[1],
        start_ms=start_ms,
        duration_ms=duration_ms,
        dt_ms=DT_MS,
        phase_rad=phase_rad,
        include_endpoint=True,
    )


class OnlineStimulationTests(unittest.TestCase):
    def test_uniform_field_supports_signed_dc_and_explicit_phase(self):
        dc = make_sinusoidal_electric_field(
            amplitude_v_per_m=0.0,
            frequency_hz=0.0,
            dc_offset_v_per_m=-0.5,
            start_ms=0.0,
            duration_ms=10.0,
            dt_ms=DT_MS,
        )
        np.testing.assert_array_equal(
            dc.field_v_per_m,
            np.full(dc.field_v_per_m.size, -0.5),
        )

        phased = make_sinusoidal_electric_field(
            amplitude_v_per_m=0.2,
            frequency_hz=10.0,
            phase_rad=np.pi / 2.0,
            dc_offset_v_per_m=-0.1,
            start_ms=0.0,
            duration_ms=100.0,
            dt_ms=DT_MS,
        )
        self.assertAlmostEqual(phased.field_v_per_m[0], 0.1)
        self.assertAlmostEqual(phased.final_phase_rad, np.pi / 2.0)

    def test_uniform_vector_field_has_correct_geometry_and_gauge(self):
        midpoints_um = np.asarray(
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        )
        axial = OnlineExtracellularController.uniform_field_potential_mV(
            midpoints_um=midpoints_um,
            field_v_per_m=np.asarray([1.0]),
            field_direction=[0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            axial[:, 0],
            np.asarray([1e-3, 0.0, -1e-3]),
        )
        transverse = OnlineExtracellularController.uniform_field_potential_mV(
            midpoints_um=midpoints_um,
            field_v_per_m=np.asarray([1.0]),
            field_direction=[1.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(transverse, np.zeros_like(transverse))

    def test_absolute_block_envelope_does_not_restart_between_windows(self):
        full_time = np.arange(0.0, 3000.0 + DT_MS, DT_MS)
        full = apply_raised_cosine_block_envelope(
            np.ones_like(full_time),
            time_ms=full_time,
            block_start_ms=0.0,
            block_stop_ms=3000.0,
            ramp_ms=250.0,
        )
        parts = []
        for start_ms in (0.0, 1000.0, 2000.0):
            window_time = np.arange(start_ms, start_ms + 1000.0 + DT_MS, DT_MS)
            window = apply_raised_cosine_block_envelope(
                np.ones_like(window_time),
                time_ms=window_time,
                block_start_ms=0.0,
                block_stop_ms=3000.0,
                ramp_ms=250.0,
            )
            parts.append(window[:-1] if start_ms < 2000.0 else window)

        np.testing.assert_array_equal(np.concatenate(parts), full)
        self.assertEqual(full[0], 0.0)
        self.assertEqual(full[-1], 0.0)
        self.assertEqual(full[int(1000.0 / DT_MS)], 1.0)

    def test_uniform_field_waveform_uses_v_per_m_without_current_conversion(self):
        waveform = make_sinusoidal_electric_field(
            amplitude_v_per_m=0.8,
            frequency_hz=10.0,
            start_ms=0.0,
            duration_ms=1000.0,
            dt_ms=DT_MS,
        )

        self.assertEqual(waveform.field_v_per_m.size, 16_001)
        self.assertAlmostEqual(np.max(waveform.field_v_per_m), 0.8)
        self.assertAlmostEqual(np.min(waveform.field_v_per_m), -0.8)

    def test_uniform_field_supports_smooth_amplitude_transition(self):
        waveform = make_sinusoidal_electric_field(
            amplitude_v_per_m=0.4,
            initial_amplitude_v_per_m=0.2,
            amplitude_transition_ms=100.0,
            frequency_hz=0.0,
            phase_rad=np.pi / 2.0,
            start_ms=0.0,
            duration_ms=200.0,
            dt_ms=DT_MS,
        )

        self.assertAlmostEqual(waveform.field_v_per_m[0], 0.2)
        midpoint = int(round(50.0 / DT_MS))
        transition_end = int(round(100.0 / DT_MS))
        self.assertAlmostEqual(waveform.field_v_per_m[midpoint], 0.3)
        self.assertAlmostEqual(waveform.field_v_per_m[transition_end], 0.4)
        self.assertAlmostEqual(waveform.field_v_per_m[-1], 0.4)

    def test_one_second_waveform_has_intervals_plus_endpoint(self):
        waveform = _waveform([0.002, 20.0], 2000.0, 1000.0, 0.0)

        self.assertEqual(waveform.time_ms.size, 16_001)
        self.assertEqual(waveform.current_nA.size, 16_001)
        self.assertEqual(waveform.time_ms[0], 2000.0)
        self.assertEqual(waveform.time_ms[-1], 3000.0)
        self.assertEqual(np.max(np.abs(waveform.current_nA)), 2000.0)

    def test_zero_actions_and_phase_continuous_boundaries(self):
        actions = [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.002, 20.0],
            [0.002, 20.0],
            [0.002, 40.0],
            [0.0, 0.0],
        ]
        duration_ms = 137.5
        phase_rad = 0.0
        previous = None
        previous_action = None

        for index, action in enumerate(actions):
            waveform = _waveform(
                action,
                index * duration_ms,
                duration_ms,
                phase_rad,
            )

            if action[0] == 0.0:
                self.assertEqual(np.count_nonzero(waveform.current_nA), 0)
            if (
                previous is not None
                and previous_action[0] != 0.0
                and action[0] != 0.0
            ):
                np.testing.assert_allclose(
                    previous[-1],
                    waveform.current_nA[0],
                    rtol=0.0,
                    atol=1e-10,
                )

            previous = waveform.current_nA
            previous_action = action
            phase_rad = waveform.final_phase_rad


if __name__ == "__main__":
    unittest.main()
