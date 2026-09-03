"""Tests for reproducible BallAndStick stochastic afferent drive."""

import unittest

import numpy as np

from setup.circuits.ballnstick.utils import (
    generate_phase_diffusion_path,
    generate_poisson_spike_train,
    generate_sinusoidally_modulated_poisson_spike_train,
    generate_split_background_spike_train,
    shared_rhythm_synapse_count,
)


class BallAndStickBackgroundTests(unittest.TestCase):
    def test_shared_rhythm_fraction_is_exact_nested_and_validated(self):
        self.assertEqual(
            shared_rhythm_synapse_count(
                n_synapses=64, shared_modulated_fraction=0.5
            ),
            32,
        )
        self.assertEqual(
            shared_rhythm_synapse_count(
                n_synapses=64, shared_modulated_fraction=1.0
            ),
            64,
        )
        with self.assertRaises(ValueError):
            shared_rhythm_synapse_count(
                n_synapses=64, shared_modulated_fraction=1.01
            )
        with self.assertRaises(ValueError):
            shared_rhythm_synapse_count(
                n_synapses=-1, shared_modulated_fraction=0.5
            )

    def test_zero_diffusion_path_is_exact_deterministic_phase(self):
        times, phase = generate_phase_diffusion_path(
            start_ms=0.0,
            stop_ms=2_000.0,
            frequency_hz=10.0,
            phase_rad=0.3,
            diffusion_rad2_per_s=0.0,
            integration_dt_ms=1.0,
            history_seed=9,
        )
        expected = 0.3 + 2.0 * np.pi * 10.0 * times / 1000.0
        np.testing.assert_allclose(phase, expected, rtol=0.0, atol=1.0e-11)

    def test_phase_diffusion_has_expected_increment_variance(self):
        times, phase = generate_phase_diffusion_path(
            start_ms=0.0,
            stop_ms=100_000.0,
            frequency_hz=10.0,
            phase_rad=0.0,
            diffusion_rad2_per_s=1.5,
            integration_dt_ms=1.0,
            history_seed=41,
        )
        residual_increments = np.diff(phase) - 2.0 * np.pi * 10.0 * (
            np.diff(times) / 1000.0
        )
        self.assertAlmostEqual(
            float(np.var(residual_increments)),
            2.0 * 1.5 * 0.001,
            delta=1.0e-4,
        )

    def test_phase_diffusion_split_preserves_history_and_continuity(self):
        common = {
            "start_ms": 0.0,
            "stop_ms": 10_000.0,
            "frequency_hz": 9.0,
            "phase_rad": 0.2,
            "diffusion_rad2_per_s": 1.0,
            "integration_dt_ms": 1.0,
            "history_seed": 51,
            "future_start_ms": 4_000.0,
        }
        times, first = generate_phase_diffusion_path(**common, future_seed=61)
        _, second = generate_phase_diffusion_path(**common, future_seed=62)
        split = int(np.flatnonzero(np.isclose(times, 4_000.0))[0])
        np.testing.assert_array_equal(first[: split + 1], second[: split + 1])
        self.assertFalse(np.array_equal(first[split + 1 :], second[split + 1 :]))
        expected_deterministic = 2.0 * np.pi * 9.0 * 0.001
        # The boundary value is inherited; there is no phase reset to phase_rad.
        self.assertGreater(abs(first[split] - (0.2 + 2.0 * np.pi * 9.0 * 4.0)), 1.0e-3)
        self.assertLess(abs((first[split + 1] - first[split]) - expected_deterministic), 0.3)

    def test_private_event_streams_can_share_one_diffusing_phase(self):
        path_times, path_phase = generate_phase_diffusion_path(
            start_ms=0.0,
            stop_ms=20_000.0,
            frequency_hz=10.0,
            phase_rad=0.0,
            diffusion_rad2_per_s=1.0,
            integration_dt_ms=1.0,
            history_seed=71,
        )
        kwargs = {
            "start_ms": 0.0,
            "stop_ms": 20_000.0,
            "interval_ms": 40.0,
            "modulation_depth": 0.2,
            "frequency_hz": 10.0,
            "phase_rad": 0.0,
            "thinning_envelope_modulation_depth": 0.2,
            "phase_path_times_ms": path_times,
            "phase_path_rad": path_phase,
        }
        first = generate_sinusoidally_modulated_poisson_spike_train(seed=81, **kwargs)
        repeat = generate_sinusoidally_modulated_poisson_spike_train(seed=81, **kwargs)
        independent = generate_sinusoidally_modulated_poisson_spike_train(seed=82, **kwargs)
        np.testing.assert_array_equal(first, repeat)
        self.assertFalse(np.array_equal(first, independent))
        self.assertAlmostEqual(first.size, 500.0, delta=80.0)

    def test_zero_depth_preserves_legacy_homogeneous_generator(self):
        expected = generate_poisson_spike_train(
            start_ms=0.0,
            stop_ms=10_000.0,
            interval_ms=40.0,
            seed=17,
        )
        observed = generate_sinusoidally_modulated_poisson_spike_train(
            start_ms=0.0,
            stop_ms=10_000.0,
            interval_ms=40.0,
            seed=17,
            modulation_depth=0.0,
            frequency_hz=60.0,
            thinning_envelope_modulation_depth=0.0,
        )

        np.testing.assert_array_equal(observed, expected)

    def test_modulated_process_has_expected_phase_and_mean_rate(self):
        duration_ms = 100_000.0
        interval_ms = 40.0
        modulation_depth = 0.8
        times = generate_sinusoidally_modulated_poisson_spike_train(
            start_ms=0.0,
            stop_ms=duration_ms,
            interval_ms=interval_ms,
            seed=23,
            modulation_depth=modulation_depth,
            frequency_hz=10.0,
            phase_rad=0.0,
            thinning_envelope_modulation_depth=1.0,
        )

        phases = 2.0 * np.pi * 10.0 * times / 1000.0
        resultant = np.mean(np.exp(1j * phases))
        expected_count = duration_ms / interval_ms

        # For p(phi)=(1+m*sin(phi))/(2*pi), E[exp(i*phi)]=i*m/2.
        self.assertAlmostEqual(resultant.real, 0.0, delta=0.04)
        self.assertAlmostEqual(
            resultant.imag,
            modulation_depth / 2.0,
            delta=0.04,
        )
        self.assertAlmostEqual(times.size, expected_count, delta=150)

    def test_common_envelope_is_reproducible_but_depth_changes_events(self):
        kwargs = {
            "start_ms": 0.0,
            "stop_ms": 10_000.0,
            "interval_ms": 40.0,
            "seed": 31,
            "frequency_hz": 20.0,
            "thinning_envelope_modulation_depth": 0.5,
        }
        asynchronous = generate_sinusoidally_modulated_poisson_spike_train(
            **kwargs,
            modulation_depth=0.0,
        )
        asynchronous_repeat = (
            generate_sinusoidally_modulated_poisson_spike_train(
                **kwargs,
                modulation_depth=0.0,
            )
        )
        rhythmic = generate_sinusoidally_modulated_poisson_spike_train(
            **kwargs,
            modulation_depth=0.5,
        )

        np.testing.assert_array_equal(asynchronous, asynchronous_repeat)
        self.assertFalse(np.array_equal(asynchronous, rhythmic))

    def test_invalid_modulation_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_sinusoidally_modulated_poisson_spike_train(
                start_ms=0.0,
                stop_ms=1000.0,
                interval_ms=40.0,
                seed=1,
                modulation_depth=1.1,
                frequency_hz=10.0,
            )
        with self.assertRaises(ValueError):
            generate_sinusoidally_modulated_poisson_spike_train(
                start_ms=0.0,
                stop_ms=1000.0,
                interval_ms=40.0,
                seed=1,
                modulation_depth=0.5,
                frequency_hz=10.0,
                thinning_envelope_modulation_depth=0.4,
            )

    def test_split_stream_preserves_history_and_varies_only_future(self):
        kwargs = {
            "start_ms": 0.0,
            "stop_ms": 10_000.0,
            "interval_ms": 40.0,
            "history_seed": 101,
            "future_start_ms": 4_000.0,
            "rhythm_enabled": True,
            "modulation_depth": 0.04,
            "frequency_hz": 10.0,
            "phase_rad": 0.3,
            "thinning_envelope_modulation_depth": 0.16,
        }
        first = generate_split_background_spike_train(
            **kwargs, future_seed=201
        )
        second = generate_split_background_spike_train(
            **kwargs, future_seed=202
        )
        np.testing.assert_array_equal(first[first < 4_000.0], second[second < 4_000.0])
        self.assertFalse(np.array_equal(
            first[first >= 4_000.0], second[second >= 4_000.0]
        ))

    def test_unsplit_dispatch_preserves_historical_generator_exactly(self):
        expected = generate_poisson_spike_train(
            start_ms=0.0, stop_ms=1_000.0, interval_ms=40.0, seed=19
        )
        observed = generate_split_background_spike_train(
            start_ms=0.0,
            stop_ms=1_000.0,
            interval_ms=40.0,
            history_seed=19,
        )
        np.testing.assert_array_equal(observed, expected)


if __name__ == "__main__":
    unittest.main()
