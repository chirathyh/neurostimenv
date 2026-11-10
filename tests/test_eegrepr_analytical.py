"""Known-law checks, not model-mirroring fixture tests."""
import unittest
import numpy as np
from experiments.eeg_representation.analytical import (
    exponential_shot_noise, generate_trial, kalman_filter, oscillator_block_forecast,
    oscillator_forks, oscillator_risk_control, oscillator_transition, poisson_events,
    poisson_forks, stream, variance_mismatch_control,
)
from experiments.eeg_representation.statistics import (
    benjamini_hochberg, independent_pair_variance, leave_one_out_oracle,
    normalized_log_budget_auc, paired_cluster_interval, per_state_fork_variance, exact_sign_flip,
)


class PoissonTests(unittest.TestCase):
    def test_event_law_and_no_global_random_mutation(self):
        before = np.random.get_state()
        rng = stream(712, "counts")
        counts = np.array([len(poisson_events(40, 0, 1, rng)) for _ in range(4000)])
        self.assertLess(abs(counts.mean() - 40), 0.4)
        self.assertLess(abs(counts.var(ddof=1) - 40), 3.0)
        events = poisson_events(40, 0, 1000, stream(712, "isi"))
        self.assertTrue(np.all(np.diff(events) > 0))
        self.assertTrue(np.all((events > 0) & (events <= 1000)))
        self.assertLess(abs(np.diff(events).mean() - 1 / 40), 0.0005)
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        np.testing.assert_array_equal(poisson_events(40, 0, 1, stream(12, "a")), poisson_events(40, 0, 1, stream(12, "a")))
        self.assertFalse(np.array_equal(poisson_events(40, 0, 1, stream(12, "a")), poisson_events(40, 0, 1, stream(12, "b"))))

    def test_causal_kernel_and_right_closed_boundary(self):
        times = [0, 0.1, 0.2, 0.3]
        values = exponential_shot_noise(times, [0.1, 0.3], tau_s=0.1, amplitude=2)
        np.testing.assert_allclose(values, [0, 2, 2 * np.exp(-1), 2 * np.exp(-2) + 2])
        changed_future = exponential_shot_noise(times, [0.1, 0.29], tau_s=0.1, amplitude=2)
        np.testing.assert_array_equal(values[:3], changed_future[:3])

    def test_known_conditional_variance_and_mean(self):
        result = poisson_forks([0.002, 0.01, 0.05, 0.2], seed=77, n_states=5, n_futures=4096)
        empirical = result["futures"].var(axis=1, ddof=1).mean(axis=0)
        np.testing.assert_allclose(empirical, result["analytic_variance"], rtol=0.06)
        for state in range(5):
            standard_error = np.sqrt(result["analytic_variance"] / 4096)
            self.assertTrue(np.all(np.abs(result["futures"][state].mean(axis=0) - result["conditional_mean"][state]) < 5 * standard_error))
        self.assertLess(result["stationary_mean_relative_truncation"], 1e-8)
        self.assertTrue(all(np.all(history <= 0) for history in result["histories"]))
        self.assertTrue(all(np.all(events > 0) for family in result["future_events"] for events in family))


class OscillatorTests(unittest.TestCase):
    def test_exact_transition_semigroup_and_stationarity(self):
        F1, Q1 = oscillator_transition(0.013)
        F2, Q2 = oscillator_transition(0.027)
        F, Q = oscillator_transition(0.04)
        np.testing.assert_allclose(F2 @ F1, F, atol=1e-14)
        np.testing.assert_allclose(F2 @ Q1 @ F2.T + Q2, Q, atol=1e-14)
        np.testing.assert_allclose(F @ F.T + Q, np.eye(2), atol=1e-14)

    def test_joint_forecast_covariance_against_independent_futures(self):
        result = oscillator_forks([0.01, 0.025, 0.05], seed=9, n_states=2, n_futures=7000)
        residuals = result["futures"] - result["conditional_mean"][:, None]
        empirical = np.cov(residuals.reshape(-1, 3), rowvar=False)
        np.testing.assert_allclose(empirical, result["analytic_covariance"], atol=0.006)
        self.assertGreaterEqual(np.linalg.eigvalsh(result["analytic_covariance"]).min(), 0)

    def test_same_target_history_information_order_and_empirical_risks(self):
        result = oscillator_risk_control(seed=112, n_trials=1600, history_samples=24, lead_samples=3)
        risks = [result[name]["exact_risk"] for name in ("state_oracle", "clean_history", "noisy_history")]
        self.assertLess(risks[0], risks[1])
        self.assertLess(risks[1], risks[2])
        for name in ("state_oracle", "clean_history", "noisy_history"):
            self.assertLess(abs(result[name]["empirical_mse"] - result[name]["exact_risk"]), 5 * result[name]["mse_mc_standard_error"])

    def test_kalman_clean_observation_does_not_reveal_hidden_coordinate(self):
        F, Q = oscillator_transition(0.01)
        mean, covariance = kalman_filter([1.0], F, Q, noise_variance=0)
        np.testing.assert_allclose(mean[0], [1, 0])
        np.testing.assert_allclose(covariance[0], [[0, 0], [0, 1]])
        _, block = oscillator_block_forecast(mean[0], covariance[0], [0.01, 0.02])
        self.assertGreater(np.linalg.eigvalsh(block).min(), 0)


class VarianceControlTests(unittest.TestCase):
    def test_classification_and_probabilistic_positive_controls(self):
        result = variance_mismatch_control(seed=98, n_trials=4000)
        self.assertGreater(result["variance_classifier_balanced_accuracy"], 0.98)
        self.assertGreater(result["nll_gain"], 0.08)
        known_class_maximum_gain = 0.5 * np.log(2.5) - 0.25 * np.log(4)
        self.assertLess(result["nll_gain"], known_class_maximum_gain + 0.02)
        self.assertEqual(result["constant_mean_mse"], result["class_aware_mean_mse"])
        self.assertAlmostEqual(result["constant_mean_mse"], 2.5, delta=0.04)


class StatisticalTests(unittest.TestCase):
    def test_fork_estimators_and_loo_finite_ensemble_correction(self):
        rng = np.random.default_rng(731)
        values = rng.normal(size=(20, 2048, 3)) * np.sqrt([1, 2, 3]) + np.arange(20)[:, None, None]
        variance = per_state_fork_variance(values)
        self.assertAlmostEqual(variance.mean(), 2, delta=0.04)
        self.assertAlmostEqual(independent_pair_variance(values).mean(), 2, delta=0.05)
        loo = leave_one_out_oracle(values)
        n = values.shape[1]
        corrected = ((values - loo) ** 2).mean(axis=(1, 2)) * (n - 1) / n
        np.testing.assert_allclose(corrected, variance)
        altered = values.copy()
        altered[:, 0] += 1e4
        np.testing.assert_allclose(leave_one_out_oracle(altered)[:, 0], loo[:, 0], atol=1e-12)

    def test_log_axis_normalization(self):
        self.assertAlmostEqual(normalized_log_budget_auc([10, 100, 1000], [0.4, 0.6, 0.8]), 0.6)
        self.assertAlmostEqual(normalized_log_budget_auc([10, 50, 250], [1, 1, 1]), 1)
        with self.assertRaises(ValueError):
            normalized_log_budget_auc([10, 10], [0.5, 0.5])

    def test_equal_structure_weight_and_exact_small_sample_resolution(self):
        summary = paired_cluster_interval([1] * 10 + [-1], [0] * 11, ["a"] * 10 + ["b"], n_bootstrap=100)
        self.assertEqual(summary["estimate"], 0)
        self.assertEqual(summary["n_structures"], 2)
        self.assertEqual(exact_sign_flip([1, 1, 1], alternative="greater")["p_value"], 0.125)
        self.assertEqual(exact_sign_flip([1, 1, 1])["p_value"], 0.25)
        np.testing.assert_allclose(benjamini_hochberg([0.01, 0.04, 0.03, 0.9]), [0.04, 0.0533333333333333, 0.0533333333333333, 0.9])


class AdapterTests(unittest.TestCase):
    def test_fork_prefix_and_complete_state_identity(self):
        base = {"simulation": {"native_dt_ms": 0.5, "burn_in_s": 0.1, "duration_s": 0.2}, "forks": {"vary": "both"}}
        for kind in ("poisson", "oscillator", "variance"):
            config = {**base, "data": {"analytical_kind": kind}}
            spec = {"trial_seed": 14, "structure_seed": 123, "regime": {"name": "Ralpha", "frequency_hz": 10}}
            original = generate_trial(config, spec)
            first = generate_trial(config, {**spec, "future_start_ms": 150, "future_seed": 21})
            second = generate_trial(config, {**spec, "future_start_ms": 150, "future_seed": 22})
            np.testing.assert_array_equal(first["eeg_v"][:300], original["eeg_v"][:300])
            np.testing.assert_array_equal(first["eeg_v"][:300], second["eeg_v"][:300])
            self.assertFalse(np.array_equal(first["eeg_v"][300:], second["eeg_v"][300:]))
            self.assertEqual(first["metadata"]["fork"], second["metadata"]["fork"])
            self.assertEqual(first["metadata"]["structure_hash"], second["metadata"]["structure_hash"])
            np.testing.assert_array_equal(first["privileged"]["cutoff_physical_state"], second["privileged"]["cutoff_physical_state"])
            with self.assertRaisesRegex(ValueError, "phase-only"):
                generate_trial({**config, "forks": {"vary": "phase"}}, {**spec, "future_start_ms": 150, "future_seed": 21})

    def test_exact_samples_float64_physical_units_and_private_state(self):
        base = {"simulation": {"native_dt_ms": 0.5, "burn_in_s": 0.1, "duration_s": 0.2}}
        for kind in ("poisson", "oscillator", "variance"):
            config = {**base, "data": {"analytical_kind": kind}}
            result = generate_trial(config, {"trial_seed": 14, "regime": {"name": "Ralpha", "frequency_hz": 10}})
            self.assertEqual(result["eeg_v"].shape, (600, 1))
            self.assertEqual(result["eeg_v"].dtype, np.float64)
            self.assertTrue(np.isfinite(result["eeg_v"]).all())
            self.assertEqual(result["time_ms"][0], 0.5)
            self.assertEqual(result["time_ms"][-1], 300)
            self.assertEqual(result["metadata"]["applied_field_v_per_m"], 0)
            self.assertFalse(result["metadata"]["dipole_available"])
            np.testing.assert_array_equal(result["eeg_v"], generate_trial(config, {"trial_seed": 14, "regime": {"name": "Ralpha", "frequency_hz": 10}})["eeg_v"])


if __name__ == "__main__":
    unittest.main()
