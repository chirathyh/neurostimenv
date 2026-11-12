"""Exact bits, operational code lengths and circuit-disjoint readout checks."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.eeg_representation.h1 import compression as c


def panel_fixture():
    rng = np.random.default_rng(41)
    panel = {"manifest": {"job_id": "fixture", "target_fingerprint": "same_target", "physical_target_fingerprint": "physical",
              "comparison_fingerprint": "comparison", "operator": "bin_average_waveform", "lead_ms": 10., "duration_ms": 32., "output_grid_hz": 500.,
              "normalization": {"x_offset": 0., "x_scale": 2e-6, "y_offset": [0.] * 4, "y_scale": 1e-6, "fit_split": "train", "id": "training"},
              "input": {"sample_rate_hz": 500., "history_s": .016, "acquisition_bits": None}}}
    for split, groups in (("train", [1, 2, 3]), ("development", [4]), ("evaluation", [5, 6])):
        structure = np.repeat(groups, 12)
        n = len(structure)
        x = rng.normal(size=(n, 8))
        labels = np.tile(np.repeat([0, 1], 6), len(groups))
        x[:, 0] = np.where(labels, 2., -2.) + rng.normal(scale=.05, size=n)
        m = 3 if split == "evaluation" else 1
        y = x[:, :4, None].transpose(0, 2, 1) + .05 * rng.normal(size=(n, m, 4))
        ids = np.array([f"{split}-{i}" for i in range(n)])
        panel[split] = dict(X=x, X_physical=x * 2e-6, Y=y, Y_physical=y * 1e-6,
                            future_mask=np.ones((n, m), bool), context_id=ids, structure_id=structure,
                            future_id=np.asarray([[f"{cid}-f{k}" for k in range(m)] for cid in ids]),
                            regime=np.where(labels, "rhythm", "homogeneous"), task_rhythm=labels,
                            task_carrier=np.where(labels, np.arange(n) % 2, -1),
                            task_phase_sin=np.sin(x[:, 1]), task_phase_cos=np.cos(x[:, 1]),
                            task_log_power=np.log(np.mean(x ** 2, axis=1)),
                            input_time_ms=np.arange(-7, 1) * 2., target_time_ms=np.arange(1, 5) * 2. + 10.)
    return panel


class ExactControlTests(unittest.TestCase):
    def test_exact_counterexample_and_equal_rate(self):
        result = c.ten_bit_counterexample()
        self.assertEqual(result["enumerated_inputs"], 1024)
        r = result["representations"]
        self.assertEqual(r["nuisance_9_bits"]["input_mi_bits"], 9)
        self.assertEqual(r["nuisance_9_bits"]["task_mi_bits"], 0)
        self.assertEqual(r["task_3_bits"]["task_mi_bits"], 1)
        self.assertEqual(r["task_3_bits"]["code_bits"], r["nuisance_equal_rate_3_bits"]["code_bits"])
        self.assertEqual(r["nuisance_equal_rate_3_bits"]["task_mi_bits"], 0)

    def test_variance_class_survives_equal_mean_mse(self):
        control = c.variance_only_counterexample()
        self.assertEqual(control["constant_representation_optimal_mse"], 5)
        self.assertEqual(control["class_representation_optimal_mse"], 5)
        self.assertEqual(control["constant_task_ce_bits"], 1)
        self.assertEqual(control["class_task_ce_bits"], 0)
        self.assertEqual(control["constant_future_discrete_ce_bits"] - control["class_future_discrete_ce_bits"], 1)

    def test_gaussian_mi_rd_and_fano_bits(self):
        result = c.gaussian_channel(4, 1)
        self.assertAlmostEqual(result["mutual_information_bits"], .5 * np.log2(5))
        self.assertAlmostEqual(result["posterior_mse"], .8)
        self.assertAlmostEqual(result["posterior_mse"], result["gaussian_rd_bound"])
        self.assertAlmostEqual(c.fano_necessary_bits(4, 0), 2)
        self.assertAlmostEqual(c.fano_necessary_bits(4, .75), 0)
        self.assertAlmostEqual(c.fano_necessary_bits(2, .1), 1 + .1 * np.log2(.1) + .9 * np.log2(.9))
        with self.assertRaises(ValueError):
            c.fano_necessary_bits(2, .6)

    def test_gaussian_conditional_mean_recovers_analytic_distortion(self):
        rng = np.random.default_rng(151)
        x = rng.normal(size=100000)
        observed = x + rng.normal(scale=.5, size=len(x))
        mse = np.mean((x - observed / 1.25) ** 2)
        self.assertLess(abs(mse - .2), 5 * .2 * np.sqrt(2 / len(x)))


class QuantizerTests(unittest.TestCase):
    def test_requested_budgets_exact_packing_and_roundtrip(self):
        rng = np.random.default_rng(81)
        for dimension in (8, 32, 128):
            for budget in (32, 128, 512):
                with self.subTest(dimension=dimension, budget=budget):
                    if budget < dimension or budget > 16 * dimension:
                        with self.assertRaises(ValueError):
                            c.allocate_bits(dimension, budget)
                        continue
                    x = rng.normal(size=(256, dimension))
                    book = c.fit_quantizer(x, budget)
                    codes, decoded, report = c.quantize(x, book)
                    packed = c.pack_codes(codes, book["bits"])
                    self.assertEqual(packed.shape, (len(x), budget // 8))
                    np.testing.assert_array_equal(c.unpack_codes(packed, book["bits"]), codes)
                    self.assertTrue(np.isfinite(decoded).all())
                    self.assertGreater(report["coordinate_clipping_fraction"], 0)

    def test_test_extreme_cannot_change_fitted_range_or_projection(self):
        train = np.arange(200, dtype=float).reshape(25, 8)
        projection = c.fit_projection(train, 8)
        book = c.fit_quantizer(c.apply_projection(train, projection), 32)
        original = copy.deepcopy(book)
        test = np.full((4, 8), 1e10)
        _, decoded, report = c.quantize(c.apply_projection(test, projection), book)
        for name in ("lower", "upper", "bits"):
            np.testing.assert_array_equal(book[name], original[name])
        self.assertGreater(report["coordinate_clipping_fraction"], 0)
        self.assertTrue(np.isfinite(decoded).all())
        with self.assertRaises(ValueError):
            c.fit_projection(train, 32)

    def test_constant_coordinates_and_invalid_codes(self):
        train = np.ones((10, 8))
        book = c.fit_quantizer(train, 32)
        codes, decoded, _ = c.quantize(train, book)
        np.testing.assert_array_equal(codes, 0)
        np.testing.assert_array_equal(decoded, train)
        with self.assertRaises(ValueError):
            c.pack_codes(np.full((2, 8), 16), [4] * 8)


class ReadoutTests(unittest.TestCase):
    def test_split_rejects_circuit_and_context_leakage(self):
        panel = panel_fixture()
        self.assertEqual(c.validate_splits(panel)["train"], 3)
        panel["evaluation"]["structure_id"][0] = 1
        with self.assertRaisesRegex(ValueError, "split leakage"):
            c.validate_splits(panel)

    def test_balanced_budget_covers_class_and_circuit(self):
        labels = np.repeat(np.arange(3), 40)
        structures = np.tile(np.repeat(np.arange(4), 10), 3)
        chosen = c._balanced_subset(labels, structures, 8, "classification", 0)
        self.assertEqual(set(labels[chosen]), {0, 1, 2})
        self.assertGreaterEqual(len(set(structures[chosen])), 2)
        self.assertEqual(len(chosen), 8)

    def test_nested_circuit_readout_numeric_rescoring(self):
        panel = panel_fixture()
        config = dict(c.DEFAULTS, regularization=[.01, 1.], flexible_features=8)
        matrices = {s: panel[s]["X"] for s in ("train", "development", "evaluation")}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            result = c._evaluate_readout(panel, matrices, "rhythm", "classification", "task_rhythm", "linear", "all", config, path)
            self.assertGreater(result["accuracy"], .95)
            self.assertEqual(result["nested_supervised_crossfit"]["status"], "complete")
            with np.load(path / "predictions.npz", allow_pickle=False) as a:
                ce = -np.log2(a["probabilities"][np.arange(len(a["observed_target"])), a["observed_target"].astype(int)])
                self.assertAlmostEqual(float(ce.mean()), result["task_ce_bits"])
                self.assertAlmostEqual(result["task_mi_lower_bound_estimate_bits"], result["empirical_task_entropy_bits"] - ce.mean())
                self.assertFalse(set(a["train_context_id"]) & set(a["context_id"]))

    def test_insufficient_nested_circuits_not_silently_claimed(self):
        panel = panel_fixture()["train"]
        keep = panel["structure_id"] != 3
        result, arrays = c.nested_crossfit(panel["X"][keep], panel["task_rhythm"][keep], panel["structure_id"][keep], "classification", "linear", c.DEFAULTS, 0)
        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(arrays)

    def test_small_pipeline_numeric_outputs_and_checksum_resume(self):
        panel = panel_fixture()
        banks = {"pca": {s: panel[s]["X"] for s in ("train", "development", "evaluation")}}
        config = {"compression": {"dimensions": [2], "code_bits": [32], "representations": ["pca"],
                                   "readouts": ["linear"], "label_budgets": ["all"], "regularization": [1.]}}
        with tempfile.TemporaryDirectory() as temp, patch("experiments.eeg_representation.h1.dataset.list_jobs", return_value=["fixture"]), \
                patch("experiments.eeg_representation.h1.dataset.load_panel", return_value=panel), \
                patch("experiments.eeg_representation.h1.benchmark.get_representations", return_value=banks, create=True):
            result = c.run_compression(config, temp)
            self.assertEqual(result["status"], "complete")
            self.assertTrue(any(r.get("task") == "future_mean" and r["status"] == "complete" for r in result["rows"]))
            second = c.run_compression(config, temp, resume=True)
            self.assertEqual(second["identity"], result["identity"])
            codes = next((Path(temp) / "compression").rglob("codes.npz"))
            codes.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "checksum"):
                c.run_compression(config, temp, resume=True)


if __name__ == "__main__":
    unittest.main()
