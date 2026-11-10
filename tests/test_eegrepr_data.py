"""Meaningful leakage, atomicity and physical-scale tests without NEURON."""
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from experiments.eeg_representation.schema import load_config, ExperimentConfig
from experiments.eeg_representation.data import (generate, derive, load_examples, assert_splits,
    assert_input_columns, trial_specs, _budget_rows)
from experiments.eeg_representation.storage import atomic_json, read_json, file_hash

ROOT = Path(__file__).resolve().parents[1]


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs/eeg_representation/smoke.yaml")

    def test_profiles_and_counts(self):
        for name, count in [("smoke", 24), ("simulator_smoke", 24), ("pilot", 640), ("confirmation", 4000)]:
            config = load_config(ROOT / f"configs/eeg_representation/{name}.yaml")
            self.assertEqual(len(trial_specs(config)), count)
            assert_splits(trial_specs(config))

    def test_rejects_unknown_overlap_seed_units_and_grid(self):
        mutations = [lambda c: c.update(typo=1),
            lambda c: c["data"]["structures"].update(test=[31000]),
            lambda c: c["data"]["structures"].update(test=[500000]),
            lambda c: c["simulation"].update(celsius=36.5),
            lambda c: c["forecast"].update(lead_ms=25),
            lambda c: c["training"].update(heldout_tasks=["rhythm"])]
        for mutate in mutations:
            c = copy.deepcopy(self.config)
            mutate(c)
            with self.assertRaises(ValueError):
                ExperimentConfig.from_dict(c)

    def test_parent_pair_noise_group_overlap_and_inputs(self):
        rows = trial_specs(self.config)
        rows[0]["split"] = "test" if rows[0]["split"] != "test" else "train"
        with self.assertRaises(ValueError):
            assert_splits(rows)
        for column in ["seed", "future_events", "spikes", "condition", "hidden_phase"]:
            with self.assertRaises(ValueError):
                assert_input_columns(["eeg", column])

    def test_nested_independent_trial_subsets(self):
        rows = trial_specs(self.config)
        small = {r["trial_id"] for r in _budget_rows(rows, 1, 7)}
        large = {r["trial_id"] for r in _budget_rows(rows, 2, 7)}
        self.assertTrue(small <= large)

    def test_optional_import_boundary(self):
        code = "import sys; import experiments.eeg_representation.cli; import experiments.eeg_representation.data; assert not any(k in sys.modules for k in ['neuron','LFPy','simnibs','torch'])"
        subprocess.run([sys.executable, "-c", code], cwd=ROOT, check=True)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name) / "dataset"
        cls.config = load_config(ROOT / "configs/eeg_representation/smoke.yaml")
        # A tiny physical amplitude must survive normalization without a 1e-8 floor.
        cls.config["analytical"] = {"amplitude_v": 1e-15}
        generate(cls.config, cls.root)
        derive(cls.config, cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_real_shards_schema_units_and_loader_permissions(self):
        import h5py
        rows = read_json(self.root / "manifest.json")["trials"]
        with h5py.File(self.root / rows[0]["path"], "r") as h:
            self.assertEqual(h["eeg_v"].dtype, np.float64)
            self.assertEqual(h["time_ms"].dtype, np.float64)
            self.assertIn("privileged", h)
            self.assertEqual(json.loads(h.attrs["metadata_json"])["units"]["eeg"], "V")
        data = load_examples(self.root, self.config, "train", "leave-task-label-out")
        self.assertEqual(data["x"].shape[1:], (1, 125))
        self.assertEqual(data["future"].shape[1:], (1, 64))
        self.assertEqual(data["x"].dtype, np.float32)
        self.assertTrue(np.isfinite(data["x"]).all())
        self.assertGreater(data["x"].std(), .1)
        self.assertTrue(np.all(data["labels"]["baseline_alpha"] == -1))
        self.assertTrue(np.isnan(data["labels"]["coherence_value"]).all())
        self.assertEqual(data["allowed_input_columns"], ["eeg"])

    def test_generator_family_excluded_from_pretraining_only(self):
        pre = load_examples(self.root, self.config, "train", "leave-generator-family-out")
        probe = load_examples(self.root, self.config, "train", "leave-generator-family-out", stage="probe")
        self.assertNotIn("Rbeta_high", {r["family"] for r in pre["metadata"]})
        self.assertIn("Rbeta_high", {r["family"] for r in probe["metadata"]})
        self.assertLess(len(pre["x"]), len(probe["x"]))

    def test_training_only_power_and_noise_parent_pairing(self):
        import h5py
        path = self.root / "derived/leave-task-label-out"
        stats = read_json(path / "normalization.json")
        self.assertLess(stats["scale_v"][0], 1e-13)
        self.assertEqual(stats["fit_split"], "train")
        rows = read_json(path / "manifest.json")["trials"]
        arrays = []
        for row in rows:
            if row["split"] == "train":
                with h5py.File(self.root / row["path"], "r") as h:
                    arrays.append(h["clean"][:])
        combined = np.concatenate(arrays)
        np.testing.assert_allclose(stats["scale_v"], combined.std(axis=0), rtol=1e-13, atol=0)
        with h5py.File(self.root / rows[0]["path"], "r") as h:
            # The same unit path produces explicitly correlated observation views.
            np.testing.assert_allclose(h["snr_0"][:] - h["clean"][:], 10 * (h["snr_20"][:] - h["clean"][:]), rtol=1e-12, atol=1e-30)

    def test_resume_refuses_incompatibility_and_preserves_hashes(self):
        rows = read_json(self.root / "manifest.json")["trials"]
        before = [file_hash(self.root / r["path"]) for r in rows]
        generate(self.config, self.root, resume=True)
        derive(self.config, self.root, resume=True)
        self.assertEqual(before, [file_hash(self.root / r["path"]) for r in rows])
        changed = copy.deepcopy(self.config)
        changed["simulation"]["duration_s"] += 1
        with self.assertRaises(ValueError):
            generate(changed, self.root, resume=True)
        with self.assertRaises(FileExistsError):
            generate(self.config, self.root)

    def test_atomic_write_and_unversioned_directory_refusal(self):
        path = Path(self.tmp.name) / "atomic.json"
        atomic_json(path, {"finished": True})
        self.assertEqual(read_json(path), {"finished": True})
        with self.assertRaises(FileExistsError):
            generate(self.config, Path(self.tmp.name))


if __name__ == "__main__":
    unittest.main()
