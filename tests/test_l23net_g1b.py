"""G1B endpoints, negative results, matching checks, and PBS submission contract."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
from scipy import signal

from experiments.l23net_analysis.g1b_analysis import (
    analyze_suite, pairing_errors, parse_pbs_snapshot, resource_and_rate_checks, spectral_views,
)
from experiments.l23net_analysis.replay_validation import canonical_json_sha256


class G1BTests(unittest.TestCase):
    def test_existing_spreadsheet_intervention_and_zero_connectivity(self):
        import pandas as pd
        from experiments.l23net_analysis.run_l23net_g1b import expected_parameters
        root = Path(__file__).resolve().parents[1]
        tables = pd.read_excel(root/"setup/circuits/L23Net/Circuit_param.xls", sheet_name=None, index_col=0)
        reference = expected_parameters(tables, False)
        mdd = expected_parameters(tables, True)
        self.assertNotIn("HL23VIP:HL23PYR", reference["connected_pairs"])
        self.assertEqual(len(reference["connected_pairs"]), 15)
        for edge, value in reference["synaptic_gmax"].items():
            ratio = .6 if edge.startswith("HL23SST:") else 1.
            self.assertAlmostEqual(mdd["synaptic_gmax"][edge], ratio * value)
        for group in ("somatic", "basal"):
            self.assertEqual(reference["tonic_g_s_per_cm2"]["HL23PYR"][group], mdd["tonic_g_s_per_cm2"]["HL23PYR"][group])
        self.assertAlmostEqual(mdd["tonic_g_s_per_cm2"]["HL23PYR"]["apical"], .6*reference["tonic_g_s_per_cm2"]["HL23PYR"]["apical"])
        for population in ("HL23SST", "HL23PV", "HL23VIP"):
            fraction = mdd["tonic_g_s_per_cm2"][population]["somatic"] / reference["tonic_g_s_per_cm2"][population]["somatic"]
            self.assertGreaterEqual(fraction, .6)
            self.assertLess(fraction, 1.)

    def make_pair(self):
        contract = {"condition": "reference", "simulation": {"MDD": False, "duration_ms": 28000}, "mpi_ranks": 624}
        a = {"status": "passed", "errors": [], "replay_contract": contract,
             "seed_manifest": {"experiment_seed": 7101}, "structure": {"hash": "same"},
             "build_audit": {"errors": [], "invariant_sha256": "same", "expected_parameters": {"gmax": 1.}}}
        b = copy.deepcopy(a)
        b["replay_contract"]["condition"] = "mdd"
        b["replay_contract"]["simulation"]["MDD"] = True
        b["build_audit"]["expected_parameters"]["gmax"] = .6
        for report in (a, b):
            report["replay_contract_sha256"] = canonical_json_sha256(report["replay_contract"])
        return a, b

    def test_pair_accepts_only_intended_changes(self):
        a, b = self.make_pair()
        self.assertEqual(pairing_errors(a, b), [])
        b["replay_contract"]["mpi_ranks"] = 312
        self.assertTrue(any("contracts differ" in x for x in pairing_errors(a, b)))

    def test_structure_or_actual_target_mismatch_rejected(self):
        a, b = self.make_pair()
        b["build_audit"]["invariant_sha256"] = "different"
        self.assertTrue(any("Actual recurrent" in x for x in pairing_errors(a, b)))
        b["structure"] = {"hash": "different"}
        self.assertTrue(any("structure" in x for x in pairing_errors(a, b)))

    def test_missing_intervention_or_failed_build_rejected(self):
        a, b = self.make_pair()
        b["build_audit"]["expected_parameters"] = a["build_audit"]["expected_parameters"]
        b["build_audit"]["errors"] = ["wrong tonic"]
        errors = pairing_errors(a, b)
        self.assertTrue(any("construction" in x for x in errors))
        self.assertTrue(any("intervention" in x for x in errors))

    def test_legacy_matches_original_and_sos_detects_known_power_ratio(self):
        # Actual production fs catches direct-form conditioning and trim errors.
        dt = .025
        t = (np.arange(1120000)+1)*dt/1000
        eeg = 1e-9*(np.sin(2*np.pi*6*t)+np.sin(2*np.pi*10*t)+np.sin(2*np.pi*14*t))
        views, spectra = spectral_views(eeg, dt)
        fs = 40000
        b, a = signal.butter(2, [.1, 100.], btype="bandpass", fs=fs, output="ba")
        expected_f, expected_p = signal.welch(signal.filtfilt(b, a, eeg[160000:])[160000:], fs=fs, nperseg=20000)
        np.testing.assert_array_equal(spectra["legacy"]["frequency_hz"], expected_f)
        np.testing.assert_array_equal(spectra["legacy"]["psd_v2_per_hz"], expected_p)
        doubled, _ = spectral_views(2*eeg, dt)
        for band in ("theta", "alpha", "low_beta"):
            self.assertAlmostEqual(doubled["corrected_sos"]["band_power_v2"][band] / views["corrected_sos"]["band_power_v2"][band], 4., places=9)
        self.assertEqual(views["legacy"]["samples"], 800000)

    def test_spectra_reject_incomplete_or_nonfinite_input(self):
        with self.assertRaises(ValueError):
            spectral_views(np.ones(100), .025)
        data = np.ones(1120000)
        data[-1] = np.nan
        with self.assertRaises(ValueError):
            spectral_views(data, .025)

    def test_resource_and_rate_gate(self):
        r = {"windows": [{"start_ms": 8000, "stop_ms": 9000, "spikes": {"counts": {"HL23PYR": 800}}}],
             "build": {"population_counts": {"HL23PYR": 800}}, "performance": {},
             "memory_snapshots": [{"simulated_ms": 8000, "rss_gib": {"sum": 100}}, {"simulated_ms": 28000, "rss_gib": {"sum": 100.1}}]}
        self.assertEqual(resource_and_rate_checks(r)["errors"], [])
        r["memory_snapshots"][-1]["rss_gib"]["sum"] = 110
        self.assertTrue(any("RSS increased" in e for e in resource_and_rate_checks(r)["errors"]))
        r["windows"][0]["spikes"]["counts"]["HL23PYR"] = 0
        self.assertTrue(any("rate" in e for e in resource_and_rate_checks(r)["errors"]))

    def test_negative_phenotype_is_not_a_success_or_a_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"submission.json").write_text(json.dumps({"commit": "fixed", "pairs": [{"seed": 7101, "project": "sj53"}, {"seed": 7102, "project": "fa32"}]}))
            for seed in (7101, 7102):
                d = root/f"seed_{seed}"
                d.mkdir()
                for name, text in {"worker_exit_code.txt": "0", "git_commit.txt": "fixed", "mechanism_sha256.txt": "same", "environment_versions.json": "{}"}.items():
                    (d/name).write_text(text)
                (d/"g1b_pair_summary.json").write_text(json.dumps({"seed": seed, "debug": False, "technical_passed": True,
                    "pilot_pair_gate_passed": seed == 7101, "structure_sha256": str(seed), "errors": []}))
            result = analyze_suite(root)
            self.assertEqual(result["status"], "direction_not_reproduced")
            self.assertFalse(result["pilot_gate_passed"])
            self.assertEqual(result["errors"], [])

    def test_pbs_cost_uses_allocated_cores_and_walltime(self):
        pbs = parse_pbs_snapshot('resources_used.walltime = 01:00:00\nresources_used.mem = 167772160kb\nresources_used.cput = 400:00:00\n')
        self.assertEqual(pbs["peak_memory_gib"], 160.)
        self.assertEqual(pbs["estimated_normal_ksu"], 1.248)
        self.assertEqual(pbs["allocated_node_hours"], 13.)

    def test_submission_project_order_dependencies_and_budget(self):
        source = Path(__file__).resolve().parents[1]/"experiments/l23net_analysis/nci"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scripts = root/"experiments/l23net_analysis/nci"
            scripts.mkdir(parents=True)
            for name in ("submit_l23net_g1b.sh", "g1b_preflight.py", "run_l23net_g1b_pair.sh", "run_l23net_g1b_summary.sh"):
                shutil.copy(source/name, scripts/name)
            (root/"results").mkdir()
            fakebin = root/"bin"
            fakebin.mkdir()
            (fakebin/"git").write_text('#!/bin/bash\nif [[ "$1" == rev-parse ]]; then echo frozencommit; fi\n')
            (fakebin/"qsub").write_text('#!/usr/bin/env python3\nimport json,os,pathlib,sys\np=pathlib.Path(os.environ["QSUB_LOG"])\nrows=json.loads(p.read_text()) if p.exists() else []\nrows.append(sys.argv[1:])\np.write_text(json.dumps(rows))\nprint(str(len(rows))+".gadi-pbs")\n')
            for p in fakebin.iterdir(): p.chmod(0o755)
            g1a = root/"g1a.json"
            g1a.write_text(json.dumps({"status": "passed", "errors": [], "structure_sha256": ["same", "same"],
                "dataset_comparisons": {k: {"exact": True} for k in ("eeg_v", "dipole_nA_um", "sample_time_ms", "field_left_boundary_time_ms", "field_left_boundary_v_per_m", "stage_code")},
                "trace_summaries": [{"committed_samples": 240000, "committed_windows": 6}]*2}))
            env = dict(os.environ, PATH=str(fakebin)+os.pathsep+os.environ["PATH"], G1A_REPORT=str(g1a), QSUB_LOG=str(root/"qsub.json"))
            run = subprocess.run(["bash", str(scripts/"submit_l23net_g1b.sh")], env=env, capture_output=True, text=True)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            jobs = json.loads((root/"qsub.json").read_text())
            self.assertEqual([args[args.index("-P")+1] for args in jobs], ["sj53", "fa32", "sj53"])
            self.assertIn("depend=afterany:1.gadi-pbs:2.gadi-pbs", jobs[2])
            manifests = list((root/"results").glob("*/submission.json"))
            self.assertEqual(len(manifests), 1)
            manifest = json.loads(manifests[0].read_text())
            self.assertEqual(manifest["maximum_pair_reservation_ksu"], 3.12)
            self.assertEqual([p["job_id"] for p in manifest["pairs"]], ["1.gadi-pbs", "2.gadi-pbs"])


if __name__ == "__main__":
    unittest.main()
