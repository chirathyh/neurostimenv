"""R1 cohort separation, inference, quota/dependency handling and failure checks."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.l23net_analysis import r1_analysis as analysis
from experiments.l23net_analysis.nci import submit_l23net_r1 as submitter
from experiments.l23net_analysis.r1_protocol import COHORTS, PROTOCOL, jobs_for_stage, save_json, sha256


def synthetic_row(seed, cohort, condition="reference", effect=.2):
    base = np.array([-19., -19.2, -19.4]) + (seed % 16)*.01
    log_power = base + (effect if condition == "mdd" else 0)
    return {"seed": seed, "condition": condition, "cohort": cohort,
            "report_sha256": f"r{seed}_{condition}", "trace_content_sha256": f"t{seed}_{condition}",
            "structure_sha256": f"s{seed}", "resources": {},
            "views": {view: {"band_power_v2": dict(zip(analysis.BANDS, 10.**log_power))}
                      for view in ("legacy", "corrected_sos")}}


class R1Tests(unittest.TestCase):
    def test_design_sizes_and_disjointness(self):
        core, extension = jobs_for_stage("core"), jobs_for_stage("extension")
        self.assertEqual((len(core), len(extension)), (24, 44))
        self.assertTrue(all(len(j["runs"]) == 2 for j in core+extension))
        seeds = [s for values in COHORTS.values() for s in values]
        self.assertEqual(len(set(seeds)), 76)
        self.assertFalse({10, 7101, 7102}.intersection(seeds))

    def test_worker_does_not_feed_manifest_to_mpirun_stdin(self):
        root = Path(__file__).resolve().parents[1]
        worker = (root/'experiments/l23net_analysis/nci/run_l23net_r1_worker.sh').read_text()
        self.assertIn('env.online.temperature_mode=configured < /dev/null', worker)

    def test_production_configuration_and_cohort_restrictions(self):
        from hydra import compose, initialize_config_dir
        from experiments.l23net_analysis.run_l23net_r1 import validate_configuration
        root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(config_dir=str(root/'configs'), version_base=None):
            cfg = compose(config_name='config', overrides=['env=hl23net', 'analysis=l23net_r1',
                          'experiment.seed=8201', 'experiment.debug=false', 'env.network.dt=0.025',
                          'env.simulation.duration=28000', 'env.simulation.obs_win_len=1000',
                          'env.simulation.MDD=false', 'env.ts.apply=false'])
        self.assertEqual(validate_configuration(cfg, 624), (28, 40000))
        cfg.experiment.seed = 7101
        with self.assertRaises(ValueError):
            validate_configuration(cfg, 624)
        cfg.experiment.seed = 8101
        cfg.analysis.cohort = 'calibration'
        cfg.analysis.condition = 'mdd'
        cfg.env.simulation.MDD = True
        with self.assertRaises(ValueError):
            validate_configuration(cfg, 624)

    def test_account_units_and_invalid_balances(self):
        for text, expected in (("Project=sj53 Avail: 40.32 KSU", 40.32),
                               ("Available: 40,320 SU", 40.32), ("Avail: 1.2 MSU", 1200.)):
            self.assertEqual(submitter.parse_balance(text), expected)
        with self.assertRaises(ValueError):
            submitter.parse_balance("Allocated: 99 KSU")

    def test_quota_priority_max_reservations_and_no_ny83_when_not_needed(self):
        jobs = submitter.allocate_projects(jobs_for_stage("core"), {"sj53": 40.32, "fa32": 34.42, "ny83": 116.93})
        self.assertEqual([sum(j['project'] == p for j in jobs) for p in submitter.PROJECTS], [12, 10, 2])
        jobs = submitter.allocate_projects(jobs_for_stage("core"), {"sj53": 100, "fa32": 100, "ny83": 100})
        self.assertEqual({j['project'] for j in jobs}, {"sj53"})
        with self.assertRaises(ValueError):
            submitter.allocate_projects(jobs_for_stage("core"), {p: 2. for p in submitter.PROJECTS})

    def test_exact_sign_flip_and_degenerate_data(self):
        self.assertEqual(analysis.paired_inference(np.ones(4))["p_one_sided"], 1/16)
        self.assertEqual(analysis.paired_inference(-np.ones(4))["p_one_sided"], 1)
        zero = analysis.paired_inference(np.zeros(4))
        self.assertEqual(zero["p_one_sided"], 1)
        self.assertIsNone(zero["paired_dz"])
        self.assertEqual(zero["mean_t_ci95"], [0, 0])

    def test_monte_carlo_is_reproducible_and_not_zero(self):
        a = analysis.paired_inference(np.ones(21))
        self.assertEqual(a, analysis.paired_inference(np.ones(21)))
        self.assertGreater(a["p_one_sided"], 0)

    def test_bh(self):
        np.testing.assert_allclose(analysis.bh_adjust([.01, .04, .03]), [.03, .04, .04])

    def target(self):
        return analysis.reference_target([synthetic_row(s, "calibration") for s in COHORTS['calibration']])

    def test_calibration_and_missing_or_duplicate_pairs(self):
        target = self.target()
        for view in target['views'].values():
            self.assertTrue(all(x >= .01 for x in view['scale_log10']))
        rows = [synthetic_row(s, 'primary', c) for s in COHORTS['primary'] for c in ('reference', 'mdd')]
        with self.assertRaises(ValueError):
            analysis.cohort_inference(rows[:-1], target, COHORTS['primary'])
        with self.assertRaises(ValueError):
            analysis.cohort_inference(rows[:-1]+[rows[0]], target, COHORTS['primary'])
        bad = [synthetic_row(s, 'calibration', 'mdd') for s in COHORTS['calibration']]
        with self.assertRaises(ValueError):
            analysis.reference_target(bad)

    def test_direction_gate_preserves_negative_result(self):
        rows = [synthetic_row(s, 'primary', c, effect=-.2) for s in COHORTS['primary'] for c in ('reference', 'mdd')]
        result = analysis.cohort_inference(rows, self.target(), COHORTS['primary'])
        self.assertFalse(result['phenotype_confirmed'])
        self.assertEqual(result['views']['legacy']['composite']['n_structures'], 16)

    def test_frozen_results_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'frozen.json'
            analysis.frozen_json(path, {'x': 1})
            analysis.frozen_json(path, {'x': 1})
            with self.assertRaises(ValueError):
                analysis.frozen_json(path, {'x': 2})

    def test_failed_reanalysis_preserves_original_summary(self):
        from experiments.l23net_analysis import analyze_l23net_r1 as cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root/'r1_summary.json'
            original.write_text('{"status":"passed"}')
            with patch.object(sys, 'argv', ['analyze', '--suite', str(root)]), \
                 patch.object(cli, 'analyze_suite', side_effect=ValueError('frozen output differs')), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(), 1)
            self.assertEqual(original.read_text(), '{"status":"passed"}')
            self.assertTrue((root/'r1_reanalysis_failure.json').exists())

    def test_final_pbs_accounting(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'pbs.out'
            self.assertFalse(analysis.pbs_epilogue(path)['final_accounting_available'])
            path.write_text('Exit Status: 0\nService Units: 1435.89\nMemory Used: 159.8GB\nWalltime Used: 01:09:02\n')
            result = analysis.pbs_epilogue(path)
            self.assertEqual(result['service_units'], 1435.89)
            self.assertAlmostEqual(result['wall_hours'], 1+9/60+2/3600)

    def test_quota_submission_throttle_and_partial_failure(self):
        for fail_at in (None, 4):
            with self.subTest(fail_at=fail_at), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                prerequisite = root/'g1b.json'
                prerequisite.write_text('{}')
                commands = []
                def output(command, **kwargs):
                    if command[0] == 'git':
                        return 'frozencommit\n'
                    commands.append(command)
                    if len(commands) == fail_at:
                        raise subprocess.CalledProcessError(1, command)
                    return f'{len(commands)}.gadi-pbs\n'
                with patch.object(submitter, 'ROOT', root), patch.object(submitter, 'verify_g1b'), \
                     patch.object(submitter.subprocess, 'check_output', side_effect=output), \
                     patch.object(submitter.subprocess, 'run'), patch.object(submitter.shutil, 'which', return_value='/bin/qsub'), \
                     contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    args = ['--stage', 'core', '--g1b', str(prerequisite), '--available-ksu',
                            'sj53=40.32', 'fa32=34.42', 'ny83=116.93', '--submit']
                    if fail_at:
                        with self.assertRaises(subprocess.CalledProcessError):
                            submitter.main(args)
                    else:
                        self.assertEqual(submitter.main(args), 0)
                manifests = list((root/'results').glob('*/submission.json'))
                self.assertEqual(len(manifests), 1)
                manifest = json.loads(manifests[0].read_text())
                if fail_at:
                    self.assertEqual(manifest['submission_status'], 'partial_failure')
                    self.assertEqual(sum('job_id' in j for j in manifest['jobs']), 3)
                else:
                    self.assertEqual(len(commands), 25)
                    self.assertIn('depend=afterany:1.gadi-pbs', commands[8])
                    self.assertIn('depend=afterany:16.gadi-pbs', commands[23])
                    self.assertIn('depend=afterany:'+':'.join(f'{i}.gadi-pbs' for i in range(1,25)), commands[24])

    def test_full_core_summary_and_missing_worker_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = jobs_for_stage('core')
            for j in jobs:
                j['project'] = 'sj53'
                folder = root/j['name']
                folder.mkdir()
                for name, value in {'worker_exit_code.txt': '0', 'git_commit.txt': 'same',
                                    'mechanism_sha256.txt': 'same', 'environment_versions.json': 'same'}.items():
                    (folder/name).write_text(value)
            save_json(root/'submission.json', {'stage': 'core', 'jobs': jobs, 'protocol': PROTOCOL, 'commit': 'same'})
            def fake_audit(directory, seed, condition, cohort):
                row = synthetic_row(seed, cohort, condition, effect=.1 + (seed % 3)*.02)
                contract = {'experiment_seed': seed, 'condition': condition, 'simulation': {'MDD': condition == 'mdd'}}
                return row, {'replay_contract': contract}, {}
            with patch.object(analysis, 'audit_run', side_effect=fake_audit), patch.object(analysis, 'pairing_errors', return_value=[]):
                result = analysis.analyze_suite(root)
                self.assertEqual(result['status'], 'passed')
                self.assertEqual(result['primary']['views']['legacy']['composite']['n_structures'], 16)
                self.assertTrue((root/'reference_target.json').exists())
                self.assertTrue((root/'core_data_frozen.json').exists())
                primary_hash = sha256(root/'primary_frozen.json')
                extension = root/'extension'
                extension.mkdir()
                extension_jobs = jobs_for_stage('extension')
                for job in extension_jobs:
                    job['project'] = 'ny83'
                    folder = extension/job['name']
                    folder.mkdir()
                    for name, value in {'worker_exit_code.txt': '0', 'git_commit.txt': 'same',
                                        'mechanism_sha256.txt': 'same', 'environment_versions.json': 'same'}.items():
                        (folder/name).write_text(value)
                save_json(extension/'submission.json', {'stage': 'extension', 'jobs': extension_jobs,
                          'protocol': PROTOCOL, 'commit': 'same', 'core_directory': str(root),
                          'core_hashes': {name: sha256(root/name) for name in submitter.CORE_FILES}})
                extended = analysis.analyze_suite(extension)
                self.assertEqual(extended['status'], 'completed_extension')
                self.assertEqual(extended['extension_44']['views']['legacy']['composite']['n_structures'], 44)
                self.assertEqual(extended['pooled_60_secondary']['views']['legacy']['composite']['n_structures'], 60)
                self.assertEqual(sha256(root/'primary_frozen.json'), primary_hash)
                (root/jobs[0]['name']/'worker_exit_code.txt').write_text('1')
                failed = analysis.analyze_suite(root)
                self.assertFalse(failed['technical_passed'])
                self.assertNotIn('primary', failed)


if __name__ == '__main__':
    unittest.main()
