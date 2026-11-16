"""Extended runner freezes code/state learners before new held-out generations."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.eeg_representation.h1.v6 import cli, config, extended_cli, information_models, models, planning
from experiments.eeg_representation.h1.v6.artifacts import initialize
from experiments.eeg_representation.h1.targets import canonical_hash
from experiments.eeg_representation.storage import atomic_json, read_json, file_hash
from tests.test_eegrepr_v6_models import panel_fixture


class ExtendedCampaignTests(unittest.TestCase):
    def test_legacy_resolved_configuration_hash_and_resume_remain_unchanged(self):
        # Recorded from the parent implementation before the amendment existed.
        legacy_hash = 'c6b89e6ae6f0cbba81ccf20c30e690377b09e242be7cdbe9aa48dd9662e06d7d'
        c = config.resolve()
        self.assertEqual(canonical_hash(c), legacy_hash)
        self.assertNotIn('extension', c); self.assertNotIn('information_models', c)
        self.assertEqual(config.resolve(c), c)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            atomic_json(root / 'resolved_config.json', c)
            atomic_json(root / 'provenance.json', dict(schema_version=config.SCHEMA, config_hash=legacy_hash))
            reloaded = config.load_config(root / 'resolved_config.json')
            self.assertEqual(initialize(root, reloaded, resume=True)['config_hash'], legacy_hash)
        amended = config.resolve(dict(extension=dict(enabled=True)))
        self.assertTrue(amended['extension']['enabled']); self.assertEqual(amended['information_models'], {})
        explicit_noop = config.resolve(dict(extension=dict(enabled=False), information_models={}))
        self.assertIn('extension', explicit_noop); self.assertIn('information_models', explicit_noop)

    def test_actual_cli_staged_information_freeze_is_required_and_survives_new_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary); root = base / 'evidence'; job = 'A0__T0_l10_d128'
            c = config.resolve(dict(claim_status='engineering', backend='analytical',
                extension=dict(enabled=True), selected_job_ids=[job], acquisition_ids=['A0'],
                data=dict(structures=dict(train=[100, 101, 102, 103], development=[104, 105], calibration=[106, 107], evaluation=[108, 109]),
                    prefixes_per_regime=dict(train=2, development=2, calibration=2, evaluation=2)),
                forks=dict(score_futures=4, high_score_futures=4, reference_futures=0, reference_circuits=[]),
                reuse=dict(enabled=False), models=dict(neural_enabled=False, run_controls=False, full_history_samples=64, pca_width=8),
                information_models=dict(forecast_enabled=False), protocol=dict(primary_job_id=job, test_circuits=2)))
            cfg = base / 'fixture.json'; atomic_json(cfg, c); initialize(root, c)
            panel, manifest, _, _ = panel_fixture(root, futures=5)
            def available(splits):
                manifest['files'] = {s: models._record(root, root / 'derived' / job / (s + '.npz')) for s in splits}
                atomic_json(root / 'derived' / job / 'manifest.json', manifest)
                atomic_json(root / 'derived/manifest.json', dict(jobs=[dict(id=job, status='complete', **models._record(root, root / 'derived' / job / 'manifest.json'))]))
            def stage(mode):
                args = cli.parser().parse_args([mode, '--root', str(root), '--config', str(cfg), '--resume'])
                return cli.stage(args)
            available(['train', 'development']); stage('fit')
            with self.assertRaisesRegex(ValueError, 'Information-model train/dev freeze'):
                planning._mean_freeze(c, root)
            stage('information-fit'); initial = planning._mean_freeze(c, root)
            freeze_sha = file_hash(root / 'information_models/freeze.json')
            self.assertIn('information_models/freeze.json', [r['path'] for r in initial['files']])
            available(['train', 'development', 'calibration']); stage('predict'); stage('calibrate'); stage('information-fit')
            available(list(panel)); stage('predict'); stage('information-fit')
            self.assertEqual(freeze_sha, file_hash(root / 'information_models/freeze.json'))
            self.assertEqual(initial, planning._mean_freeze(c, root))
            self.assertEqual(information_models.verify_freeze(c, root)['fitted_method_count'], 4)
            rows = read_json(root / 'information_models/manifest.json')['rows']
            self.assertTrue(all('evaluation' in r['counts'] for r in rows if r['status'] == 'complete'))

    def test_failed_information_freeze_stops_before_calibration_or_test_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); source = root / 'old.zip'; source.write_bytes(b'existing evidence fixture')
            c = config.resolve(dict(extension=dict(enabled=True)))
            cfg = root / 'config.json'; atomic_json(cfg, c)
            args = extended_cli.parser().parse_args(['complete', '--root', str(root / 'campaign'), '--config', str(cfg),
                '--source-archive', str(source), '--native-source', str(root / 'native'), '--execute'])
            proposal = dict(status='unchanged_candidate', phase0_complete=True, resolved_config=c)
            calls = []
            def stage(a):
                calls.append((a.mode, a.split))
                if a.mode == 'plan': return dict(status='planned', eligible_numerical_execution=True, eligible_for_generation=True)
                if a.mode == 'information-fit': raise ValueError('Information freeze qualification failed')
                return dict(status='complete')
            with patch.object(cli, 'stage', side_effect=stage), patch('experiments.eeg_representation.h1.v6.precision.certificate', return_value={}), patch('experiments.eeg_representation.h1.v6.precision.propose_design', return_value=proposal):
                with self.assertRaisesRegex(ValueError, 'Information freeze qualification failed'):
                    extended_cli.dispatch(args)
            self.assertEqual([split for mode, split in calls if mode == 'generate'], ['train', 'development'])
            self.assertNotIn('calibrate', [mode for mode, split in calls])
            self.assertNotIn('confirm-statistics', [mode for mode, split in calls])

    def test_report_only_resolves_actual_pretest_cohort_and_rejects_config_disagreement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); c = config.resolve(dict(extension=dict(enabled=True)))
            c['protocol']['test_circuits'] = 257; c['data']['structures']['evaluation'] = list(range(37052, 37309))
            selected = root / 'selected-config.json'; atomic_json(selected, c)
            core = root / 'confirmation-n257'; atomic_json(core / 'resolved_config.json', c)
            actual, cfg = extended_cli.existing_root(root)
            self.assertEqual(actual, core); self.assertEqual(cfg, core / 'resolved_config.json')
            changed = copy.deepcopy(c); changed['budget']['max_generation_hours'] = 499
            atomic_json(core / 'resolved_config.json', changed)
            with self.assertRaisesRegex(ValueError, 'configuration disagree'): extended_cli.existing_root(root)

    def test_status_is_read_only_and_does_not_assert_process_liveness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'not_started'
            result = extended_cli.status(root)
            self.assertEqual(result['status'], 'not_started'); self.assertFalse(root.exists())
            root.mkdir(); atomic_json(root / 'campaign_progress.json', dict(stage='generate', status='running'))
            result = extended_cli.status(root)
            self.assertIn('not checked', result['process_liveness'])
            self.assertTrue(result['read_only'])


if __name__ == '__main__': unittest.main()
