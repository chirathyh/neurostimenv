"""Future-code readouts learn, freeze before test use and keep state partial."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.eeg_representation.h1.v6 import information_models as im
from experiments.eeg_representation.h1.v6 import models
from experiments.eeg_representation.h1.artifacts import atomic_npz
from experiments.eeg_representation.storage import atomic_json, read_json, file_hash
from tests.test_eegrepr_v6_models import panel_fixture


class InformationModelTests(unittest.TestCase):
    def test_readout_known_signal_complete_grid_mean_fallback_and_no_test_tuning(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); panel, manifest, _, _ = panel_fixture(root)
            values = {s: panel[s]['X'][:, :8].copy() for s in panel}
            for s in panel:
                panel[s]['Y'] = (2 * values[s][:, :1] - values[s][:, 1:2])[:, None]
            first = root / 'first'
            owner = self
            class Guard(dict):
                def __getitem__(self, key):
                    owner.assertTrue((first / 'selection.json').exists(), 'Readout touched test before selection freeze')
                    return super().__getitem__(key)
            original = panel['evaluation']; panel['evaluation'] = Guard(original)
            result = im.fit_readout(panel, values, manifest, first)
            self.assertEqual(len(result['candidates']), 8)
            self.assertEqual({r['alpha'] for r in result['candidates'] if r['alpha'] is not None}, set(im.DEFAULTS['ridge_alphas']))
            self.assertNotEqual(result['selected_id'], 'train_mean')
            predictions = im._numeric(first / 'predictions.npz')
            risk = models.circuit_loss(panel['evaluation'], predictions['evaluation_prediction'])
            baseline = models.circuit_loss(panel['evaluation'], predictions['evaluation__train_mean'])
            self.assertLess(risk, .001 * baseline)
            panel['evaluation'] = copy.deepcopy(original); panel['evaluation']['Y'] += 10000
            changed = dict(values); changed['evaluation'] = values['evaluation'] * -1000
            other = root / 'other'; second = im.fit_readout(panel, changed, manifest, other)
            self.assertEqual(result['selected_id'], second['selected_id'])
            a, b = im._numeric(first / 'state.npz'), im._numeric(other / 'state.npz')
            for key in a: np.testing.assert_array_equal(a[key], b[key])
            # Deliberately adversarial dev relationship selects explicit training mean.
            panel['development']['Y'] *= -1
            fallback = im.fit_readout(panel, values, manifest, root / 'fallback')
            self.assertEqual(fallback['selected_id'], 'train_mean')

    def test_staged_pipeline_packing_freeze_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); panel, manifest, config, job = panel_fixture(root, futures=5)
            config['information_models'] = dict(forecast_enabled=False)
            def available(splits):
                manifest['files'] = {s: models._record(root, root / 'derived' / job / (s + '.npz')) for s in splits}
                atomic_json(root / 'derived' / job / 'manifest.json', manifest)
                atomic_json(root / 'derived/manifest.json', dict(jobs=[dict(id=job, status='complete', **models._record(root, root / 'derived' / job / 'manifest.json'))]))
            available(['train', 'development'])
            first = im.fit(config, root)
            self.assertEqual(len([r for r in first['rows'] if r['status'] == 'complete']), 4)
            freeze = file_hash(root / 'information_models/freeze.json')
            self.assertEqual(im.verify_freeze(config, root)['status'], 'frozen')
            paths = {r['weights_path']: file_hash(root / r['weights_path']) for r in first['rows'] if r['status'] == 'complete'}
            available(list(panel)); second = im.fit(config, root, resume=True)
            self.assertEqual(freeze, file_hash(root / 'information_models/freeze.json'))
            timing = read_json(root / 'information_models/timing.json')
            self.assertEqual(timing['summary']['fit_call_count'], 1)
            self.assertFalse(timing['calls'][-1]['includes_new_fitting'])
            self.assertGreater(timing['summary']['measured_fit_elapsed_seconds'], 0.)
            for p, digest in paths.items(): self.assertEqual(digest, file_hash(root / p))
            for row in second['rows']:
                if row['status'] != 'complete': continue
                a = im._numeric(root / row['prediction_path'])
                self.assertEqual(a['evaluation_prediction'].dtype, np.float64)
                np.testing.assert_array_equal(a['evaluation_context_id'], panel['evaluation']['context_id'])
            for p in (root / 'information_models/banks').rglob('codes.npz'):
                a = im._numeric(p)
                self.assertEqual(a['evaluation_packed'].shape[1], 4)
                self.assertEqual(a['evaluation_decoded'].shape[1], 8)
            path = root / next(iter(paths)); arrays = im._numeric(path); arrays['baseline'] += 1
            atomic_npz(path, **arrays)
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'): im.verify_freeze(config, root)

    def test_partial_state_uses_stable_training_only_units_and_excludes_privileged_future(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); panel, _, _, _ = panel_fixture(root / 'panels')
            rng = np.random.default_rng(57); ids = np.concatenate([p['context_id'] for p in panel.values()])
            structures = np.concatenate([p['structure_id'] for p in panel.values()])
            a = rng.normal(size=(len(ids), 47)); a[:, 4] *= 1e-117
            source = root / 'source'
            atomic_npz(source / 'geometry/privileged_state_geometry.npz', context_id=ids,
                structure_id=structures, membrane_gate_state=a,
                aligned_aggregate_state=np.column_stack([a, np.full((len(a), 2), 1e100)]),
                future_rng_seed=np.arange(len(a)), future_schedule=np.full((len(a), 5), 1e99))
            values, meta = im.partial_state(panel, root, source)
            self.assertEqual(meta['original_coordinates'], 47)
            self.assertEqual(meta['retained_coordinates'], 46)
            self.assertIn('pending recurrent deliveries', meta['omitted'])
            self.assertLess(max(np.max(np.abs(v)) for v in values.values()), 10)
            frozen_scaling = meta['scale_arrays']['scale_floor'].copy()
            # Held-out excursions cannot alter scaling or retained training axes.
            eval_start = sum(len(p['X']) for s, p in panel.items() if s != 'evaluation')
            a[eval_start:, 0] *= 1e6
            atomic_npz(source / 'geometry/privileged_state_geometry.npz', context_id=ids, structure_id=structures, membrane_gate_state=a)
            _, changed = im.partial_state(panel, root, source)
            np.testing.assert_array_equal(frozen_scaling, changed['scale_arrays']['scale_floor'])

    def test_native_source_state_reading_without_geometry_export(self):
        import json
        import h5py
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); panel, _, _, _ = panel_fixture(root / 'panels')
            source = root / 'native_source'; rows = []; rng = np.random.default_rng(76)
            for data in panel.values():
                for cid, sid in zip(data['context_id'], data['structure_id']):
                    path = source / 'native' / (str(cid) + '.h5'); path.parent.mkdir(parents=True, exist_ok=True)
                    with h5py.File(path, 'w') as f:
                        f.attrs['metadata_json'] = json.dumps(dict(population_sizes=dict(E=1, I=0)))
                        p = f.create_group('privileged')
                        p.create_dataset('cutoff_physical_state', data=rng.normal(size=47))
                        p.create_dataset('synapse_parameters', data=np.empty((0, 6)))
                        p.create_dataset('future_rng_seed', data=999999999)
                    rows.append(dict(path=str(path.relative_to(source)), context_id=str(cid), structure_id=int(sid),
                                     sha256=file_hash(path), bytes=path.stat().st_size, pool='score', future_index=0))
            atomic_json(source / 'native/manifest.json', dict(trials=rows))
            values, metadata = im.partial_state(panel, root, source)
            self.assertEqual(metadata['status'], 'complete')
            self.assertEqual(values['train'].shape[1], 47)
            self.assertEqual(metadata['original_coordinates'], 47)
            self.assertEqual(len(metadata['source_files']), sum(len(p['X']) for p in panel.values()))

    def test_state_source_requires_actual_generation_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            source = Path(d); atomic_json(source / 'resolved_config.json', dict(name='one-generation'))
            with im.EvidenceSource(source) as evidence: provenance = evidence.provenance()
            c = dict(claim_status='exploratory_existing')
            self.assertEqual(im.verify_state_source(dict(source_provenance=provenance), source, c)['status'], 'verified')
            with self.assertRaisesRegex(ValueError, 'IDs alone are insufficient'):
                im.verify_state_source(dict(source_content_id='different-generation'), source, c)
            runtime = im.runtime_identity(dict(claim_status='engineering'))
            self.assertIn('h1/v6/models.py', runtime['source_dependencies'])
            self.assertIn('models.py', runtime['source_dependencies'])

    def test_forecast_bank_consumes_full_history_and_staged_weights_are_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); panel, _, config, _ = panel_fixture(root, length=2000)
            config['information_models'] = dict(forecast_seeds=[0], forecast_updates=1)
            options = im.settings(config); directory = root / 'bank'
            fitting = {s: panel[s] for s in ('train', 'development')}
            values, meta = im._forecast_bank(root, fitting, options, 0, directory, 'cpu', False)
            self.assertEqual(meta['consumed_history_samples'], 2000)
            self.assertEqual(values['train'].shape, (len(panel['train']['X']), 128))
            digest = file_hash(directory / 'state.npz')
            appended, _ = im._forecast_bank(root, panel, options, 0, directory, 'cpu', True)
            self.assertEqual(digest, file_hash(directory / 'state.npz'))
            np.testing.assert_array_equal(values['train'], appended['train'])
            self.assertIn('evaluation', appended)
            changed = copy.deepcopy(panel); changed['evaluation']['Y'] += 999
            _, _ = im._forecast_bank(root, changed, options, 0, directory, 'cpu', True)
            self.assertEqual(digest, file_hash(directory / 'state.npz'))

    def test_scientific_settings_preserve_matched_grid_and_cost(self):
        config = dict(claim_status='prospective_confirmation', selected_job_ids=['A0__T0_l10_d128', 'A1__T0_l10_d128'])
        options = im.settings(config); cost = im.cost(config)
        self.assertEqual(cost['code_methods_per_target'], 10)
        self.assertEqual(cost['code_ridge_candidates'], 70)
        self.assertEqual(cost['new_simulation_episodes'], 0)
        for override in (dict(ridge_alphas=[1]), dict(dimension=32), dict(forecast_updates=1), dict(forecast_enabled=False)):
            with self.assertRaises(ValueError): im.settings(dict(config, information_models=override))


if __name__ == '__main__': unittest.main()
