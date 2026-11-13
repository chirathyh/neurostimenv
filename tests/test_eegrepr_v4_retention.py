import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.eeg_representation.h1.v4.retention import (
    balanced_label_subset, projection_fit, project, fit_range_rule,
    quantization_attribution, token_views, evaluate_readout, _new_banks, DEFAULTS, matched_code_contrast,
)
from experiments.eeg_representation.h1.compression import unpack_codes


class RetentionTests(unittest.TestCase):
    def test_balancing_requires_every_class_and_circuit(self):
        labels = np.tile([0, 1, 2], 4); circuits = np.repeat([1, 2, 3, 4], 3)
        selected, meta = balanced_label_subset(labels, circuits, 11)
        self.assertIsNone(selected); self.assertEqual(meta['status'], 'unavailable')
        selected, meta = balanced_label_subset(labels, circuits, 12)
        self.assertEqual(len(selected), 12)
        selected, meta = balanced_label_subset(labels[:-1], circuits[:-1], 'all')
        self.assertIsNone(selected); self.assertIn('coverage', meta['reason'])

    def test_payload_clipping_and_error_decomposition(self):
        rng = np.random.default_rng(4); train = rng.normal(size=(100, 8))
        test = rng.normal(size=(30, 8)); test[0] = 20
        for rule in ['quantile_005_995', 'robust_mad6']:
            q = fit_range_rule(train, 32, rule)
            payload, decoded, report = quantization_attribution(test, q)
            self.assertEqual(payload.shape, (30, 4))
            self.assertEqual(unpack_codes(payload, q['bits']).shape, test.shape)
            self.assertAlmostEqual(report['latent_dequantization_mse'],
                report['clipping_only_mse'] + report['rounding_after_clipping_mse'] + report['clipping_rounding_cross_term'])
            q_more = fit_range_rule(train, 128, rule)
            _, _, more = quantization_attribution(test, q_more)
            self.assertEqual(report['coordinate_clipping_fraction'], more['coordinate_clipping_fraction'])
            self.assertTrue(np.array_equal(q['lower'], q_more['lower']))

    def test_projection_no_fake_dimensions_and_inverse(self):
        rng = np.random.default_rng(3); x = rng.normal(size=(40, 18))
        with self.assertRaises(ValueError): projection_fit(x, 32)
        p = projection_fit(x, 18)
        np.testing.assert_allclose(project(x, p) @ p['components'] + p['center'], x, atol=1e-12)
        small = projection_fit(x, 8)
        self.assertGreater(np.mean((project(x, small) @ small['components'] + small['center'] - x) ** 2), .1)

    def test_pooling_shapes_and_information_difference(self):
        rng = np.random.default_rng(13); tokens = rng.normal(size=(3, 63, 128))
        views = token_views(tokens, 4)
        self.assertEqual(views['full_tokens'].shape, (3, 8064))
        self.assertEqual(views['pooled'].shape, (3, 128))
        self.assertEqual(views['fixed_token_readout'].shape, (3, 512))
        changed = tokens[:, ::-1]
        np.testing.assert_allclose(token_views(changed)['pooled'], views['pooled'], atol=1e-14)
        self.assertFalse(np.allclose(token_views(changed)['fixed_token_readout'], views['fixed_token_readout']))

    def test_decoder_evaluation_cannot_select_and_saves_classes(self):
        rng = np.random.default_rng(99); panel = {}; values = {}
        for i, s in enumerate(('train', 'development', 'evaluation')):
            x = rng.normal(size=(24, 4)); y = (x[:, 0] > 0).astype(int)
            # Make both classes present in each circuit.
            y[:2] = [0, 1]; y[12:14] = [0, 1]
            panel[s] = dict(X=x, structure_id=np.repeat([2*i, 2*i+1], 12), context_id=np.array([f'{s}{k}' for k in range(24)]), task_rhythm=y)
            values[s] = x
        with tempfile.TemporaryDirectory() as td:
            first = evaluate_readout(panel, values, 'rhythm', 'all', [.01, 1., 100.], Path(td) / 'a')
            panel['evaluation']['task_rhythm'] = 1 - panel['evaluation']['task_rhythm']
            second = evaluate_readout(panel, values, 'rhythm', 'all', [.01, 1., 100.], Path(td) / 'b')
            self.assertEqual(first['selected_alpha'], second['selected_alpha'])
            with np.load(Path(td) / 'a/predictions.npz', allow_pickle=False) as f:
                np.testing.assert_array_equal(f['class_order'], [0, 1])
                self.assertEqual(f['evaluation_selected_prediction'].shape, (24, 2))
            self.assertIn('balanced_accuracy', first['evaluation'])

    def test_scalar_and_all_future_readout_scoring(self):
        rng = np.random.default_rng(39); panel = {}; values = {}
        for i, split in enumerate(('train', 'development', 'evaluation')):
            x = rng.normal(size=(12, 4)); y = x[:, None, :2] + rng.normal(size=(12, 4, 2))
            panel[split] = dict(X=x, Y=y, future_mask=np.ones((12, 4), bool),
                structure_id=np.repeat([2*i, 2*i+1], 6), context_id=np.array([f'{split}{k}' for k in range(12)]),
                task_log_power=np.square(x).mean(1))
            values[split] = x
        with tempfile.TemporaryDirectory() as td:
            for task in ('future_mean', 'log_power'):
                path = Path(td) / task
                result = evaluate_readout(panel, values, task, 'all', [1., 100.], path)
                self.assertEqual(result['status'], 'complete')
                with np.load(path / 'predictions.npz', allow_pickle=False) as f:
                    pred = f['evaluation_selected_prediction']
                    self.assertEqual(pred.dtype, np.float32)
                    if task == 'future_mean':
                        loss = ((pred.astype(float)[:, None] - panel['evaluation']['Y']) ** 2).mean((1, 2))
                    else:
                        loss = (pred[:, 0].astype(float) - panel['evaluation']['task_log_power']) ** 2
                    np.testing.assert_allclose(loss, f['evaluation_selected_context_loss'], atol=1e-12)

    def test_non_neural_smoke_does_not_train_unrequested_banks(self):
        rng = np.random.default_rng(44); panel = {'manifest': {}}
        for s in ('train', 'development', 'evaluation'):
            x = rng.normal(size=(8, 128)).astype(np.float32)
            panel[s] = dict(X=x, Y=x[:, None, :4], future_mask=np.ones((8, 1), bool))
        options = dict(DEFAULTS, banks=['feature18', 'pca'], optimization_seeds=[0])
        with tempfile.TemporaryDirectory() as td:
            banks, audit = _new_banks(panel, Path(td), options, 'cpu')
            self.assertEqual(set(banks), {'feature18', 'pca'})
            self.assertEqual(audit, [])

    def test_frozen_contrast_averages_optimizer_repeats_within_circuits(self):
        spec = dict(banks=['feature18', 'forecast'], dimension=8, bits=32, readout='linear',
                    label_budget='all', task='carrier', range_rule='quantile_005_995',
                    pooling='native pooled', job_id='A0__T0_l10_d128')
        def row(name, seed, loss):
            return dict(status='complete', representation=name, representation_family=name.split('__')[0],
                optimization_seed=seed, dimension=8, code_bits=32, task='carrier', label_budget='all',
                range_rule='quantile_005_995', evaluation=dict(class_order=[0, 1, 2],
                per_circuit_loss={str(c): loss for c in range(4)}))
        rows = [row('feature18', None, 1.)] + [row(f'forecast__seed_{s}', s, 2.+s) for s in range(3)]
        result = matched_code_contrast(rows, spec, job_id=spec['job_id'], historical=False, required_seeds=[0,1,2])
        self.assertEqual(result['status'], 'complete'); self.assertEqual(result['circuit_count'], 4)
        self.assertEqual(result['optimizer_repeat_counts']['forecast'], 3)
        self.assertEqual(result['mean_difference'], 2.)
        self.assertEqual(result['signflip']['p_value'], .125)
        missing = matched_code_contrast(rows[:-1], spec, job_id=spec['job_id'], historical=False, required_seeds=[0,1,2])
        self.assertEqual(missing['status'], 'unavailable')
        bad = dict(spec, pooling='full tokens')
        self.assertEqual(matched_code_contrast(rows, bad, job_id=spec['job_id'], historical=False, required_seeds=[0,1,2])['status'], 'unavailable')

    def test_prospective_banks_three_seed_exposure_contract(self):
        rng = np.random.default_rng(20); panel = {'manifest': {}}
        for s in ('train', 'development', 'evaluation'):
            x = rng.normal(size=(8, 128)).astype(np.float32)
            panel[s] = dict(X=x, Y=x[:, None, :4], future_mask=np.ones((8, 1), bool))
        options = dict(DEFAULTS, representation_steps=1, batch_size=4,
            encoder_config=dict(architecture='transformer', channels=1, embedding_dim=8, layers=1, heads=2, patch_samples=32))
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):
                _new_banks(panel, Path(td), dict(options, optimization_seeds=[0]), 'cpu')
            banks, audit = _new_banks(panel, Path(td), options, 'cpu')
            for seed in [0, 1, 2]:
                a = banks[f'masked_mse__seed_{seed}']['metadata']; b = banks[f'forecast__seed_{seed}']['metadata']
                self.assertEqual(a['forward_examples'], b['forward_examples'])
                self.assertEqual(a['optimization_seed'], seed)
            self.assertEqual(len(audit), 9)


if __name__ == '__main__': unittest.main()
