"""Temporary marked fixtures test inference mechanics, never empirical results."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from experiments.eeg_representation.aggregation import summarize, _recipes
from experiments.eeg_representation.schema import DEFAULTS


class AggregationTests(unittest.TestCase):
    def fixture(self, root, claim_status='pilot', capped=False):
        config = deepcopy(DEFAULTS)
        config['claim_status'] = claim_status
        config['training'].update(recipes=['joint', 'progressive'], seeds=[0, 1], label_budgets=[10, 100])
        config['evaluation'].update(tasks=['baseline_alpha', 'baseline_beta_low'], probe_types=['linear'], bootstrap_samples=100)
        protocol = 'leave-task-label-out'
        directory = Path(root) / 'evaluations' / protocol
        directory.mkdir(parents=True)
        areas, rows = [], []
        for recipe in ['joint', 'progressive']:
            for seed in [0, 1]:
                run = Path(root) / 'models' / protocol / recipe / f'seed_{seed}' / 'budget_None'
                run.mkdir(parents=True)
                (run / 'run.json').write_text(json.dumps(dict(recipe=recipe, seed=seed, protocol=protocol, budget=None,
                    status='complete', examples_processed=200, optimizer_updates=10, training_parent_hash='fixture-parent-hash',
                    accounting={'label_exposures': {'rhythm': 200}}, elapsed_seconds=1.25, cpu_seconds=0.75, peak_process_rss_bytes=12345, device='cpu')))
                for task in config['evaluation']['tasks']:
                    # Equal task averaging gives joint=.5, progressive=.4.
                    base = .4 if task == 'baseline_alpha' else .6
                    effect = (0.1 if task == 'baseline_alpha' else -0.3) if recipe == 'progressive' else 0
                    score = {'a': base + .2 * seed + effect, 'b': 1 - base - .2 * seed + effect}
                    area_status = 'pending: insufficient or capped budget axis' if capped and recipe == 'progressive' and seed == 1 else 'complete'
                    areas.append(dict(recipe=recipe, optimizer_seed=seed, pretraining_budget='None', task=task, probe='linear',
                                      budgets=[10, 100], status=area_status, normalized_log_budget_area=sum(score.values()) / 2,
                                      structure_scores=score))
                    for budget in [10, 100]:
                        rows.append(dict(recipe=recipe, optimizer_seed=seed, pretraining_budget=None, task=task, probe='linear',
                                         budget=budget, status='complete', kind='classification'))
        (directory / 'probe_results.json').write_text(json.dumps(dict(status='functional smoke' if claim_status == 'smoke' else 'exploratory', rows=rows)))
        (directory / 'transfer_areas.json').write_text(json.dumps(areas))
        return config

    def test_equal_tasks_structures_and_optimizer_repeats_preserve_negative(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.fixture(root)
            result = summarize(config, root)
            primary = [r for r in result['primary'] if r['status'] == 'complete']
            self.assertEqual(len(primary), 2)
            self.assertAlmostEqual(next(r for r in primary if r['recipe'] == 'joint')['area'], .5)
            self.assertAlmostEqual(next(r for r in primary if r['recipe'] == 'progressive')['area'], .4)
            comparison = next(r for r in result['comparisons'] if r['task'] == 'primary_equal_task_area' and r['status'] == 'complete')
            self.assertAlmostEqual(comparison['estimate'], -.1)
            self.assertEqual(comparison['n_structures'], 2)
            self.assertTrue(comparison['exposure_audit']['matched'])
            self.assertEqual(result['training_efficiency'][0]['cpu_seconds'], 0.75)
            self.assertEqual(result['training_efficiency'][0]['peak_process_rss_bytes'], 12345)
            self.assertTrue(result['missing_jobs'])  # Absent second protocol remains explicitly missing.
            self.assertTrue((Path(root) / 'summary' / 'transfer_table.tex').exists())
            self.assertEqual(summarize(config, root)['primary'], result['primary'])

    def test_random_orders_capped_by_distinct_task_permutations(self):
        config=deepcopy(DEFAULTS)
        config['training'].update(recipes=['joint','random_order'],source_tasks=['rhythm','carrier'],random_orders=3)
        self.assertEqual(_recipes(config), ['joint','random_order','random_order__order_1'])
        config['training'].update(source_tasks=['rhythm','carrier','coherence'],random_orders=5)
        self.assertEqual(len(_recipes(config)), 6)

    def test_smoke_never_enters_scientific_aggregation(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.fixture(root, claim_status='smoke')
            result = summarize(config, root)
            self.assertFalse(result['scientific_aggregation_enabled'])
            self.assertTrue(all(r['area'] is None for r in result['primary']))
            self.assertFalse(result['comparisons'])
            self.assertTrue(result['smoke_diagnostics'])
            self.assertIn('Pending', (Path(root) / 'summary' / 'transfer_table.tex').read_text())

    def test_capped_budget_axis_cannot_be_pooled_with_complete_axis(self):
        with tempfile.TemporaryDirectory() as root:
            config = self.fixture(root, capped=True)
            result = summarize(config, root)
            row = next(r for r in result['primary'] if r['protocol'] == 'leave-task-label-out' and r['recipe'] == 'progressive')
            self.assertIsNone(row['area'])
            self.assertEqual(row['status'], 'Pending')


if __name__ == '__main__':
    unittest.main()
