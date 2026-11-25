"""Protect the staged allocation and the structure-level analysis contract."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import reference60_protocol as reference
from experiments.l23net_analysis import repeatability_protocol as p
from experiments.l23net_analysis.analyze_l23net_repeatability import arm_summaries, compare_histories
from experiments.l23net_analysis.nci.submit_l23net_repeatability import reservation_for_wave, worker_command
from experiments.l23net_analysis.r1_protocol import save_json


def make_rows(effects):
    return [{'seed': seed, 'arm': arm, 'eligible': True, 'rate_safe': True,
             'benefit': value, 'late_benefit': value, 'excluded_benefit': value,
             'shift_alignment': value, 'washout_benefit': 0., 'band_log10_sham_minus_active': [0., 0., value]}
            for arm, values in zip(p.ACTIVE_ARMS, effects) for seed, value in zip(p.SEEDS, values)]


class RepeatabilityTests(unittest.TestCase):
    def test_one_history_per_structure_matched_across_arms(self):
        first, second = (p.jobs(wave) for wave in p.WAVES)
        self.assertEqual(len(first), 18)
        self.assertEqual(len(second), 18)
        for jobs in (first, second):
            self.assertEqual(len({job['history_seed'] for job in jobs}), 6)
            for seed in p.SEEDS:
                pair = [job for job in jobs if job['seed'] == seed]
                self.assertEqual(tuple(job['arm'] for job in pair), p.ARMS)
                self.assertEqual(len({job['history_seed'] for job in pair}), 1)
        self.assertFalse({job['history_seed'] for job in first} & {job['history_seed'] for job in second})
        self.assertEqual(tuple(p.load_archive()), p.SEEDS)

    def test_large_effect_on_one_structure_cannot_release_second_wave(self):
        rows = make_rows([[0., 0., 0., 4., 0., -.5], [-.2] * 6])
        first = arm_summaries(rows, p.load_archive())['dose_02']
        self.assertTrue(first['repeatability_checks']['practical_mean'])
        self.assertFalse(first['repeatability_checks']['leave_one_out'])
        self.assertFalse(first['passes_repeatability_rule'])

    def test_rejected_histories_stay_in_primary_denominator(self):
        rows = make_rows([[0., 0., .25, .6, .45, .4], [.1] * 6])
        for row in rows:
            if row['seed'] in p.SEEDS[:2]:
                row.update(eligible=False, benefit=0.)
        first = arm_summaries(rows, p.load_archive())['dose_02']
        self.assertEqual(first['all_candidate_benefit']['n_structures'], 6)
        self.assertEqual(first['eligible_benefit']['n_structures'], 4)
        self.assertFalse(first['repeatability_checks']['coverage'])
        self.assertFalse(first['passes_repeatability_rule'])

    def test_two_histories_are_averaged_before_inference(self):
        archive = p.load_archive()
        first, second = (make_rows([[value] * 6, [.1] * 6]) for value in (.4, .6))
        a = {'rows': first, 'arms': arm_summaries(first, archive)}
        b = {'rows': second, 'arms': arm_summaries(second, archive)}
        self.assertTrue(a['arms']['dose_02']['passes_repeatability_rule'])
        result = compare_histories(a, b)['by_dose']['dose_02']
        self.assertEqual(result['mean_of_two_new_histories']['n_structures'], 6)
        self.assertAlmostEqual(result['mean_of_two_new_histories']['mean'], .5)
        self.assertAlmostEqual(result['within_structure_sd_among_jointly_eligible'], .2 / 2**.5)

    def test_wave_two_requires_complete_frozen_positive_summary(self):
        value = {'wave': 1, 'status': 'history_complete', 'errors': [], 'completed_valid_runs': 18,
                 'protocol': p.PROTOCOL, 'code_sha256': p.code_hashes(),
                 'mechanism_sha256': dose.mechanism_hashes(), 'second_wave_allowed': False}
        with TemporaryDirectory() as folder:
            path = Path(folder) / p.SUMMARY
            save_json(path, reference.signed(value))
            with self.assertRaisesRegex(ValueError, 'Futility rule'):
                p.load_first_wave(path)
            value['second_wave_allowed'] = True
            save_json(path, reference.signed(value))
            self.assertTrue(p.load_first_wave(path)['second_wave_allowed'])
            value['completed_valid_runs'] = 17
            save_json(path, reference.signed(value))
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                p.load_first_wave(path)

    def test_pbs_budget_and_dependency(self):
        with self.assertRaisesRegex(ValueError, '45.928 KSU'):
            reservation_for_wave(1, 45.9)
        self.assertEqual(reservation_for_wave(1, 45.928), 44.928)
        manifest = {'wave': 1, 'max_concurrent': 8, 'commit': 'abc', 'jobs': p.jobs(1)}
        manifest['jobs'][0]['job_id'] = '246.gadi-pbs'
        first = worker_command(Path('/example'), manifest, 0)
        ninth = worker_command(Path('/example'), manifest, 8)
        self.assertNotIn('-W', first)
        self.assertIn('depend=afterok:246.gadi-pbs', ninth)
        self.assertEqual(ninth[ninth.index('-P') + 1], 'fa32')


if __name__ == '__main__':
    unittest.main()
