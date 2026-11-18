"""Real scoring producers and report display validation share float64 identities.

All data are synthetic numeric arrays. No simulator, fitting, or historical
scientific artifact is changed. Supplied unrounded metrics must survive reporting.
"""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.eeg_representation.h1.v6.information import summarize
from experiments.eeg_representation.h1.v6_pilot import analysis, reporting


def _production_row(target, scale, offset, shift):
    dimension = 64 if target.startswith('T0_') else 1
    rng = np.random.default_rng(212)
    cids = np.asarray(['s40-p0000-R0', 's40-p0000-Ralpha',
                       's41-p0000-R0', 's41-p0000-Ralpha'])
    y = (.3 + rng.uniform(.1, 1.1, size=(4, 8, dimension))) * scale + offset
    evaluation = dict(Y_physical=y, context_id=cids,
        structure_id=np.asarray([40, 40, 41, 41]),
        regime=np.asarray(['R0', 'Ralpha', 'R0', 'Ralpha']),
        future_id=np.asarray([[cid + f'-score-{future:04d}' for future in range(8)]
                              for cid in cids]), future_mask=np.ones((4, 8), bool))
    predicted = np.full((4, dimension), shift)
    baseline = np.zeros(dimension)
    manifest = dict(normalization=dict(y_scale=scale, y_offset=[offset] * dimension))
    options = dict(score_futures=8, descriptive_confidence=.95)
    row, _ = analysis._prediction_row(options, dict(evaluation=evaluation), manifest,
        'A1__' + target, 'synthetic_fixed_prediction', predicted, baseline,
        'Synthetic numeric population; no simulation or fitting', [])
    direct, _ = summarize(y, predicted * scale + offset, baseline * scale + offset,
        evaluation, method_id='synthetic_fixed_prediction', confidence=.95)
    return row, direct


def _display_row(reference, ratio):
    risk = float(reference * ratio)
    return dict(job='synthetic_numeric_case', method='fixed_prediction',
        status='measured', evidence_type='fitted', risk_physical=risk,
        reference_risk_physical=float(reference), error_baseline100=100 * risk / reference,
        relative_mse_reduction=100 * (1 - risk / reference))


def _assert_float_preserved(test, actual, expected):
    # hex also detects a changed signed zero; JSON roundtrip preserves float64.
    test.assertEqual(float(actual).hex(), float(expected).hex())


class PilotReportMetricTests(unittest.TestCase):
    def test_actual_scoring_producers_accept_zero_and_small_signed_gains_at_all_units(self):
        cases = [('T0_l0_d128', 1e-9, 0., 'V^2'),
                 ('T4_energy_alpha_l0_d1000', 1e-19, 1e-18, 'V^4'),
                 ('T4_log_energy_alpha_l0_d1000', .2, -48.,
                  'dimensionless squared log units')]
        for target, scale, offset, units in cases:
            for shift in (0., -1e-8, 1e-8):
                with self.subTest(target=target, shift=shift):
                    row, direct = _production_row(target, scale, offset, shift)
                    self.assertEqual(row['risk_units'], units)
                    self.assertEqual(row['risk_physical'], direct['physical_mse'])
                    self.assertEqual(row['reference_risk_physical'], direct['reference_mse'])
                    self.assertEqual(row['relative_mse_reduction'], direct['relative_mse_reduction_pp'])
                    self.assertEqual(np.sign(row['relative_mse_reduction']), np.sign(shift))
                    original = deepcopy(row)
                    result = reporting.normalize_results(dict(tables=dict(h1_information=[row])))
                    normalized = result['tables']['h1_information'][0]
                    for key in ('risk_physical', 'reference_risk_physical', 'error_baseline100',
                                'relative_mse_reduction'):
                        _assert_float_preserved(self, normalized[key], original[key])
                    _assert_float_preserved(self, normalized['relative_mse_reduction_pp'],
                                           original['relative_mse_reduction'])
                    self.assertEqual(row, original, 'Normalization must not mutate source records')
                    # Also exercise canonical producer field names directly, including
                    # physical-MSE alias migration and no loss of nested diagnostics.
                    direct_row = dict(direct, reference_risk_physical=direct['reference_mse'])
                    direct_result = reporting.normalize_results(
                        dict(tables=dict(h4_targets=[direct_row])))['tables']['h4_targets'][0]
                    _assert_float_preserved(self, direct_result['relative_mse_reduction_pp'],
                                           direct['relative_mse_reduction_pp'])
                    self.assertEqual(direct_result['conditional_mc'], direct['conditional_mc'])

    def test_known_baseline_cancellation_and_adjacent_ratios_preserve_supplied_values(self):
        cases = [1.234567891234e-18, 5.934503211004323e-19, 1.3e-40, 2.]
        saw_subtraction_mismatch = False
        for reference in cases:
            for ratio in (1., np.nextafter(1., 0.), np.nextafter(1., 2.), 1. - 1e-8, 1. + 1e-8):
                with self.subTest(reference=reference, ratio=ratio):
                    row = _display_row(reference, ratio)
                    saw_subtraction_mismatch |= row['relative_mse_reduction'] != 100 - row['error_baseline100']
                    output = reporting.normalize_results(dict(tables=dict(h1_information=[row])))
                    actual = output['tables']['h1_information'][0]
                    _assert_float_preserved(self, actual['error_baseline100'], row['error_baseline100'])
                    _assert_float_preserved(self, actual['relative_mse_reduction_pp'], row['relative_mse_reduction'])
        self.assertTrue(saw_subtraction_mismatch, 'Fixture must reproduce cancellation in the old validator')

    def test_genuine_sign_percentage_and_tiny_unit_mismatches_are_rejected(self):
        for reference in (1.3e-40, 1.234567891234e-18, 2.):
            for ratio in (.9, 1.1, 1. - 1e-8):
                valid = _display_row(reference, ratio)
                changes = [dict(relative_mse_reduction=-valid['relative_mse_reduction']),
                           dict(relative_mse_reduction=100 * valid['relative_mse_reduction']),
                           dict(error_baseline100=valid['error_baseline100'] / 100),
                           dict(reference_risk_physical=reference * 1.01),
                           dict(risk_physical=valid['risk_physical'] * 1.01)]
                for changed in changes:
                    with self.subTest(reference=reference, ratio=ratio, changed=changed):
                        with self.assertRaisesRegex(ValueError, 'mismatch'):
                            reporting.normalize_results(dict(tables=dict(
                                h1_information=[dict(valid, **changed)])))

    def test_every_table_normalizes_idempotently_and_keeps_unrounded_numeric_exports(self):
        row = _display_row(1.234567891234e-18, 1.)
        source = dict(tables={name: [deepcopy(row)] for name in reporting.TABLES})
        original = deepcopy(source)
        once = reporting.normalize_results(source)
        twice = reporting.normalize_results(once)
        self.assertEqual(once, twice)
        self.assertEqual(source, original)
        with tempfile.TemporaryDirectory() as temporary:
            reporting.write_tables(temporary, once)
            for name in reporting.TABLES:
                with self.subTest(table=name):
                    path = Path(temporary)
                    saved = json.loads((path / (name + '.json')).read_text())[0]
                    with (path / (name + '.csv')).open() as stream:
                        csv_row = next(csv.DictReader(stream))
                    with np.load(path / (name + '.npz'), allow_pickle=False) as arrays:
                        for field in ('risk_physical', 'reference_risk_physical', 'error_baseline100'):
                            _assert_float_preserved(self, saved[field], row[field])
                            _assert_float_preserved(self, float(csv_row[field]), row[field])
                            _assert_float_preserved(self, arrays[field][0], row[field])
                        _assert_float_preserved(self, arrays['relative_mse_reduction_pp'][0],
                                               row['relative_mse_reduction'])

    def test_expanded_history_comparison_is_idempotent_with_near_baseline_values(self):
        full = _display_row(1.234567891234e-18, 1.)
        short = _display_row(1.234567891234e-18, 1. + 1e-8)
        contrast = dict(job='A1__T0_l0_d128', population='synthetic',
            short_error=short['error_baseline100'], full_error=full['error_baseline100'],
            relative_mse_reduction=full['relative_mse_reduction'],
            paired_difference=full['relative_mse_reduction'] - short['relative_mse_reduction'])
        once = reporting.normalize_results(dict(tables=dict(h1_comparisons=[contrast])))
        twice = reporting.normalize_results(once)
        self.assertEqual(once, twice)
        expanded = once['tables']['h1_comparisons']
        self.assertEqual([r['history'] for r in expanded], ['short', 'full'])
        self.assertEqual([r['error_baseline100'] for r in expanded],
                         [short['error_baseline100'], full['error_baseline100']])


if __name__ == '__main__':
    unittest.main()
