"""Synthetic report-to-receipt-to-export integration; zero simulator calls.

The temporary engineering fixture supplies 60 job identities and synthetic gate
artifacts solely to exercise the completion handshake. Changing its purpose tag
is a unit-test setup, not evidence of a real completed experiment. All temporary
files are deleted, and every fixture finding explicitly identifies synthetic data.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from experiments.eeg_representation.h1.v6.evidence import EvidenceSource
from experiments.eeg_representation.h1.v6_pilot import cli, config, exporting, reporting
from experiments.eeg_representation.h1.v4.restore_export import verify_archive
from tests.test_eegrepr_v6_pilot_reporting import engineering_fixture


class PilotReportHandoffIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(reporting.dependencies()['status'] == 'complete',
                         'Complete report/PDF rendering dependencies unavailable')
    def test_real_report_verification_and_export_preserve_producer_metrics(self):
        with tempfile.TemporaryDirectory(prefix='eegrepr-synthetic-report-handoff-') as temporary:
            base = Path(temporary)
            root = base / 'synthetic_fixture'
            results = engineering_fixture(root)
            # Only the in-memory fixture's tag changes so the real completion
            # predicate is exercised. No fabricated data leave this temp directory.
            results['purpose'] = 'exploratory_reporting_pilot'
            results['data_origin'] = 'UNIT TEST: synthetic engineering records, not experimental evidence'
            results['findings_notice'] = 'Every number is an explicitly synthetic test fixture'
            self.assertNotIn('learner_protocol', results)
            self.assertEqual(len({r['job'] for r in results['tables']['h1_information']}), 60)
            options = config.resolve()
            selected = [r for r in results['tables']['h1_information']
                        if r['method'] == 'eeg_full'][:3]
            for row, ratio in zip(selected, (1., 1. - 1e-8, 1. + 1e-8)):
                reference = 1.234567891234e-18
                risk = reference * ratio
                row.update(risk_physical=risk, reference_risk_physical=reference,
                    error_baseline100=100 * risk / reference,
                    relative_mse_reduction=100 * (1 - risk / reference),
                    state_variance_physical=reference * .892,
                    data_origin='Synthetic arithmetic regression, no fitted model')
            # This exact baseline is the former production failure: division and
            # multiplication round error slightly above 100 but gain remains zero.
            self.assertNotEqual(selected[0]['relative_mse_reduction'],
                                100 - selected[0]['error_baseline100'])
            raw = json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + '\n'
            (root / 'pilot_results.json').write_text(raw)
            (root / 'pilot_config.json').write_text(json.dumps(options))
            (root / 'pilot_provenance.json').write_text(json.dumps(dict(
                purpose='synthetic_engineering_unit_test', neuron_invocations=0)))
            expected = deepcopy(results)
            receipt = reporting.report(root, options, results)
            self.assertEqual(results, expected, 'Reporting must not mutate its numerical input')
            self.assertEqual((root / 'pilot_results.json').read_text(), raw)
            self.assertEqual(receipt['status'], 'complete')
            self.assertTrue(receipt['experimental_completion_eligible'])
            self.assertTrue(receipt['pdf_verification']['all_pages_rendered'])
            self.assertEqual(receipt['pdf_verification']['errors'], [])
            self.assertTrue(all(page.get('render_sha256') for page in receipt['pdf_verification']['pages']))
            # Invoke the actual CLI verifier: no mocks, no substituted hashes or
            # fake PDF receipt. This catches config-enrichment hash divergence.
            self.assertEqual(cli.verify_report(root, options, receipt), receipt)
            prepared = reporting.prepare_report_results(results, options)
            self.assertEqual(receipt['results_hash'], reporting._canonical(prepared))
            self.assertEqual(receipt['config_hash'], reporting._canonical(options))
            self.assertIn('learner_protocol', prepared)
            self.assertNotIn('learner_protocol', json.loads((root / 'pilot_results.json').read_text()))
            for original in selected:
                row = next(r for r in prepared['tables']['h1_information']
                           if r.get('job') == original['job'] and r.get('method') == original['method'])
                for name in ('risk_physical', 'reference_risk_physical', 'error_baseline100'):
                    self.assertEqual(float(row[name]).hex(), float(original[name]).hex())
                self.assertEqual(float(row['relative_mse_reduction_pp']).hex(),
                                 float(original['relative_mse_reduction']).hex())
            # Existing valid PDF/receipt may not endorse altered settings or results.
            changed_config = dict(options, neural_max_steps=options['neural_max_steps'] - 1)
            with self.assertRaisesRegex(ValueError, 'different numerical results or configuration'):
                cli.verify_report(root, changed_config, receipt)
            changed_results = deepcopy(results)
            changed_results['tables']['findings'][0]['observed'] += ' altered after report'
            (root / 'pilot_results.json').write_text(json.dumps(changed_results))
            with self.assertRaisesRegex(ValueError, 'different numerical results or configuration'):
                cli.verify_report(root, options, receipt)
            (root / 'pilot_results.json').write_text(raw)
            self.assertEqual(cli.verify_report(root, options, receipt), receipt)
            destination = base / 'synthetic-evidence.zip'
            exported = exporting.export(root, destination)
            self.assertEqual(exported['status'], 'complete')
            self.assertEqual(verify_archive(destination)['status'], 'complete')
            self.assertTrue(destination.with_suffix('.zip.sha256').is_file())
            with EvidenceSource(destination) as evidence:
                self.assertEqual(evidence.read_json('pilot_results.json'), results)
                self.assertEqual(evidence.read_json('report/report_input.json'), prepared)
                self.assertEqual(evidence.read_json('report/receipt.json'), receipt)
                self.assertEqual(evidence.read_bytes('report/pilot_findings.pdf'),
                                 (root / 'report/pilot_findings.pdf').read_bytes())
                for table in reporting.TABLES:
                    name = 'report/tables/' + table + '.npz'
                    restored = evidence.read_npz(name)
                    with np.load(root / name, allow_pickle=False) as local:
                        self.assertEqual(set(restored), set(local.files))
                        for key in local.files:
                            self.assertEqual(restored[key].dtype, local[key].dtype)
                            self.assertEqual(restored[key].shape, local[key].shape)
                            np.testing.assert_array_equal(restored[key], local[key])
                self.assertEqual(evidence.verify_all()['status'], 'complete')


if __name__ == '__main__':
    unittest.main()
