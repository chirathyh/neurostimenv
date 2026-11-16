"""Report claims must follow measured numeric records, not a planned cohort."""
import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
from tests.test_eegrepr_v6_models import panel_fixture
from experiments.eeg_representation.h1.v6 import models,information,extended_reporting
from experiments.eeg_representation.h1.v6.artifacts import read_json

class ReadableEvidenceTests(unittest.TestCase):
    def test_report_keeps_unknowns_physical_risks_and_cohort_status(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p,m,c,job=panel_fixture(root,length=2000,futures=4)
            models.fit(c,root);information.run(c,root)
            extended_reporting.report(c,root)
            record=read_json(root/'report/main_analysis.json');s=record['information_tables'][0]
            rows={r['id']:r for r in s['rows']};self.assertIsNone(rows['eeg_ideal']['physical_mse'])
            self.assertEqual(rows['complete_state_exact_future_theory']['status'],'theoretical')
            self.assertEqual(rows['train_mean']['error_baseline100'],100)
            self.assertEqual(rows['eeg_full_selected']['n_scoring_episodes'],len(p['evaluation']['X'])*4)
            with (root/'report/tables/information_error.csv').open() as f:csvrows=list(csv.DictReader(f))
            emitted=next(r for r in csvrows if r['id']=='eeg_full_selected')
            self.assertEqual(float(emitted['physical_mse']),rows['eeg_full_selected']['physical_mse'])
            self.assertEqual(next(r for r in csvrows if r['id']=='eeg_ideal')['physical_mse'],'')
            text=(root/'report/MAIN_ANALYSIS.md').read_text();self.assertIn('Not run in this evidence root',text)
            self.assertIn('linear-only local analysis',text);self.assertIn('first four registered futures',text)
            self.assertEqual(text,(root/'report/FINDINGS.md').read_text())
            self.assertTrue((root/'report/figures/target_timing.pdf').stat().st_size>1000)
            self.assertTrue((root/'report/figures/information_error.svg').stat().st_size>1000)
            self.assertFalse((root/'report/figures/h2_finite_reference.pdf').exists())
            self.assertEqual(record['completed_information_jobs'],1)

    def test_unrun_report_has_design_figure_and_no_invented_measured_results(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);c={'claim_status':'prospective_confirmation','selected_job_ids':['A0__T0_l0_d128'],'models':{'neural_enabled':True}}
            result=extended_reporting.report(c,root)
            self.assertEqual(result['completed_information_jobs'],0)
            record=read_json(root/'report/main_analysis.json')
            self.assertTrue(all(r['status']=='not yet run' for r in record['completion']))
            self.assertFalse((root/'report/figures/information_error.pdf').exists())
            self.assertIn('No measured forecast outcomes',(root/'report/manuscript_results.md').read_text())

if __name__=='__main__':unittest.main()
