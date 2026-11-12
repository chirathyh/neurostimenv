"""Phase qualifications, independent numeric rescoring and portable source export."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from experiments.eeg_representation.h1 import artifacts, reporting
from experiments.eeg_representation.h1.scoring import score_panel
from experiments.eeg_representation.storage import atomic_json, file_hash
from tests.test_eegrepr_h1_compression import panel_fixture


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='h1-reporting-')
        self.base=Path(self.temp.name);self.root=self.base/'evidence';self.root.mkdir()
        self.config={'name':'fixture','backend':'analytical','claim_status':'engineering','jobs':[{'id':'fixture'}]}
        panel=panel_fixture();ev=panel['evaluation'];self.data=ev
        mean=np.zeros((len(ev['X']),ev['Y'].shape[-1]));variance=np.ones_like(mean)
        report,arrays=score_panel(ev['Y'],mean,variance,np.zeros(mean.shape[1]),ev['structure_id'],
            future_mask=ev['future_mask'],target_fingerprint='same_target',draws=4)
        arrays.update(context_id=ev['context_id'],structure_id=ev['structure_id'],future_id=ev['future_id'])
        directory=self.root/'benchmarks/fixture';self.raw=directory/'scores/mean.npz'
        artifacts.atomic_npz(self.raw,**arrays)
        report.update(model='mean',prediction_file='scores/mean.npz')
        self.report=dict(schema_version='eegrepr-h1-benchmark-v2',target_fingerprint='same_target',models=[report],
            artifacts={'scores/mean.npz':file_hash(self.raw)},
            diagnostics={'few_example_overfit':{'passed':False,'ratio':.8}})
        atomic_json(directory/'summary.json',self.report)
        atomic_json(self.root/'benchmarks/manifest.json',dict(schema_version='eegrepr-h1-benchmark-index-v2',completed_jobs=1,
            jobs=[{'job_id':'fixture','path':'benchmarks/fixture/summary.json'}]))
        atomic_json(self.root/'panels/manifest.json',{'completed_jobs':1,'planned_jobs':1,'jobs':[{'job_id':'fixture'}]})
        atomic_json(self.root/'controls/controls.json',{'passed':True})
        atomic_json(self.root/'replication/summary.json',{'status':'executed available analyses; inspect pending references/allocations',
            'jobs':[{'reference_status':'pending reference RMSE: independent reference absent',
                     'pooled_one_future':{'status':'pending: insufficient training circuits'}}]})
        atomic_json(self.root/'compression/summary.json',{'status':'complete','executed_jobs':['fixture'],
            'rows':[{'status':'unsupported','reason':'constant training class'}]})

    def tearDown(self):self.temp.cleanup()

    def test_phase_presence_does_not_hide_pending_and_failed_diagnostics(self):
        phases=reporting.phase_summaries(self.config,self.root)
        self.assertEqual(phases['P1_exact_controls']['status'],'complete')
        self.assertEqual(phases['P1_matched_benchmark']['status'],'complete_with_qualifications')
        self.assertEqual(phases['P1_matched_benchmark']['status_record_counts']['failed_diagnostic'],1)
        self.assertGreaterEqual(phases['P2_replication']['status_record_counts']['pending'],2)
        self.assertEqual(phases['P3_compression']['status_record_counts']['unsupported'],1)
        self.assertEqual(phases['P4_scale']['status'],'not_run')
        report=reporting.summarize(self.config,self.root)
        self.assertTrue(report['raw_reaggregation_checks'][0]['mse_recomputed_from_truth_and_mean'])
        self.assertEqual(report['n_per_circuit_model_rows'],2)

    def test_rescore_catches_prediction_tamper_even_if_file_hash_updated(self):
        with np.load(self.raw,allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
        arrays['mean']+=1
        artifacts.atomic_npz(self.raw,**arrays)
        self.report['artifacts']['scores/mean.npz']=file_hash(self.raw)
        atomic_json(self.root/'benchmarks/fixture/summary.json',self.report)
        with self.assertRaisesRegex(ValueError,'numeric predictions'):
            reporting.reaggregate_numeric(self.root)

    def test_export_source_closure_and_subprocess_rescore_without_checkout(self):
        native=self.root/'native/example.h5';native.parent.mkdir();native.write_bytes(b'raw fixture not a real HDF5')
        output=self.base/'review.zip'
        result=reporting.export_evidence(self.config,self.root,output)
        self.assertEqual(result['sha256'],file_hash(output))
        extracted=self.base/'extracted'
        with zipfile.ZipFile(output) as archive:
            names=set(archive.namelist())
            self.assertNotIn('native/example.h5',names)
            for name in ('source/configs/env/ballnstick.yaml','source/experiments/eeg_representation/h1/reporting.py',
                         'source/requirements.txt','source/experiments/__init__.py','RESCORE_README.txt'):
                self.assertIn(name,names)
            self.assertTrue(any(n.startswith('source/setup/circuits/ballnstick/') and n.endswith('.hoc') for n in names))
            manifest=json.loads(archive.read('EVIDENCE_MANIFEST.json'))
            self.assertEqual(manifest['phase_status']['P3_compression']['status_record_counts']['unsupported'],1)
            archive.extractall(extracted)
        expected=artifacts.source_hash()
        with patch.object(artifacts,'ROOT',extracted/'source'):
            self.assertEqual(artifacts.source_hash(),expected)
        environment=dict(os.environ,PYTHONPATH=str(extracted/'source'))
        command=[sys.executable,'-m','experiments.eeg_representation.h1.reporting','--rescore-root','.',
                 '--output','independent_rescore.json']
        executed=subprocess.run(command,cwd=extracted,env=environment,text=True,capture_output=True,check=False)
        self.assertEqual(executed.returncode,0,executed.stderr)
        report=json.loads((extracted/'independent_rescore.json').read_text())
        self.assertEqual(report['status'],'complete')
        self.assertTrue(report['checks'][0]['mse_recomputed_from_truth_and_mean'])
        again=subprocess.run(command,cwd=extracted,env=environment,text=True,capture_output=True,check=False)
        self.assertNotEqual(again.returncode,0)

    def test_export_cannot_claim_absent_phases_complete_or_overwrite(self):
        empty=self.base/'empty';empty.mkdir()
        output=self.base/'empty.zip'
        reporting.export_evidence(self.config,empty,output)
        with zipfile.ZipFile(output) as archive:
            manifest=json.loads(archive.read('EVIDENCE_MANIFEST.json'))
        self.assertTrue(all(v['status']=='not_run' for v in manifest['phase_status'].values()))
        self.assertNotIn('contains_all_panel_targets_predictions_and_codes',manifest)
        with self.assertRaises(FileExistsError):reporting.export_evidence(self.config,empty,output)


if __name__=='__main__':unittest.main()
