"""Staged CLI inference/report/export integration with cheap, explicit numeric fixtures."""
import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
from tests.test_eegrepr_v6_models import panel_fixture
from experiments.eeg_representation.h1.v6 import cli,config,models,retention,scoring,reporting,exporting
from experiments.eeg_representation.h1.v6.artifacts import initialize
from experiments.eeg_representation.h1.v4.restore_export import verify_archive,restore
from experiments.eeg_representation.storage import atomic_json,read_json,file_hash


class IntegrationTests(unittest.TestCase):
    def test_staged_cli_lossless_expert_export(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);root=base/'evidence';job='A0__T0_l10_d128'
            c=config.resolve(dict(claim_status='engineering',backend='analytical',
                data=dict(structures=dict(train=[100,101,102,103],development=[104,105],calibration=[106,107],evaluation=[108,109]),
                    prefixes_per_regime=dict(train=2,development=2,calibration=2,evaluation=2)),
                forks=dict(score_futures=4,high_score_futures=4,reference_futures=0,reference_circuits=[]),
                reuse=dict(enabled=False),acquisition_ids=['A0'],selected_job_ids=[job],
                models=dict(neural_enabled=False,run_controls=False,full_history_samples=64,pca_width=8),
                retention=dict(legacy_forecast=False,dimensions=[8],code_bits=[32],range_rules=['quantile_005_995']),
                protocol=dict(primary_job_id=job,test_circuits=2)))
            cfg=base/'fixture.json';atomic_json(cfg,c);initialize(root,c)
            panel,manifest,_,_=panel_fixture(root,futures=5)
            def available(splits):
                manifest['files']={s:models._record(root,root/'derived'/job/(s+'.npz')) for s in splits}
                atomic_json(root/'derived'/job/'manifest.json',manifest)
                atomic_json(root/'derived/manifest.json',dict(status='complete',jobs=[dict(id=job,status='complete',**models._record(root,root/'derived'/job/'manifest.json'))]))
            def stage(mode,**extra):
                args=cli.parser().parse_args([mode,'--root',str(root),'--config',str(cfg),'--resume'])
                for key,value in extra.items():setattr(args,key,value)
                return cli.stage(args)
            available(['train','development']);stage('fit')
            selection=root/'models'/job/'selection.json';mean_hash=file_hash(selection)
            available(['train','development','calibration']);stage('predict');stage('calibrate')
            available(['train','development','calibration','evaluation']);stage('predict')
            self.assertEqual(mean_hash,file_hash(selection))
            stage('score');stage('retention')
            # Exercise the confirmation math using a labelled unit fixture. The
            # public CLI correctly refuses fresh-confirmation claims for engineering.
            atomic_json(root/'protocol.json',dict(status='frozen',protocol_hash='unit-fixture-not-scientific-evidence'))
            confirmation=copy.deepcopy(c);confirmation['claim_status']='prospective_confirmation'
            r=scoring.confirm_statistics(confirmation,root)
            self.assertEqual(r['familywise_alpha'],.05)
            with self.assertRaisesRegex(ValueError,'Inspected circuits'):stage('confirm-statistics')
            report=stage('report');self.assertEqual(report['claim_status'],'engineering')
            self.assertTrue((root/'report/forecast_grid.pdf').is_file())
            self.assertTrue((root/'report/task_reconstruction_contrast.pdf').is_file())
            archive=base/'compact.zip';export=stage('export',destination=str(archive))
            self.assertEqual(export['status'],'complete')
            checked=verify_archive(archive);self.assertTrue(checked['npy_bytes_exact'])
            self.assertGreater(checked['transformed_npz_files'],0)
            self.assertTrue(any(x['path'].endswith('/state.npz') for x in checked['manifest']['omitted']))
            self.assertFalse(any(x['path'].endswith('/state.npz') for x in checked['manifest']['included']))
            restored=base/'restored';restore(archive,restored)
            originals=[
                'derived/'+job+'/evaluation.npz',
                'models/'+job+'/train_mean/evaluation.npz',
                'calibration/'+job+'/train_mean/variance.npz',
                'retention/banks/feature18/d8_B32_quantile_005_995/codes.npz',
                'retention/banks/feature18/d8_B32_quantile_005_995/carrier/labels_all/predictions.npz',
                'scores/'+job+'/train_mean__risk_components.npz']
            for relative in originals:
                with zipfile.ZipFile(root/relative) as a,zipfile.ZipFile(restored/'evidence'/relative) as b:
                    self.assertEqual(set(a.namelist()),set(b.namelist()))
                    for name in a.namelist():self.assertEqual(a.read(name),b.read(name))
            self.assertEqual(read_json(restored/'evidence/report/findings.json')['claim_status'],'engineering')

    def test_readout_freeze_precedes_evaluation_access_and_no_test_tuning(self):
        with tempfile.TemporaryDirectory() as temporary:
            base=Path(temporary);panel,manifest,_,_=panel_fixture(base/'arrays')
            panel={s:panel[s] for s in retention.SPLITS};panel['manifest']=manifest
            for s in retention.SPLITS:panel[s]['task_four_class']=panel[s]['task_carrier']+1
            values={s:np.eye(4)[panel[s]['task_four_class']] for s in retention.SPLITS}
            destination=base/'readout'
            class Guarded(dict):
                def __getitem__(self,key):
                    self_outer.assertTrue((destination/'selection.json').exists(),'evaluation accessed before mean/readout freeze')
                    return super().__getitem__(key)
            self_outer=self;original=panel['evaluation'];panel['evaluation']=Guarded(original)
            first=retention.evaluate_readout(panel,values,'four_class','all',[.01,1.,100.],destination)
            panel['evaluation']=copy.deepcopy(original)
            panel['evaluation']['task_four_class']=(panel['evaluation']['task_four_class']+1)%4
            values['evaluation']=values['evaluation']*2+50.
            second=retention.evaluate_readout(panel,values,'four_class','all',[.01,1.,100.],base/'changed_test')
            self.assertEqual(first['selected_alpha'],second['selected_alpha'])
            with np.load(destination/'readout_weights.npz',allow_pickle=False) as a,np.load(base/'changed_test/readout_weights.npz',allow_pickle=False) as b:
                for key in a.files:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
