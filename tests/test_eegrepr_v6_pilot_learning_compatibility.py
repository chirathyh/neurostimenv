"""Exact-source ridge reuse and the production TCN completion/resume path."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from experiments.eeg_representation.h1.artifacts import file_hash
from experiments.eeg_representation.h1.v6_pilot import config,learning
from experiments.eeg_representation.h1.v6_pilot import learning_compatibility as compatibility
from tests.test_eegrepr_v6_pilot_learning import panel


def snapshots(folder):
    return {p.relative_to(folder).as_posix():(file_hash(p),p.stat().st_mtime_ns)
            for p in folder.rglob('*') if p.is_file()}


class LegacyRidgeCompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.panel,self.values,self.manifest=panel();self.config=config.resolve()
    def tearDown(self):self.temp.cleanup()
    def legacy(self):
        return learning.ridge(self.panel,self.values,self.manifest,self.root,self.config,
            signature_implementation_sha256=compatibility.LEGACY_SOURCE_SHA256)
    def resume(self):
        return compatibility.compatible_ridge(learning.ridge,self.panel,self.values,self.manifest,
            self.root,self.config,resume=True)

    def test_exact_approved_source_patch(self):
        record=compatibility.compatibility_record()
        self.assertEqual(record['legacy_learning_sha256'],compatibility.LEGACY_SOURCE_SHA256)
        self.assertEqual(record['current_learning_sha256'],file_hash(Path(learning.__file__)))
        self.assertNotEqual(record['legacy_learning_sha256'],record['current_learning_sha256'])
        self.assertFalse(record['tcn_legacy_resume'])

    def test_legacy_completed_resume_keeps_all_file_bytes_and_timestamps(self):
        expected=self.legacy();before=snapshots(self.root)
        with patch.object(learning,'fit_design',side_effect=AssertionError('must not refit')):
            actual=self.resume()
        self.assertEqual(expected,actual);self.assertEqual(before,snapshots(self.root))

    def test_changed_input_still_fails_without_overwrite(self):
        self.legacy();before=snapshots(self.root)
        self.values['development'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'inputs/settings changed'):self.resume()
        self.assertEqual(before,snapshots(self.root))

    def test_changed_settings_still_fail_without_overwrite(self):
        self.legacy();before=snapshots(self.root);self.config['ridge_alphas']=[.2,10.,1000.]
        with self.assertRaisesRegex(ValueError,'inputs/settings changed'):self.resume()
        self.assertEqual(before,snapshots(self.root))

    def test_corrupt_legacy_output_still_fails_without_overwrite(self):
        self.legacy();p=self.root/'predictions.npz';p.write_bytes(p.read_bytes()+b'corrupt')
        before=snapshots(self.root)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):self.resume()
        self.assertEqual(before,snapshots(self.root))

    def test_new_fits_and_current_resume_use_current_source_identity(self):
        expected=compatibility.compatible_ridge(learning.ridge,self.panel,self.values,self.manifest,self.root,self.config)
        settings=dict(alphas=self.config['ridge_alphas'],predeclared_alpha=self.config['state_candidate_alpha'],mechanism_scaled=False)
        self.assertEqual(expected['signature'],learning.signature(self.panel,self.values,self.manifest,settings))
        before=snapshots(self.root)
        with patch.object(compatibility,'compatibility_record',side_effect=AssertionError('current signature needs no override')):
            self.assertEqual(expected,self.resume())
        self.assertEqual(before,snapshots(self.root))

    def test_any_unrelated_source_edit_is_rejected(self):
        source=Path(learning.__file__).read_text();copy=self.root/'learning.py'
        for modified in (source+'\n# unrelated edit\n',source.replace("best=float('inf')","best=float('Inf')")):
            copy.write_text(modified)
            with self.assertRaisesRegex(ValueError,'Unapproved learner source'):
                compatibility.compatibility_record(copy)
        copy.write_text(source.replace('dict(account,status=', 'dict(status='))
        with self.assertRaisesRegex(ValueError,'Unapproved learner source'):
            compatibility.compatibility_record(copy)

    def test_unapproved_source_cannot_resume_legacy(self):
        self.legacy();before=snapshots(self.root)
        with patch.object(compatibility,'compatibility_record',side_effect=ValueError('Unapproved learner source')):
            with self.assertRaisesRegex(ValueError,'Unapproved learner source'):self.resume()
        self.assertEqual(before,snapshots(self.root))


class CorrectedTCNCompletionTests(unittest.TestCase):
    def test_cpu_short_and_full_history_save_and_resume_without_refit(self):
        p,_,manifest=panel();rng=np.random.default_rng(19118)
        for split,d in p.items():
            level=d['X'][:,0];x=level[:,None]+rng.normal(0,.05,(len(level),2000))
            y=level[:,None]*(.4+.2*np.cos(np.linspace(0,2*np.pi,64,endpoint=False)))[None,:]
            d.update(X=x,X_physical=x,Y=y[:,None,:],Y_physical=y[:,None,:])
        manifest=deepcopy(manifest);manifest['risk_contract']['target']='synthetic64'
        with tempfile.TemporaryDirectory() as folder:
            for history,steps in ((64,25),(2000,10)):
                with self.subTest(history=history):
                    destination=Path(folder)/str(history);c=config.resolve({'neural_max_steps':steps})
                    result=learning.tcn(p,manifest,destination,c,history,device='cpu')
                    self.assertEqual(result['status'],'complete')
                    self.assertEqual(result['history_samples'],history)
                    self.assertEqual(result['device'],'cpu')
                    self.assertEqual(result['selection_split'],'development')
                    self.assertLessEqual(result['optimizer_updates'],steps)
                    before=snapshots(destination)
                    with patch.object(learning.v6,'train_tcn',side_effect=AssertionError('resume must not fit')):
                        resumed=learning.tcn(p,manifest,destination,c,history,device='cpu',resume=True)
                    self.assertEqual(result,resumed);self.assertEqual(before,snapshots(destination))
                    predictions=learning.numeric(destination/'predictions.npz')
                    for split,d in p.items():
                        self.assertEqual(predictions[split+'_prediction'].shape,(len(d['X']),64))
                        self.assertTrue(np.isfinite(predictions[split+'_prediction']).all())


if __name__=='__main__':unittest.main()
