"""Synthetic numeric tests; no native simulator is invoked."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import precision
from experiments.eeg_representation.h1.artifacts import atomic_npz
from experiments.eeg_representation.storage import atomic_json,write_trial,file_hash
from experiments.eeg_representation.h1.v6.targets import extract_physical_target


def native(values,structure=1):
    values=np.asarray(values,float);future=np.repeat(values,32);past=np.zeros(2048)
    eeg=np.r_[past,future][:,None];t=6000-128+np.arange(1,len(eeg)+1)*.0625
    return dict(time_ms=t,eeg_v=eeg,dipole_na_um=np.zeros((len(t),3)),privileged={},
        metadata=dict(fork=dict(cutoff_ms=6000.),rng_namespaces=dict(structure=structure)))


class PilotPrecisionTests(unittest.TestCase):
    def test_h2_uses_actual_l32_keeps_m1_unavailable_and_negative_corrected_values(self):
        rng=np.random.default_rng(15)
        context=dict(context_id='s1-p0000-R0',score=[(f's1-p0000-R0-score-{i:04d}',native(rng.normal(size=64)*1e-9)) for i in range(32)],
            reference=[(f's1-p0000-R0-reference-{i:04d}',native(rng.normal(size=64)*1e-9)) for i in range(32)])
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);jobs=['A0__T0_l0_d128','A1__T0_l0_d128']
            for job in jobs:atomic_json(root/'matched/derived'/job/'manifest.json',dict(normalization=dict(y_scale=1e-9,y_offset=[0.]*64,id='same-norm'),target_fingerprint='same-target'))
            with patch.object(precision.generation,'h2_trials',return_value=iter([context])):
                out=precision.finite_future_tables(dict(h2_m=[1,2,4,8,16,32],h2_bootstrap=10),root,jobs)
            self.assertEqual(len(out['h2_precision']),12)
            for r in out['h2_precision']:
                self.assertEqual(r['L'],32);self.assertFalse(r['reference_is_truth']);self.assertTrue(r['groups_share_reference'])
                self.assertEqual(r['variance_error'] is None,r['M']==1)
                self.assertAlmostEqual(r['mean_error_normalized'],r['mean_error_corrected']/1e-18)
                self.assertEqual(r['variance_error_units'],'(V^2)^2')
            self.assertEqual(len(out['figures']['same_state_futures']['futures']),32)
            for a,b in zip(out['h2_precision'][:6],out['h2_precision'][6:]):self.assertEqual(a['mean_error_corrected'],b['mean_error_corrected'])
            # M32 is one fixed full pool; its uncertainty is still finite-reference uncertainty.
            self.assertTrue(out['h2_precision'][-1]['full_pool_does_not_mean_zero_mc_uncertainty'])

    def test_replay_comparators_use_exact_same_single_future_population(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);job='A1__T0_l0_d128';cids=np.array(['s1-p0000-R0','s2-p0000-Ralpha']);tids=np.array([x+'-score-0001' for x in cids])
            trials=[native(np.linspace(0,1,64)*1e-9,1),native(np.linspace(1,2,64)*1e-9,2)];rows=[];audits=[];physical=[]
            for cid,tid,trial in zip(cids,tids,trials):
                original=root/'original'/str(tid)/'trial.h5';replica=root/'replicas'/str(tid)/'trial.h5'
                for p in (original,replica):write_trial(p,trial,dict(metadata=trial['metadata']))
                rows.append(dict(trial_id=str(tid),context_id=str(cid),structure_id=int(trial['metadata']['rng_namespaces']['structure']),source_root=str(original.parent),path=original.name,bytes=original.stat().st_size,sha256=file_hash(original)))
                audits.append(dict(context_id=str(cid),original_trial_id=str(tid),replica_path=str(replica),replay_native_sha256=file_hash(replica)))
                physical.append(extract_physical_target(trial['eeg_v'],trial['time_ms'],'T0_l0_d128'))
            atomic_json(root/'replay_controls.json',dict(audits=audits));atomic_json(root/'design.json',dict(historical_rows=rows))
            y=np.asarray(physical);evalp=dict(context_id=cids,structure_id=np.array([1,2]),future_id=np.stack([np.array([c+'-score-0000',t]) for c,t in zip(cids,tids)]),future_mask=np.ones((2,2),bool),Y_physical=np.stack([y,y],axis=1))
            train=dict(context_id=np.array(['s3-p0000-R0','s4-p0000-Ralpha']),structure_id=np.array([3,4]),regime=np.array(['R0','Ralpha']),Y=np.zeros((2,1,64)),future_mask=np.ones((2,1),bool))
            panel=dict(train=train,evaluation=evalp);manifest=dict(normalization=dict(y_scale=1e-9,y_offset=[0.]*64,id='norm'),target_fingerprint='target',risk_contract={'target':'same'})
            directory=root/'matched/learners'/job/'complete_state';preds=dict(evaluation_context_id=cids,evaluation_prediction=y/1e-9*.5,evaluation_candidate_prediction=y/1e-9*.25,risk_contract_json=np.asarray(json.dumps(manifest['risk_contract'])))
            atomic_npz(directory/'predictions.npz',**preds);atomic_json(directory/'selection.json',dict(files={'predictions.npz':file_hash(directory/'predictions.npz')}))
            with patch.object(precision,'load_panel',return_value=(panel,manifest)),patch.object(precision.vm,'context_weights',return_value=np.array([.5,.5])):
                out=precision.replay_tables({},root,[job])
            data={r['method']:r for r in out['replay_information']}
            self.assertEqual(data['measured_replay']['risk_physical'],0.)
            self.assertEqual(data['training_mean']['error_baseline100'],100.)
            self.assertAlmostEqual(data['complete_state_selected']['error_baseline100'],25.)
            self.assertAlmostEqual(data['complete_state_candidate']['error_baseline100'],56.25)
            self.assertEqual(data['state_variance']['status'],'unavailable')
            self.assertIsNone(data['state_variance']['risk_physical'])
            self.assertEqual(data['theoretical_future_input']['evidence_type'],'theoretical')
            self.assertEqual(data['measured_replay']['evidence_type'],'privileged_replay')
            self.assertTrue(all(r['futures']==1 and r['contexts']==2 and r['circuits']==2 for r in data.values()))

if __name__=='__main__':unittest.main()
