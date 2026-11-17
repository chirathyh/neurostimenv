"""Resume must verify, never silently overwrite bank or full-state inputs."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import analysis,banks,learning,state
from experiments.eeg_representation.h1.artifacts import atomic_json,file_hash


def panel_fixture():
    rng=np.random.default_rng(19221);out={}
    for split,n,base in [('train',12,0),('development',4,20),('calibration',4,30),('evaluation',4,40)]:
        x=rng.normal(size=(n,2000));y=rng.normal(size=(n,2,4))
        out[split]=dict(X=x,X_physical=x*1e-9,Y=y,Y_physical=y*1e-9,
            future_mask=np.ones((n,2),bool),context_id=np.asarray([f'{split}-{i}' for i in range(n)]),
            structure_id=np.asarray([base+i//2 for i in range(n)]),regime=np.asarray(['R0','Ralpha']*(n//2)))
    return out


def forecast_fixture(root,panel,options,seed,destination,device,resume):
    return {split:np.asarray(p['X'][:,:16]) for split,p in panel.items()},dict(method='test deterministic code',optimizer_updates=options['forecast_updates'])


def state_fixture(value,cid):
    p=state._synthetic_payload();p['binding']['context_id']=cid;p['binding']['structure_id']=cid
    state._add(p,'synapses/contact/A',value,'synapses','engineering fixture')
    state._add(p,'drive/phase_rad',value*.2,'drive','engineering fixture')
    return state._seal_synthetic(p)


def make_states(root):
    cohorts=dict(train=['t0','t1'],development=['d0'],calibration=['c0'],evaluation=['e0']);rows=[]
    for i,cid in enumerate(x for ids in cohorts.values() for x in ids):
        path=root/'source_states'/cid;state.export_state(path,state_fixture(.001*i,cid))
        rows.append(dict(context_id=cid,state_directory=str(path)))
    atomic_json(root/'state/manifest.json',dict(status='complete',contexts=rows))
    return cohorts


def snapshots(root):return {p.relative_to(root).as_posix():(file_hash(p),p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}


class ImmutableBankTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.panel=panel_fixture();self.config=dict(forecast_updates=1,forecast_seed=0)
    def tearDown(self):self.temp.cleanup()
    def build(self,resume=False):
        with patch.object(banks,'_forecast_bank',side_effect=forecast_fixture):
            return banks.build(self.config,self.root,self.panel,None,resume=resume)

    def test_completed_resume_loads_saved_codes_without_refitting(self):
        first=self.build();before=snapshots(self.root)
        with patch.object(banks,'projection_fit',side_effect=AssertionError('must not refit')),patch.object(banks,'_forecast_bank',side_effect=AssertionError('must not refit')):
            second=banks.build(self.config,self.root,self.panel,None,resume=True)
        self.assertEqual(before,snapshots(self.root))
        for key in first:
            for split in self.panel:np.testing.assert_array_equal(first[key]['values'][split],second[key]['values'][split])

    def test_changed_input_settings_or_source_fail_before_overwrite(self):
        self.build();before=snapshots(self.root)
        self.config['forecast_updates']=2
        with self.assertRaisesRegex(ValueError,'contract changed'):self.build(resume=True)
        self.assertEqual(before,snapshots(self.root));self.config['forecast_updates']=1
        self.panel['evaluation']['X'][0,0]+=1
        with self.assertRaisesRegex(ValueError,'contract changed'):self.build(resume=True)
        self.assertEqual(before,snapshots(self.root))

    def test_changed_verified_source_identity_is_rejected(self):
        with patch.object(banks,'_source_identity',return_value={'content_id':'original'}):self.build()
        before=snapshots(self.root)
        with patch.object(banks,'_source_identity',return_value={'content_id':'changed'}):
            with self.assertRaisesRegex(ValueError,'contract changed'):self.build(resume=True)
        self.assertEqual(before,snapshots(self.root))

    def test_corrupt_saved_code_is_never_replaced(self):
        self.build();path=self.root/'codes/forecast.npz';path.write_bytes(path.read_bytes()+b'corrupt')
        before=snapshots(self.root)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):self.build(resume=True)
        self.assertEqual(before,snapshots(self.root))

    def test_missing_manifest_partial_outputs_are_compared_not_overwritten(self):
        self.build();(self.root/'codes/manifest.json').unlink()
        before=snapshots(self.root);self.build(resume=True)
        after=snapshots(self.root)
        for path,fingerprint in before.items():self.assertEqual(after[path],fingerprint)

    def test_imported_forecast_procedure_is_checked_before_reuse(self):
        source=self.root/'source';folder=source/'retention/banks/forecast__seed_0';folder.mkdir(parents=True)
        (folder/'representations.npz').write_bytes(b'declared-representations')
        (source/'retention/task_targets.npz').write_bytes(b'declared-targets')
        atomic_json(folder/'metadata.json',dict(representation_family='masked',optimization_seed=0,optimizer_updates=200,native_width=128))
        with self.assertRaisesRegex(ValueError,'does not match the declared'):
            banks._source_identity(source)

    def test_manifest_metadata_tampering_fails(self):
        self.build();path=self.root/'codes/manifest.json';r=json.loads(path.read_text());r['forecast_reused']=True;atomic_json(path,r)
        before=snapshots(self.root)
        with self.assertRaisesRegex(ValueError,'manifest content changed'):self.build(resume=True)
        self.assertEqual(before,snapshots(self.root))


class ImmutableStateFeatureTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.cohorts=make_states(self.root)
    def tearDown(self):self.temp.cleanup()

    def test_completed_resume_verifies_and_never_refits_or_overwrites(self):
        values,audit=analysis.state_features({},self.root,self.cohorts);before=snapshots(self.root)
        with patch.object(state,'fit_adapter',side_effect=AssertionError('must not refit')),patch.object(state,'transform_adapter',side_effect=AssertionError('must not recompute')):
            again,reloaded=analysis.state_features({},self.root,self.cohorts)
        self.assertEqual(audit,reloaded);self.assertEqual(before,snapshots(self.root))
        for split in values:np.testing.assert_array_equal(values[split],again[split])

    def test_changed_state_export_fails_before_adapter_overwrite(self):
        analysis.state_features({},self.root,self.cohorts)
        state.export_state(self.root/'source_states/e0',state_fixture(.123,'e0'))
        before=snapshots(self.root)
        with self.assertRaisesRegex(ValueError,'input/cohort/schema contract changed'):analysis.state_features({},self.root,self.cohorts)
        self.assertEqual(before,snapshots(self.root))

    def test_unrelated_invocation_ledger_update_does_not_refit_state_adapter(self):
        analysis.state_features({},self.root,self.cohorts)
        path=self.root/'state/manifest.json';m=json.loads(path.read_text());m['ledger']={'calls':106};atomic_json(path,m)
        before=snapshots(self.root)
        with patch.object(state,'fit_adapter',side_effect=AssertionError('must not refit')):
            analysis.state_features({},self.root,self.cohorts)
        self.assertEqual(before,snapshots(self.root))

    def test_corrupt_matrix_fails_before_learner_can_use_it(self):
        analysis.state_features({},self.root,self.cohorts)
        path=self.root/'state/adapter_inputs.npz';path.write_bytes(path.read_bytes()+b'corrupt');before=snapshots(self.root)
        with self.assertRaisesRegex(ValueError,'checksum mismatch'):analysis.state_features({},self.root,self.cohorts)
        self.assertEqual(before,snapshots(self.root))

    def test_partial_matrix_continuation_verifies_existing_artifacts(self):
        analysis.state_features({},self.root,self.cohorts);(self.root/'state/adapter_freeze.json').unlink()
        before=snapshots(self.root);analysis.state_features({},self.root,self.cohorts);after=snapshots(self.root)
        for path,fingerprint in before.items():self.assertEqual(after[path],fingerprint)


class ProductionStateLearningControl(unittest.TestCase):
    def test_production_ridge_learns_previously_omitted_fields(self):
        rng=np.random.default_rng(18013);cohorts={};values={};panel={};payloads={}
        sizes=dict(train=48,development=16,calibration=16,evaluation=16)
        for si,(split,n) in enumerate(sizes.items()):
            z=rng.normal(size=(n,5));rows=[]
            for i,row in enumerate(z):
                p=state._synthetic_payload();p['binding']['context_id']=f'{split}-{i}'
                for key,v in zip(('synapses/contact/A','synapses/contact/B','drive/phase_rad','pending/recurrent/0/arrival_ms','connectivity/contact/weight_uS'),row):
                    state._add(p,key,v,key.split('/')[0],'known omitted-field target control')
                rows.append(state._seal_synthetic(p))
            payloads[split]=rows;y=z@np.array([.7,-.4,.3,.2,-.6])
            panel[split]=dict(Y=y[:,None,None],future_mask=np.ones((n,1),bool),context_id=np.asarray([f'{split}-{i}' for i in range(n)]),
                structure_id=np.asarray([si*1000+i//4 for i in range(n)]),regime=np.asarray(['R0']*n))
        adapter=state.fit_adapter(payloads['train'],all_schema=[p for rows in payloads.values() for p in rows])
        values={s:state.transform_adapter(adapter,rows) for s,rows in payloads.items()}
        manifest=dict(risk_contract={'scope':'engineering only','target':'known linear omitted-field function'},target_fingerprint='engineering-control')
        config=dict(ridge_alphas=[.1,10.,1000.],state_candidate_alpha=10.)
        with tempfile.TemporaryDirectory() as d:
            result=learning.ridge(panel,values,manifest,d,config,mechanism_scaled=True)
            pred=learning.numeric(Path(d)/'predictions.npz');y=panel['evaluation']['Y'][:,0,0]
            measured=float(np.mean((pred['evaluation_prediction'][:,0]-y)**2));baseline=float(np.mean((np.mean(panel['train']['Y'])-y)**2))
            candidate=float(np.mean((pred['evaluation_candidate_prediction'][:,0]-y)**2))
            self.assertLess(measured,.001*baseline)
            self.assertLess(candidate,.10*baseline)
            self.assertTrue(result['actually_fitted_state_dependent'])
            self.assertNotEqual(result['selected_id'],'train_mean')
            self.assertTrue((Path(d)/'learning_curve.json').is_file())
            with patch.object(learning,'fit_design',side_effect=AssertionError('must not refit')):
                resumed=learning.ridge(panel,values,manifest,d,config,mechanism_scaled=True,resume=True)
            self.assertEqual(result,resumed)

if __name__=='__main__':unittest.main()
