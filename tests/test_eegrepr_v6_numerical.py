"""Semantic numerical/generation/state-geometry controls; no costly simulator panel."""
from copy import deepcopy
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from experiments.eeg_representation.h1.v6.config import DEFAULTS,contexts,counts
from experiments.eeg_representation.h1.v6.generation import generation_contract,trial_spec,execute_panel,row_path
from experiments.eeg_representation.h1.v6.simulator import common_output_grid,verify_coupling
from experiments.eeg_representation.h1.v6.numerical import evaluate_paired,panel_config
from experiments.eeg_representation.h1.v6.geometry import analyze_state
from experiments.eeg_representation.h1.v4.targets import TargetSpec,extract_target
from experiments.eeg_representation.storage import file_hash


class NumericalQualificationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(718)
        self.y=rng.normal(size=(4,4,16,5));self.p=rng.normal(scale=.1,size=(5,4,4,5))
    def test_identical_law_qualifies_but_missing_predictor_cannot(self):
        ys={.0625:self.y,.03125:self.y,.015625:self.y};ps={k:self.p for k in ys}
        result=evaluate_paired(ys,ps,np.zeros(5))
        self.assertTrue(result['passed']);self.assertEqual(result['circuits'],4)
        self.assertEqual(result['pairs'][0]['checks']['gain_difference_pp']['df'],3)
        self.assertFalse(evaluate_paired(ys)['passed'])
    def test_variance_bias_and_ambiguous_small_m_fail_closed(self):
        result=evaluate_paired({.0625:self.y*1.04,.03125:self.y},
                              {.0625:self.p,.03125:self.p},np.zeros(5))
        self.assertFalse(result['passed'])
        check=result['pairs'][0]['checks']['floor_relative_change']
        self.assertAlmostEqual(check['estimate'],1.04**2-1,places=12)
        rng=np.random.default_rng(920)
        noisy=self.y+rng.normal(scale=.1,size=self.y.shape)
        ambiguous=evaluate_paired({.0625:noisy,.03125:self.y},{.0625:self.p,.03125:self.p},np.zeros(5))
        self.assertFalse(ambiguous['passed'])
    def test_member_risks_are_averaged_not_ensemble_predictions(self):
        pred=np.stack([np.ones((4,4,5)), -np.ones((4,4,5))])
        result=evaluate_paired({.0625:self.y,.03125:self.y},{.0625:pred,.03125:pred},np.zeros(5))
        row=result['physical_moments']['0.0625']
        np.testing.assert_allclose(np.asarray(row['risk'])-row['baseline'],1.)
    def test_common_grid_keeps_future_fourier_physical_normalization(self):
        targets=[];spec=TargetSpec(family='T3',cutoff_ms=20.,lead_ms=10.,duration_ms=1000.)
        for dt in (.0625,.03125,.015625):
            time=np.arange(1,round(1030/dt)+1)*dt
            wave=np.sin(2*np.pi*10*(time-30-.0625)/1000)
            trial=dict(time_ms=time,eeg_v=wave[:,None],dipole_na_um=np.zeros((len(time),3)),metadata={})
            common=common_output_grid(trial,dt)
            self.assertEqual(len(common['time_ms']),16480)
            targets.append(extract_target(common['eeg_v'],common['time_ms'],spec))
        for target in targets[1:]:np.testing.assert_array_equal(target,targets[0])
        self.assertAlmostEqual(float(np.sum(targets[0]**2)),8000.,places=8)
    def test_changed_forcing_is_not_accepted_as_numerical_comparison(self):
        keys=('external_event_times_ms','external_event_offsets','phase_time_ms','phase_rad',
              'structure_geometry','synapse_parameters','recurrent_netcons','external_netcons')
        a={'time_ms':np.arange(3)*.0625,'privileged':{k:np.arange(4,dtype=float) for k in keys}}
        b=deepcopy(a);self.assertTrue(verify_coupling([a,b])['passed'])
        b['privileged']['external_event_times_ms'][1]+=.1
        with self.assertRaisesRegex(ValueError,'rerolled'):verify_coupling([a,b])
    def test_full_allocation_and_common_forcing_namespaces(self):
        c=deepcopy(DEFAULTS);n=counts(c)
        self.assertEqual((n['total_episodes'],n['new_episodes'],n['reused_episodes'],n['numerical_episodes']),(13056,12736,320,768))
        ref=[r for r in contexts(c) if r['reference_futures']]
        self.assertEqual(len(ref),32);self.assertTrue(all(r['prefix']==0 and r['score_futures']==32 and r['reference_futures']==128 for r in ref))
        specs=[]
        for dt in c['numerical']['dts_ms']:
            q=panel_config(c,dt);r=contexts(q)[0];specs.append(trial_spec(q,r,'score',0))
        self.assertEqual(specs[0],specs[1]);self.assertEqual(specs[1],specs[2])
        self.assertNotEqual(trial_spec(c,ref[0],'score',0)['future_seed'],trial_spec(c,ref[0],'reference',0)['future_seed'])
        a=generation_contract(c);c['models']={'changed_report_grid':True}
        self.assertEqual(a,generation_contract(c));c['simulation']['native_dt_ms']=.03125
        self.assertNotEqual(a,generation_contract(c))


class GenerationResumeTests(unittest.TestCase):
    def test_four_splits_process_workers_resume_and_corruption(self):
        c=deepcopy(DEFAULTS);c['claim_status']='engineering';c['backend']='analytical';c['analytical']['kind']='variance'
        c['reuse']['enabled']=False;c['data']['regimes']=c['data']['regimes'][:1]
        c['data']['structures']={s:[37000+i] for i,s in enumerate(('train','development','calibration','evaluation'))}
        c['data']['prefixes_per_regime']={s:1 for s in c['data']['structures']}
        c['data']['workers']=2;c['forks'].update(reference_circuits=[],score_futures=2,calibration_futures=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);result=execute_panel(c,root,contexts(c))
            self.assertEqual(result['continuations'],6)
            manifest=json.loads((root/'native/manifest.json').read_text())
            self.assertEqual({r['split'] for r in manifest['trials']},set(c['data']['structures']))
            before={r['trial_id']:file_hash(row_path(root,r)) for r in manifest['trials']}
            c['models']={'analysis_only_edit':True};resumed=execute_panel(c,root,contexts(c),resume=True)
            self.assertEqual(resumed['generated_this_call'],0)
            self.assertEqual(before,{r['trial_id']:file_hash(row_path(root,r)) for r in manifest['trials']})
            first=manifest['trials'][0];(root/'records'/(first['trial_id']+'.json')).unlink()
            recovered=execute_panel(c,root,contexts(c),resume=True)
            self.assertEqual(recovered['continuations'],6);self.assertEqual(file_hash(row_path(root,first)),first['sha256'])
            with row_path(root,first).open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(ValueError,'altered'):execute_panel(c,root,contexts(c),resume=True)


class GeometrySensitivityTests(unittest.TestCase):
    def test_train_only_rules_expose_low_variance_without_evaluation_fit(self):
        rng=np.random.default_rng(104);x=rng.normal(size=(40,49));x[:,1]=0;x[:20,-1]*=1e-100
        train=np.arange(40)<20;circuits=np.repeat(np.arange(10),4);regimes=np.tile(np.arange(4),10)
        result,scales=analyze_state(x,train,circuits,regimes,47)
        self.assertEqual(result['pathology_diagnostic']['dropped_coordinates'],2)
        self.assertEqual(result['pathology_diagnostic']['positive_sd_below_1e50'],1)
        self.assertGreater(result['scalings']['train_sd_drop_near_constant']['pooled']['participation_ratio'],10)
        changed=x.copy();changed[20:,-1]*=1e8
        other,other_scales=analyze_state(changed,train,circuits,regimes,47)
        for key in ('training_sd','scale_floor','keep','offset'):np.testing.assert_array_equal(scales[key],other_scales[key])
        self.assertEqual(result['populations']['circuits'],10)
        for r in result['scalings'].values():self.assertLessEqual(r['pooled']['numerical_rank'],39)


class ExecutionGateTests(unittest.TestCase):
    def _gate_fixture(self,root,c):
        from experiments.eeg_representation.h1.v6.artifacts import atomic_json,environment
        from experiments.eeg_representation.h1.v6.generation import generation_contract,cpu_topology
        from experiments.eeg_representation.h1.targets import canonical_hash
        saved=deepcopy(c);saved['claim_status']='timing_only';saved['namespace']='different-timing-namespace'
        contract=generation_contract(saved);identity=dict(generation_contract=contract,generation_contract_hash=canonical_hash(contract),context_allocation=[],allocation_hash=canonical_hash([]))
        rows=[]
        for i in range(8):
            p=root/'generation/evaluation/native'/f'{i}.h5';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(str(i).encode())
            rows.append(dict(path='native/'+p.name,source_root=str(root/'generation/evaluation'),trial_id=str(i),sha256=file_hash(p),bytes=p.stat().st_size,generation_contract_hash=identity['generation_contract_hash']))
        atomic_json(root/'resolved_config.json',saved);atomic_json(root/'provenance.json',{'environment':environment()})
        atomic_json(root/'generation/evaluation/generation_contract.json',identity)
        atomic_json(root/'generation/native/manifest.json',{'trials':rows})
        atomic_json(root/'generation/timing_audit.json',dict(continuations=8,new_episodes=8,complete=True,cpu_topology=cpu_topology(),requested_splits=['evaluation'],
            results=[dict(integration_dt_ms=.0625,full_prefix_replay=True,state_caching=False)],mean_continuation_seconds=100.,native_bytes=8,workers=1,peak_process_rss_bytes=100000000))
        from experiments.eeg_representation.h1.v6.models import settings
        from experiments.eeg_representation.h1.v6 import controls,models
        payload=dict(settings=settings(saved),environment=environment(),device='cpu',source=file_hash(Path(controls.__file__)),models_source=file_hash(Path(models.__file__)),hardware={})
        atomic_json(root/'fit_throughput.json',dict(status='complete',host=environment()['host'],signature=canonical_hash(payload),settings=payload['settings'],environment=payload['environment'],device='cpu',hardware={}))
    def test_timing_physical_source_edits_invalidate_measured_gate(self):
        from unittest.mock import patch
        from experiments.eeg_representation.h1.v6.planning import measured_gate
        c=deepcopy(DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self._gate_fixture(root,c)
            self.assertEqual(measured_gate(c,root)['continuations'],8)
            altered=generation_contract(c);altered['physics_source']['changed']='different'
            with patch('experiments.eeg_representation.h1.v6.generation.generation_contract',return_value=altered):
                with self.assertRaisesRegex(ValueError,'physical-source'):measured_gate(c,root)
    def test_numeric_execution_revalidates_stale_phase0_hash(self):
        from unittest.mock import patch
        from experiments.eeg_representation.h1.v6.planning import require_execution
        from experiments.eeg_representation.h1.targets import canonical_hash
        from experiments.eeg_representation.h1.v6.artifacts import atomic_json
        c=deepcopy(DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=dict(config_hash=canonical_hash(c),eligible_numerical_execution=True,timing_root=tmp,phase0_root=tmp,
                measured_gate={'evidence_hash':'gate'},phase0={'evidence_hash':'old'})
            p['plan_hash']=canonical_hash(p);atomic_json(root/'plan.json',p)
            with patch('experiments.eeg_representation.h1.v6.planning.measured_gate',return_value={'evidence_hash':'gate'}),patch('experiments.eeg_representation.h1.v6.planning.qualify_phase0',return_value={'passed':True,'evidence_hash':'changed'}):
                with self.assertRaisesRegex(ValueError,'Phase0 artifacts changed'):require_execution(c,root,True,numerical=True)
    def test_missing_old_native_blocks_core_but_not_numerical_qualification(self):
        from unittest.mock import patch
        from experiments.eeg_representation.h1.v6.planning import plan
        c=deepcopy(DEFAULTS)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with patch('experiments.eeg_representation.h1.v6.planning.qualify_phase0',return_value={'passed':True,'evidence_hash':'phase'}),patch('experiments.eeg_representation.h1.v6.planning.measured_gate',return_value={'mean_continuation_seconds':100.,'native_bytes':8000000,'workers':1,'fit_throughput':{'status':'complete'}}),patch('experiments.eeg_representation.h1.v6.planning._precision_binding',return_value={'certificate_sha256':'precision','source_archive':'numeric.zip'}),patch('experiments.eeg_representation.h1.v6.planning.shutil.disk_usage',return_value=type('Usage',(),{'free':1000000000000})()):
                result=plan(c,root,phase0_root=tmp,timing_root=tmp)
            self.assertTrue(result['eligible_numerical_execution']);self.assertFalse(result['eligible_for_generation'])
            self.assertTrue(any('explicit native source' in r for r in result['blocked_reasons']))
    def test_mean_weights_cannot_change_after_calibration_freeze(self):
        from experiments.eeg_representation.h1.v6.planning import _mean_freeze
        from experiments.eeg_representation.h1.v6.artifacts import atomic_json
        c=deepcopy(DEFAULTS);c['selected_job_ids']=['A1__T0_l2_d32']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);d=root/'models/A1__T0_l2_d32';d.mkdir(parents=True)
            atomic_json(d/'selection.json',{'status':'frozen','selection_split':'development','selected_candidate':'mean'})
            atomic_json(d/'fit_identity.json',{'signature':'frozen'})
            p=d/'state.npz';p.write_bytes(b'initial weights')
            def manifest():
                row=dict(path=str(p.relative_to(root)),sha256=file_hash(p),bytes=p.stat().st_size)
                atomic_json(d/'manifest.json',{'selected_candidate':'mean','candidates':[{'state_file':row['path'],'files':[row]}]})
            manifest();_mean_freeze(c,root);p.write_bytes(b'refit weights');manifest()
            with self.assertRaisesRegex(ValueError,'changed after calibration'):_mean_freeze(c,root)


class DerivationStagingTests(unittest.TestCase):
    def test_four_split_append_preserves_training_bytes_and_risk_contract(self):
        from experiments.eeg_representation.h1.v6.generation import generate
        from experiments.eeg_representation.h1.v6.derivation import derive,load_panel
        c=deepcopy(DEFAULTS);c['claim_status']='engineering';c['backend']='analytical';c['analytical']['kind']='variance';c['reuse']['enabled']=False
        c['data']['regimes']=c['data']['regimes'][:1];c['data']['workers']=1
        c['data']['structures']={'train':[37000,37001],'development':[37002],'calibration':[37003],'evaluation':[37004]}
        c['data']['prefixes_per_regime']={s:1 for s in c['data']['structures']}
        c['forks'].update(reference_circuits=[],score_futures=2,calibration_futures=2)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);generate(c,root,execute=True,split=['train','development']);first=derive(c,root)
            self.assertEqual(first['completed_jobs'],27)
            frozen={};fingerprints={}
            for job in c['selected_job_ids']:
                panel,m=load_panel(root,job);self.assertEqual(set(panel),{'train','development'})
                fingerprints[job]=(m['target_fingerprint'],m['normalization'])
                for split in panel:frozen[job+'/'+split]=file_hash(root/'derived'/job/(split+'.npz'))
            for split in ('calibration','evaluation'):
                generate(c,root,execute=True,split=split,resume=True);derive(c,root,resume=True)
                for job in c['selected_job_ids']:
                    panel,m=load_panel(root,job);self.assertIn(split,panel)
                    self.assertEqual((m['target_fingerprint'],m['normalization']),fingerprints[job])
                    for existing in ('train','development'):self.assertEqual(file_hash(root/'derived'/job/(existing+'.npz')),frozen[job+'/'+existing])
            manifest=root/'generation/native/manifest.json';bad=json.loads(manifest.read_text());bad['trials'].pop();manifest.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'only append|incomplete native allocation'):derive(c,root,resume=True)

class NumericArchiveIntegrityTests(unittest.TestCase):
    def test_descriptor_hash_and_duplicate_array_keys_are_verified(self):
        import hashlib,io,zipfile
        from experiments.eeg_representation.h1.v6.evidence import EvidenceSource
        buf=io.BytesIO();np.save(buf,np.arange(3),allow_pickle=False);data=buf.getvalue()
        rec={'key':'x','member_name':'x.npy','array_path':'numeric_arrays/x.npy','bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'shape':[3],'dtype':'int64'}
        descriptor={'original_path':'evidence/arrays.npz','original_sha256':'original','original_bytes':100,'arrays':[rec]}
        with tempfile.TemporaryDirectory() as tmp:
            def archive(name,duplicate=False,mutate=False):
                d=deepcopy(descriptor)
                if duplicate:d['arrays'].append(rec)
                desc=json.dumps(d).encode();payload={'evidence/resolved_config.json':b'{}','numeric_arrays/x.npy':data,'numeric_descriptors/a.json':desc}
                manifest={'included':[dict(path=k,sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for k,v in payload.items()],
                    'transformed_numeric':[{'original_path':'evidence/arrays.npz','original_sha256':'original','original_bytes':100,'descriptor_path':'numeric_descriptors/a.json'}]}
                if mutate:payload['numeric_descriptors/a.json']=desc+b' '
                path=Path(tmp)/name
                with zipfile.ZipFile(path,'w') as z:
                    for k,v in payload.items():z.writestr(k,v)
                    z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest))
                return path
            with EvidenceSource(archive('ok.zip')) as s:np.testing.assert_array_equal(s.read_npz('arrays.npz')['x'],np.arange(3))
            with EvidenceSource(archive('bad.zip',mutate=True)) as s:
                with self.assertRaisesRegex(ValueError,'checksum'):s.read_npz('arrays.npz')
            with EvidenceSource(archive('duplicate.zip',duplicate=True)) as s:
                with self.assertRaisesRegex(ValueError,'Duplicate numeric'):s.read_npz('arrays.npz')


class ResourceGateTests(unittest.TestCase):
    def test_worker_topology_and_resume_storage_credit(self):
        from unittest.mock import patch
        from experiments.eeg_representation.h1.v6.planning import _resources
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'numerical/dt/native/a.h5';p.parent.mkdir(parents=True);p.write_bytes(b'x'*100)
            with patch('experiments.eeg_representation.h1.v6.planning.shutil.disk_usage',return_value=type('Usage',(),{'free':10})()),patch('experiments.eeg_representation.h1.v6.planning.os.cpu_count',return_value=2),patch('pathlib.Path.read_text',return_value='MemTotal: 100 kB\nMemAvailable: 1 kB\n'):
                r=_resources(root,4,1000,numerical_root=root)
            self.assertEqual(r['completed_native_storage_credit_bytes'],100)
            self.assertEqual(r['conservative_worker_ram_budget_bytes'],8000)
            self.assertEqual(len(r['blocked_reasons']),2)

if __name__=='__main__':unittest.main()
