"""Small integrity fixtures for immutable roots, evidence imports and expert exports."""
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.v4 import artifacts, reporting
from experiments.eeg_representation.h1.v4.evidence import EvidenceSource, import_evidence
from experiments.eeg_representation.h1.v4.models import fit
from experiments.eeg_representation.h1.v4.restore_export import restore,verify_archive
from experiments.eeg_representation.h1.targets import canonical_hash


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def record(path,root):
    data=Path(path).read_bytes()
    return dict(path=Path(path).relative_to(root).as_posix(),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def config():
    return dict(claim_status='engineering',backend='analytical',simulation={'population_e':2,'population_i':1})


def evidence(directory,*,missing_native=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    write(directory/'resolved_config.json',{'source':'frozen'})
    write(directory/'provenance.json',{'source_hash':'original-source','config_hash':'original-config'})
    raw=directory/'native/train/context/score-0000.h5';raw.parent.mkdir(parents=True,exist_ok=True)
    raw.write_bytes(b'fixture-native-bytes')
    trial=record(raw,directory);trial['trial_id']='context-score-0000'
    write(directory/'native/manifest.json',{'source_hash':'original-source','config_hash':'original-config','trials':[trial]})
    for name in ('compression/task/readout.npz','compression/representations/forecast_model.npz','benchmarks/model.npz',
                 'compression/task/predictions.npz','compression/task/logits.npz','compression/codes.npz'):
        p=directory/name;p.parent.mkdir(parents=True,exist_ok=True);np.savez(p,value=np.arange(4))
    recs=[record(p,directory) for p in directory.rglob('*') if p.is_file()]
    write(directory/'EVIDENCE_MANIFEST.json',{'files':recs})
    if missing_native:raw.unlink()
    return directory


class EvidenceIntegrityTests(unittest.TestCase):
    def test_source_identity_lock_and_same_content_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);a=evidence(base/'a');b=evidence(base/'b');root=base/'analysis'
            first=import_evidence(a,root)
            moved=import_evidence(b,root)
            self.assertEqual(first['identity']['content_id'],moved['identity']['content_id'])
            self.assertEqual(json.loads((root/'imports/current.json').read_text())['source_path'],str(b.resolve()))
            write(b/'extra.json',{'changed':True})
            with self.assertRaisesRegex(ValueError,'source changed'):import_evidence(b,root)
            with self.assertRaisesRegex(ValueError,'separate'):import_evidence(a,a/'new-analysis')

    def test_directories_without_native_manifest_still_bind_numeric_content(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir();write(source/'resolved_config.json',{})
            np.savez(source/'panel.npz',x=np.array([1.]))
            first=EvidenceSource(source).identity['content_id']
            np.savez(source/'panel.npz',x=np.array([2.]))
            second=EvidenceSource(source).identity['content_id']
            self.assertNotEqual(first,second)

    def test_conflicting_records_and_archive_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);source=evidence(base/'source')
            manifest=json.loads((source/'native/manifest.json').read_text());manifest['trials'][0]['sha256']='0'*64
            write(source/'native/manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'Conflicting'):EvidenceSource(source)
            archive=base/'unsafe.zip'
            with zipfile.ZipFile(archive,'w') as z:
                z.writestr('resolved_config.json','{}');z.writestr('../escape','bad')
            with self.assertRaisesRegex(ValueError,'Unsafe'):EvidenceSource(archive)

    def test_numeric_npz_rejects_object_pickle_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temp:
            source=evidence(Path(temp)/'source')
            np.savez(source/'objects.npz',object=np.array([{'secret':1}],dtype=object))
            with EvidenceSource(source) as src:
                with self.assertRaises(ValueError):src.read_npz('objects.npz')
                (source/'compression/task/predictions.npz').write_bytes(b'corrupt')
                with self.assertRaisesRegex(ValueError,'checksum'):src.verify('compression/task/predictions.npz')


class RootIntegrityTests(unittest.TestCase):
    def test_resume_config_source_environment_and_stage_checks(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'analysis';cfg=config()
            env={'python':'3.10','versions':{'numpy':'1'},'host':'local','platform':'test','logical_cpus':1}
            with patch.object(artifacts,'source_hash',return_value='source1'),patch.object(artifacts,'environment',return_value=env):
                artifacts.initialize(root,cfg)
                artifacts.initialize(root,cfg,resume=True)
                with self.assertRaises(FileExistsError):artifacts.initialize(root,cfg)
                with self.assertRaisesRegex(ValueError,'configuration/source'):artifacts.initialize(root,{**cfg,'backend':'neuron'},True)
                write(root/'resolved_config.json',{'tampered':True})
                with self.assertRaisesRegex(ValueError,'resolved configuration'):artifacts.initialize(root,cfg,True)
                write(root/'resolved_config.json',cfg)
                changed={**env,'versions':{'numpy':'2'}}
                with patch.object(artifacts,'environment',return_value=changed):
                    with self.assertRaisesRegex(ValueError,'environment'):artifacts.initialize(root,cfg,True)
                with self.assertRaises(FileNotFoundError):artifacts.stage_done(root,'missing',[root/'missing'],{'status':'complete'})
                out=root/'value.json';write(out,{'x':1});artifacts.stage_done(root,'good',[out],{'status':'complete'})
                artifacts.resume_stage(root,'good',True)
                out.write_text('changed')
                with self.assertRaisesRegex(ValueError,'corrupt'):artifacts.resume_stage(root,'good',True)
            with patch.object(artifacts,'source_hash',return_value='source2'):
                with self.assertRaisesRegex(ValueError,'configuration/source'):artifacts.initialize(root,cfg,True)


class ExportIntegrityTests(unittest.TestCase):
    def check_zip(self,path):
        with zipfile.ZipFile(path) as z:
            manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
            self.assertEqual(set(z.namelist()),{'EXPORT_MANIFEST.json'}|{r['path'] for r in manifest['included']})
            for r in manifest['included']:
                data=z.read(r['path']);self.assertEqual(len(data),r['bytes']);self.assertEqual(hashlib.sha256(data).hexdigest(),r['sha256'])
            return manifest

    def test_compact_full_omission_and_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'analysis';root.mkdir();source=evidence(base/'source')
            write(root/'resolved_config.json',config())
            p=root/'models/job/candidate';p.mkdir(parents=True)
            np.savez(p/'state.npz',weight=np.ones(4));np.savez(p/'evaluation.npz',mean=np.ones(4),variance=np.ones(4))
            (p/'unsafe.pt').write_bytes(b'never-loaded')
            compact=base/'compact.zip';reporting.export(config(),root,compact,source=source,include_imported=True)
            manifest=self.check_zip(compact);omitted={r['path'] for r in manifest['omitted']};included={r['path'] for r in manifest['included']}|{r['original_path'] for r in manifest.get('transformed_numeric',[])}
            for name in ('compression/task/readout.npz','compression/representations/forecast_model.npz','benchmarks/model.npz','native/train/context/score-0000.h5'):
                self.assertIn('imported/'+name,omitted)
            self.assertIn('evidence/models/job/candidate/state.npz',omitted)
            self.assertIn('evidence/models/job/candidate/unsafe.pt',omitted)
            for name in ('compression/task/predictions.npz','compression/task/logits.npz','compression/codes.npz'):
                self.assertIn('imported/'+name,included)
            self.assertIn('evidence/models/job/candidate/evaluation.npz',included)
            self.assertEqual(manifest['source']['generation_source_hash'],'original-source')
            default=base/'default-compact.zip';reporting.export(config(),root,default,source=source)
            dm=self.check_zip(default);di={r['path'] for r in dm['included']}|{r['original_path'] for r in dm.get('transformed_numeric',[])};do={r['path'] for r in dm['omitted']}
            self.assertIn('evidence/models/job/candidate/evaluation.npz',di)
            self.assertIn('imported/provenance.json',di)
            self.assertIn('imported/resolved_config.json',di)
            self.assertIn('imported/compression/task/predictions.npz',do)
            self.assertNotIn('imported/compression/task/predictions.npz',di)
            full=base/'full.zip';reporting.export(config(),root,full,source=source,full_native=True)
            fm=self.check_zip(full);self.assertEqual(fm['native_files_included'],1)
            self.assertIn('imported/compression/task/readout.npz',{r['path'] for r in fm['included']})
            self.assertEqual((full.with_suffix('.zip.sha256')).read_text().split()[0],hashlib.sha256(full.read_bytes()).hexdigest())
            with self.assertRaises(FileExistsError):reporting.export(config(),root,full,source=source)
            with self.assertRaisesRegex(ValueError,'outside'):reporting.export(config(),root,root/'bad.zip',source=source)

    def test_lossless_npz_array_dedup_and_standard_library_restore(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'analysis';root.mkdir()
            shared=np.arange(120,dtype=np.float64).reshape(30,4)
            np.savez_compressed(root/'one.npz',X=shared,labels=np.array(['R0','Ralpha']))
            np.savez_compressed(root/'two.npz',X=shared,payload=np.array([0,17,255],dtype=np.uint8))
            archive=base/'deduplicated.zip';result=reporting.export(config(),root,archive)
            manifest=self.check_zip(archive)
            self.assertEqual(result['transformed_npz_files'],2)
            self.assertEqual(result['unique_numeric_arrays'],3)
            transformed=manifest['transformed_numeric']
            self.assertEqual(len(transformed),2)
            self.assertFalse(any(r['path'].endswith('one.npz') for r in manifest['omitted']))
            verify_archive(archive)
            restored=base/'restored';summary=restore(archive,restored)
            self.assertTrue(summary['npy_bytes_exact'])
            self.assertEqual(len(summary['restored_numeric']),2)
            for name in ('one.npz','two.npz'):
                original=root/name;new=restored/'evidence'/name
                with np.load(original,allow_pickle=False) as a,np.load(new,allow_pickle=False) as b:
                    self.assertEqual(a.files,b.files)
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])
                with zipfile.ZipFile(original) as a,zipfile.ZipFile(new) as b:
                    self.assertEqual(a.namelist(),b.namelist())
                    for key in a.namelist():self.assertEqual(a.read(key),b.read(key))
            with self.assertRaises(FileExistsError):restore(archive,restored)
            plain=base/'plain.zip';reporting.export(config(),root,plain,deduplicate_npz=False)
            pm=self.check_zip(plain);self.assertFalse(pm['transformed_numeric'])
            with zipfile.ZipFile(plain) as z:self.assertEqual(z.read('evidence/one.npz'),(root/'one.npz').read_bytes())

    def test_full_native_requires_available_raw_and_limit_is_atomic(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'analysis';root.mkdir();source=evidence(base/'source',missing_native=True)
            with self.assertRaisesRegex(ValueError,'Full-native export unavailable'):
                reporting.export(config(),root,base/'bad-full.zip',source=source,full_native=True)
            self.assertFalse((base/'bad-full.zip').exists())
            with self.assertRaisesRegex(ValueError,'exceeds requested limit'):
                reporting.export(config(),root,base/'too-big.zip',source=source,max_bytes=1)
            self.assertFalse((base/'too-big.zip').exists());self.assertFalse(list(base.glob('.too-big.zip*')))

    def test_nested_generated_source_included_once(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp);root=base/'analysis';root.mkdir();source=evidence(root/'generation')
            output=base/'full.zip';reporting.export(config(),root,output,source=source,full_native=True)
            manifest=self.check_zip(output)
            self.assertEqual(manifest['native_files_included'],1)
            self.assertFalse(any(r['path'].startswith('imported/') for r in manifest['included']))


class ModelFeatureIsolationTests(unittest.TestCase):
    def fixture(self,root,changed=False):
        rng=np.random.default_rng(21);job='A1__T0_l2_d8';directory=root/'derived'/job;directory.mkdir(parents=True)
        files={};contract=dict(target_physical_hash='target',population_id='population',acquisition_hash='A1',normalization_id='norm',units='V',dimension=4,support=[6002,6010])
        for s,n,start in [('train',8,1),('development',4,3),('evaluation',4,5)]:
            x=rng.normal(size=(n,32));m=4 if s=='evaluation' else 1;y=x[:,-4:,None].transpose(0,2,1)+rng.normal(size=(n,m,4))*.1
            if changed and s=='evaluation':y+=1234
            ids=np.array([f'{s}-{i}' for i in range(n)])
            arrays=dict(X=x.astype(np.float32),X_physical=x,Y=y,Y_physical=y,
                reference_Y=np.full((n,2,4),1e8 if changed else 0.),reference_Y_physical=np.full((n,2,4),1e8 if changed else 0.),
                future_mask=np.ones((n,m),bool),reference_mask=np.ones((n,2),bool),
                future_id=np.array([[f'{i}-score{j}' for j in range(m)] for i in ids]),
                reference_future_id=np.array([[f'{i}-reference{j}' for j in range(2)] for i in ids]),
                context_id=ids,structure_id=np.arange(n)%2+start,regime=np.array(['R0']*n),
                privileged_future_seed=np.full(n,999999 if changed else 0),
                privileged_future_drive=np.full((n,128),1e10 if changed else 0.))
            p=directory/f'{s}.npz';np.savez(p,**arrays);files[s]=record(p,root)
        manifest=dict(normalization=dict(fit_split='train',x_offset=0.,x_scale=1.,y_offset=[0.]*4,y_scale=1.),
            target_family='T0',risk_contract=contract,target_fingerprint=canonical_hash(contract),files=files)
        write(directory/'manifest.json',manifest);write(root/'derived/manifest.json',{'jobs':[{'id':job,'status':'complete'}]})
        return job

    def test_fitted_states_and_predictions_ignore_privileged_reference_and_evaluation_outcomes(self):
        with tempfile.TemporaryDirectory() as temp:
            root1=Path(temp)/'one';root2=Path(temp)/'two';job=self.fixture(root1);self.fixture(root2,True)
            cfg={'claim_status':'engineering','models':{'neural_enabled':False,'state_ranks':[2]}}
            fit(cfg,root1,[job]);fit(cfg,root2,[job])
            for path in (root1/'models'/job).rglob('*.npz'):
                other=root2/path.relative_to(root1)
                with np.load(path,allow_pickle=False) as a,np.load(other,allow_pickle=False) as b:
                    self.assertEqual(set(a.files),set(b.files))
                    self.assertFalse(any('privileged' in k for k in a.files))
                    for key in a.files:np.testing.assert_array_equal(a[key],b[key])


if __name__=='__main__':unittest.main()
