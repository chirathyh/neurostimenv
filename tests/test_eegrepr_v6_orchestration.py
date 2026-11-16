"""Runner contracts: no silent simulation or post-test sample-size revision."""
import tempfile,unittest,json,copy,zipfile,hashlib
from pathlib import Path
from unittest.mock import patch
from experiments.eeg_representation.h1.v6 import cli,config
from experiments.eeg_representation.h1.v6.controls import target_controls
from experiments.eeg_representation.h1.v6.artifacts import initialize,read_json,canonical_hash

class RunnerContracts(unittest.TestCase):
 def test_json_scientific_float_and_root_integrity(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'config.json';c=config.resolve();p.write_text(json.dumps(c))
   self.assertEqual(config.load_config(p),c)
   root=Path(d)/'root';initialize(root,c);initialize(root,c,resume=True)
   q=root/'resolved_config.json';changed=read_json(q);changed['energy']['log_floor_v2']=1e-30;q.write_text(json.dumps(changed))
   with self.assertRaisesRegex(ValueError,'Stored resolved'):initialize(root,c,resume=True)
 def test_future_only_physical_targets_and_causal_acquisitions(self):
  self.assertTrue(target_controls()['passed'])
 def test_protocol_margins_cannot_disagree_with_scoring(self):
  for key,value in [('task_margin_bits',.1),('forecast_interval_level',.95),('primary_job_id','A0__T0_l10_d128')]:
   with self.assertRaisesRegex(ValueError,'Frozen statistical'):config.resolve({'protocol':{key:value}})
 def test_run_requires_explicit_execution(self):
  a=cli.parser().parse_args(['run','--root','/tmp/never-v6-execute'])
  with self.assertRaisesRegex(ValueError,'requires --execute'):cli.campaign(a)
 def test_precision_revision_precedes_test_and_a_failed_gate_never_generates(self):
  with tempfile.TemporaryDirectory() as d:
   source=Path(d)/'old.zip';source.write_bytes(b'historical fixture')
   a=cli.parser().parse_args(['run','--root',d,'--source-archive',str(source),'--native-source','/explicit/old','--execute'])
   candidate=config.resolve();candidate['protocol']['test_circuits']=257;candidate['data']['structures']['evaluation']=list(range(37052,37309))
   proposal={'status':'proposed','phase0_complete':True,'resolved_config':candidate};calls=[]
   def stage(args):
    calls.append(args.mode)
    if args.mode=='plan':return {'status':'planned','eligible_numerical_execution':False,'blocked_reasons':['measured cap failure']}
    return {'status':'complete'}
   with patch.object(cli,'stage',side_effect=stage),patch('experiments.eeg_representation.h1.v6.precision.certificate',return_value={}),patch('experiments.eeg_representation.h1.v6.precision.propose_design',return_value=proposal):
    r=cli.campaign(a)
   self.assertEqual(r['status'],'blocked');self.assertNotIn('generate',calls);self.assertNotIn('numerical-check',calls)
   self.assertEqual(read_json(Path(d)/'selected-config.json')['protocol']['test_circuits'],257)
   changed=copy.deepcopy(proposal);changed['resolved_config']['protocol']['test_circuits']=258;changed['resolved_config']['data']['structures']['evaluation'].append(37309)
   with patch.object(cli,'stage',side_effect=stage),patch('experiments.eeg_representation.h1.v6.precision.certificate',return_value={}),patch('experiments.eeg_representation.h1.v6.precision.propose_design',return_value=changed):
    with self.assertRaisesRegex(ValueError,'already fixed'):cli.campaign(a)


 def test_complete_campaign_resume_verifies_receipt_and_skips_every_stage(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);source=root/'old.zip';source.write_bytes(b'historical fixture')
   cfg=root/'request.json';cfg.write_text(json.dumps(config.resolve()))
   a=cli.parser().parse_args(['run','--root',str(root/'campaign'),'--config',str(cfg),'--source-archive',str(source),'--native-source',str(root/'native'),'--execute'])
   proposal={'status':'unchanged_candidate','phase0_complete':True,'resolved_config':config.resolve()};calls=[]
   def stage(args):
    calls.append(args.mode)
    if args.mode=='plan':return {'status':'planned','eligible_numerical_execution':True,'eligible_for_generation':True}
    if args.mode=='export':
     path=Path(args.destination);payload=b'Complete numerical evidence fixture.'
     manifest={'schema_version':'eegrepr-v6-export-1','status':'complete','full_native':False,'included':[{'path':'evidence/result.txt','bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}]}
     with zipfile.ZipFile(path,'w') as z:
      z.writestr('evidence/result.txt',payload);z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest))
     return {'status':'complete','path':str(path),'bytes':path.stat().st_size,'sha256':cli.file_hash(path),'full_native':False}
    return {'status':'complete'}
   with patch.object(cli,'stage',side_effect=stage),patch('experiments.eeg_representation.h1.v6.precision.certificate',return_value={}),patch('experiments.eeg_representation.h1.v6.precision.propose_design',return_value=proposal):
    first=cli.campaign(a)
   self.assertIn('generate',calls);self.assertIn('calibrate',calls);self.assertEqual(calls[-1],'export')
   self.assertEqual(a.config,str(cfg))
   receipt=read_json(root/'campaign/campaign_completed.json')
   self.assertEqual(receipt['execution_request']['resolved_original_config'],config.resolve())
   self.assertEqual(receipt['export_verification']['payload_files_verified'],1)
   with patch.object(cli,'stage',side_effect=AssertionError('A completed resume must not rerun any stage')):
    with self.assertRaisesRegex(FileExistsError,'use --resume'):cli.campaign(a)
    a.resume=True;second=cli.campaign(a)
    self.assertEqual(first,second)
    for key,value in [('workers',2),('device','cuda'),('full_native',True),('max_bytes',1),('destination',str(root/'other.zip')),('native_source',str(root/'other-native'))]:
     changed=copy.copy(a);setattr(changed,key,value)
     with self.subTest(key=key),patch('experiments.eeg_representation.h1.v6.planning._fit_hardware',return_value={}):
      with self.assertRaisesRegex(ValueError,'request/config/source/implementation/environment changed'):cli.campaign(changed)
    config_changed=config.resolve();config_changed['budget']['max_generation_hours']=499
    cfg.write_text(json.dumps(config_changed))
    with self.assertRaisesRegex(ValueError,'request/config/source/implementation/environment changed'):cli.campaign(a)
    cfg.write_text(json.dumps(config.resolve()))
    source.write_bytes(b'changed historical fixture')
    with self.assertRaisesRegex(ValueError,'request/config/source/implementation/environment changed'):cli.campaign(a)
    source.write_bytes(b'historical fixture')
    with patch.object(cli,'environment',return_value={'changed':'environment'}):
     with self.assertRaisesRegex(ValueError,'request/config/source/implementation/environment changed'):cli.campaign(a)
    with patch.object(cli,'_campaign_implementation',return_value={'changed':'implementation'}):
     with self.assertRaisesRegex(ValueError,'request/config/source/implementation/environment changed'):cli.campaign(a)
    archive=Path(first['path']);archive.write_bytes(archive.read_bytes()+b'corrupt')
    with self.assertRaisesRegex(ValueError,'export missing or checksum changed'):cli.campaign(a)

 def test_completed_receipt_cannot_substitute_for_archive_payload_verification(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'out.zip'
   with zipfile.ZipFile(path,'w') as z:
    z.writestr('evidence/data',b'wrong')
    z.writestr('EXPORT_MANIFEST.json',json.dumps({'schema_version':'eegrepr-v6-export-1','status':'complete','full_native':False,
     'included':[{'path':'evidence/data','bytes':5,'sha256':'0'*64}]}))
   request={'destination':str(path),'full_native':False,'max_bytes':None}
   result={'status':'complete','path':str(path),'bytes':path.stat().st_size,'sha256':cli.file_hash(path),'full_native':False}
   receipt={'schema_version':'eegrepr-v6-completed-campaign-1','status':'complete','execution_request':request,
    'execution_request_hash':canonical_hash(request),'export':result}
   receipt['receipt_hash']=canonical_hash(receipt)
   (root/'campaign_completed.json').write_text(json.dumps(receipt))
   a=cli.parser().parse_args(['run','--root',str(root),'--resume'])
   with self.assertRaisesRegex(ValueError,'Checksum mismatch'):cli._completed_campaign(a,root,request)
   receipt['export']['bytes']+=1;(root/'campaign_completed.json').write_text(json.dumps(receipt))
   with self.assertRaisesRegex(ValueError,'receipt invalid or changed'):cli._completed_campaign(a,root,request)

 def test_export_without_completed_receipt_is_never_implicitly_accepted(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);path=root/'orphan.zip';path.write_bytes(b'completed export before receipt interruption')
   a=cli.parser().parse_args(['run','--root',str(root),'--resume'])
   with self.assertRaisesRegex(FileExistsError,'without a verified completed-campaign receipt'):
    cli._completed_campaign(a,root,{'destination':str(path)})

if __name__=='__main__':unittest.main()
