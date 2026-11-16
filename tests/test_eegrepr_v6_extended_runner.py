"""User-facing safety contracts for the complete extended workstation wrapper."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from experiments.eeg_representation.h1.v6 import extended_cli as cli
from experiments.eeg_representation.h1.v6.config import resolve
from experiments.eeg_representation.h1.v6.artifacts import atomic_json

class ExtendedRunnerTests(unittest.TestCase):
 def test_complete_and_resume_cannot_launch_without_explicit_execution(self):
  with tempfile.TemporaryDirectory() as tmp:
   for mode in ('complete','resume'):
    root=Path(tmp)/mode
    args=cli.parser().parse_args([mode,'--root',str(root)])
    with self.assertRaisesRegex(ValueError,'requires --execute'):cli.dispatch(args)
    self.assertFalse(root.exists())
 def test_resume_retains_explicit_sources_and_complete_extended_config(self):
  args=cli.parser().parse_args(['resume','--root','/tmp/never-extended-mocked','--execute','--source-archive','/archive.zip','--native-source','/native'])
  def campaign(a):
   self.assertEqual(a.mode,'run');self.assertTrue(a.resume);self.assertTrue(a.execute)
   self.assertEqual((a.source_archive,a.native_source),('/archive.zip','/native'))
   from experiments.eeg_representation.h1.v6.config import load_config
   self.assertEqual(len(load_config(a.config)['selected_job_ids']),60)
   return {'status':'blocked','stage':'test-controlled gate'}
  with patch.object(cli.cli,'campaign',side_effect=campaign),patch.object(cli,'atomic_json'):
   self.assertEqual(cli.dispatch(args)['status'],'blocked')
 def test_status_is_read_only_and_does_not_mistake_stale_progress_for_liveness(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'missing'
   args=cli.parser().parse_args(['status','--root',str(root)])
   with patch.object(cli.cli,'stage',side_effect=AssertionError('read-only status must not stage')):
    self.assertEqual(cli.dispatch(args)['status'],'not_started')
   self.assertFalse(root.exists())
   root.mkdir();atomic_json(root/'campaign_progress.json',{'status':'running','updated_unix':1,'stage':'generate'})
   before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()};result=cli.status(root)
   self.assertIn('not proof',result['process_liveness']);self.assertEqual(result['status'],'observed')
   self.assertEqual(before,{p:p.read_bytes() for p in root.rglob('*') if p.is_file()})
 def test_report_only_uses_frozen_actual_N_and_cannot_refit_or_reset_to192(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);c=resolve({'extension':{'enabled':True}});c['protocol']['test_circuits']=257;c['data']['structures']['evaluation']=list(range(37052,37309))
   atomic_json(root/'selected-config.json',c);core=root/'confirmation-n257';atomic_json(core/'resolved_config.json',c)
   args=cli.parser().parse_args(['report-only','--root',str(root),'--workers','4'])
   with patch.object(cli.cli,'stage',return_value={'status':'complete'}) as stage,patch.object(cli.cli,'campaign',side_effect=AssertionError('report cannot run campaign')):
    cli.dispatch(args)
   called=stage.call_args.args[0];self.assertEqual(called.mode,'report');self.assertEqual(called.root,str(core));self.assertEqual(called.config,str(core/'resolved_config.json'));self.assertTrue(called.resume);self.assertIsNone(called.workers)
   c['protocol']['test_circuits']=258;c['data']['structures']['evaluation'].append(37309);atomic_json(core/'resolved_config.json',c)
   with self.assertRaisesRegex(ValueError,'disagree'):cli.existing_root(root)
 def test_old_protocol_is_not_silently_relabelled_as_extended(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'old.json';atomic_json(p,resolve())
   args=cli.parser().parse_args(['doctor','--root',str(Path(tmp)/'root'),'--config',str(p)])
   with patch.object(cli.cli,'stage',side_effect=AssertionError('must reject first')):
    with self.assertRaisesRegex(ValueError,'extension.enabled'):cli.dispatch(args)

if __name__=='__main__':unittest.main()
