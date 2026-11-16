"""Physical-support controls for the secondary immediate-prediction amendment."""
from copy import deepcopy
from pathlib import Path
import json
import shutil
import tempfile
import unittest
import numpy as np

from experiments.eeg_representation.h1.v6.config import DEFAULTS, resolve, required_job_ids, load_config
from experiments.eeg_representation.h1.v6.targets import TargetSpec, target_metadata, extract_physical_target, spec, energy_from_coefficients
from experiments.eeg_representation.h1.v4.targets import TargetSpec as LegacyTargetSpec, extract_target as legacy_extract
from experiments.eeg_representation.h1.v4.measurement import causal_history


class ImmediateSupportTests(unittest.TestCase):
 def setUp(self):
  self.time=np.arange(0,7010.0625,.0625,dtype=np.float64)
 def test_time_ramp_bins_have_zero_gap_exclude_cutoff_and_use_exact32samples(self):
  for h in (0,2,10):
   for w in (32,128):
    tid=f'T0_l{h}_d{w}';y=extract_physical_target(self.time,self.time,tid)
    expected=6000+h+np.arange(w//2)*2+(1+32)*.0625/2
    np.testing.assert_array_equal(y,expected)
    m=target_metadata(tid)
    self.assertEqual(m['first_bin_end_offset_ms'],h+2)
    self.assertEqual(m['native_sample_support']['cutoff_relative_first_index'],h*16+1)
    self.assertEqual(m['native_sample_support']['cutoff_relative_last_index'],(h+w)*16)
    altered=self.time.copy();altered[self.time<=6000]=-1e50
    np.testing.assert_array_equal(y,extract_physical_target(altered,self.time,tid))
 def test_legacy_values_and_operator_fingerprints_are_unchanged(self):
  rng=np.random.default_rng(117);x=rng.normal(size=len(self.time))*1e-6
  for h,w,f in ((2,32,'T0'),(10,128,'T0'),(10,1000,'T3'),(2,128,'T0'),(10,32,'T0')):
   old=LegacyTargetSpec(family=f,lead_ms=h,duration_ms=w);new=TargetSpec(family=f,lead_ms=h,duration_ms=w)
   self.assertEqual(new.to_dict(),old.to_dict());self.assertEqual(new.fingerprint,old.fingerprint)
   np.testing.assert_array_equal(extract_physical_target(x,self.time,new.id),legacy_extract(x,self.time,old))
  # The published energy operator identity must remain the exact V6 definition.
  self.assertEqual(spec('alpha')['id'],'T4_energy_alpha_l10_d1000')
  self.assertEqual(spec('alpha')['physical_support_ms'],{'left_open':6010.,'right_closed':7010.})
 def test_future_only_fourier_and_direct_energy_are_exact_same_physical_definition(self):
  for h in (0,10):
   x=2e-6*np.cos(2*np.pi*10*(self.time-6000-h-.0625)/1000)
   y=extract_physical_target(x,self.time,f'T3_l{h}_d1000')
   self.assertAlmostEqual(float(np.sum(y*y)/16000),2e-12,places=24)
   energy=extract_physical_target(x,self.time,f'T4_energy_alpha_l{h}_d1000')
   self.assertAlmostEqual(float(energy[0]),2e-12,places=24)
   log=extract_physical_target(x,self.time,f'T4_log_energy_alpha_l{h}_d1000')
   self.assertAlmostEqual(float(log[0]),np.log(2e-12),places=12)
   x[self.time<=6000+h]=1e50
   np.testing.assert_array_equal(y,extract_physical_target(x,self.time,f'T3_l{h}_d1000'))
 def test_post_cutoff_changes_cannot_change_any_acquisition(self):
  rng=np.random.default_rng(48);x=1e-6*(1+np.sin(2*np.pi*10*self.time/1000)+.01*rng.normal(size=len(self.time)))
  changed=x.copy();changed[self.time>6000]=1e40
  for a in ('A0','A1','A2'):
   before,t,_=causal_history(x,self.time,cutoff_ms=6000.,history_s=4.,acquisition_id=a)
   after,_,_=causal_history(changed,self.time,cutoff_ms=6000.,history_s=4.,acquisition_id=a)
   np.testing.assert_array_equal(before,after)
   self.assertEqual(t[-1],0.)
 def test_incomplete_or_backdated_support_fails(self):
  with self.assertRaisesRegex(ValueError,'support is absent'):
   extract_physical_target(self.time[:-1],self.time[:-1],'T3_l10_d1000')
  with self.assertRaises(ValueError):TargetSpec(lead_ms=-2)
  with self.assertRaises(ValueError):target_metadata('T0_l0_d0')
 def test_complete_matrix_is_opt_in_and_primary_family_unchanged(self):
  old=resolve();new=resolve({'extension':{'enabled':True}})
  self.assertEqual(len(old['selected_job_ids']),27);self.assertEqual(len(new['selected_job_ids']),60)
  self.assertTrue(set(old['selected_job_ids']).issubset(new['selected_job_ids']))
  self.assertEqual(old['protocol'],new['protocol']);self.assertEqual(old['data'],new['data'])
  new['selected_job_ids']=old['selected_job_ids']
  with self.assertRaisesRegex(ValueError,'Complete60'):resolve(new)


class NativeAndSavedDerivationTests(unittest.TestCase):
 def test_all60_native_jobs_and33_saved_jobs_preserve_exact_legacy_targets(self):
  from experiments.eeg_representation.h1.v6.generation import generate
  from experiments.eeg_representation.h1.v6.derivation import derive, load_panel
  c=resolve({'extension':{'enabled':True},'claim_status':'engineering','backend':'analytical','reuse':{'enabled':False}})
  c['data']['regimes']=c['data']['regimes'][:1]
  c['data']['structures']={'train':[37000,37001],'development':[37002],'calibration':[37003],'evaluation':[37004]}
  c['data']['prefixes_per_regime']={s:1 for s in c['data']['structures']};c['forks'].update(reference_circuits=[],score_futures=2,calibration_futures=2)
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)/'native';generate(c,root,execute=True);result=derive(c,root)
   self.assertEqual(result['completed_jobs'],60);self.assertEqual(result['unavailable_jobs'],[])
   for target in ('T0_l0_d32','T0_l2_d128','T3_l0_d1000','T4_energy_alpha_l0_d1000'):
    a,am=load_panel(root,'A0__'+target)
    for acquisition in ('A1','A2'):
     b,bm=load_panel(root,acquisition+'__'+target)
     self.assertEqual(am['target_normalization'],bm['target_normalization'])
     for split in a:
      np.testing.assert_array_equal(a[split]['Y_physical'],b[split]['Y_physical'])
      self.assertTrue(all(not key.startswith('partial_state') for key in b[split]))
   saved=Path(tmp)/'saved';saved.mkdir()
   # Model an archive that contains all V4 delayed supports but no native files.
   for target in ('T0_l2_d32','T0_l2_d128','T0_l10_d32','T0_l10_d128','T3_l10_d1000'):
    for a in ('A0','A1','A2'):
     d=saved/'derived'/(a+'__'+target);shutil.copytree(root/'derived'/(a+'__'+target),d)
   # A compact archive retains native provenance even though recordings are omitted.
   (saved/'native').mkdir();(saved/'native/manifest.json').write_text(json.dumps({'trials':[{'path':'native/omitted.h5'}]}))
   derived=derive(c,Path(tmp)/'imported',source=saved)
   self.assertEqual(derived['status'],'partial');self.assertEqual(derived['completed_jobs'],33)
   self.assertEqual(len(derived['unavailable_jobs']),27)
   self.assertTrue(all('_l0_' in j['id'] for j in derived['unavailable_jobs']))
   for target in ('T0_l2_d32','T0_l10_d128','T3_l10_d1000','T4_energy_alpha_l10_d1000'):
    for a in ('A0','A1','A2'):
     before,bm=load_panel(root,a+'__'+target);after,am=load_panel(Path(tmp)/'imported',a+'__'+target)
     self.assertEqual(am['physical_target_fingerprint'],bm['physical_target_fingerprint'])
     for split in before:np.testing.assert_array_equal(before[split]['Y_physical'],after[split]['Y_physical'])
   # Only two original waveform anchors still suffice for both delayed windows.
   for a in ('A0','A1','A2'):
    for t in ('T0_l2_d128','T0_l10_d32'):shutil.rmtree(saved/'derived'/(a+'__'+t))
   fallback=derive(c,Path(tmp)/'composed',source=saved)
   self.assertEqual(fallback['completed_jobs'],33)
   for a in ('A0','A1','A2'):
    for t in ('T0_l2_d128','T0_l10_d32'):
     direct,_=load_panel(root,a+'__'+t);composed,_=load_panel(Path(tmp)/'composed',a+'__'+t)
     for split in direct:np.testing.assert_array_equal(direct[split]['Y_physical'],composed[split]['Y_physical'])

if __name__=='__main__':unittest.main()
