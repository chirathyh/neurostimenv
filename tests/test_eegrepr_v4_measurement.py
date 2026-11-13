"""Scientific support/causality controls; no NEURON generation or archive pickle loads."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import signal
from experiments.eeg_representation.h1.acquisition import causal_history as legacy_history
from experiments.eeg_representation.h1.targets import TargetSpec as LegacyTarget, extract_target as legacy_target, canonical_hash
from experiments.eeg_representation.h1.v4.measurement import (metadata, causal_history, apply_prefix,
    same_sensor_future_decomposition)
from experiments.eeg_representation.h1.v4.targets import (TargetSpec, extract_target, from_binned,
    full_real_fft, inverse_full_real_fft, axes, floor_activations)
from experiments.eeg_representation.h1.v4.derivation import derive, load_panel


class AcquisitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t=np.arange(1,112161)*.0625
        rng=np.random.default_rng(8)
        cls.x=2.+np.sin(2*np.pi*10*cls.t/1000)+rng.normal(0,.1,len(cls.t))

    def test_legacy_and_native_bins_exact(self):
        a,_,_=causal_history(self.x,self.t,cutoff_ms=6000,acquisition_id='A0')
        b,_,_=legacy_history(self.x,self.t,cutoff_ms=6000,history_s=4)
        np.testing.assert_array_equal(a,b)
        x,tm,_=causal_history(self.x,self.t,cutoff_ms=6000,acquisition_id='A1')
        native=self.x[(self.t>2000)&(self.t<=6000)].reshape(2000,32).mean(1)
        np.testing.assert_array_equal(x,native)
        self.assertEqual(tm[-1],0.)
        self.assertAlmostEqual(metadata('A0')['dc_gain'],.9282472260233312)
        delays=metadata('A0')['group_delay_ms_by_frequency']
        self.assertTrue(all(d==64. for d in delays if d is not None))
        a2=metadata('A2');first=[d for d in a2['group_delay_ms_by_frequency'] if d is not None][:2]
        self.assertTrue(all(0<d<500 for d in first))
        self.assertLess(abs(first[0]-first[1]),1.)

    def test_postcut_perturbation_invariant_for_every_input(self):
        changed=self.x.copy();changed[self.t>6000]=1e8
        for acquisition in ('A0','A1','A2'):
            a,_,_=causal_history(self.x,self.t,cutoff_ms=6000,acquisition_id=acquisition)
            b,_,_=causal_history(changed,self.t,cutoff_ms=6000,acquisition_id=acquisition)
            np.testing.assert_array_equal(a,b)
            changed[self.t>6000]=np.nan
            c,_,_=causal_history(changed,self.t,cutoff_ms=6000,acquisition_id=acquisition)
            np.testing.assert_array_equal(a,c)

    def test_response_constant_impulse_sinusoid_and_startup(self):
        m=metadata('A2');self.assertTrue(m['design_measurement_pass'])
        self.assertIsNone(m['time_support_ms'])
        self.assertLess(m['unit_step_settling_ms'],2000)
        constant=apply_prefix(np.ones(64000),'A2')
        self.assertLess(np.max(abs(constant[32000:])),1e-4)
        impulse=np.zeros(64000);impulse[0]=1
        ir=apply_prefix(impulse,'A2')
        np.testing.assert_allclose(ir,signal.sosfilt(np.asarray(m['coefficients']),impulse),rtol=0,atol=0)
        t=np.arange(64000)/16000
        for freq in (6.,10.,24.,250.,1000.):
            x=np.sin(2*np.pi*freq*t);y=apply_prefix(x,'A2')
            empirical=np.sqrt(np.mean(y[32000:]**2)/np.mean(x[32000:]**2))
            expected=abs(signal.sosfreqz(np.asarray(m['coefficients']),worN=[freq],fs=16000)[1][0])
            self.assertAlmostEqual(empirical,expected,places=7)
        with self.assertRaisesRegex(ValueError,'precedes measured settling'):
            causal_history(self.x,self.t,cutoff_ms=4500,history_s=4,acquisition_id='A2')

    def test_inherited_filter_state_split(self):
        for acquisition in ('A0','A2'):
            result=same_sensor_future_decomposition(self.x,self.t,cutoff_ms=6000,acquisition_id=acquisition)
            np.testing.assert_allclose(result['whole'],result['inherited_past']+result['post_cutoff_contribution'],rtol=2e-10,atol=2e-10)
            self.assertGreater(np.linalg.norm(result['inherited_past'][self.t>6000]),0)
            np.testing.assert_array_equal(result['post_cutoff_contribution'][self.t<=6000],0)


class TargetTests(unittest.TestCase):
    def setUp(self):
        self.t=np.arange(1,112161)*.0625
        self.x=np.random.default_rng(2).normal(size=len(self.t))

    def test_frozen_t0_exact_and_support(self):
        for lead in (2.,10.):
            for duration in (32.,128.):
                spec=TargetSpec(lead_ms=lead,duration_ms=duration)
                a=extract_target(self.x,self.t,spec)
                b=legacy_target(self.x,self.t,LegacyTarget(lead_ms=lead,duration_ms=duration))
                np.testing.assert_array_equal(a,b)
        with self.assertRaisesRegex(ValueError,'support is absent'):
            extract_target(self.x[:98208],self.t[:98208],TargetSpec(family='T3',duration_ms=1000))

    def test_centered_power_identity_and_fixed_floor(self):
        y=np.random.default_rng(3).normal(size=(8,64))+2
        centered=from_binned(y,TargetSpec(family='T1'))
        ms=from_binned(y,TargetSpec(family='T2_mean_square'))[...,0]
        ac=from_binned(y,TargetSpec(family='T2_ac_variance'))[...,0]
        np.testing.assert_allclose(ms,y.mean(-1)**2+ac,atol=1e-14)
        np.testing.assert_allclose(centered.mean(-1),0,atol=1e-14)
        s=TargetSpec(family='T2_log_ac_variance')
        self.assertEqual(floor_activations(np.ones((3,64)),s),3)
        np.testing.assert_array_equal(from_binned(np.ones((3,64)),s),np.full((3,1),np.log(1e-40)))

    def test_t3_orthonormal_future_only_frequency_axes(self):
        spec=TargetSpec(family='T3',duration_ms=1000)
        a=extract_target(self.x,self.t,spec)
        changed=self.x.copy();changed[self.t<=6010]=123456.
        np.testing.assert_array_equal(a,extract_target(changed,self.t,spec))
        changed[self.t<=6010]=np.nan
        np.testing.assert_array_equal(a,extract_target(changed,self.t,spec))
        keep=self.t>6010
        np.testing.assert_array_equal(a,extract_target(self.x[keep],self.t[keep],spec))
        n=np.arange(16000);freq=np.arange(4,31)
        basis=np.stack((np.cos(2*np.pi*freq[:,None]*n/16000),np.sin(2*np.pi*freq[:,None]*n/16000)),axis=1).reshape(54,16000)*np.sqrt(2/16000)
        np.testing.assert_allclose(basis@basis.T,np.eye(54),atol=1e-13)
        raw=self.x[(self.t>6010)&(self.t<=7010)]
        np.testing.assert_allclose(a,basis@raw,atol=1e-13)
        aa=axes(spec);self.assertEqual(len(aa['target_time_ms']),0)
        self.assertEqual(aa['target_component'][1],'sin')
        self.assertEqual(len(aa['target_frequency_hz']),54)

    def test_full_fft_parseval_and_covariance_trace(self):
        for n in (63,64):
            y=np.random.default_rng(6).normal(size=(20,n));z=full_real_fft(y)
            np.testing.assert_allclose(inverse_full_real_fft(z),y,atol=2e-15)
            np.testing.assert_allclose(np.sum(z*z,axis=-1),np.sum(y*y,axis=-1),rtol=1e-14)
            self.assertAlmostEqual(np.var(y,axis=0,ddof=1).sum(),np.var(z,axis=0,ddof=1).sum(),places=12)


class SavedArrayDerivationTests(unittest.TestCase):
    def test_missing_native_supported_work_and_anchor_preservation(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';root=Path(temp)/'analysis';source.mkdir()
            (source/'resolved_config.json').write_text('{}')
            paneldir=source/'panels/primary';paneldir.mkdir(parents=True)
            rng=np.random.default_rng(10);norm={'x_offset':0.,'x_scale':1.,'y_offset':[0.]*64,'y_scale':1.,'fit_split':'train'}
            norm['id']=canonical_hash(norm)
            (paneldir/'manifest.json').write_text(json.dumps({'normalization':norm,'input':{'history_s':4},'target_fingerprint':'original'}))
            originals={}
            for si,split in enumerate(('train','development','evaluation')):
                n=4;m=2 if split=='evaluation' else 1
                xp=rng.normal(size=(n,2000));yp=rng.normal(size=(n,m,64));ids=np.array([f'{split}{i}' for i in range(n)])
                p={'X':xp.astype(np.float32),'X_physical':xp,'Y':yp.copy(),'Y_physical':yp,
                   'reference_Y':np.zeros((n,0,64)),'reference_Y_physical':np.zeros((n,0,64)),
                   'future_mask':np.ones((n,m),bool),'reference_mask':np.zeros((n,0),bool),
                   'future_id':np.array([[f'{c}-score-{j}' for j in range(m)] for c in ids]),'reference_future_id':np.full((n,0),'',dtype='U128'),
                   'context_id':ids,'structure_id':np.arange(n)//2+si*2,'regime':np.array(['R0','R1','R0','R1']),
                   'input_time_ms':np.arange(-3998,1,2,dtype=float),'target_time_ms':np.arange(12,139,2,dtype=float),
                   'acquisition_clipping_fraction':np.zeros(n),'task_rhythm':np.array([0,1,0,1]),'task_carrier':np.array([-1,0,-1,0])}
                np.savez_compressed(paneldir/f'{split}.npz',**p);originals[split]=p
            result=derive({},root,source)
            self.assertEqual(result['completed_jobs'],8)
            self.assertFalse(result['native_verified'])
            panels,manifest=load_panel(root,'A0__T0_l10_d128')
            self.assertTrue(manifest['frozen_legacy_anchor_arrays_preserved'])
            for split,p in originals.items():
                for key,value in p.items():np.testing.assert_array_equal(panels[split][key],value)
            status={r['id']:r['status'] for r in result['jobs']}
            self.assertEqual(status['A2__T0_l10_d128'],'unavailable')
            self.assertEqual(status['A0__T3_l10_d1000'],'unavailable')
            centered,_=load_panel(root,'A0__T1_l10_d128')
            np.testing.assert_allclose(centered['evaluation']['Y_physical'].mean(-1),0,atol=1e-14)
            # Changing only evaluation labels cannot change training normalization.
            p=originals['evaluation'];p['Y_physical']*=10;p['Y']*=10
            np.savez_compressed(paneldir/'evaluation.npz',**p)
            with self.assertRaisesRegex(ValueError,'dependency changed|fingerprint mismatch'):
                derive({},root,source,resume=True)
            derive({},Path(temp)/'analysis2',source)
            _,other=load_panel(Path(temp)/'analysis2','A0__T1_l10_d128')
            _,first=load_panel(root,'A0__T1_l10_d128')
            self.assertEqual(other['normalization'],first['normalization'])


if __name__=='__main__':unittest.main()
