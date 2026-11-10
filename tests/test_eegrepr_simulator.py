"""Causal data plumbing and optional actual NEURON integration."""
import importlib.util
import os
import random
import unittest
import numpy as np
from experiments.eeg_representation.preprocessing import causal_resample, PhysicalNormalizer, add_observation_noise
from experiments.eeg_representation.simulator import burst_envelope, generate_trial, _burst_events
from experiments.eeg_representation.forks import validate_fork_prefixes, conditional_variance_summary, target_block


class CausalPreprocessingTests(unittest.TestCase):
    def test_future_perturbation_does_not_change_causal_inputs(self):
        t=np.arange(1,16001)*.0625
        x=np.random.default_rng(31).normal(size=(len(t),1))*1e-14
        y=x.copy(); y[t>500]*=100000
        a,ta,meta=causal_resample(x,t)
        b,tb,_=causal_resample(y,t)
        np.testing.assert_array_equal(a[ta<=500],b[tb<=500])
        self.assertEqual(len(a),500); self.assertEqual(ta[0],2.)
        self.assertEqual(meta['group_delay_ms'],64.)

    def test_no_generic_floor_or_trial_standardization(self):
        a=np.array([[1e-20],[2e-20],[4e-20]])
        norm=PhysicalNormalizer.fit([a])
        self.assertLess(norm.scale_v[0],1e-19)
        self.assertAlmostEqual(norm.transform(a).var(),1.,places=6)
        with self.assertRaises(ValueError): PhysicalNormalizer.fit([np.ones((10,1))])
        with self.assertRaises(ValueError): PhysicalNormalizer.fit([a],split='test')

    def test_noise_uses_shared_training_reference(self):
        x=np.zeros((100000,1))
        out,meta=add_observation_noise(x,snr_db=10,reference_power_v2=[1e-20],seed=7)
        self.assertAlmostEqual(out.var()/1e-21,1.,delta=.02)
        self.assertEqual(meta['noise_covariance_v2'],[[1e-21]])

    def test_burst_fork_preserves_state_and_history(self):
        a=burst_envelope(stop_ms=10000,seed=14,depth=.16,future_start_ms=5000,future_seed=1)
        b=burst_envelope(stop_ms=10000,seed=14,depth=.16,future_start_ms=5000,future_seed=2)
        for i in (0,1): np.testing.assert_array_equal(a[i][a[0]<=5000],b[i][b[0]<=5000])
        self.assertTrue(np.all((a[1]>=0)&(a[1]<=.16)))
        self.assertFalse(np.array_equal(a[0],b[0]))

    def test_burst_future_cannot_change_the_thinning_bound_or_prefix(self):
        # Both histories are completely off. Only one future has a burst, so
        # deriving the thinning bound from the realized full envelope leaks.
        common=dict(start_ms=0.,stop_ms=200.,interval_ms=4.,seed=1,
            modulation_depth=.8,frequency_hz=10.,phase_rad=.3,
            envelope_times=np.array([0.,100.,200.]),envelope_max_depth=.8)
        a=_burst_events(**common,envelope_depths=np.array([0.,0.,0.]))
        b=_burst_events(**common,envelope_depths=np.array([0.,.8,.8]))
        np.testing.assert_array_equal(a[a<100],b[b<100])
        self.assertTrue(len(a[a<100])>0)

    def test_variance_and_leave_one_out_identity(self):
        x=np.random.default_rng(14).normal(size=(4000,5))*3
        result=conditional_variance_summary(x,bootstrap_samples=20)
        self.assertAlmostEqual(result['conditional_variance_trace_per_dimension'],9.,delta=.25)
        self.assertAlmostEqual(result['conditional_variance_trace_per_dimension'],result['loo_finite_ensemble_corrected_mse'])
        t=np.arange(2,2002,2); y=target_block(t[:,None],t,cutoff_ms=1000,lead_ms=2,block_ms=128)
        self.assertEqual(y.shape,(64,1)); self.assertEqual(y[0,0],1002)
        with self.assertRaises(ValueError): target_block(t[:,None],t,cutoff_ms=1900,lead_ms=2,block_ms=128)


@unittest.skipUnless(os.environ.get('EEGREPR_RUN_NEURON')=='1','Set EEGREPR_RUN_NEURON=1 for real integration')
class ActualSimulatorTests(unittest.TestCase):
    def test_exact_prefix_state_independent_future_streams(self):
        cfg={'simulation':{'burn_in_s':.05,'duration_s':.15,'population_sizes':{'E':2,'I':1},'external_synapses_per_cell':8}}
        spec={'structure_seed':31000,'trial_seed':310010,'phase_seed':310011,
              'regime':{'name':'Ralpha','frequency_hz':10.,'modulation_depth':.16,'diffusion':2.},
              'future_start_ms':100.,'future_seed':410010,'future_phase_seed':410011}
        np.random.seed(193); random.seed(33)
        nstate=np.random.get_state(); pstate=random.getstate()
        a=generate_trial(cfg,spec)
        np.testing.assert_array_equal(nstate[1],np.random.get_state()[1]); self.assertEqual(pstate,random.getstate())
        b=generate_trial(cfg,{**spec,'future_seed':410012})
        c=generate_trial(cfg,{**spec,'future_phase_seed':410013})
        report=validate_fork_prefixes([a,b,c])
        self.assertTrue(report['passed']); self.assertEqual(a['eeg_v'].shape,(3200,1))
        self.assertEqual(a['dipole_na_um'].shape,(3200,3)); self.assertEqual(a['eeg_v'].dtype,np.float64)
        np.testing.assert_array_equal(a['privileged']['phase_rad'],b['privileged']['phase_rad'])
        self.assertFalse(np.array_equal(a['privileged']['phase_rad'],c['privileged']['phase_rad']))
        self.assertNotEqual(a['metadata']['external_events_hash'],b['metadata']['external_events_hash'])
        self.assertEqual(a['metadata']['applied_field_v_per_m'],0.)
        self.assertEqual(a['metadata']['configured_celsius'],a['metadata']['effective_celsius'])
        with self.assertRaises(ValueError): generate_trial(cfg,{**spec,'future_start_ms':100.5})
        fine=generate_trial({'simulation':{**cfg['simulation'],'phase_grid_dt_ms':.5}},spec)
        fine2=generate_trial({'simulation':{**cfg['simulation'],'phase_grid_dt_ms':.5}},{**spec,'future_phase_seed':410013})
        self.assertTrue(validate_fork_prefixes([fine,fine2])['passed'])

    def test_pending_recurrent_events_are_preserved_in_actual_40_cell_model(self):
        cfg={'simulation':{'burn_in_s':.01,'duration_s':.09}}
        spec={'structure_seed':31000,'trial_seed':310010,'phase_seed':310011,
              'regime':{'name':'Ralpha','frequency_hz':10.,'modulation_depth':.04,'diffusion':.5},
              'future_start_ms':46.,'future_seed':410010,'future_phase_seed':410011}
        a=generate_trial(cfg,spec)
        b=generate_trial(cfg,{**spec,'future_seed':410012,'future_phase_seed':410013})
        report=validate_fork_prefixes([a,b])
        self.assertGreater(report['pending_recurrent_count'],0)
        # LFPy schedules direct NetCon.event(tdeliver), which ignores its delay.
        self.assertEqual(report['pending_external_count'],0)
        self.assertEqual(a['metadata']['population_sizes'],{'E':32,'I':8})

    def test_tonic_burst_and_kinetics_smokes(self):
        cfg={'simulation':{'burn_in_s':.01,'duration_s':.04,'population_sizes':{'E':2,'I':1},'external_synapses_per_cell':4}}
        base={'structure_seed':31001,'trial_seed':310020,'phase_seed':310021,
              'regime':{'name':'R0','frequency_hz':0,'modulation_depth':0}}
        a=generate_trial(cfg,base)
        self.assertEqual(a['metadata']['native_sample_count'],800)
        b=generate_trial(cfg,{**base,'regime':{'name':'burst','frequency_hz':16,'modulation_depth':.08,'burst':{'on_mean_ms':10,'off_mean_ms':10},'kinetics_scale':1.2,'shared_fraction':.5}})
        self.assertIn('burst_depth',b['privileged']); self.assertTrue(np.isfinite(b['eeg_v']).all())

if __name__=='__main__': unittest.main()
