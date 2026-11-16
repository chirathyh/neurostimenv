"""V6 inferential contracts, finite-reference controls and precision gating."""
import unittest
import json
import numpy as np
from scipy.stats import t
from experiments.eeg_representation.h1.v6.statistics import (
    family_a,family_b,stable_mean,stable_sd,aggregate_circuit_losses,
    average_fitting_repeats,select_primary_futures,validate_split_ids,
    paired_ratio_difference,ratio_of_means_interval)
from experiments.eeg_representation.h1.v6.replication import (
    finite_reference_comparison,analyze_futures,paired_regime_future_indices,
    paired_future_bootstrap_losses,reference_controls,analyze_panel,synchronized_jackknife)
from experiments.eeg_representation.h1.v6.precision import (
    upper_sd,required_n,planned_half_width,build_precision_certificate,synthetic_controls,propose_design)
from experiments.eeg_representation.h1.v6.audit import audit_native_metadata


class InferenceContracts(unittest.TestCase):
    def test_intersection_union_requires_both_practical_margins(self):
        z=np.linspace(-1,1,192)
        fail=family_a(.5+.05*z,.01+.01*z)
        self.assertFalse(fail['passed'])
        self.assertEqual(fail['iut_p'],max(fail['task']['one_sided_p_greater_than_margin'],fail['reconstruction']['one_sided_p_greater_than_margin']))
        good=family_a(.5+.05*z,.05+.01*z)
        self.assertTrue(good['passed'])
        self.assertAlmostEqual(good['task']['critical_t'],t.ppf(.975,191))
        self.assertEqual(good['alpha'],.025)
        json.dumps(good,allow_nan=False)
        self.assertEqual(good['circuit_bootstrap_sensitivity']['n_circuits'],192)
        self.assertLess(good['iut_p'],.025)
        with self.assertRaises(ValueError):family_a(.5+.05*z,.05+.01*z,alpha=.05)

    def test_reconstruction_denominator_influence(self):
        base=np.linspace(.2,3,192);raw=.05*base+np.linspace(-.01,.01,192)
        out=family_a(np.full(192,.5),raw,reconstruction_baseline_risks=base)
        ratio=raw.mean()/base.mean();influence=(raw-ratio*base)/base.mean()
        self.assertAlmostEqual(out['reconstruction']['estimate'],ratio)
        self.assertAlmostEqual(out['reconstruction']['standard_error'],influence.std(ddof=1)/np.sqrt(192))
        self.assertIn('denominator uncertainty included',out['reconstruction_denominator_scope'])

    def test_ratio_is_ratio_of_means_not_average_of_ratios(self):
        a=np.array([.5,9,1.,8.]);b=np.array([1.,10,2.,10.])
        out=family_b(a,b)
        self.assertAlmostEqual(out['estimate'],100*(1-a.mean()/b.mean()))
        self.assertGreater(abs(out['estimate']-np.mean(100*(1-a/b))),5.)
        influence=-100*(a-(a.mean()/b.mean())*b)/b.mean()
        self.assertAlmostEqual(out['standard_error'],influence.std(ddof=1)/2)
        self.assertAlmostEqual(out['critical_t'],t.ppf(.9875,3))

    def test_equivalence_is_not_nonsignificance(self):
        base=np.linspace(.5,2,192)
        self.assertEqual(family_b(base,base)['decision'],'practical equivalence')
        self.assertEqual(family_b(.98*base,base)['decision'],'meaningful improvement')
        self.assertEqual(family_b(1.02*base,base)['decision'],'meaningful worsening')
        noisy=np.ones(8)+np.array([-.2,.2]*4)
        self.assertEqual(family_b(noisy,np.ones(8))['decision'],'inconclusive')
        self.assertEqual(paired_ratio_difference(.98*base,base,.98*base,base)['within_tolerance'],True)
        with self.assertRaises(ValueError):family_b(np.ones(8),np.zeros(8))
        with self.assertRaises(ValueError):paired_ratio_difference(np.ones(8),np.zeros(8),np.ones(8),np.ones(8))

    def test_five_seed_losses_average_within_circuit(self):
        losses=np.stack([np.ones(12)*v for v in [.5,.7,.9,1.1,1.3]])
        out=family_b(losses,np.ones(12),neural=True)
        self.assertEqual(out['n_circuits'],12)
        self.assertAlmostEqual(out['estimate'],10.)
        json.dumps(out,allow_nan=False)
        self.assertEqual(out['fitting_repeats_averaged'],5)
        with self.assertRaises(ValueError):family_b(losses[:3],np.ones(12),neural=True)
        # This averages losses (0.9), not squared ensemble predictions.
        np.testing.assert_allclose(average_fitting_repeats(losses,neural=True),.9)

    def test_fixed_condition_prefix_weighting_and_missing_cells(self):
        losses=np.array([[1.,3.],[3.,5.],[10.,12.],[2.,4.],[4.,6.],[20.,22.]])
        sid=['a','a','a','b','b','b'];conditions=['r0','r0','r1']*2;prefix=[0,1,0]*2
        out=aggregate_circuit_losses(losses,sid,conditions,prefix)
        np.testing.assert_allclose(out['values'],[7.,12.5])
        with self.assertRaises(ValueError):aggregate_circuit_losses(losses,sid,conditions,prefix,expected_prefixes=2)
        with self.assertRaises(ValueError):aggregate_circuit_losses(losses,sid,conditions,prefix,expected_conditions=['r0','r1','r2'])

    def test_primary_futures_are_registered_first_four_and_immutable_to_extras(self):
        ids=np.array([[f'c-score-{j:04d}'for j in [5,3,0,7,1,4,6,2]]])
        y=np.array([[[j]for j in [5,3,0,7,1,4,6,2]]],float)
        panel=dict(Y=y,future_id=ids,future_mask=np.ones((1,8),bool))
        out=select_primary_futures(panel)
        np.testing.assert_array_equal(out['Y'][0,:,0],[0,1,2,3])
        panel['Y'][0,[0,3,5,6],0]=1e20
        np.testing.assert_array_equal(select_primary_futures(panel)['Y'],out['Y'])
        panel['future_mask'][0,0]=False;panel['Y'][0,0]=np.nan
        np.testing.assert_array_equal(select_primary_futures(panel)['Y'],out['Y'])
        panel['future_mask'][0,7]=False
        with self.assertRaises(ValueError):select_primary_futures(panel)
        with self.assertRaises(ValueError):select_primary_futures(out,count=8)

    def test_calibration_and_test_are_distinct_circuits(self):
        self.assertTrue(validate_split_ids(dict(train=[1],development=[2],calibration=[3],test=[4]),inspected_test_ids=[8])['pairwise_disjoint'])
        with self.assertRaises(ValueError):validate_split_ids(dict(train=[1],development=[2],calibration=[4],test=[4]))
        with self.assertRaises(ValueError):validate_split_ids(dict(test=[8]),inspected_test_ids=[8])

    def test_float64_scale_safe_mean_and_sd(self):
        x=np.array([1e200,1e200+1e190,1e200-1e190])
        self.assertTrue(np.isfinite(stable_mean(x)))
        self.assertTrue(np.isfinite(stable_sd(x)))
        self.assertAlmostEqual(stable_sd(x)/1e190,1.,places=5)
        with self.assertRaises(ValueError):family_b([np.nan,1],[1,1])


class FuturePrecision(unittest.TestCase):
    def test_finite_L_correction_retains_negative_and_primary_separate(self):
        x=np.array([[-1.],[1.]]);r=np.tile([[-2.],[2.]],(64,1))
        out=finite_reference_comparison(x,r,scoring_ids=['s0','s1'],reference_ids=[f'r{i}'for i in range(128)],bootstrap=0)
        self.assertLess(out['reference_noise_corrected_mean_mse'],0)
        self.assertAlmostEqual(out['reference_noise_corrected_mean_mse'],-r.var(ddof=1)/128)
        self.assertIn('reference_variance_jackknife_sampling_variance',out)
        self.assertFalse(out['reference_is_exact_truth'])
        with self.assertRaises(ValueError):finite_reference_comparison(x,r[:32],scoring_ids=['s0','s1'],reference_ids=[f'r{i}'for i in range(32)],bootstrap=0)
        with self.assertRaises(ValueError):finite_reference_comparison(x,r,scoring_ids=['r0','s1'],reference_ids=[f'r{i}'for i in range(128)],bootstrap=0)

    def test_nested_and_disjoint_groups_record_shared_reference_dependence(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(32,2));ref=rng.normal(size=(128,2))
        out=analyze_futures(x,ref,scoring_ids=[f's{i}'for i in range(32)],reference_ids=[f'r{i}'for i in range(128)],prediction=np.zeros(2),bootstrap=0)
        previous=[]
        for row in out['rows']:
            self.assertEqual(row['nested_indices'][:len(previous)],previous);previous=row['nested_indices']
            seen=[]
            for g in row['disjoint_scoring_groups']:seen.extend(g['indices'])
            self.assertEqual(len(set(seen)),len(seen));self.assertTrue(row['groups_share_reference'])
            self.assertFalse(row['groups_independent_for_inference'])
            self.assertFalse(row['new_future_uncertainty_is_zero'])
            if row['M']>1:self.assertAlmostEqual(row['nested']['finite_m_identity_residual'],0.,places=12)
        self.assertEqual(out['rows'][-1]['n_disjoint_scoring_groups'],1)

    def test_regime_pairing_cannot_fake_more_mc_precision(self):
        rng=np.random.default_rng(7);loss=rng.uniform(size=(2,16))
        first=paired_future_bootstrap_losses(loss,[0,1],['r0','r0'],[0,0],repeats=200)
        repeated=paired_future_bootstrap_losses(np.repeat(loss,4,0),np.repeat([0,1],4),list(range(4))*2,[0]*8,repeats=200)
        self.assertAlmostEqual(first['conditional_mc_standard_error'],repeated['conditional_mc_standard_error'])
        indices=paired_regime_future_indices([0,0,1,1],16,repeats=20)
        np.testing.assert_array_equal(indices[:,0],indices[:,1]);np.testing.assert_array_equal(indices[:,2],indices[:,3])
        self.assertFalse(np.array_equal(indices[:,0],indices[:,2]))
        prefix=paired_regime_future_indices([0,0,0,0],16,prefix_ids=[0,0,1,1],repeats=20)
        np.testing.assert_array_equal(prefix[:,0],prefix[:,1]);np.testing.assert_array_equal(prefix[:,2],prefix[:,3])
        self.assertFalse(np.array_equal(prefix[:,0],prefix[:,2]))

    def test_jackknife_pairs_regimes_and_adds_independent_prefix_variances(self):
        values=np.array([.5,1,2,3.])
        def panel(regimes,prefixes):
            cid=np.array([f's1-p{p:04d}-{r}'for r,p in zip(regimes,prefixes)])
            return dict(structure_id=np.ones(len(cid),int),context_id=cid,regime=np.array(regimes),
                future_id=np.array([[f'{c}-score-{j:04d}'for j in range(4)]for c in cid]),future_mask=np.ones((len(cid),4),bool))
        p1=panel(['R0'],[0]);one=synchronized_jackknife(values[None,:,None],np.zeros((1,1)),p1)
        p4=panel(['R0','Ralpha','Rbeta_low','Rbeta_high'],[0]*4)
        four=synchronized_jackknife(np.tile(values[None,:,None],(4,1,1)),np.zeros((4,1)),p4)
        p8=panel(['R0','Ralpha','Rbeta_low','Rbeta_high']*2,[0]*4+[1]*4)
        eight=synchronized_jackknife(np.tile(values[None,:,None],(8,1,1)),np.zeros((8,1)),p8)
        json.dumps(eight,allow_nan=False)
        for key in ('mean_risk','state_floor','corrected_excess'):
            self.assertAlmostEqual(one['pooled_conditional_mc_se'][key],four['pooled_conditional_mc_se'][key])
            self.assertAlmostEqual(one['pooled_conditional_mc_se'][key]/np.sqrt(2),eight['pooled_conditional_mc_se'][key])
        self.assertAlmostEqual(one['pooled_conditional_mc_se']['mean_risk'],(values**2).std(ddof=1)/2)

    def test_panel_future_order_preserves_regime_pairing(self):
        rng=np.random.default_rng(8);x=rng.normal(size=(32,2));r=rng.normal(size=(128,2))
        sx=rng.permutation(32);rx=rng.permutation(128)
        cids=['s1-p0000-R0','s1-p0000-Ralpha']
        panel=dict(Y=np.stack([x,x[sx]]),reference_Y=np.stack([r,r[rx]]),
            future_mask=np.ones((2,32),bool),reference_mask=np.ones((2,128),bool),
            future_id=np.array([[f'{cids[0]}-score-{i:04d}'for i in range(32)],
                                [f'{cids[1]}-score-{i:04d}'for i in sx]]),
            reference_future_id=np.array([[f'{cids[0]}-reference-{i:04d}'for i in range(128)],
                                          [f'{cids[1]}-reference-{i:04d}'for i in rx]]),
            context_id=np.array(cids),structure_id=np.array([1,1]))
        out=analyze_panel(panel,np.zeros((2,2)),selected_context_ids=cids,bootstrap=0)
        self.assertEqual(out['n_circuits'],1)
        json.dumps(out,allow_nan=False)
        first,second=out['records']
        for a,b in zip(first['rows'],second['rows']):
            self.assertEqual(a['nested_indices'],b['nested_indices'])
            self.assertEqual(a['nested']['reference_noise_corrected_mean_mse'],b['nested']['reference_noise_corrected_mean_mse'])
        panel['reference_future_id'][1,0]=panel['reference_future_id'][1,1]
        with self.assertRaises(ValueError):analyze_panel(panel,np.zeros((2,2)),selected_context_ids=cids,bootstrap=0)

    def test_independent_gaussian_poisson_reference_controls(self):
        out=reference_controls(repetitions=2000)
        json.dumps(out,allow_nan=False)
        for row in out['controls']:
            self.assertTrue(row['unbiased_mean_control_pass'])
            self.assertTrue(row['variance_correction_control_pass'])
            self.assertTrue(.92<row['actual_welch_interval_coverage']<.98)
            self.assertGreater(row['known_law_interval_coverage'],.93)
            self.assertLess(row['known_law_interval_coverage'],.995)


class PrecisionPlanning(unittest.TestCase):
    def test_upper_sd_and_t_required_N_not_favorable_choice(self):
        bound=upper_sd(1.575,8)
        self.assertAlmostEqual(bound['inflation'],1.797,delta=.002)
        self.assertAlmostEqual(bound['upper_sd'],2.831,delta=.004)
        n=required_n(bound['upper_sd'],.5)
        self.assertLessEqual(planned_half_width(bound['upper_sd'],n),.5)
        self.assertGreater(planned_half_width(bound['upper_sd'],n-1),.5)
        self.assertLess(planned_half_width(bound['upper_sd'],192),.47)

    def test_historical_certificate_cannot_freeze_missing_phase0(self):
        components=[]
        for name,family,sd,width in [('task','A',.5,.1),('reconstruction','A',.02,.01),('risk','B',2.,.5)]:
            components.append(dict(name=name,family=family,upper_sd=sd,standardized_upper_sd=sd/width,target_half_width=width))
        out=build_precision_certificate(components,N=192,phase0_complete=False,simulation_repetitions=200)
        self.assertFalse(out['eligible_to_freeze']);self.assertEqual(out['status'],'provisional')
        self.assertEqual(out['worst_standardized_uncertainty_component'],'task')
        incomplete=build_precision_certificate(components,N=192,phase0_complete=True,phase0_reasons=['one declared policy incomplete'],simulation_repetitions=200)
        self.assertFalse(incomplete['eligible_to_freeze'])
        components[-1]['upper_sd']=10;components[-1]['standardized_upper_sd']=20
        out=build_precision_certificate(components,N=192,phase0_complete=True,simulation_repetitions=200)
        self.assertFalse(out['precision_sufficient']);self.assertEqual(out['planned_N'],192)
        self.assertGreater(out['minimum_N_for_all_targets'],192)
        self.assertEqual(out['controls']['status'],'incomplete')
        self.assertTrue(any(r.get('status')=='unavailable'for r in out['controls']['results']))

    def test_secondary_uncertainty_is_reported_but_not_primary_gate(self):
        components=[dict(name=n,family=f,upper_sd=sd,standardized_upper_sd=sd/w,target_half_width=w)
            for n,f,sd,w in [('task','A',.1,.1),('reconstruction','A',.001,.01),('risk','B',1.,.5),('old','secondary',10.,.1)]]
        out=build_precision_certificate(components,N=192,phase0_complete=True,simulation_repetitions=200)
        self.assertTrue(out['precision_sufficient'])
        self.assertFalse(out['components'][-1]['required_for_primary_precision'])
        self.assertNotEqual(out['worst_standardized_uncertainty_component'],'old')

    def test_certificate_adapter_requires_complete_phase0_not_just_risk_rows(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from unittest.mock import patch
        from experiments.eeg_representation.h1.v6.precision import certificate
        from experiments.eeg_representation.h1.v6.config import resolve
        provisional=dict(status='complete',phase0_complete=True,eligible_to_freeze=True,flags=[])
        with TemporaryDirectory() as directory:
            root=Path(directory)
            with patch('experiments.eeg_representation.h1.v6.precision.certificate_from_archive',return_value=dict(provisional,flags=[])), patch('experiments.eeg_representation.h1.v6.planning.qualify_phase0',return_value=dict(status='incomplete',passed=False,missing=['same neural path control missing'],evidence_hash='verified-artifacts')):
                out=certificate(resolve(),root/'output',root/'source.zip',root/'phase0')
            self.assertFalse(out['phase0_complete']);self.assertFalse(out['eligible_to_freeze'])
            self.assertEqual(out['status'],'provisional');self.assertEqual(out['phase0_evidence_hash'],'verified-artifacts')
            self.assertTrue(any('neural path' in f for f in out['flags']))
            json.dumps(out,allow_nan=False)
            with patch('experiments.eeg_representation.h1.v6.precision.certificate_from_archive',return_value=dict(provisional,flags=[])), patch('experiments.eeg_representation.h1.v6.planning.qualify_phase0',side_effect=ValueError('artifact mismatch')):
                out=certificate(resolve(),root/'output',root/'source.zip',root/'phase0')
            self.assertFalse(out['eligible_to_freeze'])
            self.assertTrue(any('artifact mismatch' in f for f in out['flags']))

    def test_precision_revision_is_registered_and_does_not_change_default(self):
        from experiments.eeg_representation.h1.v6.config import resolve
        original=resolve()
        cert=dict(minimum_N_for_all_targets=257,worst_standardized_uncertainty_component='M4',phase0_complete=False)
        out=propose_design(original,cert)
        json.dumps(out,allow_nan=False)
        self.assertEqual(out['status'],'proposed');self.assertEqual(out['proposed_N'],257)
        self.assertEqual(original['protocol']['test_circuits'],192)
        self.assertEqual(out['resolved_config']['protocol']['test_circuits'],257)
        self.assertEqual(out['counts']['new_episodes'],14816)
        self.assertFalse(out['execution_authorized']);self.assertFalse(out['phase0_complete'])
        self.assertEqual(len(set(out['resolved_config']['data']['structures']['evaluation'])),257)
        cert['minimum_N_for_all_targets']=500
        self.assertEqual(propose_design(original,cert)['status'],'unavailable')

    def test_independent_coverage_power_and_skew_controls(self):
        out=synthetic_controls(N=192,repetitions=2500)
        json.dumps(out,allow_nan=False)
        normal=[r for r in out['results']if r['law']=='Gaussian']
        for row in normal:
            if row['family']=='B':self.assertTrue(.955<row['coverage']<.99)
            elif 'null'in row['scenario']:self.assertLess(row['conjunction_rejection_rate'],.04)
        null=next(r for r in normal if r['family']=='B'and r['true_G_pp']==0)
        alternate=next(r for r in normal if r['family']=='B'and r['true_G_pp']==2)
        self.assertGreater(null['equivalence_rate'],.95)
        self.assertGreater(alternate['meaningful_improvement_rate'],.95)
        self.assertEqual(set(r['law']for r in out['results']),{'Gaussian','skew_exponential','rare_outlier'})
        self.assertFalse(out['universal_coverage_claim'])


class NativeMetadataAudit(unittest.TestCase):
    def test_actual_nested_fork_fields_and_future_streams_are_required(self):
        from copy import deepcopy
        rows=[]
        for regime in ['R0','Ralpha']:
            cid='s1-p0000-'+regime
            for i,pool in enumerate(['score','reference']):
                spec=dict(trial_id=f'{cid}-{pool}-0000',history_id=cid,structure_id='1',structure_seed=1,
                    future_id=pool+'-0000',split='evaluation',regime={'name':regime},future_start_ms=6000.,
                    trial_seed=10,phase_seed=20,envelope_seed=30,future_seed=40+i,future_phase_seed=50+i,future_burst_seed=60+i)
                fork={k:'a'*64 for k in ('physical_state_hash','pending_recurrent_hash','pending_external_hash','external_history_hash','prefix_spike_hash')}
                fork.update(cutoff_ms=6000,future_schedules_are_oracle_only=True)
                meta=dict(fork=fork,rng_namespaces=dict(structure=1,events=10,phase=20,burst=30,future_events=40+i,future_phase=50+i,future_burst=60+i),
                    native_dt_ms=.0625,total_duration_ms=7010,native_sample_count=112160,configured_celsius=6.3,effective_celsius=6.3,
                    initializations_per_episode=1,sample_boundary_convention='(start,stop]',regime={'name':regime})
                rows.append(dict(spec=spec,metadata=meta,context_id=cid,structure_id=1,trial_id=spec['trial_id'],
                    pool=pool,future_index=0,split='evaluation',path=spec['trial_id']+'.h5'))
        out=audit_native_metadata(rows)
        self.assertTrue(out['score_reference_streams_disjoint'])
        self.assertEqual(out['fork_contexts'],2)
        missing=deepcopy(rows);del missing[0]['metadata']['fork']['physical_state_hash']
        with self.assertRaises(ValueError):audit_native_metadata(missing)
        overlap=deepcopy(rows);overlap[1]['spec']['future_phase_seed']=50;overlap[1]['metadata']['rng_namespaces']['future_phase']=50
        with self.assertRaises(ValueError):audit_native_metadata(overlap)


if __name__=='__main__':unittest.main()
