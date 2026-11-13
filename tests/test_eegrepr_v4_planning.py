"""Bounded panel, seed/pool support and explicit execution-gate contracts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from experiments.eeg_representation.h1.v4.config import resolve,load_config,contexts,counts
from experiments.eeg_representation.h1.v4.generation import adapter,generate
from experiments.eeg_representation.h1.v4.planning import (
    registry_check,plan,require_execution,measured_gate,freeze_protocol)
from experiments.eeg_representation.h1.v4.artifacts import environment,source_hash
from experiments.eeg_representation.h1.dataset import trial_spec
from experiments.eeg_representation.h1.config import context_specs,cutoff_ms,stop_ms
from experiments.eeg_representation.h1.targets import canonical_hash
from experiments.eeg_representation.storage import atomic_json,read_json

CONFIG=Path(__file__).resolve().parents[1]/'configs/eeg_representation/v4'


class BoundedPlanningTests(unittest.TestCase):
    def test_default_design_exact_counts_support_and_disjoint_reference_rule(self):
        c=resolve({});n=counts(c)
        self.assertEqual(n['total_episodes'],832)
        self.assertEqual(n['score_by_split'],dict(train=256,development=64,evaluation=256))
        self.assertEqual(n['reference_episodes'],256);self.assertEqual(n['reference_contexts'],8)
        reference=[r for r in contexts(c) if r['reference_futures']]
        self.assertEqual({r['structure_id'] for r in reference},{36012,36013,36014,36015})
        self.assertEqual({r['regime']['name'] for r in reference},{'R0','Ralpha'})
        old=adapter(c);self.assertEqual(cutoff_ms(old),6000.);self.assertEqual(stop_ms(old),7010.)
        self.assertEqual({r['context_id'] for r in context_specs(old)},{r['context_id'] for r in contexts(c)})
        self.assertEqual(sum(r['score_futures']+r['reference_futures'] for r in context_specs(old)),832)
        self.assertFalse(old['fixed_budget_extension']['enabled'])

    def test_adapter_seed_pairing_and_reference_independence(self):
        old=adapter(resolve({}));rows=context_specs(old)
        pair=[r for r in rows if r['structure_id']==36012 and r['prefix']==0]
        specs=[trial_spec(old,r,'score',0) for r in pair]
        self.assertEqual(len({s['trial_seed'] for s in specs}),1)
        self.assertEqual(len({s['phase_seed'] for s in specs}),1)
        self.assertEqual(len({s['future_seed'] for s in specs}),1)
        reference=trial_spec(old,pair[0],'reference',0)
        self.assertEqual(reference['trial_seed'],specs[0]['trial_seed'])
        for key in ('future_seed','future_phase_seed','future_burst_seed'):
            self.assertNotEqual(reference[key],specs[0][key])
        for pool,total in [('score',8),('reference',32)]:
            seeds={trial_spec(old,pair[0],pool,f)['future_seed'] for f in range(total)}
            self.assertEqual(len(seeds),total)
        a={trial_spec(old,pair[0],'score',f)['future_seed'] for f in range(8)}
        b={trial_spec(old,pair[0],'reference',f)['future_seed'] for f in range(32)}
        self.assertFalse(a&b)
        training=[r for r in rows if r['structure_id']==36000 and r['regime']['name']=='R0']
        self.assertNotEqual(trial_spec(old,training[0],'score',0)['trial_seed'],trial_spec(old,training[1],'score',0)['trial_seed'])

    def test_registry_and_timing_design(self):
        c=resolve({});self.assertEqual(len(registry_check(c)['circuit_ids']),20)
        bad=deepcopy(c);bad['data']['structures']['evaluation'][-1]=36080
        with self.assertRaises(ValueError):registry_check(bad)
        gate=load_config(CONFIG/'timing_gate.yaml');self.assertEqual(counts(gate)['total_episodes'],8)
        self.assertEqual(registry_check(gate)['circuit_ids'],[36080,36081])
        self.assertEqual(gate['simulation']['population_e'],32)
        self.assertEqual(gate['simulation']['population_i'],8)
        self.assertEqual(gate['simulation']['external_synapses_per_cell'],64)
        self.assertEqual(stop_ms(adapter(gate)),7010.)
        with self.assertRaises(ValueError):resolve({'stop_ms':7000.})
        with self.assertRaises(ValueError):resolve({'simulation':{'celsius':36.5}})

    def test_plan_cost_and_explicit_execution_are_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=resolve({});p=plan(c,root)
            self.assertAlmostEqual(p['summed_generation_hours'],832*88.71645*(7010/6138)/3600)
            self.assertFalse(p['eligible_for_generation'])
            self.assertIn('NOT execution authorization',p['cost_status'])
            with self.assertRaises(ValueError):require_execution(c,root,False)
            with self.assertRaises(ValueError):require_execution(c,root,True)
            # A boolean flip in a saved plan cannot authorize expensive generation.
            p['eligible_for_generation']=True;atomic_json(root/'plan.json',p)
            with self.assertRaisesRegex(ValueError,'plan hash'):require_execution(c,root,True)
            with patch('experiments.eeg_representation.h1.v4.generation.legacy_generate') as engine:
                with self.assertRaises(ValueError):generate(c,root)
                self.assertFalse(engine.called)

    def test_engineering_execution_still_needs_matching_saved_plan(self):
        with tempfile.TemporaryDirectory() as temp:
            c=load_config(CONFIG/'analytic_smoke.yaml');root=Path(temp);plan(c,root)
            self.assertTrue(require_execution(c,root,True)['eligible_for_generation'])
            with self.assertRaises(ValueError):require_execution(c,root,True,timing=True)
            changed=deepcopy(c);changed['namespace']+='-changed'
            with self.assertRaises(ValueError):require_execution(changed,root,True)
            with patch('experiments.eeg_representation.h1.v4.generation.legacy_generate',return_value={'status':'engineering fixture'}) as engine:
                result=generate(c,root,execute=True)
                self.assertEqual(result['status'],'engineering fixture');self.assertTrue(engine.called)

    def test_measured_gate_rejects_host_duration_and_incomplete_native_record(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);c=resolve({});gate=load_config(CONFIG/'timing_gate.yaml')
            atomic_json(root/'resolved_config.json',gate)
            p=dict(analysis_source_hash=source_hash(),config_hash=canonical_hash(gate),environment=environment())
            p['environment']['host']='not-this-host';atomic_json(root/'provenance.json',p)
            with self.assertRaisesRegex(ValueError,'host changed'):measured_gate(c,root)
            p['environment']=environment();atomic_json(root/'provenance.json',p)
            changed=deepcopy(c);changed['stop_ms']+=1
            with self.assertRaisesRegex(ValueError,'duration mismatch'):measured_gate(changed,root)
            with self.assertRaises((ValueError,FileNotFoundError)):measured_gate(c,root)

    def test_failed_qualification_never_freezes(self):
        with tempfile.TemporaryDirectory() as temp:
            q=Path(temp)/'q';out=Path(temp)/'out'
            atomic_json(q/'measurement/qualification.json',{'passed':False})
            atomic_json(q/'models/manifest.json',{'neural_path_control':{'passed':True}})
            with self.assertRaises(ValueError):freeze_protocol(resolve({}),out,q)
            self.assertFalse((out/'protocol.json').exists())


class CommonEventAdapterTests(unittest.TestCase):
    def setUp(self):
        import os
        self.environment=patch.dict(os.environ,{'MAIN_PATH':str(CONFIG.parents[2])})
        self.environment.start();self.addCleanup(self.environment.stop)

    def test_default_is_exact_and_unsplit_common_events_are_valid(self):
        from setup.circuits.ballnstick.utils import generate_split_background_spike_train as original
        from experiments.eeg_representation.h1.v4.driver import split_common_events
        kwargs=dict(start_ms=0.,stop_ms=1000.,interval_ms=40.,history_seed=41)
        np.testing.assert_array_equal(original(**kwargs),split_common_events(original,kwargs,fraction=0,history_seed=1,future_seed=2))
        values=split_common_events(original,kwargs,fraction=.02,history_seed=1,future_seed=2)
        self.assertTrue(np.all(np.diff(values)>=0))
        self.assertTrue(np.all((values>=0)&(values<1000)))

    def test_split_prefix_and_equal_rate_groups(self):
        from setup.circuits.ballnstick.utils import generate_split_background_spike_train as original
        from experiments.eeg_representation.h1.v4.driver import split_common_events
        kwargs=dict(start_ms=0.,stop_ms=2000.,interval_ms=40.,history_seed=41,future_start_ms=1000.,future_seed=42)
        a=split_common_events(original,kwargs,fraction=.02,history_seed=1,future_seed=2)
        b=split_common_events(original,dict(kwargs,future_seed=92),fraction=.02,history_seed=1,future_seed=12)
        np.testing.assert_array_equal(a[a<1000],b[b<1000])
        self.assertFalse(np.array_equal(a[a>=1000],b[b>=1000]))
        cache={}
        first=split_common_events(original,kwargs,fraction=1,history_seed=1,future_seed=2,cache=cache)
        second=split_common_events(original,dict(kwargs,history_seed=999),fraction=1,history_seed=1,future_seed=2,cache=cache)
        np.testing.assert_array_equal(first,second)
        split_common_events(original,dict(kwargs,interval_ms=30.),fraction=1,history_seed=1,future_seed=2,cache=cache)
        self.assertEqual(len(cache),2)



class CommonRegimePairingTests(unittest.TestCase):
    def test_common_rng_pairing_across_regimes_and_separate_rate_groups(self):
        from experiments.eeg_representation.h1.v4.driver import split_common_events
        seen=[]
        def original(**kwargs):
            seen.append(kwargs.copy())
            return np.asarray([])
        base=dict(start_ms=0.,stop_ms=2000.,interval_ms=40.,history_seed=4,
                  future_start_ms=1000.,future_seed=5)
        for rate,frequency,depth in [(40.,0.,0.),(40.,10.,.04),(30.,10.,.04)]:
            split_common_events(original,dict(base,interval_ms=rate,frequency_hz=frequency,
                modulation_depth=depth,rhythm_enabled=depth>0),fraction=.02,history_seed=10,future_seed=20)
        self.assertEqual(seen[0]['history_seed'],seen[2]['history_seed'])
        self.assertEqual(seen[0]['future_seed'],seen[2]['future_seed'])
        self.assertNotEqual(seen[2]['history_seed'],seen[4]['history_seed'])
        self.assertNotEqual(seen[2]['future_seed'],seen[4]['future_seed'])

if __name__=='__main__':unittest.main()
