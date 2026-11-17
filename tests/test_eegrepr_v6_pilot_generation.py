import json
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import multiprocessing
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import budget,design,generation


def charge(args):
    root,index=args
    try:
        attempt=budget.begin(root,'test',{'index':index})
        budget.finish(root,attempt,status='failed' if index%2 else 'complete',seconds=.01)
        return True
    except RuntimeError:return False


def fixture_rows():
    rows=[]
    regimes=[dict(name='R0',frequency_hz=0,modulation_depth=0),dict(name='Ralpha',frequency_hz=10,modulation_depth=.04),
        dict(name='Rbeta_low',frequency_hz=16,modulation_depth=.04),dict(name='Rbeta_high',frequency_hz=24,modulation_depth=.04)]
    seed=1
    for split,ids in [('train',range(36000,36008)),('development',range(36008,36012)),('evaluation',range(36012,36020))]:
        for sid in ids:
            for regime in regimes:
                cid=f's{sid}-p0000-{regime["name"]}'
                for pool,count in [('score',8 if split=='evaluation' else 1),('reference',32 if split=='evaluation' and sid<36016 and regime['name'] in ('R0','Ralpha') else 0)]:
                    for future in range(count):
                        spec=dict(history_id=cid,trial_id=cid+f'-{pool}-{future:04d}',future_id=f'{pool}-{future:04d}',structure_seed=sid,structure_id=str(sid),
                            regime=regime,split=split,future_start_ms=6000.,trial_seed=sid*2,phase_seed=sid*3,envelope_seed=sid*4,
                            future_seed=seed,future_phase_seed=seed+10000,future_burst_seed=seed+20000)
                        rows.append(dict(context_id=cid,trial_id=spec['trial_id'],structure_id=sid,split=split,pool=pool,future_index=future,spec=spec));seed+=1
    return rows


def fixture_trial(stop=12):
    t=np.arange(1,stop+1,dtype=float)
    return dict(time_ms=t,eeg_v=t[:,None],dipole_na_um=np.tile(t[:,None],(1,3)),
        metadata=dict(structure_hash='structure',fork=dict(cutoff_ms=10.,physical_state_hash='physical',external_history_hash='past',pending_recurrent_hash='pendingr',pending_external_hash='pendinge',prefix_spike_hash='spikes')),
        privileged=dict(cutoff_physical_state=np.arange(10),pending_recurrent_events=np.empty((0,6)),pending_external_events=np.empty((0,6)),phase_time_ms=t,phase_rad=t*.2))


class PilotGenerationTests(unittest.TestCase):
    def test_immutable_hard_budget_counts_failure_and_concurrent_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            budget.initialize(root,5)
            with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn')) as pool:
                results=list(pool.map(charge,[(root,i) for i in range(12)]))
            self.assertEqual(sum(results),5)
            summary=budget.summary(root)
            self.assertEqual(summary['attempted_native_calls'],5)
            self.assertEqual(summary['remaining_native_calls'],0)
            self.assertEqual(summary['failed_calls']+summary['successful_calls'],5)
            with self.assertRaises(RuntimeError):budget.begin(root,'retry',{})
            with self.assertRaises(ValueError):budget.initialize(root,6)
            with self.assertRaises(ValueError):budget.initialize(root,161)

    def test_interrupted_invocation_is_never_refunded(self):
        with tempfile.TemporaryDirectory() as root:
            budget.initialize(root,2);budget.begin(root,'interrupted',{})
            self.assertEqual(budget.summary(root)['unresolved_calls'],1)
            with self.assertRaises(RuntimeError):budget.require_capacity(root,2,'two new calls')
            budget.initialize(root,2)
            self.assertEqual(budget.summary(root)['attempted_native_calls'],1)

    def test_documented_implementation_calls_are_charged_once(self):
        with tempfile.TemporaryDirectory() as root:
            budget.initialize(root,160)
            evidence={'path':'engineering_invocations.json','sha256':'documented'}
            budget.record_implementation_debit(root,21,evidence)
            budget.record_implementation_debit(root,21,evidence)
            current=budget.summary(root)
            self.assertEqual(current['attempted_native_calls'],21)
            self.assertEqual(current['local_attempted_native_calls'],0)
            self.assertEqual(current['remaining_native_calls'],139)
            attempt=budget.begin(root,'new',{})
            self.assertEqual(budget.summary(root)['attempted_native_calls'],22)
            self.assertEqual(budget.summary(root)['local_attempted_native_calls'],1)
            with self.assertRaises(ValueError):budget.record_implementation_debit(root,20,evidence)
            with self.assertRaises(RuntimeError):budget.require_capacity(root,139,'cannot disregard implementation calls')

    def test_context_registration_is_outcome_blind_and_circuit_disjoint(self):
        rows=fixture_rows();a=design.select_contexts(rows);b=design.select_contexts(list(reversed(rows)))
        self.assertEqual(a,b)
        self.assertEqual(a['state_counts'],dict(train=32,development=8,calibration=8,evaluation=8))
        self.assertEqual([x['context_id'] for x in a['h2_contexts']],['s36012-p0000-R0','s36013-p0000-Ralpha'])
        self.assertEqual(len(a['concurrency_contexts']),8)
        ids=[sid for group in a['state_circuits'].values() for sid in group]
        self.assertEqual(len(ids),len(set(ids)))
        for split in a['state_counts']:
            selected=[x for x in a['state_cohort'] if x['split']==split]
            self.assertEqual(len(selected),a['state_counts'][split])
            self.assertEqual(len({x['regime']['name'] for x in selected}),4)

    def test_future_extension_copies_all_history_fields(self):
        old=fixture_rows()[0]['spec'];spec=design.continuation_spec(old,'original-namespace',8)
        changed={'future_seed','future_phase_seed','future_burst_seed','future_id','trial_id'}
        self.assertEqual({k:v for k,v in spec.items() if k not in changed},{k:v for k,v in old.items() if k not in changed})
        self.assertEqual(spec['future_id'],'score-0008')
        with self.assertRaises(ValueError):design.continuation_spec(old,'ns',32)
        collision={**spec,'future_seed':old['future_seed']}
        with self.assertRaises(ValueError):design.verify_pool_seeds([{'spec':old}],[collision])

    def test_prefix_recovery_allows_shorter_tail_but_rejects_any_past_difference(self):
        original=fixture_trial(12);recovered=fixture_trial(11)
        self.assertTrue(generation.verify_prefix(original,recovered,10)['passed'])
        changed=deepcopy(recovered);changed['eeg_v'][8,0]+=1
        with self.assertRaises(ValueError):generation.verify_prefix(original,changed,10)
        changed=deepcopy(recovered);changed['metadata']['fork']['pending_recurrent_hash']='different'
        with self.assertRaises(ValueError):generation.verify_prefix(original,changed,10)
        changed=deepcopy(recovered);changed['privileged']['cutoff_physical_state'][0]=99
        with self.assertRaises(ValueError):generation.verify_prefix(original,changed,10)
        changed=deepcopy(recovered);changed['eeg_v'][-1,0]+=1
        self.assertTrue(generation.verify_prefix(original,changed,10)['passed'])

    def test_hardware_blocker_does_not_silently_reduce_workers(self):
        with tempfile.TemporaryDirectory() as root,patch.object(generation.os,'sched_getaffinity',return_value={0,1,2,3}):
            result=generation.preflight(dict(workers=8,threads_per_worker=1,minimum_available_ram_bytes=0,minimum_free_bytes=0),root)
            self.assertFalse(result['passed']);self.assertEqual(result['requested_workers'],8)
            self.assertIn('CPU affinity',result['blockers'][0])
            with self.assertRaises(ValueError):generation.preflight(dict(workers=4),root)

    def test_completed_orphan_resume_does_not_invoke_simulator_again(self):
        from experiments.eeg_representation.h1.v6_pilot import state
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root,160)
            task=dict(root=str(root),stage='mocked-engineering',key='one',spec=dict(history_id='c',structure_seed=1,future_start_ms=10.),
                legacy_config={},original_config_hash='historical',design_hash='frozen',reference=None)
            trial=fixture_trial();payload=state._seal_synthetic(state._synthetic_payload())
            with patch.object(state,'generate_with_state',return_value=(trial,payload)) as generate:
                first=generation._worker(task)
                second=generation._worker(task)
                self.assertEqual(generate.call_count,1)
                self.assertEqual(first['native_sha256'],second['native_sha256'])
                generation._record_path(root,task['stage'],task['key']).unlink()
                recovered=generation._worker(task)
                self.assertTrue(recovered['recovered_complete_orphan'])
                self.assertEqual(generate.call_count,1)
                self.assertEqual(budget.summary(root)['attempted_native_calls'],1)

    def test_completed_resume_at_hard_cap_recovers_orphan_before_capacity_or_hardware(self):
        from experiments.eeg_representation.h1.v6_pilot import state
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root,1)
            task=dict(root=str(root),stage='recovery',key='one',spec=dict(history_id='c',structure_seed=1,future_start_ms=10.),
                legacy_config={},original_config_hash='historical',design_hash='frozen',reference=None)
            payload=state._seal_synthetic(state._synthetic_payload())
            with patch.object(state,'generate_with_state',return_value=(fixture_trial(),payload)) as generate:
                original=generation._worker(task)
                generation._record_path(root,'recovery','one').unlink()
                with patch.object(generation,'preflight',side_effect=AssertionError('No new work needs hardware')):
                    resumed=generation._run({},root,'recovery',[task])
                self.assertEqual(resumed['generated_this_call'],0)
                self.assertEqual(resumed['reused_completed_calls'],1)
                self.assertEqual(generate.call_count,1)
                self.assertEqual(resumed['ledger']['remaining_native_calls'],0)
                self.assertEqual(resumed['results'][0]['native_sha256'],original['native_sha256'])

    def test_cached_allocations_revalidate_capture_identity_and_numeric_state_binding(self):
        from experiments.eeg_representation.h1.v6_pilot import state
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);budget.initialize(root,1)
            contract=generation.adapter_contract()
            task=dict(root=str(root),stage='cache',key='one',spec=dict(history_id='c',structure_seed=1,future_start_ms=10.),
                legacy_config={},original_config_hash='historical',design_hash='frozen',reference=None,adapter_contract=contract)
            payload=state._seal_synthetic(state._synthetic_payload())
            with patch.object(state,'generate_with_state',return_value=(fixture_trial(),payload)) as generate:
                record=generation._worker(task);cached=dict(status='complete',results=[record])
                generation._validate_cached_run(cached,[task])
                with patch.object(generation,'adapter_contract',return_value={**contract,'state_schema':'changed'}):
                    with self.assertRaisesRegex(ValueError,'identity changed'):generation._validate_cached_run(cached,[task])
                changed=deepcopy(payload);first=next(iter(changed['arrays']));changed['arrays'][first]=changed['arrays'][first]+1
                changed=state._seal_synthetic(changed);state.export_state(record['state_directory'],changed)
                with self.assertRaisesRegex(ValueError,'state binding changed'):generation._validate_cached_run(cached,[task])
                self.assertEqual(generate.call_count,1)
                self.assertEqual(budget.summary(root)['attempted_native_calls'],1)

    def test_allocation_cap_reserves_fallback_without_hidden_expansion(self):
        self.assertLess(8+48+48+2,160)
        self.assertLess(8+48+64+2,160)
        with tempfile.TemporaryDirectory() as root:
            budget.initialize(root,160)
            for i in range(97):budget.begin(root,'previous-attempt',dict(i=i))
            with self.assertRaises(RuntimeError):budget.require_capacity(root,64,'fresh fallback')

if __name__=='__main__':unittest.main()
