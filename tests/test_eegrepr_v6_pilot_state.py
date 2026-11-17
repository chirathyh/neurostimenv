"""Lossless complete-state coverage and adapter controls; no hidden simulations."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.eeg_representation.h1.v6_pilot import state


def payload(values=(), context='c', queue=0):
    p=state._synthetic_payload();p['binding']['context_id']=context
    for k,v in values:
        state._add(p,k,v,k.split('/')[0],'explicit fixture')
    for i in range(queue):
        state._add(p,f'pending/recurrent/{i}/arrival_ms',2.+i,'pending','fixture precise event')
    state._add(p,'pending/recurrent_count',queue,'pending','fixture count')
    p['inventory']['pending_recurrent_count']=queue
    return state._seal_synthetic(p)


class CompleteStateTests(unittest.TestCase):
    def test_every_declared_coordinate_preserved_with_presence_mask(self):
        p=payload([('synapses/contact/A',.001),('drive/phase_rad',.4)])
        q=payload([('synapses/contact/A',.003),('drive/phase_rad',.8)],queue=3)
        a=state.fit_adapter([p],all_schema=[p,q]);x=state.transform_adapter(a,[p,q]);n=len(a['keys'])
        self.assertEqual(x.shape,(2,2*n))
        for j,v in enumerate([p,q]):
            for key,original,_ in state._coordinates(v):
                i=a['keys'].index(key)
                self.assertEqual(x[j,n+i],1.)
                self.assertAlmostEqual(x[j,i]*a['scale'][i]+a['offset'][i],original,places=13)
        i=a['keys'].index('pending/recurrent/2/arrival_ms#0')
        self.assertEqual(x[0,n+i],0.);self.assertEqual(x[1,n+i],1.)

    def test_held_out_numeric_values_never_fit_scaling(self):
        train=[payload([('synapses/contact/A',.001)]),payload([('synapses/contact/A',.001)])]
        q=payload([('synapses/contact/A',.1)],queue=1)
        r=payload([('synapses/contact/A',1e6)],queue=1)
        a=state.fit_adapter(train,all_schema=train+[q]);b=state.fit_adapter(train,all_schema=train+[r])
        self.assertEqual(a['fingerprint'],b['fingerprint'])
        i=a['keys'].index('synapses/contact/A#0')
        self.assertGreaterEqual(a['scale'][i],.001)
        self.assertTrue(np.isfinite(state.transform_adapter(a,[r])).all())

    def test_unregistered_queue_fails_without_truncation(self):
        p=payload(queue=1);q=payload(queue=5);a=state.fit_adapter([p])
        with self.assertRaisesRegex(ValueError,'refusing truncation'):state.transform_adapter(a,[q])
        k=state.sparse_kernel([p,q]);self.assertEqual(k.shape,(2,2))
        self.assertGreater(k[1,1],k[0,0]);np.testing.assert_allclose(k,k.T)

    def test_lossless_state_and_adapter_roundtrip(self):
        p=payload([('synapses/contact/A',np.array([1e-200,.001,1e5]))],queue=2)
        with tempfile.TemporaryDirectory() as d:
            state.export_state(Path(d)/'state',p);q=state.load_state(Path(d)/'state')
            self.assertEqual(p['input_hash'],q['input_hash'])
            for k in p['arrays']:np.testing.assert_array_equal(p['arrays'][k],q['arrays'][k])
            a=state.fit_adapter([p]);state.save_adapter(Path(d)/'adapter',a)
            b=state.load_adapter(Path(d)/'adapter')
            np.testing.assert_array_equal(state.transform_adapter(a,[p]),state.transform_adapter(b,[q]))

    def test_missing_group_and_future_encoding_rejected(self):
        p=payload();p['arrays'].pop('drive/control');p['fields'].pop('drive/control');state._seal_synthetic(p)
        with self.assertRaisesRegex(ValueError,'Missing causal'):state.validate(p)
        with self.assertRaisesRegex(ValueError,'Forbidden future'):state._add(payload(),'drive/future_seed',17,'drive','forbidden')

    def test_corrupt_state_hash_rejected(self):
        p=payload([('synapses/contact/A',.001)]);p['arrays']['synapses/contact/A']=np.array(.002)
        with self.assertRaisesRegex(ValueError,'fingerprint mismatch'):state.validate(p)

    def test_partial_historical_snapshot_not_complete(self):
        result=state.from_native('/nonexistent/old-native.h5')
        self.assertEqual(result['status'],'unavailable');self.assertTrue(result['requires_cutoff_recovery'])

    def test_omitted_fields_learnable_control(self):
        r=state.learnable_control();self.assertTrue(r['passed']);self.assertLess(r['physical_mse'],1e-10)
        self.assertEqual(r['neuron_invocations'],0)

    def test_actual_pending_queue_preserves_exact_connection_and_arrival(self):
        p=payload();p.pop('schema_hash');p.pop('input_hash')
        p['_recurrent_connections']=[dict(connection_index=0,source_gid=0,target_contact_index=4,
            target_gid=1,section_index=0,segment_index=0,x=.5,delay_ms=2.,threshold_mV=-10.,
            weight_uS=.003,active=1,source_voltage_mV=-60.,threshold_above=0)]
        p['_actual_recurrent_pending']=[dict(connection_index=0,arrival_ms=31.,recurrent_queue_order=0)]
        t=np.arange(1,481)*.0625
        trial=dict(time_ms=t,eeg_v=np.zeros((len(t),1)),privileged=dict(spike_times_ms=np.asarray([29.]),
            spike_gids=np.asarray([0]),phase_time_ms=np.asarray([0.,30.]),phase_rad=np.asarray([0.,.6]),
            cutoff_physical_state=np.zeros(2),pending_recurrent_events=np.asarray([[0,1,0,.5,31.,.003]]),pending_external_events=np.empty((0,6))),
            metadata=dict(software={'neuron':'test','lfpy':'test'},fork={'external_history_hash':'history'},structure_hash='structure'))
        result=state._finish(p,trial,dict(future_start_ms=30.,history_id='bound-history',structure_id='1',structure_seed=1,regime={'modulation_depth':.04}))
        self.assertEqual(result['binding']['context_id'],'bound-history')
        self.assertEqual(result['arrays']['pending/recurrent/0/target_contact_index'],4)
        self.assertEqual(result['arrays']['pending/recurrent/0/arrival_ms'],31.)
        self.assertEqual(result['arrays']['pending/recurrent/0/emission_ms'],29.)
        self.assertEqual(result['arrays']['pending/recurrent/0/weight_uS'],.003)

    def test_pending_queue_mismatch_fails_before_returning_export(self):
        p=payload();p['_recurrent_connections']=[]
        p['_actual_recurrent_pending']=[dict(connection_index=0,arrival_ms=31.,recurrent_queue_order=0)]
        trial=dict(time_ms=np.asarray([30.]),privileged=dict(spike_times_ms=np.empty(0),spike_gids=np.empty(0)))
        with self.assertRaisesRegex(ValueError,'Actual recurrent queue differs'):
            state._finish(p,trial,dict(future_start_ms=30.))

    def test_future_invariance_rejects_present_change(self):
        p=payload([('drive/phase_rad',.5)]);q=deepcopy(p)
        for key in ('prefix_eeg_hash','legacy_physical_state_hash','legacy_pending_recurrent_hash','external_history_hash','structure_hash'):
            p['binding'][key]=q['binding'][key]='equal'
        self.assertTrue(state.validate_future_invariance(p,q)['passed'])
        q['arrays']['drive/phase_rad']=np.asarray(.6);state._seal_synthetic(q)
        with self.assertRaisesRegex(ValueError,'changed with unknown future'):state.validate_future_invariance(p,q)

if __name__=='__main__':unittest.main()
