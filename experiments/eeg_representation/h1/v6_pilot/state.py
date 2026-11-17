"""Versioned complete *declared* HH/passive/Exp2Syn present-state ML input.

This adapter is opt-in and never changes the legacy native generator. It records
live fields at its existing cutoff hook. It is not a restorable NEURON checkpoint.
Only this audited fixed-step, zero-field model is supported; unknown mechanisms,
RxD species, non-Exp2Syn contacts or extra NetCon weights fail the coverage gate.
All extracellular layers are retained (the older partial export saved layer 0).

Unknown future external events, presampled phase paths, history/future RNG seeds,
and RNG states are deliberately absent from the input. Pending RECURRENT events
are retained with exact contact and connection identities and all attributes.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np

from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json, file_hash, read_json
from ...simulator import array_hash

SCHEMA_VERSION = 'eegrepr-pilot-complete-present-state-1.0'
ADAPTER_VERSION = 'eegrepr-pilot-lossless-state-adapter-1.0'
_ALLOWED_DENSITY = {'hh', 'pas', 'extracellular'}
_REQUIRED_GROUPS = ('segments', 'mechanisms', 'ions', 'synapses', 'connectivity',
                    'pending', 'drive', 'geometry', 'forward_model', 'solver', 'acquisition')
_FORBIDDEN_TOKENS = ('seed', 'rng', 'future_event', 'future_phase', 'future_schedule')


def _unit(path):
    """Mechanism-aware base scales are conservative physical units, never eps SD."""
    last = path.split('/')[-1]
    if any(s in path for s in ('phase_rad', 'diffusion_rad2')): return 'rad or rad^2/s', 1.
    if 'acquisition/' in path: return 'V', 1e-9
    if last.endswith('_mV') or '/vext' in path or last in ('v_mV','e','el','ena','ek','e_extracellular','voltage_mV','threshold_mV'):
        return 'mV', 1.
    if last in ('A','B','g','weight_uS') and ('synapse' in path or 'pending' in path or 'connectivity' in path): return 'uS', .001
    if 'hh/' in path and last in ('m','h','n','minf','hinf','ninf'): return 'dimensionless', .1
    if last in ('tau1','tau2','delay_ms','arrival_ms','emission_ms','cutoff_ms','dt_ms','last_spike_ms') or last.endswith('tau'):
        return 'ms', 1.
    if last in ('x','y','z','length','d','L','diam','pts3d') or path.startswith('geometry/'):
        return 'um (area: um^2)', 100.
    if last == 'i' and path.startswith('synapses/'): return 'nA', .001
    if last in ('i','il','ina','ik','i_membrane'): return 'mA/cm^2', .001
    if last in ('dina_dv_','dik_dv_','gna','gk'): return 'S/cm^2', .001
    if last.endswith('_ms'): return 'ms', 1.
    if last == 'celsius': return 'deg C', 1.
    if last in ('cm','xc'): return 'uF/cm^2', 1.
    if last == 'Ra': return 'ohm cm', 100.
    if last in ('gnabar','gkbar','gl') or ('/pas/' in path and last == 'g'): return 'S/cm^2', .001
    if last in ('nai','nao','ki','ko'): return 'mM', 1.
    if last == 'xraxial': return 'MOhm/cm', 1e9
    if last == 'xg': return 'S/cm^2', 1e9
    if 'rate_hz' in last or 'frequency_hz' in last: return 'Hz', 1.
    if any(x in last for x in ('index','id','count','mask','active','parent','nseg','gid')): return 'index/dimensionless', 1.
    return 'NEURON native units; see exact source field', 1.


def _add(payload, name, value, group, source, *, unit=None, scale=None):
    a = np.asarray(value, dtype=np.float64)
    if not np.isfinite(a).all(): raise ValueError(f'Nonfinite complete-state field: {name}')
    if any(t in name.lower() for t in _FORBIDDEN_TOKENS):
        raise ValueError(f'Forbidden future/randomness encoding in state input: {name}')
    u, s = _unit(name)
    payload['arrays'][name] = a.copy()
    payload['fields'][name] = dict(shape=list(a.shape), dtype='float64', units=unit or u,
        minimum_scale=float(s if scale is None else scale), group=group, source=source)


def _flatten_numeric(payload, prefix, values, group, source):
    if isinstance(values, dict):
        for k, v in sorted(values.items()): _flatten_numeric(payload, prefix+'/'+str(k), v, group, source)
    elif isinstance(values, (list, tuple, np.ndarray, int, float, bool, np.number)):
        _add(payload, prefix, values, group, source)
    elif values is not None:
        raise ValueError(f'Non-numeric physical parameter not explicitly encoded: {prefix}={values}')


def _capture(network, env, spec):
    """Inspect live mechanisms and map every contact without global HOC names."""
    from neuron import h
    from omegaconf import OmegaConf
    p = dict(schema_version=SCHEMA_VERSION, arrays={}, fields={}, inventory={}, binding={})
    cells = [cell for pop in network.population_names for cell in network.populations[pop].cells]
    sections = {str(sec):(int(c.gid),j) for c in cells for j,sec in enumerate(c.allseclist)}
    contact_ids = {}; source_cells = {int(c.gid):c for c in cells}
    mechanisms = set(); nsegments=0; ncontacts=0; all_rec=[]; all_ext=[]; detectors={}
    for c in cells:
        gid=int(c.gid)
        for key in ('x','y','z','area','d','length'):
            _add(p, f'geometry/cell{gid}/{key}', getattr(c,key), 'geometry', f'LFPy cell.{key}')
        for si, sec in enumerate(c.allseclist):
            prefix=f'cell{gid}/section{si}'
            ps=sec.psection(); found=set(ps['density_mechs']); mechanisms |= found
            if found - _ALLOWED_DENSITY: raise ValueError('Unaudited density mechanisms: '+str(found-_ALLOWED_DENSITY))
            if ps.get('species') or ps.get('regions'): raise ValueError('RxD state is outside the declared complete-state schema')
            if found != ({'hh','extracellular'} if si==0 else {'pas','extracellular'}):
                raise ValueError(f'Unexpected section mechanism inventory at {prefix}: {found}')
            parent=ps['morphology'].get('parent')
            parent_info=[-1.,-1.,-1.] if parent is None else [*sections[str(parent.sec)],parent.x]
            _add(p,f'geometry/{prefix}/parent',parent_info,'geometry','Section.psection morphology.parent')
            _add(p,f'geometry/{prefix}/Ra',ps['Ra'],'geometry','Section.Ra')
            _add(p,f'geometry/{prefix}/nseg',ps['nseg'],'geometry','Section.nseg')
            _add(p,f'geometry/{prefix}/cm',ps['cm'],'geometry','Section.cm')
            for key in ('L','diam','pts3d'):
                _add(p,f'geometry/{prefix}/{key}',ps['morphology'][key],'geometry','Section.psection morphology.'+key)
            segments=list(sec); nsegments+=len(segments)
            _add(p,f'segments/{prefix}/x', [s.x for s in segments],'segments','segment.x; spatial compartment identity')
            _add(p,f'segments/{prefix}/v_mV',[s.v for s in segments],'segments','segment.v')
            for mechanism, fields in sorted(ps['density_mechs'].items()):
                for field, values in sorted(fields.items()):
                    _add(p,f'mechanisms/{prefix}/{mechanism}/{field}',values,'mechanisms','Section.psection density_mechs.'+mechanism+'.'+field)
            for ion, fields in sorted(ps['ions'].items()):
                for field, values in sorted(fields.items()):
                    _add(p,f'ions/{prefix}/{ion}/{field}',values,'ions','Section.psection ions.'+ion+'.'+field)
                _add(p,f'ions/{prefix}/{ion}/ion_style',h.ion_style(ion+'_ion',sec=sec),'ions','h.ion_style; concentration/reversal update rules')
            for sj, seg in enumerate(segments):
                for pi, pp in enumerate(seg.point_processes()):
                    kind=pp.hname().split('[')[0]
                    if kind!='Exp2Syn': raise ValueError('Unaudited point process: '+kind)
                    key=f'synapses/{prefix}/segment{sj}/contact{pi}'
                    idx=ncontacts; ncontacts+=1; contact_ids[pp.hname()]=(key,idx,gid,si,sj,seg.x)
                    _add(p,key+'/identity',[gid,si,sj,seg.x,pi,idx],'synapses','canonical cell/section/segment/contact ordinal; not global HOC instance index')
                    for field in ('A','B','tau1','tau2','e','g','i'):
                        _add(p,key+'/'+field,getattr(pp,field),'synapses','Exp2Syn.'+field)
        source_detectors=list(c._hoc_sd_netconlist)
        if len(source_detectors)!=1:raise ValueError('Expected exactly one registered LFPy source detector per physical cell')
        sd=source_detectors[0];source_segment=sd.preseg()
        if source_segment is None:raise ValueError('Source detector has no exposed physical voltage site')
        sgid,ssi=sections[str(source_segment.sec)]
        if sgid!=gid:raise ValueError('Source detector GID/physical source mismatch')
        detector=dict(source_gid=gid,section_index=ssi,x=float(source_segment.x),
            source_voltage_mV=float(source_segment.v),threshold_mV=float(sd.threshold),
            threshold_above=int(source_segment.v>=sd.threshold),active=int(sd.active()),
            delay_ms=float(sd.delay),weight_uS=float(sd.weight[0]),detector_index=0)
        detectors[gid]=detector
        _flatten_numeric(p,f'connectivity/source_detector/{gid}',detector,'connectivity',
            'cell._hoc_sd_netconlist[-1] registered by pc.cell; true emitting voltage detector, not gid_connect target NetCon')
        for ni,nc in enumerate(c._hoc_netconlist):
            pp=nc.syn(); key,idx,tgid,si,sj,x=contact_ids[pp.hname()]
            if int(nc.wcnt())!=1: raise ValueError('Only single-weight Exp2Syn NetCons are audited')
            ext=dict(connection_index=len(all_ext),target_contact_index=idx,target_gid=tgid,
                section_index=si,segment_index=sj,x=x,delay_ms=float(nc.delay),weight_uS=float(nc.weight[0]),
                active=int(nc.active()),contact_within_cell=ni)
            all_ext.append(ext)
            _flatten_numeric(p,f'connectivity/external/{len(all_ext)-1}',ext,'connectivity','LFPy background direct NetCon.event delivery; delay does not add causal queue')
    # All contact IDs now exist before recurrent topology is serialized.
    for ni,nc in enumerate(network._hoc_netconlist):
        pp=nc.syn(); key,idx,gid,si,sj,x=contact_ids[pp.hname()]
        if int(nc.wcnt())!=1: raise ValueError('Only single-weight recurrent Exp2Syn NetCons are audited')
        source=int(nc.srcgid())
        if source not in source_cells: raise ValueError('Recurrent source not a local physical cell')
        detector=detectors[source]
        row=dict(connection_index=ni,source_gid=source,target_contact_index=idx,target_gid=gid,
            section_index=si,segment_index=sj,x=x,delay_ms=float(nc.delay),threshold_mV=detector['threshold_mV'],
            target_netcon_threshold_mV=float(nc.threshold),source_detector_index=detector['detector_index'],
            weight_uS=float(nc.weight[0]),active=int(nc.active()),source_voltage_mV=detector['source_voltage_mV'],
            threshold_above=detector['threshold_above'])
        all_rec.append(row)
        _flatten_numeric(p,f'connectivity/recurrent/{ni}',row,'connectivity','LFPy recurrent NetCon and source soma threshold detector')
    cfg=OmegaConf.to_container(env.args,resolve=True) if hasattr(env,'args') else None
    # env_online stores the actual expanded configuration as self.args.
    if cfg is None: raise ValueError('Live environment physical configuration unavailable')
    e=cfg['env']; ff=dict(eeg=e['eeg'],source_position_um=e['network']['position'])
    # Only numerical forward-model parameters enter the predictor; flags encoded 0/1.
    _flatten_numeric(p,'forward_model',ff,'forward_model','actual OnlineNeuronEnv cfg.env.eeg and network.position')
    for ni, transform in enumerate(network._online_transforms):
        _add(p,f'forward_model/probe_transform{ni}',transform,'forward_model','actual LFPy online segment-current transform; dipole aggregation included')
    _flatten_numeric(p,'solver',dict(cutoff_ms=float(h.t),dt_ms=float(h.dt),celsius=float(h.celsius),
        secondorder=int(h.secondorder),cvode_active=int(h.CVode().active()),hh_usetable_exposed=int(hasattr(h,"usetable_hh"))),
        'solver','actual NEURON h solver settings')
    if hasattr(h,'usetable_hh'):_add(p,'solver/hh_usetable',int(h.usetable_hh),'solver','actual exposed NEURON HH table setting; absent when not exposed, never substituted zero')
    for name in ('E','I'):
        bg=e['network']['background'][name]; rhythm=bg['rhythm']
        n=int(bg['n_synapses']); nmod=int(np.floor(n*float(rhythm['shared_modulated_fraction'])+.5)) if rhythm['enabled'] else 0
        row=dict(baseline_rate_hz=1000/float(bg['interval_ms']),start_ms=float(bg['start_ms']),
            contacts_per_cell=n,modulated_contacts_per_cell=nmod,modulation_depth=float(rhythm['modulation_depth']),
            frequency_hz=float(rhythm['frequency_hz']) if rhythm['enabled'] else 0.,
            diffusion_rad2_s=float(rhythm['phase_diffusion_rad2_per_s']),phase_grid_ms=float(rhythm['phase_diffusion_integration_dt_ms']))
        _flatten_numeric(p,'drive/'+name,row,'drive','actual background drive law; local contact ordinal < modulated_contacts selects shared-rate group')
    # Inspect the actual queue, but retain ONLY physically caused recurrent
    # NetCons. External prescheduled Poisson deliveries are unknown inputs.
    tv=h.Vector();queued=h.List();h.CVode().event_queue_info(2,tv,queued)
    rec_by_name={nc.hname():i for i,nc in enumerate(network._hoc_netconlist)}
    actual_pending=[]
    for arrival,nc in zip(tv,queued):
        if nc.hname() in rec_by_name:
            actual_pending.append(dict(connection_index=rec_by_name[nc.hname()],arrival_ms=float(arrival),
                recurrent_queue_order=len(actual_pending)))
    flags=h.Vector();self_times=h.Vector();self_targets=h.List()
    h.CVode().event_queue_info(3,self_times,flags,self_targets)
    if len(self_times):raise ValueError('Unaudited self-events in declared passive/HH/Exp2Syn model')
    p['_actual_recurrent_pending']=actual_pending
    p['_recurrent_connections']=all_rec
    p['inventory']=dict(backend='neuron_lfpy_real',segment_count=nsegments,contact_count=ncontacts,recurrent_connections=len(all_rec),
        external_connections=len(all_ext),density_mechanisms=sorted(mechanisms),point_process_types=['Exp2Syn'],
        extracellular_layers=int(h.nlayer_extracellular()),n_cells=len(cells),source_detector_count=len(detectors),
        dynamic_mechanism_contract={'hh':['m','h','n'],'pas':[], 'extracellular':['all vext layers'], 'Exp2Syn':['A','B']},
        includes_derived_redundant_fields=True,coordinate_count_is_intrinsic_dimension=False,
        hh_table_contract='Actual table setting exported only if exposed; otherwise the immutable installed NEURON mechanism implementation is version-bound, with no fabricated numeric setting.',
        source_detector_contract='Actual cell._hoc_sd_netconlist source detector registered by pc.cell, distinct from target gid_connect NetCons. Fixed-step voltage threshold; current source voltage, above-threshold flag at cutoff and cutoff-dt, threshold, last emitted spike and caused deliveries retained. No refractory mechanism in this model.',
        delivery_contract='All recurrent contacts are Exp2Syn with one constant weight; NET_RECEIVE has no plasticity or per-connection mutable state. Direct external deliveries before cutoff are already applied. Actual recurrent queue membership/arrival/contact order is cross-checked against caused spikes. Equal-time Exp2Syn deliveries are additive; no external queue positions/counts are retained.',
        unknown_future_inputs_excluded=['all external event schedules','future phase/envelope paths','all RNG state and seeds'],
        target_contract='Native future-only waveform/Fourier/energy targets; all A0/A1/A2 acquisition states additionally retained for explicit accounting')
    return p


def _finish(payload, trial, spec):
    from scipy import signal
    from ..v4.measurement import _design
    cutoff=float(spec['future_start_ms']); p=trial['privileged']; t=np.asarray(trial['time_ms'])
    # Recurrent deliveries are indexed by exact target contact, never collapsed by location.
    st=np.asarray(p['spike_times_ms']); sg=np.asarray(p['spike_gids'])
    pending=[]
    for row in payload.pop('_recurrent_connections'):
        source=row['source_gid']; emissions=st[(sg==source)&(st<=cutoff)]
        last=float(emissions[-1]) if len(emissions) else -1.
        _add(payload,f'connectivity/recurrent/{row["connection_index"]}/last_spike_ms',last,
             'connectivity','last observed source spike at or before cutoff; -1 means absent')
        for emitted in emissions[emissions+row['delay_ms']>cutoff]:
            event=dict(row,emission_ms=float(emitted),arrival_ms=float(emitted+row['delay_ms']))
            pending.append(event)
    actual_pending=payload.pop('_actual_recurrent_pending')
    expected=sorted((int(x['connection_index']),float(x['arrival_ms'])) for x in pending)
    actual=sorted((int(x['connection_index']),float(x['arrival_ms'])) for x in actual_pending)
    if len(expected)!=len(actual) or any(i!=j or not np.isclose(t,u,rtol=0,atol=1e-9) for (i,t),(j,u) in zip(expected,actual)):
        raise ValueError('Actual recurrent queue differs from caused pre-cutoff spike/delay reconstruction')
    # Preserve actual recurrent queue order without leaking any external queue
    # locations/counts. Equal-time Exp2Syn arrivals are additive; connection IDs
    # and relative recurrent delivery order remain explicit.
    events={(x['connection_index'],round(x['arrival_ms'],9)):x for x in pending}
    pending=[dict(events[(x['connection_index'],round(x['arrival_ms'],9))],
        arrival_ms=x['arrival_ms'],recurrent_queue_order=x['recurrent_queue_order']) for x in actual_pending]
    _add(payload,'pending/recurrent_count',len(pending),'pending','number of causally pending pre-cutoff recurrent deliveries')
    for ei, event in enumerate(pending):
        _flatten_numeric(payload,f'pending/recurrent/{ei}',event,'pending','pre-cutoff source spike + exact recurrent NetCon/contact identity')
    _add(payload,'pending/external_count',0,'pending','LFPy direct NetCon.event(tdeliver): no additional delay after arrival')
    phase=float(np.interp(cutoff,p['phase_time_ms'],p['phase_rad']))
    _add(payload,'drive/phase_rad',phase,'drive','shared phase evaluated at exact aligned cutoff; future phase samples excluded')
    if 'burst_times_ms' in p:
        bt=np.asarray(p['burst_times_ms']); bi=int(np.searchsorted(bt,cutoff,side='right')-1)
        depth=float(np.asarray(p['burst_depth'])[bi]); burst=spec['regime'].get('burst',{})
        burst={} if burst is True else burst
        _flatten_numeric(payload,'drive/envelope',dict(depth=depth,on_mean_ms=float(burst.get('on_mean_ms',500)),
            off_mean_ms=float(burst.get('off_mean_ms',1000)),low_depth=float(burst.get('low_depth',0))),
            'drive','present CTMC envelope and exponential dwell-law parameters; future residual dwell excluded')
    else: _add(payload,'drive/envelope/depth',float(spec['regime'].get('modulation_depth',0)),'drive','constant modulation envelope')
    x=np.asarray(trial['eeg_v'],dtype=np.float64)[t<=cutoff]
    if x.ndim==2 and x.shape[1]==1:x=x[:,0]
    if x.ndim!=1 or not len(x) or t[t<=cutoff][-1]!=cutoff:raise ValueError('Complete acquisition prefix/cutoff unavailable')
    for name in ('A0','A1','A2'):
        kind,coef=_design(name)
        if kind=='sos': _,zi=signal.sosfilt(coef,x,zi=np.zeros((len(coef),2)))
        else: _,zi=signal.lfilter(coef,[1.],x,zi=np.zeros(len(coef)-1))
        _add(payload,f'acquisition/{name}/state',zi,'acquisition','zero-start causal filter state preserved through entire prefix',unit='V',scale=1e-9)
        _add(payload,f'acquisition/{name}/coefficients',coef,'acquisition','frozen exact input operator coefficients',unit='dimensionless',scale=1.)
    payload['inventory']['pending_recurrent_count']=len(pending)
    payload['inventory']['software']=dict(trial['metadata']['software'])
    source_root=Path(__file__).resolve().parents[4]
    source_paths=('setup/circuits/ballnstick/BallAndStick.hoc','setup/circuits/ballnstick/BallAndStickTemplate.hoc',
        'setup/circuits/ballnstick/utils.py','env/models/neuron/networkenv_online.py','configs/env/ballnstick.yaml')
    payload['inventory']['bound_model_source_sha256']={path:file_hash(source_root/path) for path in source_paths}
    payload['inventory']['globally_constant_law']='Canonical NEURON hh/pas/Exp2Syn equations bound to recorded NEURON version and source hashes; all instance RANGE parameters exported.'
    payload['binding']=dict(context_id=str(spec.get('context_id',spec.get('history_id',spec.get('trial_id','unspecified')))),
        structure_id=str(spec.get('structure_id',spec['structure_seed'])),cutoff_ms=cutoff,
        legacy_physical_state_hash=array_hash(p['cutoff_physical_state']),
        legacy_pending_recurrent_hash=array_hash(p['pending_recurrent_events']),
        legacy_pending_external_hash=array_hash(p['pending_external_events']),
        prefix_eeg_hash=array_hash(np.asarray(trial['eeg_v'])[t<=cutoff]),
        prefix_time_hash=array_hash(t[t<=cutoff]),
        external_history_hash=trial['metadata']['fork']['external_history_hash'],
        structure_hash=trial['metadata']['structure_hash'],
        recovery_method='live field capture at exact cutoff during deterministic full-prefix reconstruction; not checkpoint restoration')
    payload['schema_hash']=canonical_hash(dict(schema_version=SCHEMA_VERSION,fields=payload['fields'],inventory=payload['inventory']))
    payload['input_hash']=input_hash(payload)
    validate(payload)
    return payload


def input_hash(payload):
    return canonical_hash({k:array_hash(v) for k,v in sorted(payload['arrays'].items())})


def generate_with_state(config, spec):
    """One actual invocation, billed by caller; capture before any future outcomes.

    Cutoff recovery can stop at cutoff+1ms because the unchanged split generator
    requires an interior cutoff. This engineering tail is excluded from the input.
    """
    from ... import simulator as legacy
    import os
    previous_main=os.environ.get('MAIN_PATH');os.environ['MAIN_PATH']=str(Path(__file__).resolve().parents[4])
    try:
        from env.models.neuron import env_online
    finally:
        if previous_main is None:os.environ.pop('MAIN_PATH',None)
        else:os.environ['MAIN_PATH']=previous_main
    from ..v6 import simulator as numerical
    captured={}; original_env=env_online.OnlineNeuronEnv; original_cp=legacy._physical_checkpoint
    def environment(*a,**kw):
        e=original_env(*a,**kw);captured['environment']=e
        advance=e.network.advance_online;captured['original_advance']=advance
        def observe_advance(*,stop_ms,before_advance=None):
            def before(left_ms):
                if before_advance is not None:before_advance(left_ms)
                if np.isclose(left_ms,float(spec['future_start_ms'])-float(e.network.dt),rtol=0,atol=1e-8):
                    captured['previous_detector_voltage']={int(cell.gid):float(cell._hoc_sd_netconlist[-1].preseg().v)
                        for name in e.network.population_names for cell in e.network.populations[name].cells}
            return advance(stop_ms=stop_ms,before_advance=before)
        e.network.advance_online=observe_advance
        return e
    def checkpoint(network):
        del network.advance_online
        captured.pop('original_advance',None)
        captured['payload']=_capture(network,captured['environment'],spec)
        if 'previous_detector_voltage' not in captured:raise ValueError('Previous fixed-step detector input unavailable')
        for gid,v in captured['previous_detector_voltage'].items():
            prefix=f'connectivity/source_detector/{gid}'
            _add(captured['payload'],prefix+'/previous_step_voltage_mV',v,'connectivity',
                'read-only before_advance callback at cutoff-dt; fixed-step source detector bookkeeping')
            threshold=float(captured['payload']['arrays'][prefix+'/threshold_mV'])
            _add(captured['payload'],prefix+'/previous_step_threshold_above',int(v>=threshold),'connectivity',
                'above-threshold source value at cutoff-dt, retained separately from cutoff voltage')
        return original_cp(network)
    with legacy._LOCK, patch.object(env_online,'OnlineNeuronEnv',environment), patch.object(legacy,'_physical_checkpoint',checkpoint):
        trial=numerical.generate_trial(config,spec)
    if 'payload' not in captured:raise ValueError('No live cutoff capture; a valid future_start_ms is mandatory')
    return trial,_finish(captured['payload'],trial,spec)


def validate(payload):
    if payload.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Unknown complete-state schema')
    if set(payload['arrays'])!=set(payload['fields']):raise ValueError('State arrays/field inventory mismatch')
    groups={v['group'] for v in payload['fields'].values()}
    if set(_REQUIRED_GROUPS)-groups:raise ValueError('Missing causal field groups: '+str(set(_REQUIRED_GROUPS)-groups))
    for name,a in payload['arrays'].items():
        field=payload['fields'][name];a=np.asarray(a)
        if list(a.shape)!=field['shape'] or not np.isfinite(a).all():raise ValueError('Invalid field shape/value: '+name)
        if any(t in name.lower() for t in _FORBIDDEN_TOKENS):raise ValueError('Unknown future encoding leaked into state: '+name)
        if field['minimum_scale']<=0:raise ValueError('Nonpositive physical scaling floor')
    inv=payload['inventory']
    if set(inv['density_mechanisms'])!=_ALLOWED_DENSITY or inv['point_process_types']!=['Exp2Syn']:
        raise ValueError('Incomplete/unknown mechanism inventory')
    if inv['extracellular_layers']<1 or inv['contact_count']!=inv['external_connections']+inv['recurrent_connections']:
        raise ValueError('Contact coverage or extracellular-layer coverage failed')
    if inv['pending_recurrent_count']!=int(payload['arrays']['pending/recurrent_count']):raise ValueError('Pending event inventory mismatch')
    if inv.get('backend')=='neuron_lfpy_real':_validate_live_coverage(payload)
    expected=canonical_hash(dict(schema_version=SCHEMA_VERSION,fields=payload['fields'],inventory=inv))
    if payload.get('schema_hash')!=expected or payload.get('input_hash')!=input_hash(payload):raise ValueError('Complete-state content/schema fingerprint mismatch')
    return dict(status='complete',schema_version=SCHEMA_VERSION,schema_hash=expected,input_hash=payload['input_hash'],
        field_count=len(payload['fields']),numeric_coordinate_count=sum(np.asarray(a).size for a in payload['arrays'].values()),
        coverage=deepcopy(inv),unknown_future_fields_present=False)



def _validate_live_coverage(p):
    """Check actual identities/counts, not merely a self-reported complete flag."""
    a=p['arrays'];inv=p['inventory'];required=[]
    voltage=[k for k in a if k.startswith('segments/') and k.endswith('/v_mV')]
    if sum(np.asarray(a[k]).size for k in voltage)!=inv['segment_count']:
        raise ValueError('Segment coverage does not match live network count')
    if len(voltage)!=2*inv['n_cells']:raise ValueError('Canonical soma/apical section identities missing')
    for key in voltage:
        sec=key.removeprefix('segments/').removesuffix('/v_mV');n=len(a[key])
        for f in ('xraxial','xg','xc','e','i_membrane','vext'):
            field=f'mechanisms/{sec}/extracellular/{f}';required.append(field)
            if f=='vext' and (field not in a or np.asarray(a[field]).shape!=(n,inv['extracellular_layers'])):
                raise ValueError('An extracellular layer is absent')
        if sec.endswith('section0'):
            required.extend(f'mechanisms/{sec}/hh/{f}' for f in ('m','h','n','gnabar','gkbar','gl','el'))
            for ion,fields in (('na',('ena','nai','nao','ion_style')),('k',('ek','ki','ko','ion_style'))):
                required.extend(f'ions/{sec}/{ion}/{f}' for f in fields)
        else:required.extend(f'mechanisms/{sec}/pas/{f}' for f in ('g','e'))
        required.extend(f'geometry/{sec}/{f}' for f in ('Ra','nseg','cm','parent','L','diam','pts3d'))
    contacts=[k.removesuffix('/identity') for k in a if k.startswith('synapses/') and k.endswith('/identity')]
    if len(contacts)!=inv['contact_count']:raise ValueError('Synaptic contact coverage mismatch')
    ids=[]
    for key in contacts:
        required.extend(key+'/'+f for f in ('A','B','tau1','tau2','e','g','i'))
        ids.append(int(a[key+'/identity'][-1]))
    if sorted(ids)!=list(range(inv['contact_count'])):raise ValueError('Synaptic contact identity is not bijective')
    if inv['source_detector_count']!=inv['n_cells']:raise ValueError('Missing emitting source detector')
    for gid in range(inv['n_cells']):
        required.extend(f'connectivity/source_detector/{gid}/'+f for f in
            ('source_gid','section_index','x','source_voltage_mV','previous_step_voltage_mV','previous_step_threshold_above','threshold_mV','threshold_above','active','delay_ms','weight_uS'))
    targets=[]
    for kind in ('external','recurrent'):
        for i in range(inv[kind+'_connections']):
            key=f'connectivity/{kind}/{i}'
            required.extend(key+'/'+f for f in ('target_contact_index','weight_uS','delay_ms','active'))
            if kind=='recurrent':required.extend(key+'/'+f for f in ('source_gid','threshold_mV','source_voltage_mV','threshold_above','last_spike_ms'))
            if key+'/target_contact_index' in a:targets.append(int(a[key+'/target_contact_index']))
    if sorted(targets)!=list(range(inv['contact_count'])):raise ValueError('Connectivity does not cover every contact exactly once')
    for i in range(inv['pending_recurrent_count']):
        key=f'pending/recurrent/{i}'
        required.extend(key+'/'+f for f in ('arrival_ms','emission_ms','connection_index','target_contact_index','weight_uS','source_gid','delay_ms','active','recurrent_queue_order'))
        if key+'/arrival_ms' in a and float(a[key+'/arrival_ms'])<=p['binding']['cutoff_ms']:
            raise ValueError('Pending event is not strictly after cutoff')
    required.extend(('drive/phase_rad','drive/envelope/depth','solver/cutoff_ms','solver/dt_ms','solver/celsius'))
    for name in ('E','I'):required.extend('drive/'+name+'/'+f for f in ('baseline_rate_hz','contacts_per_cell','modulated_contacts_per_cell','modulation_depth','frequency_hz','diffusion_rad2_s','phase_grid_ms'))
    for name in ('A0','A1','A2'):required.extend('acquisition/'+name+'/'+f for f in ('state','coefficients'))
    missing=[k for k in required if k not in a]
    if missing:raise ValueError('Incomplete native-state causal coverage: '+str(missing[:20]))
    if not inv.get('software',{}).get('neuron') or not inv.get('bound_model_source_sha256'):
        raise ValueError('Constitutive model constants are not bound to software/source')


def validate_future_invariance(a,b):
    validate(a);validate(b)
    if a['binding']['context_id']!=b['binding']['context_id']:raise ValueError('Future-invariance comparison requires one context')
    if a['input_hash']!=b['input_hash'] or a['schema_hash']!=b['schema_hash']:
        changed=[k for k in set(a['arrays'])|set(b['arrays']) if k not in a['arrays'] or k not in b['arrays'] or not np.array_equal(a['arrays'][k],b['arrays'][k])]
        raise ValueError('Present input changed with unknown future draw: '+str(changed[:20]))
    for key in ('prefix_eeg_hash','legacy_physical_state_hash','legacy_pending_recurrent_hash','external_history_hash','structure_hash'):
        if a['binding'][key]!=b['binding'][key]:raise ValueError('Sibling prefix/structure mismatch: '+key)
    return dict(passed=True,input_hash=a['input_hash'],schema_hash=a['schema_hash'],excluded='Unknown future event/phase paths and all seeds/RNG state')


def export_state(path,payload):
    path=Path(path);path.mkdir(parents=True,exist_ok=True);audit=validate(payload)
    names=sorted(payload['arrays']);arrays={f'field_{i:06d}':payload['arrays'][name] for i,name in enumerate(names)}
    atomic_npz(path/'fields.npz',**arrays)
    meta={k:v for k,v in payload.items() if k!='arrays'}
    meta.update(array_keys={name:f'field_{i:06d}' for i,name in enumerate(names)},fields_sha256=file_hash(path/'fields.npz'),audit=audit)
    atomic_json(path/'schema.json',meta)
    return dict(path=str(path),schema_path=str(path/'schema.json'),fields_path=str(path/'fields.npz'),
        schema_sha256=file_hash(path/'schema.json'),fields_sha256=meta['fields_sha256'],input_hash=payload['input_hash'])


def load_state(path):
    path=Path(path);meta=read_json(path/'schema.json')
    if file_hash(path/'fields.npz')!=meta['fields_sha256']:raise ValueError('Corrupt state numeric export')
    with np.load(path/'fields.npz',allow_pickle=False) as f:
        arrays={name:f[key].copy() for name,key in meta['array_keys'].items()}
    payload={k:v for k,v in meta.items() if k not in ('array_keys','fields_sha256','audit')};payload['arrays']=arrays
    validate(payload);return payload


def from_native(path):
    """Older snapshots are incomplete; never promote their partial vector silently."""
    path=Path(path)
    if path.is_dir() and (path/'schema.json').exists():return load_state(path)
    return dict(status='unavailable',requires_cutoff_recovery=True,
        reason='Historical native cutoff vector lacks exhaustive live mechanism/layer/contact schema and exact pending contact IDs; retain its native target arrays and recover only the prefix.')


def _coordinates(payload):
    """Every scalar plus an explicit presence coordinate; reversible, no pooling."""
    for name in sorted(payload['fields']):
        a=np.asarray(payload['arrays'][name]);floor=payload['fields'][name]['minimum_scale']
        for index,value in enumerate(a.reshape(-1)):
            yield name+'#'+str(index),float(value),float(floor)


def fit_adapter(train_payloads, all_schema=None):
    """Register field paths/shapes, then fit numerical scaling on TRAIN ONLY.

    all_schema may enumerate registered cohort structures/queues, but its values
    are never inspected for offsets/scales. No queue is truncated or aggregated.
    New, unregistered fields explicitly fail; use sparse_kernel for open schemas.
    """
    train_payloads=list(train_payloads);all_schema=train_payloads if all_schema is None else list(all_schema)
    if not train_payloads:raise ValueError('Training present states required')
    names={};shapes={}
    for p in all_schema:
        validate(p)
        for name,field in p['fields'].items():
            for i in range(int(np.prod(field['shape'],dtype=int)) if field['shape'] else 1):
                key=name+'#'+str(i);floor=float(field['minimum_scale'])
                if key in names and names[key]!=floor:raise ValueError('Physical units/scale changed across state fields')
                names[key]=floor
    keys=sorted(names);lookup={k:i for i,k in enumerate(keys)};n=len(keys)
    raw=np.zeros((len(train_payloads),n));present=np.zeros_like(raw)
    for j,p in enumerate(train_payloads):
        for k,v,_ in _coordinates(p):
            if k not in lookup:raise ValueError('Training field absent from structural registry')
            raw[j,lookup[k]]=v;present[j,lookup[k]]=1
    count=present.sum(axis=0);offset=np.divide(raw.sum(axis=0),count,out=np.zeros(n),where=count>0)
    centered=(raw-offset)*present
    sd=np.sqrt(np.divide((centered**2).sum(axis=0),np.maximum(count-1,1)))
    floor=np.asarray([names[k] for k in keys]);scale=np.maximum(sd,floor)
    adapter=dict(version=ADAPTER_VERSION,keys=keys,offset=offset,scale=scale,physical_floor=floor,
        registry_policy='Field paths/shapes from registered cohort only; all numerical scaling uses training states only.',
        training_input_hashes=[p['input_hash'] for p in train_payloads],
        dimensions=2*n,model_input='All scaled numeric coordinates followed by per-coordinate presence masks; no dropped coordinates, clipping, queue aggregation or projection.',
        unseen_coordinate_policy='Fail closed; register a complete structural schema or use lossless sparse kernel.')
    adapter['fingerprint']=_adapter_hash(adapter);return adapter


def _adapter_hash(a):
    return canonical_hash(dict(version=a['version'],keys=a['keys'],offset=array_hash(a['offset']),
        scale=array_hash(a['scale']),physical_floor=array_hash(a['physical_floor']),training_input_hashes=a['training_input_hashes']))


def transform_adapter(adapter,payloads):
    if adapter.get('fingerprint')!=_adapter_hash(adapter):raise ValueError('Adapter fingerprint mismatch')
    lookup={k:i for i,k in enumerate(adapter['keys'])};n=len(lookup);out=np.zeros((len(payloads),2*n),dtype=np.float64)
    for j,p in enumerate(payloads):
        validate(p)
        for k,v,_ in _coordinates(p):
            if k not in lookup:raise ValueError('Unregistered state coordinate; refusing truncation: '+k)
            i=lookup[k];out[j,i]=(v-adapter['offset'][i])/adapter['scale'][i];out[j,n+i]=1.
    if not np.isfinite(out).all():raise ValueError('Nonfinite scaled complete-state input')
    return out


def save_adapter(path,adapter):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    arrays={k:adapter[k] for k in ('offset','scale','physical_floor')};atomic_npz(path/'scaling.npz',**arrays)
    meta={k:v for k,v in adapter.items() if k not in arrays};meta['scaling_sha256']=file_hash(path/'scaling.npz')
    atomic_json(path/'adapter.json',meta);return meta


def load_adapter(path):
    path=Path(path);meta=read_json(path/'adapter.json')
    if file_hash(path/'scaling.npz')!=meta.pop('scaling_sha256'):raise ValueError('Corrupt state adapter scaling')
    with np.load(path/'scaling.npz',allow_pickle=False) as f:meta.update({k:f[k] for k in f.files})
    if meta['fingerprint']!=_adapter_hash(meta):raise ValueError('State adapter schema changed')
    return meta


def sparse_kernel(left,right=None):
    """Lossless open-schema linear feature kernel; all fields have fixed unit scale.

    Distinct exact coordinates and presence bits remain distinct in an unbounded
    sparse feature space. This avoids any maximum queue length. A fitted linear
    kernel model has zero trained weight on fields never seen during training;
    that is a model limitation, not deletion from the input or claimed oracle.
    """
    right=left if right is None else right
    def row(p):
        validate(p);return {k:v/floor for k,v,floor in _coordinates(p)}
    a=[row(p) for p in left];b=[row(p) for p in right];out=np.zeros((len(a),len(b)))
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i,j]=sum(v*y[k]+1. for k,v in x.items() if k in y)
    return out


def learnable_control(destination=None):
    """Cheap linear target uses formerly omitted A/B, phase and exact event data."""
    rng=np.random.default_rng(19073);n=96;values=rng.normal(size=(n,5))
    # Reuses the production adapter's scaling/feature path rather than a bespoke fit.
    payloads=[]
    for row in values:
        p=_synthetic_payload()
        for key,value in zip(('synapses/contact/A','synapses/contact/B','drive/phase_rad',
                              'pending/recurrent/0/arrival_ms','connectivity/contact/weight_uS'),row):
            _add(p,key,value,'synapses' if key.startswith('synapses') else key.split('/')[0],'known engineering control')
        _seal_synthetic(p);payloads.append(p)
    adapter=fit_adapter(payloads[:64],all_schema=payloads)
    x=transform_adapter(adapter,payloads);y=values@np.asarray([.7,-.4,.3,.2,-.6])
    # Standard dual ridge with a training intercept, no target labels in the adapter.
    mean=x[:64].mean(0);z=x-mean;ym=y[:64].mean()
    alpha=np.linalg.solve(z[:64]@z[:64].T+1e-8*np.eye(64),y[:64]-ym)
    pred=z[64:]@z[:64].T@alpha+ym;mse=float(np.mean((pred-y[64:])**2));base=float(np.mean((ym-y[64:])**2))
    result=dict(status='complete',passed=mse<1e-10,control='engineering; deterministic target depends on synaptic A/B, phase, pending arrival and connection weight',
        training_contexts=64,evaluation_contexts=32,physical_mse=mse,mean_baseline_mse=base,
        relative_mse_reduction=1-mse/base,adapter_dimensions=adapter['dimensions'],neuron_invocations=0,
        interpretation='Tests lossless adapter and fitted state-dependent ridge path; not biological predictability or pilot completion.')
    if not result['passed']:raise AssertionError('Complete-state learnable control failed')
    if destination is not None:atomic_json(Path(destination),result)
    return result


def _synthetic_payload():
    p=dict(schema_version=SCHEMA_VERSION,arrays={},fields={},inventory={},binding={'context_id':'control','structure_id':'control'})
    for group in _REQUIRED_GROUPS:_add(p,group+'/control',0.,group,'synthetic engineering fixture')
    _add(p,'pending/recurrent_count',0.,'pending','synthetic empty pending queue')
    p['inventory']=dict(density_mechanisms=sorted(_ALLOWED_DENSITY),point_process_types=['Exp2Syn'],
        extracellular_layers=2,contact_count=1,external_connections=1,recurrent_connections=0,pending_recurrent_count=0)
    return p


def _seal_synthetic(p):
    p['schema_hash']=canonical_hash(dict(schema_version=SCHEMA_VERSION,fields=p['fields'],inventory=p['inventory']))
    p['input_hash']=input_hash(p)
    return p
