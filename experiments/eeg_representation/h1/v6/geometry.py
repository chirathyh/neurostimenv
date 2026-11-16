"""Secondary state geometry sensitivity with train-only, unit-aware scale rules."""
from pathlib import Path
import json
import numpy as np
from ...storage import atomic_json, file_hash
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from .evidence import EvidenceSource
from ..v4.geometry import covariance_geometry, deduplicate_prefixes, geometry_controls


def mechanism_axes(membrane_coordinates,total_coordinates):
    if membrane_coordinates%47 or total_coordinates<membrane_coordinates or (total_coordinates-membrane_coordinates)%2:
        raise ValueError('Expected audited 47-coordinate soma/apical membrane layout and A/B synapses')
    names=[]
    for _ in range(membrane_coordinates//47):
        names.extend(['voltage','extracellular','hh_gate','hh_gate','hh_gate'])
        names.extend(['voltage','extracellular']*21)
    names.extend(['exp2syn_conductance']*(total_coordinates-membrane_coordinates))
    # Numerical comparison units, not empirical human scales or biological DOF.
    units={'voltage':'mV','extracellular':'mV','hh_gate':'dimensionless','exp2syn_conductance':'uS'}
    references={'voltage':1.,'extracellular':1.,'hh_gate':1.,'exp2syn_conductance':1.}
    absolute_floors={'voltage':1e-6,'extracellular':1e-6,'hh_gate':1e-8,'exp2syn_conductance':1e-12}
    return np.asarray(names),units,references,absolute_floors


def scaling_rules(training,groups,*,relative_floor=1e-3):
    """Freeze rules from training only, irrespective of held-out excursions."""
    training=np.asarray(training,dtype=np.float64);groups=np.asarray(groups)
    if training.ndim!=2 or len(training)<2 or len(groups)!=training.shape[1] or not np.isfinite(training).all():
        raise ValueError('Finite training matrix with at least two contexts and matching mechanisms required')
    _,units,references,absolute=mechanism_axes(47,47)
    # Standard deviation computed after reference scaling so heterogeneous units
    # never determine the relative threshold in another mechanism family.
    sd=np.std(training,axis=0,ddof=1);floors=np.empty(training.shape[1]);fixed=np.empty_like(floors);rules={}
    for group in np.unique(groups):
        if group not in units:raise ValueError('Unknown mechanism unit '+group)
        mask=groups==group;positive=sd[mask][sd[mask]>absolute[group]]
        median=float(np.median(positive)) if len(positive) else 0.
        floor=max(absolute[group],relative_floor*median);floors[mask]=floor;fixed[mask]=references[group]
        rules[group]=dict(unit=units[group],fixed_reference=references[group],absolute_sd_floor=absolute[group],
            relative_median_sd_floor=relative_floor,training_median_active_sd=median,effective_sd_floor=floor,
            coordinates=int(mask.sum()),below_floor=int(np.sum(sd[mask]<=floor)),
            exactly_constant=int(np.sum(sd[mask]==0)))
    return dict(offset=training.mean(0),training_sd=sd,scale_floor=np.maximum(sd,floors),
                fixed_scale=fixed,keep=sd>floors,floors=floors,rules=rules)


def _geometry(x):
    # Normalizing the common matrix magnitude changes no PR/rank/count. Retain
    # explicit covariance unit factor and avoid ever squaring a huge raw matrix.
    x=np.asarray(x,dtype=np.float64)
    if not x.shape[1]:return dict(status='unavailable',reason='No coordinates survive train-only rule',samples=len(x),coordinates=0,rank_ceiling=0)
    magnitude=float(np.max(np.abs(x))) or 1.
    result=covariance_geometry(x/magnitude)
    result['covariance_eigenvalue_unit_factor']=magnitude*magnitude
    result['eigenvalues_scaled_for_numerical_stability']=True
    result['scale_factor']=magnitude
    if result['status']=='complete':result['covariance_trace_original_scaled_units']=result['covariance_trace']*magnitude*magnitude
    return result


def analyze_state(values,training_mask,circuits,regimes,membrane_coordinates):
    x=np.asarray(values,dtype=np.float64);train=np.asarray(training_mask,bool);circuits=np.asarray(circuits);regimes=np.asarray(regimes)
    if x.ndim!=2 or not np.isfinite(x).all() or train.shape!=(len(x),) or circuits.shape!=train.shape or regimes.shape!=train.shape:
        raise ValueError('State populations/axes mismatch')
    groups,units,_,_=mechanism_axes(membrane_coordinates,x.shape[1]);scales=scaling_rules(x[train],groups)
    maps={'fixed_mechanism_units':x/scales['fixed_scale'],
          'train_sd_floored':(x-scales['offset'])/scales['scale_floor'],
          'train_sd_drop_near_constant':(x[:,scales['keep']]-scales['offset'][scales['keep']])/scales['training_sd'][scales['keep']]}
    reports={}
    for name,a in maps.items():
        report=dict(pooled=_geometry(a),within_circuit=[],within_circuit_regime=[],sample_size_sensitivity=[])
        residual=a.copy()
        for circuit in np.unique(circuits):
            mask=circuits==circuit;report['within_circuit'].append(dict(circuit=str(circuit),**_geometry(a[mask])))
            for regime in np.unique(regimes[mask]):
                select=mask&(regimes==regime)
                report['within_circuit_regime'].append(dict(circuit=str(circuit),regime=str(regime),**_geometry(a[select])))
                residual[select]-=residual[select].mean(0)
        report['pooled_within_circuit_regime_centered']=_geometry(residual)
        rng=np.random.default_rng(20611)
        for n in sorted(set([min(len(a),16),min(len(a),32),min(len(a),64),len(a)])):
            if n>=2:
                r=_geometry(a[np.sort(rng.choice(len(a),n,replace=False))]);report['sample_size_sensitivity'].append({k:r[k] for k in ('samples','coordinates','rank_ceiling','participation_ratio') if k in r})
        reports[name]=report
    positive=scales['training_sd'][scales['training_sd']>0]
    diagnostic=dict(minimum_positive_training_sd=float(positive.min()) if len(positive) else None,
        positive_sd_below_1e50=int(np.sum((scales['training_sd']>0)&(scales['training_sd']<1e-50))),
        dropped_coordinates=int(np.sum(~scales['keep'])),total_coordinates=x.shape[1],
        interpretation='Sparse nearly inactive training coordinates can explode ordinary coordinate-standardized covariance; rules do not use evaluation magnitudes')
    per_mechanism={g:dict(unit=units[g],pooled_physical=_geometry(x[:,groups==g])) for g in np.unique(groups)}
    return dict(status='complete',populations=dict(contexts=len(x),training_contexts=int(train.sum()),circuits=len(np.unique(circuits)),
                training_circuits=len(np.unique(circuits[train])),regimes=np.unique(regimes).tolist()),
        rules=scales['rules'],pathology_diagnostic=diagnostic,scalings=reports,per_mechanism=per_mechanism,
        interpretation='Descriptive linear covariance geometry; mixed-unit/scaling choices matter; not intrinsic dimension or retained information',
        sampling='One snapshot per distinct context; prefixes/futures nested within circuits; sample-size sensitivity is not population confidence',
        excluded_state='Variable-length pending event queues and phase are not represented in the numeric covariance matrix; future RNG/drive is never a learner input'),scales


def run(c,root,source=None,resume=False):
    root=Path(root);dest=root/'geometry';dest.mkdir(parents=True,exist_ok=True)
    owned=source is not None and not hasattr(source,'read_npz');src=EvidenceSource(source) if owned else source
    try:
        array_name='geometry/privileged_state_geometry.npz';inventory_name='geometry/runtime_inventory.json'
        if src is None or not src.exists(array_name):
            result=dict(status='unavailable',reason='Saved privileged aligned state arrays absent; no invented trajectories or nonlinear intrinsic dimension',controls=geometry_controls())
            atomic_json(dest/'summary.json',result);return result
        source_record=src.verify(array_name);arrays=src.read_npz(array_name)
        config=src.read_json('resolved_config.json') if src.exists('resolved_config.json') else c
        training_ids=set(map(str,config['data']['structures']['train']))
        ids=arrays['context_id'].astype(str);x=arrays['aligned_aggregate_state'];keep=deduplicate_prefixes(x,ids)
        circuits=arrays['structure_id'].astype(str)[keep];regimes=arrays['regime'].astype(str)[keep]
        train=np.isin(circuits,list(training_ids));membrane=arrays['membrane_gate_state'].shape[1]
        signature=canonical_hash(dict(source=source_record,training_circuits=sorted(training_ids),module=file_hash(Path(__file__))))
        marker=dest/'summary.json'
        if marker.exists():
            previous=json.loads(marker.read_text())
            if not resume or previous.get('signature')!=signature:raise ValueError('Geometry exists or analysis identity changed')
            for row in previous.get('artifacts',[]):
                if file_hash(dest/row['path'])!=row['sha256']:raise ValueError('Geometry artifact checksum mismatch')
            return previous
        report,scales=analyze_state(x[keep],train,circuits,regimes,membrane)
        controls=geometry_controls()
        if not controls['passed']:raise AssertionError('Known geometry controls failed')
        atomic_json(dest/'state_scaling_sensitivity.json',report)
        atomic_npz(dest/'train_scaling.npz',context_id=ids[keep],structure_id=circuits,regime=regimes,training_mask=train,
                   **{k:v for k,v in scales.items() if isinstance(v,np.ndarray)})
        historical=src.verify(inventory_name) if src.exists(inventory_name) else None
        result=dict(status='complete',signature=signature,source=source_record,historical_inventory=historical,
            historical_pathology_policy='Original v4 files preserved unchanged; new descriptive sensitivity does not relabel or replace their result',
            controls=controls,methods=list(report['scalings']),populations=report['populations'],
            nonlinear_id=dict(status='unavailable',reason='No qualified nonlinear estimator; no universal intrinsic dimension claimed'),
            innovation=dict(status='unavailable',reason='No new cross-covariance inference in state scaling audit; original noisy non-PSD estimates preserved'),
            artifacts=[dict(path=p.name,sha256=file_hash(p),bytes=p.stat().st_size) for p in sorted(dest.iterdir()) if p.is_file() and p.name!='summary.json'])
        atomic_json(marker,result);return result
    finally:
        if owned:src.close()
