"""Immutable scientific freeze and same-host measured execution gates."""
from pathlib import Path
import platform,math
from .config import counts,contexts
from .generation import adapter
from .measurement import metadata
from .artifacts import atomic_json,read_json,file_hash,source_hash,environment,ROOT
from ..targets import canonical_hash
from ..planning import timing_reference
from .evidence import EvidenceSource


def registry_check(c):
    registry=read_json(ROOT/'configs/eeg_representation/v4/seed_registry.json')
    names={'prospective_bounded':'bounded_prospective','timing_only':'timing_gate'}
    group=names.get(c['claim_status'],c['name'])
    allowed=registry['ranges'].get(group)
    ids={r['structure_id'] for r in contexts(c)}
    if c['claim_status'] in names and (allowed is None or not ids<=set(range(allowed[0],allowed[1]+1))):
        raise ValueError('Unregistered prospective/timing circuits; reserve a new disjoint namespace first')
    return {'registry_sha256':file_hash(ROOT/'configs/eeg_representation/v4/seed_registry.json'),'reservation':group,'circuit_ids':sorted(ids)}


def freeze_protocol(c,root,qualification_root):
    root=Path(root);q=Path(qualification_root)
    measurement=read_json(q/'measurement/qualification.json');models=read_json(q/'models/manifest.json')
    if not measurement.get('passed') or not models.get('neural_path_control',{}).get('passed'):
        raise ValueError('Measurement and SAME neural-path qualification must pass before freezing')
    if not models.get('jobs'):raise ValueError('No completed EEG model qualification')
    # The engineering control is bound both as an artifact and in the fit index.
    control=read_json(q/'models/neural_path_control.json')
    if control!=models['neural_path_control'] or not control.get('same_neural_path') or not control.get('risk',float('inf'))<.1*control.get('baseline_risk',0):
        raise ValueError('Known neural-path qualification is missing, inconsistent or failed')
    for a in c['acquisition_ids']:
        if measurement.get('conditions',{}).get(a,{}).get('operator_hash')!=metadata(a)['operator_hash']:
            raise ValueError('Qualified acquisition operator differs from frozen operator')
    frozen_files=['measurement/qualification.json','models/manifest.json','models/neural_path_control.json','derived/manifest.json']
    inspected=set()
    for job in models['jobs']:
        if job['status']!='complete':raise ValueError('Qualification model jobs incomplete')
        model=read_json(q/job['path'])
        if model.get('complete_ridge_candidate_count')!=14 or not model.get('mean_fallback_retained'):raise ValueError('Incomplete model grid or missing mean fallback')
        frozen_files.append(job['path'])
        from .derivation import load_panel
        panels,derived_manifest=load_panel(q,job['id'])
        from .models import _signature
        if model.get('signature')!=_signature(c,derived_manifest,q,job['id']):
            raise ValueError('Qualified learner grid/source differs from prospective learner grid')
        for p in panels.values():inspected.update(map(int,p['structure_id']))
        # Bind actual numeric qualification predictions/state, not only a summary label.
        for record in model.get('files',[]):
            if file_hash(q/record['path'])!=record['sha256']:raise ValueError('Qualification artifact corrupted')
    prospective=set(registry_check(c)['circuit_ids'])
    if prospective&inspected:raise ValueError('Prospective circuit IDs overlap inspected qualification')
    if c['claim_status']!='prospective_bounded':raise ValueError('Only prospective bounded config can be scientifically frozen')
    if c['optional']['common_event_fraction'] or c['optional']['phase_diffusion']:raise ValueError('Optional robustness must use a separate named protocol, outside core832')
    contrast=c['protocol']['matched_code_contrast']
    if contrast['banks']!=['feature18','forecast'] or contrast['dimension']!=8 or contrast['bits']!=32:raise ValueError('Freeze ONE supported matched-code contrast')
    retention=c['retention']
    if (contrast.get('job_id')!=retention['job_id'] or contrast.get('task')!='carrier' or
        contrast.get('readout')!='linear' or contrast.get('pooling')!='native pooled' or
        contrast.get('range_rule') not in retention['range_rules'] or contrast['task'] not in retention['tasks'] or
        contrast['dimension'] not in retention['dimensions'] or contrast['bits'] not in retention['code_bits'] or
        contrast['label_budget'] not in retention['label_budgets'] or not set(contrast['banks'])<=set(retention['banks']) or
        len(set(retention['optimization_seeds']))<3):
        raise ValueError('Matched-code contrast is not supported by the frozen retention grid')
    from .targets import default_specs,TargetSpec
    from .models import settings
    specs=([TargetSpec(**{**v,'cutoff_ms':c['cutoff_ms']}) for v in c['target_specs']]
           if c.get('target_specs') else default_specs(c['cutoff_ms']))
    available={f'{a}__{v.id}' for a in c['acquisition_ids'] for v in specs}
    if c['protocol']['primary_job_id'] not in available or contrast['job_id'] not in available or not set(c['selected_job_ids'])<=available:
        raise ValueError('Primary/selected jobs do not match frozen operator/target specifications')
    if c['protocol']['primary_job_id'] not in c['selected_job_ids']:
        raise ValueError('Prospective primary job must be included in fitting/scoring jobs')
    if any(v.cutoff_ms+v.lead_ms+v.duration_ms>c['stop_ms'] for v in specs):
        raise ValueError('Frozen target support exceeds generated episode support')
    payload={'schema_version':c['schema_version'],'status':'frozen','config_hash':canonical_hash(c),'analysis_source_hash':source_hash(),
        'registry':registry_check(c),'namespace':c['namespace'],'operators':{a:metadata(a)['operator_hash'] for a in c['acquisition_ids']},
        'target_specs':[s.to_dict() for s in specs],
        'learner_grid':settings(c),'calibration':'Primary matched_calibrated: identical development-fitted context-independent diagonal family for every mean. Preserve uncalibrated and multiplicative-calibrated diagnostics; multiplicative access is matched but raw heteroscedastic expressivity is not.',
        'normalization':'target-specific training-only mean/scale; no evaluation normalization',
        'primary_job_id':c['protocol']['primary_job_id'],'matched_code_contrast':contrast,
        'qualification_files':[{'path':n,'sha256':file_hash(q/n)} for n in frozen_files],
        'qualification_root':str(q.resolve()),'inspected_circuits':sorted(inspected),'counts':counts(c),
        'interpretation':c['protocol']['claim'],'positive_evaluation_effect_required':False,
        'later_confirmation':'separate powered frozen plan using circuit heterogeneity, practical effect/CI width and multiplicity'}
    payload['protocol_hash']=canonical_hash(payload)
    path=root/'protocol.json'
    if path.exists() and read_json(path)!=payload:raise ValueError('Frozen protocol differs; use a new root/namespace')
    atomic_json(path,payload);return payload


def measured_gate(c,path):
    path=Path(path);p=read_json(path/'provenance.json');gate=read_json(path/'resolved_config.json')
    if p['analysis_source_hash']!=source_hash() or p['config_hash']!=canonical_hash(gate):raise ValueError('Timing source/config mismatch')
    if gate['claim_status']!='timing_only' or gate['optional']!=c['optional']:raise ValueError('Timing must match default full-network driver')
    if gate['simulation']!=c['simulation'] or gate['cutoff_ms']!=c['cutoff_ms'] or gate['stop_ms']!=c['stop_ms']:raise ValueError('Timing integration, contacts or duration mismatch')
    current=environment();old=p['environment']
    if any(old.get(k)!=current.get(k) for k in ('host','platform','python','versions','logical_cpus')):raise ValueError('Timing hardware/software host changed; remeasure on current host')
    t=timing_reference(adapter(c),path/'generation')
    return {**t,'root':str(path.resolve()),'environment':old,'generation_manifest_sha256':file_hash(path/'generation/native/manifest.json')}


def plan(c,root,timing_root=None,qualification_root=None):
    root=Path(root);design=counts(c);registry=registry_check(c)
    protocol=freeze_protocol(c,root,qualification_root) if qualification_root else (read_json(root/'protocol.json') if (root/'protocol.json').exists() else None)
    if protocol:
        if protocol['protocol_hash']!=canonical_hash({k:v for k,v in protocol.items() if k!='protocol_hash'}) or protocol['config_hash']!=canonical_hash(c) or protocol['analysis_source_hash']!=source_hash():
            raise ValueError('Existing protocol config/source/hash mismatch; choose a new root')
    gate=measured_gate(c,timing_root) if timing_root else None
    seconds=design['total_episodes']*(gate['mean_continuation_seconds'] if gate else c['budget']['provisional_trial_seconds']*c['stop_ms']/c['budget']['historical_duration_ms'])
    result={'status':'planned','config_hash':canonical_hash(c),'analysis_source_hash':source_hash(),'backend':c['backend'],
        'claim_status':c['claim_status'],'counts':design,'circuit_counts':{k:len(v) for k,v in c['data']['structures'].items()},
        'registry':registry,'stop_ms':c['stop_ms'],'cutoff_ms':c['cutoff_ms'],'protocol_hash':protocol['protocol_hash'] if protocol else None,
        'measured_gate':gate,'summed_generation_hours':seconds/3600,'serial_wall_hours_estimate':seconds/3600,'workers':1,
        'native_bytes_estimate':round(design['total_episodes']*gate['native_bytes']/8) if gate else round(c['budget']['provisional_native_bytes']*design['total_episodes']/832),
        'cost_status':'measured same-host extrapolation' if gate else 'provisional historical extrapolation; NOT execution authorization',
        'excludes':['fitting','derivation','export duplication','optional instrumentation','optional biological sensitivity','extra engineering replay control'],
        'fit_jobs':c['selected_job_ids'],'ridge_candidates_per_job':14,'neural_fitting_repeats_per_job':len(c['models']['neural_hidden'])*len(c['models']['neural_seeds'])*len(c['models']['neural_losses']),
        'gpu_note':'CUDA fits learners only; it does not accelerate this NEURON engine',
        'eligible_for_generation':c['claim_status']=='engineering' or c['claim_status']=='timing_only' or bool(gate and protocol),
        'claim':'8 evaluation circuits are bounded prospective evidence, not powered confirmation; negative results retained; old expanded plans are not executed'}
    result['plan_hash']=canonical_hash(result)
    # Keep every plan revision; a measured plan supersedes a provisional plan explicitly.
    atomic_json(root/'plans'/f"{result['plan_hash']}.json",result);atomic_json(root/'plan.json',result)
    return result


def require_execution(c,root,execute,timing=False):
    if not execute:raise ValueError('Generation requires a previously saved plan and explicit --execute')
    root=Path(root);p=read_json(root/'plan.json')
    if p.get('plan_hash')!=canonical_hash({k:v for k,v in p.items() if k!='plan_hash'}):raise ValueError('Saved measured plan hash mismatch')
    if p['config_hash']!=canonical_hash(c) or p['analysis_source_hash']!=source_hash():raise ValueError('Plan source/config changed')
    if timing != (c['claim_status']=='timing_only'):raise ValueError('timing-gate mode requires timing-only config; generate requires non-timing config')
    if c['claim_status']=='exploratory_existing':raise ValueError('Existing-data config cannot generate')
    if not p['eligible_for_generation']:raise ValueError('Freeze qualified protocol and record measured same-host gate before prospective generation')
    if c['claim_status']=='prospective_bounded':
        protocol=read_json(root/'protocol.json');expected=canonical_hash({k:v for k,v in protocol.items() if k!='protocol_hash'})
        if expected!=protocol['protocol_hash'] or protocol['protocol_hash']!=p['protocol_hash'] or protocol['config_hash']!=canonical_hash(c) or protocol['analysis_source_hash']!=source_hash():raise ValueError('Frozen protocol mismatch')
        measured_gate(c,p['measured_gate']['root'])
    return p
