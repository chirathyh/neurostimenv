"""Outcome-blind registration of historical pilot populations and RNG extensions."""
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import json
import re
from ..artifacts import rng_seed
from ..targets import canonical_hash
from ..v6.evidence import EvidenceSource
from ...storage import atomic_json, file_hash

SCHEMA='eegrepr-reporting-pilot-design-1'
FRESH_CIRCUIT=37700
FRESH_NAMESPACE='eegrepr-v6-reporting-pilot-fallback-20261117'
EXPECTED_NATIVE_MANIFEST='c9ac6a52ea5f8d6ea17bedd608194cabb42cb90fd864930c342a8da3520bcd3a'
EXPECTED_NATIVE_CONFIG='5c9e98d2651d141be31e0a5af68ed1938c631ec63a651b418e99ab4cc9b05387'


def source_metadata(source, require_native=True):
    source=Path(source).expanduser().resolve()
    if not source.is_dir(): raise ValueError('Native reuse requires a directory containing actual HDF5 recordings, not a compact ZIP')
    directory=source/'generation' if (source/'generation/native/manifest.json').is_file() else source
    if not (directory/'native/manifest.json').is_file(): raise ValueError('Native source lacks native/manifest.json')
    manifest=json.loads((directory/'native/manifest.json').read_text())
    cfg=json.loads((directory/'resolved_config.json').read_text())
    provenance=json.loads((directory/'provenance.json').read_text())
    if canonical_hash(manifest)!=EXPECTED_NATIVE_MANIFEST or canonical_hash(cfg)!=EXPECTED_NATIVE_CONFIG:
        raise ValueError('Native source differs from the frozen 832-episode historical export; never replace its generation provenance with current source')
    if manifest.get('config_hash')!=canonical_hash(cfg) or cfg.get('backend')!='neuron' or len(manifest.get('trials',[]))!=832:
        raise ValueError('Historical native source configuration/count/backend mismatch')
    rows=deepcopy(manifest['trials']);seen=set()
    for row in rows:
        if row['trial_id'] in seen:raise ValueError('Duplicate historical trial ID')
        seen.add(row['trial_id'])
        native_root=Path(row.get('source_root',directory)).resolve();path=(native_root/row['path']).resolve()
        if not path.is_relative_to(native_root):raise ValueError('Native path escapes original root')
        if require_native and (not path.is_file() or path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']):
            raise ValueError('Missing or altered original native recording: '+str(path))
        row['source_root']=str(native_root)
        row['generation_origin']='historical_reused'
        row['original_generation_provenance']=provenance
    return dict(source_root=str(directory), original_configuration=cfg,
        original_generation_provenance=provenance, manifest_sha256=file_hash(directory/'native/manifest.json'), rows=rows)


def _context(rows):
    row=min(rows,key=lambda x:(x['pool']!='score',x['future_index']))
    match=re.search(r'-p(\d+)-',row['context_id'])
    if match is None:raise ValueError('Context lacks explicit prefix index')
    return dict(context_id=row['context_id'],structure_id=int(row['structure_id']),prefix=int(match.group(1)),
        split=row['split'],regime=deepcopy(row['spec']['regime']),cutoff_ms=float(row['spec']['future_start_ms']),
        score_futures=sum(r['pool']=='score' for r in rows),reference_futures=sum(r['pool']=='reference' for r in rows),
        representative=deepcopy(row),original_trial_ids=sorted(r['trial_id'] for r in rows))


def select_contexts(rows):
    grouped=defaultdict(list)
    for row in rows:grouped[row['context_id']].append(row)
    contexts=[_context(v) for v in grouped.values()]
    contexts.sort(key=lambda x:(x['structure_id'],x['prefix'],x['regime']['name']))
    regimes=sorted({x['regime']['name'] for x in contexts})
    if len(regimes)!=4:raise ValueError('Historical pilot requires all four original regimes')
    ids={s:sorted({x['structure_id'] for x in contexts if x['split']==s}) for s in ('train','development','evaluation')}
    if len(ids['train'])<8 or len(ids['development'])<4 or len(ids['evaluation'])<2:
        raise ValueError('Insufficient disjoint historical circuits for 32/8/8/8 state cohort')
    circuits=dict(train=ids['train'][:8],development=ids['development'][:2],calibration=ids['development'][2:4],evaluation=ids['evaluation'][:2])
    selected=[]
    for split,structures in circuits.items():
        for sid in structures:
            for regime in regimes:
                candidates=[x for x in contexts if x['structure_id']==sid and x['regime']['name']==regime]
                if not candidates:raise ValueError('Missing regime coverage in matched state cohort')
                chosen=deepcopy(min(candidates,key=lambda x:(x['prefix'],x['context_id'])))
                chosen['historical_split']=chosen['split'];chosen['split']=split;selected.append(chosen)
    high=[x for x in contexts if x['split']=='evaluation' and x['score_futures']==8 and x['reference_futures']==32]
    unmod=[x for x in high if float(x['regime'].get('modulation_depth',0))==0]
    if not unmod:raise ValueError('No unmodulated M8/L32 historical context')
    first=deepcopy(unmod[0]);mod=[x for x in high if x['structure_id']!=first['structure_id'] and float(x['regime'].get('frequency_hz',0))==10 and float(x['regime'].get('modulation_depth',0))>0]
    if not mod:raise ValueError('No different-circuit 10-Hz M8/L32 historical context')
    h2=[first,deepcopy(mod[0])]
    # Timing contexts cover both H2 prefix checks and distinct selected state exports.
    cohort_by_id={x['context_id']:x for x in selected}
    if any(x['context_id'] not in cohort_by_id for x in h2):raise ValueError('Registered H2 contexts must lie in matched evaluation cohort')
    timing=[deepcopy(cohort_by_id[x['context_id']]) for x in h2]
    for split,n in [('train',4),('development',1),('calibration',1)]:
        timing.extend(deepcopy([x for x in selected if x['split']==split][:n]))
    if len(timing)!=8 or len({x['context_id'] for x in timing})!=8:raise ValueError('Concurrency check must use exactly eight distinct contexts')
    return dict(state_cohort=selected,state_circuits=circuits,h2_contexts=h2,concurrency_contexts=timing,
        selection_rule='Ascending circuit, prefix and regime IDs, without reading EEG or outcomes; first eight train circuits × four regimes; development circuits split two/two for development/calibration; first two evaluation circuits × four regimes',
        state_counts={'train':32,'development':8,'calibration':8,'evaluation':8},
        populations=dict(ordinary='historical eight-evaluation-circuit inspected population',complete_state='registered 56-context cohort with two evaluation circuits',h2='two registered states on different evaluation circuits; conditional Monte Carlo diagnostics'))


def continuation_spec(original_spec, namespace, future):
    """Copy every history/physics field. Only the independent future draw changes."""
    spec=deepcopy(original_spec);sid=int(spec['structure_seed'])
    match=re.search(r'-p(\d+)-',spec['history_id'])
    if not match:raise ValueError('History prefix index unavailable')
    prefix=int(match.group(1));future=int(future)
    if not 8<=future<32:raise ValueError('Pilot extends exactly scoring futures 8 through 31')
    for key,kind in [('future_seed','future-events'),('future_phase_seed','future-phase'),('future_burst_seed','future-envelope')]:
        spec[key]=rng_seed(namespace,kind,sid,prefix,'score',future)
    spec['future_id']=f'score-{future:04d}';spec['trial_id']=spec['history_id']+'-'+spec['future_id']
    return spec


def verify_pool_seeds(rows,specs):
    by_context=defaultdict(set)
    for row in rows:
        spec=row.get('spec',row);cid=spec['history_id'];seed=int(spec['future_seed'])
        if seed in by_context[cid]:raise ValueError('Historical scoring/reference future seeds overlap')
        by_context[cid].add(seed)
    for spec in specs:
        cid=spec['history_id'];seed=int(spec['future_seed'])
        if seed in by_context[cid]:raise ValueError('New scoring draw overlaps an existing scoring/reference draw')
        by_context[cid].add(seed)
    return {'passed':True,'contexts':len(by_context),'counts':{k:len(v) for k,v in by_context.items()}}


def register(config,root,source,require_native=True):
    root=Path(root);original=source_metadata(source,require_native);selection=select_contexts(original['rows'])
    if FRESH_CIRCUIT in {int(r['structure_id']) for r in original['rows']}:raise ValueError('Fallback circuit ID already used by imported evidence')
    namespace=original['original_configuration']['namespace']
    additions=[]
    for context in selection['h2_contexts']:
        additions.extend(continuation_spec(context['representative']['spec'],namespace,f) for f in range(8,32))
    disjoint=verify_pool_seeds(original['rows'],additions)
    result=dict(schema_version=SCHEMA,purpose='exploratory pilot/template-development; no power/sample-size expansion',
        selection=selection,source={k:v for k,v in original.items() if k!='rows'},
        historical_rows=original['rows'],h2_additional_specs=additions,seed_pool_audit=disjoint,
        fallback_registration=dict(circuit_id=FRESH_CIRCUIT,namespace=FRESH_NAMESPACE,prefix=0,regime=selection['h2_contexts'][1]['regime'],score_futures=32,reference_futures=32,
            condition='Only when registered original prefix/physical-state checks fail; separate population, never blended'),
        budget_plan=dict(hard_cap=160,concurrency_checks=8,maximum_cutoff_recoveries=56,cutoff_recoveries_after_reusing_eight_captured_checks=48,h2_additional_episodes=48,exact_forcing_replays=2,
            planned_calls_with_captured_checks=106,conservative_calls_without_capture_reuse=114,fallback_calls_with_captured_checks=122,
            note='Every failed, interrupted, retry, gate and replay native call remains charged; capacity checked before each stage and simulator call'))
    result['design_hash']=canonical_hash(result)
    marker=root/'design.json'
    if marker.exists():
        if json.loads(marker.read_text())!=result:raise ValueError('Pilot design/source changed; choose a new output root')
    else:atomic_json(marker,result)
    return result
