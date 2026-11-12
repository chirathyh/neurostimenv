"""Actual-job planning and measured full-network resource gates."""
from pathlib import Path
import platform
from .config import context_specs,cutoff_ms,stop_ms
from .targets import canonical_hash
from .dataset import target_spec
from .artifacts import read_json,source_hash,verify_record


def timing_reference(c,timing_root):
    """Use the completed gate's checksum-bound timing, never a mutable alias."""
    if timing_root is None:
        if c['cost_gate']['required']:raise ValueError('A measured full-network timing audit is required; run the eight-continuation gate and pass --timing-root')
        return None
    root=Path(timing_root);p=read_json(root/'provenance.json')
    gate=read_json(root/'resolved_config.json')
    expected_hash=canonical_hash(gate)
    if p.get('schema_version')!='eegrepr-h1-2.0' or p.get('source_hash')!=source_hash():raise ValueError('Timing audit schema/source differs from this implementation')
    if p.get('config_hash')!=expected_hash:raise ValueError('Timing gate resolved configuration does not match provenance')
    contexts=context_specs(gate)
    structures={x['structure_id'] for x in contexts};regimes={x['regime']['name'] for x in contexts}
    pairs={(x['structure_id'],x['regime']['name']) for x in contexts}
    full_network=(gate.get('backend')=='neuron' and
                  (gate['simulation']['population_e'],gate['simulation']['population_i'])==(32,8))
    design=(len(contexts)==4 and len(structures)==2 and len(regimes)==2 and len(pairs)==4 and
            all(x['split']=='evaluation' and x['score_futures']==2 and x['reference_futures']==0 for x in contexts))
    if not full_network or not design:raise ValueError('Timing gate must be the full 32E/8I, two-circuit by two-regime, two-future design (eight continuations)')
    candidates=[root/'stages/generate-evaluation.json',root/'stages/generate-all.json']
    stages=[path for path in candidates if path.exists()]
    if not stages:raise ValueError('No completed timing generation stage')
    expected_shards={f"native/evaluation/{x['context_id']}/score-{f:04d}.h5" for x in contexts for f in range(2)}
    # The CLI creates generate-evaluation; generate-all is supported only when
    # its configuration is this same exact gate with no training/dev contexts.
    stage=read_json(stages[0])
    if (stage.get('schema_version')!='eegrepr-h1-2.0' or stage.get('status')!='complete' or
            stage.get('source_hash')!=source_hash() or stage.get('config_hash')!=expected_hash):
        raise ValueError('Timing stage schema/config/source is stale or incomplete')
    records={row['path']:row for row in stage['outputs']}
    timing_rel=f"native/timing-{stage['stage']}.json"
    audit_rel=f"native/prefix_validation-{stage['stage']}.json"
    if not expected_shards.issubset(records) or timing_rel not in records or audit_rel not in records:
        raise ValueError('Timing stage lacks the complete eight continuations and bound timing/prefix audits')
    for record in records.values():verify_record(root,record)
    t=read_json(root/timing_rel);audit=read_json(root/audit_rel)
    audited={row['context_id']:row for row in audit.get('contexts',[])}
    if set(audited)!={x['context_id'] for x in contexts} or not all(
            row.get('fork_audit_status')=='passed' and row.get('audit',{}).get('passed') is True
            for row in audited.values()):raise ValueError('Timing gate prefix audit is incomplete or failed')
    if (t.get('backend')!='neuron' or (t.get('population_e'),t.get('population_i'))!=(32,8) or
            not t.get('complete') or t.get('contexts')!=4 or t.get('continuations')!=8):
        raise ValueError('Timing measurement does not match the complete full-network gate')
    import math
    seconds=t.get('mean_continuation_seconds');summed=t.get('summed_generation_seconds')
    if (not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<=0 or
            not isinstance(summed,(int,float)) or not math.isfinite(summed) or
            not math.isclose(seconds*8,summed,rel_tol=1e-9)):
        raise ValueError('Invalid or inconsistent measured timing')
    if t.get('episode_stop_ms')!=stop_ms(gate) or t.get('cutoff_ms')!=cutoff_ms(gate):
        raise ValueError('Timing duration/cutoff differs from the verified gate configuration')
    if t.get('native_bytes')!=sum(records[path]['bytes'] for path in expected_shards):
        raise ValueError('Timing storage measurement differs from verified native shards')
    return {**t,'verified_timing_artifact':timing_rel,'verified_gate_config_hash':expected_hash}


def plan(c,timing_root=None):
    contexts=context_specs(c);counts={s:sum(x['split']==s for x in contexts) for s in ('train','development','evaluation')}
    continuations={s:sum(x['score_futures']+x['reference_futures'] for x in contexts if x['split']==s) for s in counts}
    stop=stop_ms(c);native_samples=round(stop/c['simulation']['native_dt_ms'])
    jobs=[]
    for j in c['jobs']:
        spec=target_spec(c,j)
        jobs.append({**j,'target_dimension':spec.dimension,'target_spec':spec.to_dict(),
          'target_fingerprint':spec.fingerprint,'input_samples':round(j['history_s']*j['input_rate_hz'])})
    measurements=timing_reference(c,timing_root)
    total=sum(continuations.values());memory_bytes=native_samples*7*8
    projection=None
    if measurements:
        ratio=stop/measurements['episode_stop_ms']
        seconds=total*measurements['mean_continuation_seconds']*ratio
        projection={'measured_mean_gate_seconds':measurements['mean_continuation_seconds'],
          'measured_gate_total_seconds':measurements['summed_generation_seconds'],
          'duration_ratio':ratio,'projected_summed_trial_hours':seconds/3600,
          'projected_serial_wall_hours':seconds/3600,'assumed_workers':1,
          'measured_gate_rss_bytes':measurements['peak_process_rss_bytes'],
          'projected_native_disk_bytes':round(total*measurements['native_bytes']/measurements['continuations']*ratio),
          'method':'linear duration extrapolation from measured full-network gate; excludes fitting/derivation, scheduling variation and extra replay controls',
          'hardware_transfer':'Valid on the measured host; remeasure after hardware/software changes'}
    trainable=counts['train']>0 and counts['development']>0
    fixed=c['fixed_budget_extension'];b=c['benchmark'];limit=b['max_candidates']
    candidates={'train_mean':1,'train_mean_unit_variance':1,'persistence':1,'ridge_wiener':min(limit,len(b['ridge_alphas'])*len(b['lags'])),
      'harmonic':min(limit,len(b['harmonic_history_fractions'])),'state_space':min(limit,len(b['state_ranks'])),
      'neural_mse':min(limit,len(b['learning_rates'])),'neural_gaussian':min(limit,len(b['learning_rates']))}
    finalists=sum(len(b['seeds']) if m.startswith('neural_') else 1 for m in b['models'])
    searches=sum(candidates[m]*(len(b['seeds']) if m.startswith('neural_') else 1) for m in b['models'])
    comp=c['compression'];task_upper=7
    head_upper=(len(comp['representations'])*len(comp['dimensions'])*len(comp['code_bits'])+1)*len(comp['readouts'])*len(comp['label_budgets'])*task_upper

    return {'schema_version':'eegrepr-h1-2.0','name':c['name'],'config_hash':canonical_hash(c),
      'backend':c['backend'],'claim_status':c['claim_status'],'circuit_counts':{s:len(v) for s,v in c['data']['structures'].items()},
      'contexts_by_split':counts,'continuations_by_split':continuations,'total_continuations':total,
      'evaluation_reference_futures':sum(x['reference_futures'] for x in contexts if x['split']=='evaluation'),
      'evaluation_score_futures':sum(x['score_futures'] for x in contexts if x['split']=='evaluation'),
      'explicit_jobs':jobs,'actual_job_count':len(jobs),
      'fit_phase_eligible':trainable,
      'fit_phase_note':'training and development contexts present' if trainable else 'timing-only panel; no trainable models or compression heads',
      'final_fitted_model_count':len(jobs)*finalists if trainable else 0,
      'benchmark_search_candidate_fits':len(jobs)*searches if trainable else 0,
      'compression_readout_heads_upper_bound':len(jobs)*head_upper if trainable else 0,
      'compression_cost_note':'Upper bound before dimensional/code feasibility and constant-task exclusions; nested training-circuit tuning multiplies readout fits, encoder fitting is additional. Readout wall time requires measurement.',
      'fit_searches':'Hyperparameter and diagnostics counts are separately recorded by the benchmark; final fit count is not all forward passes',
      'cutoff_ms':cutoff_ms(c),'stop_ms':stop,'native_samples_per_continuation':native_samples,
      'uncompressed_signal_bytes':total*native_samples*5*8,
      'minimum_one_trial_working_array_bytes':memory_bytes,'memory_note':'Array lower bound; NEURON events, checkpoint states and model memory need measured RSS',
      'storage_note':'Includes native EEG/time/3-dipole columns only before compression; event/state files, NPZ/model outputs and backups require additional disk',
      'measured_gate':measurements,'resource_projection':projection,
      'historical_planning_only':{'old_trial_mean_seconds':210.86,'old_duration_seconds':14.,
          '3584_full_14s_replays_summed_trial_hours':3584*210.86/3600,
          'interpretation':'historical arithmetic, not measured H1 runtime or concurrent wall time'},
      'fixed_budget_extension_enabled':fixed['enabled'],
      'eligible_for_large_generation':bool(measurements) or not c['cost_gate']['required'],
      'confirmation':'planned only; >=12 circuits is provisional and must be refined from development precision, not a power guarantee'}
