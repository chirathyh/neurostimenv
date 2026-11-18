"""Pilot M32/L32 finite-future diagnostics and separately matched replay risks.

H2 uses fixed original/fallback contexts, never automatic reference expansion.
Replay comparators are rescored on the same two single-future observations; their
reference risks are not borrowed from the main eight-future matched cohort.
"""
from pathlib import Path
import json
import numpy as np
from ..artifacts import atomic_json,atomic_npz,file_hash
from ..targets import canonical_hash
from ..v6.targets import extract_physical_target,target_metadata,parse_target_id,timing_metadata,energy_from_coefficients,target_spec,axes
from ..v6.replication import analyze_futures
from ..v6.derivation import load_panel
from ..v6 import models as vm
from ...forks import _read_fork
from . import generation
from .learning import numeric


def _read(path):return json.loads(Path(path).read_text())


def _record(root,path):
    path=Path(path);return dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size)


def _risk_units(target):
    if target.startswith('T4_log_energy'):return 'squared natural-log units'
    if target.startswith('T4_energy'):return 'V^4'
    return 'V^2'


def _normalization(root,job):
    marker=Path(root)/'matched/derived'/job/'manifest.json';m=_read(marker)
    return m,m['normalization'],_record(root,marker)


def finite_future_tables(config,root,jobs):
    root=Path(root);targets=list(dict.fromkeys(job.split('__')[1] for job in jobs));rows=[];artifacts=[];figures={}
    summaries=[]
    for context_index,context in enumerate(generation.h2_trials(config,root)):
        cid=context['context_id'];score_ids=np.asarray([r[0] for r in context['score']]);ref_ids=np.asarray([r[0] for r in context['reference']])
        if set(score_ids)&set(ref_ids):raise ValueError('H2 scoring and reference IDs overlap')
        sid=int(context['score'][0][1]['metadata']['rng_namespaces']['structure'])
        cutoff=float(context['score'][0][1]['metadata']['fork']['cutoff_ms'])
        for target in targets:
            score=np.stack([extract_physical_target(t['eeg_v'],t['time_ms'],target,cutoff) for _,t in context['score']])
            reference=np.stack([extract_physical_target(t['eeg_v'],t['time_ms'],target,cutoff) for _,t in context['reference']])
            if score.shape[0]!=32 or reference.shape[0]!=32:raise ValueError('Pilot H2 requires M32/L32')
            # Physical diagnostics are computed once; acquisition only changes the
            # input, and may not silently change physical target/normalization.
            analysis=analyze_futures(score,reference,scoring_ids=score_ids,reference_ids=ref_ids,
                m_values=config['h2_m'],expected_reference_count=32,bootstrap=int(config['h2_bootstrap']),seed=814+context_index)
            analysis.update(protocol='reporting pilot M32/L32',primary_score_policy='Pilot ordinary information tables retain eight original scoring futures; these additions serve conditional H2 diagnostics only')
            directory=root/'precision/h2'/cid/target
            atomic_npz(directory/'targets.npz',scoring_physical=score,reference_physical=reference,
                scoring_future_id=score_ids,reference_future_id=ref_ids,context_id=np.asarray(cid),structure_id=np.asarray(sid))
            atomic_json(directory/'finite_reference.json',dict(context_id=cid,structure_id=sid,target=target,target_metadata=target_metadata(target,cutoff),**analysis))
            source=[_record(root,directory/'targets.npz'),_record(root,directory/'finite_reference.json')];artifacts.extend(source)
            normalized_reference=None
            for job in [j for j in jobs if j.split('__')[1]==target]:
                manifest,norm,normrecord=_normalization(root,job);scale=float(norm['y_scale']);offset=np.asarray(norm['y_offset'])
                this=(scale,tuple(offset))
                if normalized_reference is not None and this!=normalized_reference:raise ValueError('H2 target normalization changes across acquisition')
                normalized_reference=this
                family,h,w=parse_target_id(target)
                for item in analysis['rows']:
                    a=item['nested'];mean_ci=a.get('independent_pool_bootstrap_mean_corrected_mse_95')
                    var_ci=a.get('independent_pool_bootstrap_variance_difference_95')
                    rows.append(dict(job=job,acquisition=job.split('__')[0],target=target,h_ms=h,duration_ms=w,
                        context_id=cid,structure_id=sid,M=item['M'],L=32,circuits=1,contexts=1,status='measured',
                        population='fixed registered H2 context; conditional Monte Carlo',
                        mean_error_corrected=a['reference_noise_corrected_mean_mse'],
                        mean_error_uncorrected=a['squared_mean_distance_to_finite_reference'],
                        mean_error_reference_correction=a['reference_mean_noise_correction'],
                        mean_error_normalized=a['reference_noise_corrected_mean_mse']/scale**2,
                        mean_error_units=_risk_units(target),
                        variance_error=a.get('reference_noise_corrected_variance_mse'),
                        variance_error_uncorrected=a.get('squared_variance_difference'),
                        variance_difference=a.get('variance_difference_to_finite_reference'),
                        variance_error_normalized=None if a.get('reference_noise_corrected_variance_mse') is None else a['reference_noise_corrected_variance_mse']/scale**4,
                        variance_error_units='('+_risk_units(target)+')^2',
                        variance_status='unavailable: no direct within-context sample variance at M=1' if item['M']==1 else 'finite-reference corrected squared variance error; jackknife correction approximate',
                        reference_variance_estimate=a['reference_estimate_variance'],
                        reference_variance_jackknife_sampling_variance=a['reference_variance_jackknife_sampling_variance'],
                        uncertainty=dict(mean_error_interval=mean_ci,variance_difference_interval=var_ci,variance_error_interval=None,
                            variance_error_interval_status='deferred; variance-difference interval is not a squared-error interval',
                            method='independent two-pool percentile bootstrap for mean error and signed variance difference; both finite pools resampled',
                            confidence=.95,assumptions='conditional exchangeable simulator futures; approximate bootstrap, no population coverage claim'),
                        n_disjoint_scoring_groups=item['n_disjoint_scoring_groups'],
                        disjoint_group_mean_error=item['disjoint_group_corrected_mean_mse'],
                        groups_share_reference=True,subsets_independent=False,full_pool_does_not_mean_zero_mc_uncertainty=True,
                        reference_is_truth=False,reference_size_completed=32,deferred_full_study_reference_size=128,
                        target_fingerprint=manifest['target_fingerprint'],normalization_id=norm['id'],normalization_scale=scale,
                        source=source+[normrecord],interpretation='Within-state Monte Carlo precision is separate from across-circuit generalization; negative corrections retained'))
            if context_index==0 and target=='T0_l0_d128':
                first=context['score'][0][1];time=np.asarray(first['time_ms']);eeg=np.asarray(first['eeg_v']).reshape(-1)
                past=eeg[(time>cutoff-128)&(time<=cutoff)]
                if len(past)!=2048:raise ValueError('Same-state past plot requires exactly 128ms native support')
                figures['same_state_futures']=dict(context_id=cid,structure_id=sid,units='V',futures=score.tolist(),times_ms=np.arange(2,130,2).tolist(),
                    past=past.reshape(64,32).mean(1).tolist(),past_times_ms=np.arange(-126,2,2).tolist(),
                    scoring_future_ids=score_ids.tolist(),source=source,
                    selection='first ID-registered H2 context, first six paths displayed; ensemble mean/range uses all 32; no outcome selection')
        summaries.append(dict(context_id=cid,structure_id=sid,scoring_futures=32,reference_futures=32,target_families=len(targets)))
    if not summaries:raise ValueError('No completed H2 native contexts')
    result=dict(status='complete',h2_precision=rows,contexts=summaries,figures=figures,artifacts=artifacts,
        interpretation='L32 is finite and uncertain; repeated groups are dependent; M1 variance unavailable; large confirmation deferred')
    atomic_json(root/'precision/h2_summary.json',result);return result


def _replay_physical_targets(trials,target):
    """Reproduce frozen target arithmetic without relaxing exact comparisons.

    T0/T3 were derived one native trial at a time. T4 was derived from a
    (contexts, futures, 54) coefficient tensor. NumPy's masked band selection
    changes strides: the multi-record reduction sums along a strided axis,
    whereas the single-vector reduction can use pairwise summation. The latter
    differs by a few float64 ULPs. Keep the original multi-record arithmetic;
    the two registered replay contexts are actual records, not new replicates.
    This changes neither the target operator nor any saved target or learner.
    """
    family,h,_=parse_target_id(target)
    base=f'T3_l{h:g}_d1000' if family.startswith('T4_') else target
    values=np.stack([extract_physical_target(t['eeg_v'],t['time_ms'],base,
        float(t['metadata']['fork']['cutoff_ms'])) for t in trials])
    if not family.startswith('T4_'):return values
    if len(values)<2:raise ValueError('Replay energy arithmetic requires the two registered contexts')
    a=axes(target_spec(base))
    return energy_from_coefficients(values[:,None,:],a['target_frequency_hz'],a['target_component'],
        family.split('energy_',1)[1],log=family.startswith('T4_log_'))[:,0]


def _verify_replay_target(root,job,cid,tid,frozen,observed):
    """Keep exact equality and retain actionable evidence for any real mismatch."""
    if np.array_equal(frozen,observed):return
    same_shape=frozen.shape==observed.shape
    delta=np.abs(frozen-observed) if same_shape else np.array([])
    finite=delta[np.isfinite(delta)]
    details=dict(status='failed',job=job,context_id=cid,future_id=tid,
        frozen_shape=list(frozen.shape),native_shape=list(observed.shape),
        frozen_dtype=str(frozen.dtype),native_dtype=str(observed.dtype),
        different_coordinates=int(np.count_nonzero(frozen!=observed)) if same_shape else None,
        max_absolute_difference=float(finite.max()) if finite.size else None,
        target_metadata=target_metadata(job.split('__')[1]),
        comparison='Exact array equality; no tolerance or replacement of frozen targets')
    directory=Path(root)/'precision/replay_target_failures'/job/tid
    atomic_npz(directory/'comparison.npz',frozen_physical=frozen,native_physical=observed)
    details['arrays']=_record(Path(root),directory/'comparison.npz')
    atomic_json(directory/'comparison.json',details)
    raise ValueError(f'Native replay target differs from frozen matched target: job={job}, '
        f'context={cid}, future={tid}, max_absolute_difference={details["max_absolute_difference"]}; '
        f'diagnostic={directory / "comparison.json"}')


def replay_tables(config,root,jobs):
    root=Path(root);controls=_read(root/'replay_controls.json');plan=_read(root/'design.json')
    raw={r['trial_id']:r for r in plan['historical_rows']};audits=controls['audits']
    if len(audits)!=2:raise ValueError('Expected exactly two registered replay contexts')
    originals=[];replicas=[];cids=[];tids=[];sids=[];provenance=[]
    for audit in audits:
        row=raw[audit['original_trial_id']];path=generation._native_path(row)
        replica=Path(audit['replica_path'])
        if file_hash(replica)!=audit['replay_native_sha256']:raise ValueError('Replay artifact hash changed')
        originals.append(_read_fork(path));replicas.append(_read_fork(replica));cids.append(audit['context_id']);tids.append(audit['original_trial_id']);sids.append(int(row['structure_id']))
        provenance.append(dict(context_id=audit['context_id'],original_trial_id=row['trial_id'],original_native_sha256=row['sha256'],replay_native_sha256=file_hash(replica),method='full-prefix exact-forcing replay'))
    rows=[];artifacts=[];population='Privileged replay population: two circuits / two contexts / one original scoring future each'
    physical={}
    for job in jobs:
        target=job.split('__')[1];family,h,w=parse_target_id(target)
        panel,manifest=load_panel(root/'matched',job);norm=manifest['normalization'];scale=float(norm['y_scale']);offset=np.asarray(norm['y_offset'])
        if target not in physical:
            physical[target]=tuple(_replay_physical_targets(pool,target) for pool in (originals,replicas))
        y,replayed=physical[target]
        indices=[];evalp=panel['evaluation']
        for cid,tid,observed in zip(cids,tids,y):
            where=np.flatnonzero(evalp['context_id']==cid)
            if len(where)!=1:raise ValueError('Replay context absent/duplicated in matched evaluation panel')
            ix=int(where[0]);future=np.flatnonzero(evalp['future_id'][ix]==tid)
            if len(future)!=1 or not evalp['future_mask'][ix,int(future[0])]:raise ValueError('Replay future absent from matched target pool')
            _verify_replay_target(root,job,cid,tid,evalp['Y_physical'][ix,int(future[0])],observed)
            indices.append(ix)
        baseline_std=vm.context_weights(panel['train'])@vm.first_targets(panel['train'])
        baseline=baseline_std*scale+offset
        base_losses=np.mean((y-baseline)**2,axis=1);reference=float(base_losses.mean())
        if not np.isfinite(reference) or reference<=0:raise ValueError('Degenerate replay-population baseline')
        methods={'measured_replay':replayed,'training_mean':np.broadcast_to(baseline,y.shape).copy()};inputs={}
        for predpath in sorted((root/'matched/learners'/job).glob('*/predictions.npz')):
            selection=_read(predpath.parent/'selection.json')
            if file_hash(predpath)!=selection['files']['predictions.npz']:raise ValueError('Replay comparator prediction checksum changed')
            pred=numeric(predpath)
            if not np.array_equal(pred['evaluation_context_id'],evalp['context_id']):raise ValueError('Replay comparator prediction IDs differ')
            if 'risk_contract_json' in pred and json.loads(str(pred['risk_contract_json']))!=manifest['risk_contract']:
                raise ValueError('Replay comparator risk contract mismatch')
            keys=[('complete_state_selected','evaluation_prediction'),('complete_state_candidate','evaluation_candidate_prediction')] if predpath.parent.name=='complete_state' else [(predpath.parent.name,'evaluation_prediction')]
            for method,key in keys:
                methods[method]=np.asarray(pred[key])[indices]*scale+offset;inputs[method]=_record(root,predpath)
        if 'complete_state_candidate' not in methods:raise ValueError('Replay comparison lacks fitted complete-state candidate')
        directory=root/'precision/replay'/job
        atomic_json(directory/'target_consistency.json',dict(status='verified',job=job,
            context_ids=cids,future_ids=tids,comparison='Exact array equality for every frozen physical target',
            native_target_arithmetic='Original per-trial T0/T3; original multi-record T4 reduction',
            frozen_targets_changed=False,implementation_sha256=file_hash(Path(__file__))))
        consistency=_record(root,directory/'target_consistency.json');artifacts.append(consistency)
        arrays=dict(Y_physical=y,replayed_physical=replayed,baseline_physical=np.broadcast_to(baseline,y.shape),context_id=np.asarray(cids),structure_id=np.asarray(sids),future_id=np.asarray(tids),
            reference_loss_per_context=base_losses)
        for method,prediction in methods.items():arrays[method+'__prediction_physical']=prediction;arrays[method+'__loss_per_context']=np.mean((y-prediction)**2,axis=1)
        atomic_npz(directory/'predictions_and_losses.npz',**arrays);artifact=_record(root,directory/'predictions_and_losses.npz');artifacts.append(artifact)
        common=dict(job=job,acquisition=job.split('__')[0],target=target,h_ms=h,duration_ms=w,population=population,
            circuits=len(set(sids)),contexts=len(cids),futures=1,context_ids=cids,future_ids=tids,physical_units=_risk_units(target),
            reference_risk_physical=reference,target_fingerprint=manifest['target_fingerprint'],normalization_id=norm['id'],
            uncertainty=dict(status='deferred',reason='Only two contexts on two circuits and one realised future each; no empirical future-variance estimate or meaningful population precision'),
            source=[artifact,consistency],replay_provenance=provenance)
        for method,prediction in methods.items():
            losses=np.mean((y-prediction)**2,axis=1);risk=float(losses.mean());evidence='privileged_replay' if method=='measured_replay' else 'fitted'
            rows.append(dict(common,method=method,information='complete present state plus actual realised future forcing' if method=='measured_replay' else method,
                evidence_type=evidence,status='measured',risk_physical=risk,error_baseline100=100*risk/reference,
                relative_mse_reduction=100*(1-risk/reference),per_context_risk_physical=losses.tolist(),
                source=common['source']+([inputs[method]] if method in inputs else []),
                state_variance_physical=None,combined_unresolved_excess=None,
                interpretation='Full-prefix privileged replay is compared with every method on exactly the same registered realised futures; no theoretical value is substituted'))
        rows.extend([dict(common,method='theoretical_future_input',information='complete state plus exact future input',evidence_type='theoretical',status='theoretical',risk_physical=0.,error_baseline100=0.,interpretation='Deterministic model limit, separate from observed replay error'),
            dict(common,method='state_variance',information='complete state with unknown future inputs',evidence_type='estimated reference',status='unavailable',risk_physical=None,error_baseline100=None,interpretation='M1 replay population cannot estimate within-context sample variance; do not borrow a floor from another future pool'),
            dict(common,method='ideal_eeg',information='EEG history only',evidence_type='unknown',status='unavailable',risk_physical=None,error_baseline100=None,interpretation='Ideal EEG prediction risk remains unknown')])
    result=dict(status='complete',replay_information=rows,artifacts=artifacts,population=population,
        observed_replay_never_replaced_with_theory=True,no_variance_estimate_from_one_future=True)
    atomic_json(root/'precision/replay_summary.json',result);return result


def run(config,root):
    root=Path(root);from .config import jobs
    selected=jobs();h2=finite_future_tables(config,root,selected);replay=replay_tables(config,root,selected)
    result=dict(status='complete',h2_precision=h2['h2_precision'],replay_information=replay['replay_information'],figures=h2['figures'],
        artifacts=h2['artifacts']+replay['artifacts'],h2_contexts=h2['contexts'],source_fingerprint=canonical_hash(dict(
            h2=file_hash(root/'h2/native_manifest.json'),replay=file_hash(root/'replay_controls.json'),implementation=file_hash(Path(__file__)))))
    atomic_json(root/'precision/summary.json',result);return result
