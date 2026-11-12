"""Finite-future estimation and legitimate pooled one-future prediction.

The inherited unbiased variance/pair/leave-one-out estimators are reused.
A state-conditioned variance is NEVER used as EEG-conditioned ground truth.
Independent reference panels remain finite, uncertain estimates.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import t as student_t, chi2

from ..analytical import stream
from ..statistics import per_state_fork_variance, independent_pair_variance, leave_one_out_oracle
from ..storage import atomic_json, file_hash, jsonable


def _targets(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim==1: x=x[:,None]
    if x.ndim<2 or not len(x) or not np.isfinite(x).all():
        raise ValueError('need finite independent future rows')
    return x.reshape(len(x),-1)


def finite_m_estimates(futures, prediction=None):
    """One fixed context; f must be frozen independently of these outcomes.

    Negative bias-corrected excess estimates are kept. M=1 cannot directly
    estimate variance or bias-corrected excess, but has a legitimate mean/MSE.
    """
    x=_targets(futures); m=len(x); mean=x.mean(axis=0)
    result=dict(m=m,target_dimension=x.shape[1],mean=mean,
        variance=None,independent_pair_half_mse=None,loo_mse=None,loo_corrected_mse=None,
        variance_status='unavailable: one future' if m==1 else 'estimated',
        conditioning='one fixed complete causal state')
    if m>1:
        variance=float(per_state_fork_variance(x[None])[0])
        loo=leave_one_out_oracle(x[None])[0]
        result.update(variance=variance,
            independent_pair_half_mse=float(independent_pair_variance(x[None]).mean()),
            loo_mse=float(np.mean((x-loo)**2)),
            loo_corrected_mse=float(np.mean((x-loo)**2)*(m-1)/m))
    if prediction is not None:
        pred=np.asarray(prediction,dtype=float).reshape(-1)
        if pred.shape!=mean.shape or not np.isfinite(pred).all():
            raise ValueError('prediction must match the target dimension')
        squared=float(np.mean((pred-mean)**2))
        result.update(model_risk=float(np.mean((x-pred)**2)),squared_distance_to_sample_mean=squared,
            bias_corrected_excess=None if m==1 else squared-result['variance']/m,
            model_independence_requirement='prediction frozen before inspecting any reference/scoring outcomes')
    return result


def check_pool_independence(scoring_ids, reference_ids):
    a=[str(x) for x in scoring_ids]; b=[str(x) for x in reference_ids]
    if any(not x for x in a+b) or len(set(a))!=len(a) or len(set(b))!=len(b):
        raise ValueError('future IDs must be nonempty and unique inside each pool')
    if set(a)&set(b):
        raise ValueError('reference and scoring futures overlap')


def replication_curve(scoring, reference=None, *, predictions=None, m_values=(1,2,4,8,16,32,64,128),
                      repetitions=32, bootstrap_samples=64, seed=0,
                      scoring_ids=None, reference_ids=None, truth_mean=None,
                      truth_variance=None, gaussian_scalar=False):
    """Nested draws from a scoring pool, with a disjoint finite reference.

    Repetitions reuse the finite pool; their spread is conditional subsampling
    variability, not extra physical circuits nor independent Monte Carlo runs.
    Exact coverage is named only when an exact truth argument is supplied.
    """
    x=_targets(scoring)
    ref=None if reference is None or not np.asarray(reference).size else _targets(reference)
    if ref is not None and ref.shape[1]!=x.shape[1]: raise ValueError('target dimensions differ')
    if (scoring_ids is None)!=(reference_ids is None): raise ValueError('supply both ID pools')
    if scoring_ids is not None:
        if len(scoring_ids)!=len(x) or len(reference_ids)!=(0 if ref is None else len(ref)):
            raise ValueError('future ID counts do not match panels')
        check_pool_independence(scoring_ids, reference_ids)
    if repetitions<2 or bootstrap_samples<2: raise ValueError('need at least two repetitions/bootstrap samples')
    requested=[int(m) for m in m_values]
    if any(m<1 for m in requested) or len(set(requested))!=len(requested):
        raise ValueError('M values must be unique positive integers')
    pred={} if predictions is None else {k:np.asarray(v,dtype=float).reshape(-1) for k,v in predictions.items()}
    for p in pred.values():
        if p.shape!=(x.shape[1],) or not np.isfinite(p).all(): raise ValueError('prediction dimensions differ')
    reference_metrics=dict(status='pending: independent reference panel absent or smaller than two')
    refmean=refvar=None
    if ref is not None and len(ref)>1:
        refmean=ref.mean(axis=0); refvar=float(per_state_fork_variance(ref[None])[0])
        rng=stream(seed,'h1/reference-bootstrap')
        boot=np.array([np.var(ref[rng.integers(0,len(ref),len(ref))],axis=0,ddof=1).mean()
                       for _ in range(bootstrap_samples)])
        reference_metrics=dict(status='finite independent reference; not exact truth',n_futures=len(ref),
            mean=refmean,variance=refvar,mean_mc_standard_error=np.sqrt(ref.var(axis=0,ddof=1)/len(ref)),
            variance_bootstrap_standard_error=float(boot.std(ddof=1)),
            variance_bootstrap_interval95=np.quantile(boot,[.025,.975]),
            interval_scope='future Monte Carlo conditional on this one state; percentile bootstrap approximate')
    rng=stream(seed,'h1/nested-score-subsamples')
    permutations=np.stack([rng.permutation(len(x)) for _ in range(repetitions)])
    rows=[]; numeric={}
    for m in sorted(requested):
        if m>len(x):
            rows.append(dict(m=m,status='pending: insufficient independent scoring futures',available=len(x)))
            continue
        samples=x[permutations[:,:m]]; means=samples.mean(axis=1)
        variances=None if m==1 else samples.var(axis=1,ddof=1).mean(axis=1)
        row=dict(m=m,status='computed',n_subsamples=repetitions,
            variance_status='unavailable: one future' if m==1 else 'estimated',
            variance_mean=None if variances is None else float(variances.mean()),
            variance_subsample_sd=None if variances is None else float(variances.std(ddof=1)),
            reference_error_status=reference_metrics['status'],models={})
        numeric[f'm{m}_mean']=means
        if variances is not None: numeric[f'm{m}_variance']=variances
        if refmean is not None:
            squared=np.mean((means-refmean)**2,axis=1)
            row.update(mean_rmse_against_finite_reference=float(np.sqrt(squared.mean())),
                mean_mse_reference_noise_corrected=float(squared.mean()-refvar/len(ref)),
                mean_bias_against_finite_reference=(means-refmean).mean(axis=0),
                variance_rmse_against_finite_reference=None if variances is None else float(np.sqrt(np.mean((variances-refvar)**2))),
                variance_bias_against_finite_reference=None if variances is None else float(variances.mean()-refvar),
                reference_error_caveat='Reference variance/mean are uncertain; corrected mean MSE may be negative and is retained.')
        if m>1:
            component_var=samples.var(axis=1,ddof=1)
            halfwidth=student_t.ppf(.975,m-1)*np.sqrt(component_var/m)
            if refmean is not None:
                row['mean_interval_reference_containment']=float(np.mean(abs(means-refmean)<=halfwidth))
                row['containment_scope']='contains estimated reference mean; not coverage of unknown exact mean'
            if truth_mean is not None:
                truth=np.asarray(truth_mean,dtype=float).reshape(-1)
                if truth.shape!=(x.shape[1],): raise ValueError('exact mean dimensions differ')
                row['mean_interval_exact_truth_coverage']=float(np.mean(abs(means-truth)<=halfwidth))
            if gaussian_scalar and truth_variance is not None:
                if x.shape[1]!=1: raise ValueError('chi-square interval is scalar Gaussian only')
                low=(m-1)*variances/chi2.ppf(.975,m-1)
                high=(m-1)*variances/chi2.ppf(.025,m-1)
                row['variance_interval_exact_truth_coverage']=float(np.mean((low<=truth_variance)&(truth_variance<=high)))
                row['variance_interval_assumption']='exact IID scalar Gaussian; not asserted for EEG'
        for name,p in pred.items():
            risk=np.mean((samples-p)**2,axis=(1,2))
            squared=np.mean((means-p)**2,axis=1)
            excess=None if variances is None else squared-variances/m
            row['models'][name]=dict(mean_model_risk=float(risk.mean()),
                bias_corrected_excess=None if excess is None else float(excess.mean()),
                excess_subsample_sd=None if excess is None else float(excess.std(ddof=1)),
                negative_excess_fraction=None if excess is None else float(np.mean(excess<0)))
            numeric[f'm{m}_{name}_model_risk']=risk
            if excess is not None: numeric[f'm{m}_{name}_excess']=excess
        rows.append(row)
    return dict(rows=rows,reference=reference_metrics,arrays=numeric,
        repetition_scope='Nested samples share a fixed scoring pool; variability describes subsampling, not independent circuit uncertainty',
        independence_status='validated disjoint future IDs' if scoring_ids is not None else 'caller must guarantee reference/scoring independence')


def fixed_budget_plan(*, available_contexts=(), circuit_ids=(0,1,2,3),
                      regimes=('R0','Ralpha','Rbeta_low','Rbeta_high'), budget=256,
                      m_values=(1,2,4,8,16), prefixes_per_stratum=16):
    """Plan a balanced, nested candidate pool; never invent measured examples.

    Each row in available_contexts names structure_id, regime, and optionally
    candidate_index (within stratum), n_futures. Only qualifying candidate
    indices contribute to available union counts. Other panels remain separate.
    """
    circuits=list(circuit_ids); regimes=list(regimes); strata=len(circuits)*len(regimes)
    if not strata or len(set(circuits))!=len(circuits) or len(set(regimes))!=len(regimes):
        raise ValueError('unique nonempty circuits and regimes required')
    if budget<1 or prefixes_per_stratum<1: raise ValueError('positive budget and candidate counts required')
    ms=sorted(set(int(m) for m in m_values))
    if len(ms)!=len(m_values) or any(m<1 or budget%m or (budget//m)%strata for m in ms):
        raise ValueError('every fixed-budget allocation must be balanced across all circuit/regime strata')
    max_k=budget//min(ms)
    if max_k>strata*prefixes_per_stratum: raise ValueError('candidate design has insufficient prefixes')
    required=[0]*prefixes_per_stratum
    allocations=[]
    for m in ms:
        k=budget//m; per=k//strata
        for j in range(per): required[j]=max(required[j],m)
        allocations.append(dict(k=k,m=m,outcomes=k*m,contexts_per_stratum=per))
    counts={(str(c),str(r)):{} for c in circuits for r in regimes}
    nonqualifying=0
    for row in available_contexts:
        key=(str(row['structure_id']),str(row['regime']))
        index=row.get('candidate_index')
        if key not in counts or index is None or int(index)<0 or int(index)>=prefixes_per_stratum:
            nonqualifying+=1; continue
        index=int(index)
        if index in counts[key]: raise ValueError('duplicate candidate context in fixed-budget pool')
        counts[key][index]=int(row.get('n_futures',0))
    available=0; existing_prefixes=0
    for per in counts.values():
        for index,n in per.items():
            if required[index]:
                existing_prefixes+=1; available+=min(max(n,0),required[index])
    union=strata*sum(required)
    return dict(status='planned allocation; not executed evidence',budget_per_allocation=budget,
        allocations=allocations,required_candidate_contexts=max_k,
        independent_circuits=len(circuits),regimes=len(regimes),prefixes_per_stratum=prefixes_per_stratum,
        required_future_counts_by_nested_prefix=required,
        unique_future_union=union,available_qualifying_contexts=existing_prefixes,
        additional_contexts_required=max_k-existing_prefixes,
        available_qualifying_union_futures=available,additional_union_futures_required=union-available,
        nonqualifying_existing_contexts=nonqualifying,
        reference_and_evaluation_futures_included=False,
        cost_status='requires separately measured prefix replay/continuation cost; independent reference/evaluation panel is additional')


def eeg_features(history):
    """Fixed low-dimensional EEG-only summaries; no state or future labels."""
    x=np.asarray(history,dtype=float)
    if x.ndim!=2 or x.shape[1]<2 or not np.isfinite(x).all(): raise ValueError('need finite EEG histories')
    parts=np.array_split(x,min(8,x.shape[1]),axis=1)
    return np.column_stack([*[p.mean(axis=1) for p in parts],x[:,-1],x.std(axis=1),
                            np.mean(x[:,1:]*x[:,:-1],axis=1)])


def fit_pooled_one_future(history, targets, structure_ids, *, ridge=1.0, context_ids=None):
    """Cross-fitted residual Gaussian from MANY independent one-future contexts.

    Mean residuals are out-of-circuit within training. Variance parameters use
    those residuals, never evaluation states/futures. Its variance is an EEG-
    conditional predictive approximation, NOT an estimate of state variance.
    """
    x=eeg_features(history); y=_targets(targets); groups=np.asarray(structure_ids)
    unique=np.unique(groups)
    if len(x)!=len(y) or groups.shape!=(len(y),) or len(unique)<3:
        raise ValueError('one-future learner needs matching rows and at least three TRAINING circuits')
    if ridge<=0: raise ValueError('ridge must be positive')
    context_ids=np.arange(len(y)).astype(str) if context_ids is None else np.asarray(context_ids).astype(str)
    if context_ids.shape!=(len(y),): raise ValueError('training context ID count differs')
    for cid in np.unique(context_ids):
        indices=np.flatnonzero(context_ids==cid)
        if len(np.unique(groups[indices]))!=1 or not np.all(x[indices]==x[indices[0]]):
            raise ValueError('Repeated training context has inconsistent circuit/history')
    center=x.mean(axis=0); scale=x.std(axis=0); scale[scale<1e-12]=1
    design=np.column_stack([np.ones(len(x)),(x-center)/scale])
    penalty=np.eye(design.shape[1])*ridge; penalty[0,0]=0
    def fit(a,b): return np.linalg.solve(a.T@a+penalty,a.T@b)
    residual=np.empty_like(y);fold_support=[]
    for group in unique:
        held=groups==group
        distinct=len(np.unique(context_ids[~held]))
        if distinct<2: raise ValueError('insufficient independent training contexts per cross-fit: at least two required')
        fold_support.append(dict(held_out_circuit=str(group),independent_contexts=distinct,
            outcome_rows=int((~held).sum()),design_columns=design.shape[1],
            regularized_underdetermined=distinct<design.shape[1]))
        # Positive ridge makes n<p estimable. We disclose that design instead
        # of pretending repeated futures add independent context coverage.
        residual[held]=y[held]-design[held]@fit(design[~held],y[~held])
    weights=fit(design,y)
    # Variance is shared across target coordinates after averaging squared
    # residuals; report this declared diagonal/isotropic model limitation.
    squared=np.mean(residual**2,axis=1)
    floor=max(float(squared.mean())*1e-8,1e-12*max(float(np.mean(y*y)),1.0))
    initial=np.zeros(design.shape[1]); initial[0]=np.log(max(squared.mean(),floor))
    def variance_for(beta):
        eta=design@beta
        exponential=np.exp(np.clip(eta,-40.,40.))
        return eta,exponential,floor+exponential
    def objective(beta):
        _,_,variance=variance_for(beta)
        return float(.5*np.mean(np.log(variance)+squared/variance)+.5*ridge/len(y)*np.sum(beta[1:]**2))
    def gradient(beta):
        eta,exponential,variance=variance_for(beta)
        derivative=exponential/variance*((eta>-40.)&(eta<40.))
        result=.5*design.T@((1-squared/variance)*derivative)/len(y)
        result[1:]+=ridge/len(y)*beta[1:]
        return result
    opt=minimize(objective,initial,jac=gradient,method='BFGS',options={'maxiter':300,'gtol':1e-7})
    if not np.isfinite(opt.fun) or not np.isfinite(opt.x).all(): raise ValueError('pooled variance optimization failed')
    return dict(feature_center=center,feature_scale=scale,mean_weights=weights,
        log_variance_weights=opt.x,variance_floor=floor,log_variance_clip=np.asarray([-40.,40.]),training_structures=unique,
        variance_fit_success=bool(opt.success),variance_fit_message=str(opt.message),
        context_qualification={'independent_training_contexts':len(np.unique(context_ids)),'folds':fold_support,
            'interpretation':'positive-ridge fit allowed for n<p; repeated outcomes do not add context coverage; fitting adequacy must be judged on held-out circuits'},
        fitting_scope='training circuits only; leave-one-training-circuit-out residuals',
        variance_conditioning='visible EEG history; includes hidden-state and residual fitting ambiguity',
        covariance_model='isotropic diagonal across target coordinates, heteroscedastic across EEG contexts')


def predict_pooled_one_future(model, history, *, structure_ids=None):
    if structure_ids is not None and set(np.asarray(structure_ids).tolist())&set(model['training_structures'].tolist()):
        raise ValueError('pooled evaluation circuits overlap training circuits')
    x=(eeg_features(history)-model['feature_center'])/model['feature_scale']
    design=np.column_stack([np.ones(len(x)),x]); mean=design@model['mean_weights']
    logvar=design@model['log_variance_weights']
    if not np.isfinite(logvar).all(): raise ValueError('nonfinite predicted log variance')
    variance=model['variance_floor']+np.exp(np.clip(logvar,*model['log_variance_clip']))
    return mean,np.broadcast_to(variance[:,None],mean.shape).copy()


def gaussian_calibration(targets, means, variances):
    y,mu,var=np.broadcast_arrays(np.asarray(targets,dtype=float),np.asarray(means,dtype=float),np.asarray(variances,dtype=float))
    if not np.isfinite(y).all() or not np.isfinite(mu).all() or not np.isfinite(var).all() or np.any(var<=0):
        raise ValueError('finite targets/means and positive variances required')
    z=(y-mu)/np.sqrt(var); phi=np.exp(-z*z/2)/np.sqrt(2*np.pi); cdf=ndtr(z)
    return dict(mse=float(np.mean((y-mu)**2)),nll_per_dimension=float(np.mean(.5*(np.log(2*np.pi*var)+z*z))),
        marginal_crps=float(np.mean(np.sqrt(var)*(z*(2*cdf-1)+2*phi-1/np.sqrt(np.pi)))),
        marginal_coverage95=float(np.mean(abs(z)<=1.959963984540054)),
        mean_marginal_sd=float(np.mean(np.sqrt(var))),pit_mean=float(cdf.mean()),
        pit_histogram=np.histogram(cdf,bins=np.linspace(0,1,11))[0].tolist(),
        uncertainty_scope='scores across correlated coordinates/futures; independent unit remains circuit')


def _load_predictions(root,job_id,panel):
    path=Path(root)/'benchmarks'/job_id/'predictions.npz'
    if not path.exists(): return {},None
    with np.load(path,allow_pickle=False) as archive:
        a={key:archive[key] for key in archive.files}
    manifest=panel['manifest']; ev=panel['evaluation']
    for key in ('target_fingerprint','comparison_fingerprint'):
        if str(a[key].item())!=manifest[key]: raise ValueError('benchmark/replication fingerprint mismatch')
    if not np.array_equal(a['context_id'],ev['context_id']): raise ValueError('prediction contexts differ')
    if not np.array_equal(a['structure_id'],ev['structure_id']): raise ValueError('prediction circuit order differs')
    if a['mean'].shape!=(len(a['model_names']),len(ev['context_id']),ev['Y'].shape[-1]):
        raise ValueError('benchmark prediction dimensions differ')
    return {str(name):a['mean'][i] for i,name in enumerate(a['model_names'])},path


def _equal_circuit(rows, key):
    grouped={}
    for row in rows:
        value=row.get(key)
        if value is not None and np.isfinite(value): grouped.setdefault(str(row['structure_id']),[]).append(float(value))
    means={k:float(np.mean(v)) for k,v in grouped.items()}
    return dict(mean=None if not means else float(np.mean(list(means.values()))),
                circuit_values=means,n_circuits=len(means),
                population_interval=None,interval_status='not estimated from nested future subsamples')


def run_replication(config, output_root, resume=False):
    """Analyze each actually generated TargetSpec panel, preserving absent data.

    Saves finite-M estimates, raw repeated estimates, frozen predictions and
    EEG-only pooled model scores. No reference panel is synthesized or borrowed
    from a different context, split, target, or normalization.
    """
    from .dataset import list_jobs,load_panel
    from .artifacts import (atomic_npz,require_compatible,resume_stage,stage_complete,
                            artifact_record,verify_record)
    root=Path(output_root); require_compatible(root,config)
    cached=resume_stage(root,'replication',resume)
    if cached is not None:
        summary=json.loads((root/'replication'/'summary.json').read_text())
        for record in summary['input_artifacts']: verify_record(root,record)
        return summary
    options=config.get('h1',config).get('replication',{})
    outputs=[]; jobs=[]; input_artifacts=[]
    for job_id in list_jobs(root):
        panel=load_panel(root,job_id,verify=True); manifest=panel['manifest']; ev=panel['evaluation']
        directory=root/'replication'/job_id; directory.mkdir(parents=True,exist_ok=True)
        predictions,prediction_path=_load_predictions(root,job_id,panel)
        if prediction_path is not None: input_artifacts.append(artifact_record(root,prediction_path))
        for record in manifest['files'].values():
            input_artifacts.append(artifact_record(root,root/record['path']))
        train=panel['train']; pooled=dict(status='disabled')
        if options.get('pooled_one_future',True):
            if len(np.unique(train['structure_id']))<3:
                pooled=dict(status='pending: at least three independent training circuits required')
            elif any(len(np.unique(train['context_id'][train['structure_id']!=sid]))<2 for sid in np.unique(train['structure_id'])):
                pooled=dict(status='pending: at least two distinct training contexts per residual cross-fit required')
            elif not np.all(np.sum(train['future_mask'],axis=1)>=1):
                pooled=dict(status='pending: training outcomes missing')
            else:
                first=np.argmax(train['future_mask'],axis=1)
                model=fit_pooled_one_future(train['X'],train['Y'][np.arange(len(first)),first],
                    train['structure_id'],ridge=float(options.get('ridge',1.)),context_ids=train['context_id'])
                means,variances=predict_pooled_one_future(model,ev['X'],structure_ids=ev['structure_id'])
                predictions['pooled_one_future_eeg']=means
                context_scores=[]
                nll=np.zeros_like(ev['Y']); pit=np.zeros_like(ev['Y']); squared=np.zeros_like(ev['Y'])
                for i,cid in enumerate(ev['context_id']):
                    mask=ev['future_mask'][i]; y=ev['Y'][i,mask]
                    scored=gaussian_calibration(y,means[i],variances[i])
                    context_scores.append(dict(context_id=str(cid),structure_id=int(ev['structure_id'][i]),**scored))
                    z=(y-means[i])/np.sqrt(variances[i])
                    nll[i,mask]=.5*(np.log(2*np.pi*variances[i])+z*z)
                    pit[i,mask]=ndtr(z); squared[i,mask]=(y-means[i])**2
                pooled=dict(status='executed',conditioning=model['variance_conditioning'],
                    variance_fit_success=model['variance_fit_success'],variance_fit_message=model['variance_fit_message'],
                    context_qualification=model['context_qualification'],
                    covariance_model=model['covariance_model'],variance_numerics={'additive_floor':model['variance_floor'],'log_variance_clip':model['log_variance_clip'].tolist()},context_scores=context_scores,
                    scores={k:_equal_circuit(context_scores,k) for k in ('mse','nll_per_dimension','marginal_crps','marginal_coverage95','mean_marginal_sd','pit_mean')},
                    state_fork_variance_rmse=None,
                    state_fork_variance_comparison='not an EEG-conditional ground truth; deliberately not scored as variance RMSE')
                model_path=directory/'pooled_one_future_model.npz'
                atomic_npz(model_path,**{k:v for k,v in model.items() if isinstance(v,np.ndarray)},
                           variance_floor=np.asarray(model['variance_floor']))
                score_path=directory/'pooled_one_future_scores.npz'
                atomic_npz(score_path,context_id=ev['context_id'],structure_id=ev['structure_id'],
                    mean=means,variance=variances,future_mask=ev['future_mask'],future_id=ev['future_id'],
                    squared_error=squared,nll_per_dimension=nll,pit=pit,
                    target_fingerprint=np.asarray(manifest['target_fingerprint']),
                    comparison_fingerprint=np.asarray(manifest['comparison_fingerprint']))
                outputs.extend([model_path,score_path])
        contexts=[]; arrays={}; counts=np.sum(ev['future_mask'],axis=1)
        reference_counts=np.sum(ev['reference_mask'],axis=1)
        for i,cid in enumerate(ev['context_id']):
            smask=ev['future_mask'][i]; rmask=ev['reference_mask'][i]
            curve=replication_curve(ev['Y'][i,smask],ev['reference_Y'][i,rmask],
                predictions={k:v[i] for k,v in predictions.items()},
                m_values=options.get('m_values',[1,2,4,8,16,32,64,128]),
                repetitions=int(options.get('repetitions',32)),bootstrap_samples=int(options.get('bootstrap_samples',64)),
                seed=int(options.get('seed',0))+i,
                scoring_ids=ev['future_id'][i,smask],reference_ids=ev['reference_future_id'][i,rmask])
            for key,value in curve.pop('arrays').items(): arrays[f'context{i}_{key}']=value
            contexts.append(dict(context_id=str(cid),structure_id=int(ev['structure_id'][i]),
                                 score_futures=int(counts[i]),reference_futures=int(reference_counts[i]),**curve))
        fixed=[]
        requested=options.get('m_values',[1,2,4,8,16,32,64,128])
        # Each maximum support value defines a genuinely fixed context subset.
        # Expanded panels yield all-context curves through M=32 and high-M-only
        # curves through M=128, never a single curve with silently changing K.
        for cap in sorted(set(int(c) for c in counts)):
            levels=[int(m) for m in requested if m<=cap]
            if not levels: continue
            indices=np.flatnonzero(counts>=cap)
            fixed_rows=[]
            for m in levels:
                selected=[]
                for index in indices:
                    row=next(r for r in contexts[index]['rows'] if r['m']==m)
                    selected.append(dict(structure_id=contexts[index]['structure_id'],**row))
                metrics={key:_equal_circuit(selected,key) for key in (
                    'variance_mean','mean_rmse_against_finite_reference','mean_mse_reference_noise_corrected',
                    'variance_rmse_against_finite_reference','variance_bias_against_finite_reference',
                    'mean_interval_reference_containment')}
                model_scores={}
                for name in predictions:
                    flattened=[dict(structure_id=r['structure_id'],**r['models'][name]) for r in selected]
                    model_scores[name]={key:_equal_circuit(flattened,key) for key in ('mean_model_risk','bias_corrected_excess','negative_excess_fraction')}
                fixed_rows.append(dict(m=m,metrics=metrics,models=model_scores))
            fixed.append(dict(max_supported_m=cap,k=len(indices),
                context_ids=[str(ev['context_id'][i]) for i in indices],rows=fixed_rows,
                reference_status='available finite references' if np.all(reference_counts[indices]>=2)
                    else 'pending reference RMSE/bias/coverage: independent reference panel absent or insufficient'))
        candidate=[]
        for i,cid in enumerate(train['context_id']):
            # Dataset IDs encode a separately seeded physical prefix. Nothing
            # about overlapping windows creates a new candidate context here.
            try: index=int(str(cid).split('-p',1)[1].split('-',1)[0])
            except (IndexError,ValueError): index=None
            candidate.append(dict(structure_id=int(train['structure_id'][i]),regime=str(train['regime'][i]),
                                  candidate_index=index,n_futures=int(np.sum(train['future_mask'][i]))))
        circuits=config['data']['structures']['train']; regimes=[r['name'] for r in config['data']['regimes']]
        try:
            allocation=fixed_budget_plan(available_contexts=candidate,circuit_ids=circuits,regimes=regimes,
                budget=int(options.get('fixed_budget',256)),m_values=options.get('fixed_budget_m',[1,2,4,8,16]),
                prefixes_per_stratum=int(options.get('prefixes_per_stratum',16)))
        except ValueError as exc:
            allocation=dict(status='unsupported balanced allocation for this engineering configuration',reason=str(exc))
        try:
            allocation_execution=fixed_budget_comparison(train,ev,
                budget=int(options.get('fixed_budget',256)),m_values=options.get('fixed_budget_m',[1,2,4,8,16]),
                prefixes_per_stratum=int(options.get('prefixes_per_stratum',16)),ridge=float(options.get('ridge',1.)))
            allocation_arrays=allocation_execution.pop('arrays')
            if allocation_arrays:
                allocation_path=directory/'fixed_budget_scores.npz'
                atomic_npz(allocation_path,**allocation_arrays,context_id=ev['context_id'],
                    structure_id=ev['structure_id'],future_mask=ev['future_mask'],future_id=ev['future_id'])
                outputs.append(allocation_path)
        except ValueError as exc:
            allocation_execution=dict(status='unsupported or insufficient candidate design',reason=str(exc))
        result=dict(schema_version='eegrepr-h1-replication-2.0',job_id=job_id,
            target_fingerprint=manifest['target_fingerprint'],comparison_fingerprint=manifest['comparison_fingerprint'],
            contexts=contexts,fixed_k_panels=fixed,pooled_one_future=pooled,fixed_budget_plan=allocation,
            fixed_budget_comparison=allocation_execution,
            physical_mse_conversion=float(manifest['normalization']['y_scale'])**2,
            interpretation='Conditional future Monte Carlo precision is separate from circuit generalization; negative excess retained.')
        arrays_path=directory/'estimates.npz'; atomic_npz(arrays_path,**arrays)
        summary_path=directory/'summary.json'; atomic_json(summary_path,result)
        outputs.extend([arrays_path,summary_path]); jobs.append(result)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        plotted=[False]*4
        endpoints=[('variance_mean','Estimated single-future state variance'),
            ('mean_rmse_against_finite_reference','Mean RMSE versus finite reference'),
            ('variance_rmse_against_finite_reference','Variance RMSE versus finite reference'),
            ('mean_interval_reference_containment','Interval containment of reference mean')]
        for fixed_panel in fixed:
            for a,(endpoint,label) in enumerate(endpoints):
                valid=[r for r in fixed_panel['rows'] if r['metrics'][endpoint]['mean'] is not None]
                if valid:
                    axes.flat[a].plot([r['m'] for r in valid],[r['metrics'][endpoint]['mean'] for r in valid],
                        marker='o',label=f"K={fixed_panel['k']}, maximum M={fixed_panel['max_supported_m']}")
                    plotted[a]=True
        for a,(_,label) in enumerate(endpoints):
            axes.flat[a].set(xscale='log',xlabel='Independent futures M',ylabel=label)
            if plotted[a]: axes.flat[a].legend(fontsize=8)
            else: axes.flat[a].text(.5,.5,'Pending independent reference panel',ha='center',va='center',transform=axes.flat[a].transAxes)
        fig.suptitle('Nested finite-pool subsampling; finite references are uncertain\n'
                     'M=1 direct variance unavailable; no circuit-population confidence claim')
        for suffix in ('png','pdf'):
            path=directory/f'replication.{suffix}';fig.savefig(path,dpi=180);outputs.append(path)
        plt.close(fig)
    summary=dict(schema_version='eegrepr-h1-replication-2.0',status='executed available analyses; inspect pending references/allocations',
        jobs=jobs,input_artifacts=input_artifacts,
        fixed_budget_execution='Only supported independent candidate pools can be compared; plans do not manufacture contexts.')
    path=root/'replication'/'summary.json';atomic_json(path,summary);outputs.append(path)
    stage_complete(root,'replication',outputs,dict(jobs=len(jobs)))
    return jsonable(summary)


def fixed_budget_comparison(training, evaluation, *, budget=256, m_values=(1,2,4,8,16),
                            prefixes_per_stratum=16, ridge=1.0):
    """Fit equal-outcome-budget learners only where actual candidate data exist.

    The same EEG feature/covariance model sees K distinct contexts and M futures
    per context. Outcomes within a context are repetitions, not new context or
    circuit coverage. All allocations score the SAME disjoint evaluation panel.
    """
    groups=np.asarray(training['structure_id']); regimes=np.asarray(training['regime'])
    if set(groups.tolist())&set(np.asarray(evaluation['structure_id']).tolist()):
        raise ValueError('fixed-budget training and evaluation circuits overlap')
    circuits=sorted(np.unique(groups).tolist()); names=sorted(np.unique(regimes).tolist())
    lookup={}; available=[]
    for i,cid in enumerate(training['context_id']):
        try: index=int(str(cid).split('-p',1)[1].split('-',1)[0])
        except (IndexError,ValueError): continue
        key=(int(groups[i]),str(regimes[i]),index)
        if key in lookup: raise ValueError('duplicate candidate prefix')
        lookup[key]=i
        available.append(dict(structure_id=key[0],regime=key[1],candidate_index=index,
                              n_futures=int(training['future_mask'][i].sum())))
    plan=fixed_budget_plan(available_contexts=available,circuit_ids=circuits,regimes=names,
        budget=budget,m_values=m_values,prefixes_per_stratum=prefixes_per_stratum)
    rows=[]; arrays={}
    for allocation in plan['allocations']:
        m=allocation['m']; wanted=[(c,r,j) for c in circuits for r in names
                                  for j in range(allocation['contexts_per_stratum'])]
        missing_contexts=[key for key in wanted if key not in lookup]
        missing_futures=sum(max(0,m-int(training['future_mask'][lookup[key]].sum()))
                            if key in lookup else m for key in wanted)
        if missing_contexts or missing_futures:
            rows.append(dict(**allocation,status='pending: candidate allocation incomplete',
                             additional_contexts=len(missing_contexts),additional_futures=missing_futures))
            continue
        indices=np.array([lookup[key] for key in wanted])
        x=[]; y=[]; gs=[]; selected_ids=[]; selected_futures=[]
        for i in indices:
            chosen=np.flatnonzero(training['future_mask'][i])[:m]
            x.extend([training['X'][i]]*m); y.extend(training['Y'][i,chosen]); gs.extend([groups[i]]*m)
            selected_ids.extend([str(training['context_id'][i])]*m)
            selected_futures.extend(training['future_id'][i,chosen].tolist())
        # Same fitting routine and ridge across allocations; its training
        # residual folds are circuit-disjoint even with repeated histories.
        model=fit_pooled_one_future(np.asarray(x),np.asarray(y),np.asarray(gs),ridge=ridge,context_ids=np.asarray(selected_ids))
        mean,var=predict_pooled_one_future(model,evaluation['X'],structure_ids=evaluation['structure_id'])
        scores=[]
        mse=np.zeros_like(evaluation['Y']); nll=np.zeros_like(evaluation['Y'])
        for i,cid in enumerate(evaluation['context_id']):
            mask=evaluation['future_mask'][i]; target=evaluation['Y'][i,mask]
            values=gaussian_calibration(target,mean[i],var[i])
            scores.append(dict(context_id=str(cid),structure_id=int(evaluation['structure_id'][i]),**values))
            mse[i,mask]=(target-mean[i])**2
            nll[i,mask]=.5*(np.log(2*np.pi*var[i])+(target-mean[i])**2/var[i])
        rows.append(dict(**allocation,status='executed',independent_training_contexts=len(indices),
            training_circuits=len(circuits),training_outcomes=len(y),context_scores=scores,
            variance_fit_success=model['variance_fit_success'],variance_fit_message=model['variance_fit_message'],
            context_qualification=model['context_qualification'],
            scores={key:_equal_circuit(scores,key) for key in ('mse','nll_per_dimension','marginal_crps','marginal_coverage95')},
            covariance_model=model['covariance_model'],variance_numerics={'additive_floor':model['variance_floor'],'log_variance_clip':model['log_variance_clip'].tolist()},variance_conditioning=model['variance_conditioning']))
        arrays[f'm{m}_mean']=mean; arrays[f'm{m}_variance']=var
        arrays[f'm{m}_squared_error']=mse; arrays[f'm{m}_nll']=nll
        arrays[f'm{m}_training_context_id']=np.asarray(selected_ids)
        arrays[f'm{m}_training_future_id']=np.asarray(selected_futures)
        for key,value in model.items():
            if isinstance(value,np.ndarray): arrays[f'm{m}_{key}']=value
    return dict(plan=plan,allocations=rows,arrays=arrays,
        conditioning='all learners see EEG histories only; no state-variance target or reference outcomes in fitting',
        evaluation='same held-out circuits/contexts/futures for every complete allocation')
