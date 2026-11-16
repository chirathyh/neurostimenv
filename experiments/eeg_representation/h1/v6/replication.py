"""H2 finite-future precision with separate L=128 reference and primary M=4.

Small-M disjoint scoring groups share the reference pool and are therefore not
fully independent replicates. Regime-paired RNG streams use synchronized future
indices. No result here changes primary prediction scoring or circuit N.
"""
from __future__ import annotations
import numpy as np
import re
from scipy.stats import norm, poisson, t
from ..v4.replication import finite_reference_comparison as _reference
from .statistics import aggregate_circuit_losses, stable_mean, stable_sd

M_VALUES=(1,2,4,8,16,32)
REFERENCE_FUTURES=128


def _pool(x,ids,name):
    a=np.asarray(x,np.float64);tags=np.asarray(ids).astype(str)
    if a.ndim!=2 or not len(a) or not a.shape[1] or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite [future,coordinate]')
    if tags.shape!=(len(a),) or len(set(tags))!=len(tags):
        raise ValueError(f'{name} requires unique, aligned future IDs')
    return a,tags


def finite_reference_comparison(scoring,reference,*,scoring_ids,reference_ids,
                                expected_reference_count=REFERENCE_FUTURES,bootstrap=200,seed=0):
    """Unbiased reference-mean correction; uncertain variance/error diagnostics.

    v_hat_ref/L is subtracted from squared distance to the finite reference mean.
    Both-pool Welch intervals and independent both-pool bootstrap diagnostics
    retain reference uncertainty. Jackknife variance-MSE correction is approximate.
    Small engineering controls can explicitly override expected_reference_count.
    """
    x,si=_pool(scoring,scoring_ids,'scoring');r,ri=_pool(reference,reference_ids,'reference')
    if x.shape[1]!=r.shape[1] or set(si)&set(ri):raise ValueError('Mismatched targets or overlapping pools')
    if len(r)!=expected_reference_count:raise ValueError('Incomplete/mismatched frozen reference L')
    if expected_reference_count<2:raise ValueError('Reference must contain at least two futures')
    out=_reference(x,r,scoring_ids=si,reference_ids=ri,bootstrap=bootstrap,seed=seed)
    out.update(protocol='v6',reference_count_matches_protocol=True,
               reference_uncertainty='finite-L mean noise and variance jackknife, plus both-pool intervals/bootstrap',
               precision_scope='fixed-context Monte Carlo; no extra independent circuits')
    return out


def _summarize_group(x,r,si,ri,pred,reference_count,bootstrap,seed):
    result=finite_reference_comparison(x,r,scoring_ids=si,reference_ids=ri,
        expected_reference_count=reference_count,bootstrap=bootstrap,seed=seed)
    if pred is not None:
        prediction=np.asarray(pred,np.float64)
        if prediction.shape!=(x.shape[1],)or not np.isfinite(prediction).all():raise ValueError('Prediction dimension mismatch')
        risk=float(np.mean((x-prediction)**2))
        result['model_risk']=risk
        if len(x)>1:
            variance=float(x.var(axis=0,ddof=1).mean())
            corrected=float(np.mean((prediction-x.mean(0))**2)-variance/len(x))
            result.update(state_variance=variance,corrected_excess=corrected,
                finite_m_identity_residual=float(risk-variance-corrected))
        else:result.update(state_variance=None,corrected_excess=None,
            within_context_variance_status='unavailable at M=1; pooled variance learning is a separate modeling assumption')
    return result


def analyze_futures(scoring,reference,*,scoring_ids,reference_ids,prediction=None,
                    m_values=M_VALUES,expected_reference_count=REFERENCE_FUTURES,
                    seed=601,bootstrap=200):
    """One nested sequence plus mutually disjoint groups at each fixed M.

    All grouping choices are outcome-independent. Groups at different M are
    nested/reused and groups within M share one reference; neither is treated as
    independently generated experimental replication.
    """
    x,si=_pool(scoring,scoring_ids,'scoring');r,ri=_pool(reference,reference_ids,'reference')
    if set(si)&set(ri):raise ValueError('Reference/scoring pools overlap')
    if len(r)!=expected_reference_count:raise ValueError('Incomplete frozen reference pool')
    if x.shape[1]!=r.shape[1]:raise ValueError('Reference/scoring dimensions differ')
    ms=[int(m)for m in m_values]
    if ms!=sorted(set(ms))or min(ms)<1 or max(ms)>len(x):raise ValueError('Nested M grid must be unique, ordered and available')
    permutation=np.random.default_rng(seed).permutation(len(x));rows=[]
    for m in ms:
        ix=permutation[:m]
        nested=_summarize_group(x[ix],r,si[ix],ri,prediction,expected_reference_count,bootstrap,seed+m)
        groups=[];losses=[]
        for group_id,start in enumerate(range(0,len(x)-m+1,m)):
            take=permutation[start:start+m]
            # Bootstrap intervals only on the prespecified nested group; repeated
            # groups retain analytic finite-reference uncertainty without costly
            # redundant bootstrapping of the same reference.
            value=_summarize_group(x[take],r,si[take],ri,prediction,expected_reference_count,0,seed+m)
            groups.append(dict(group_id=group_id,indices=take.tolist(),scoring_ids=si[take].tolist(),**value))
            losses.append(value['reference_noise_corrected_mean_mse'])
        rows.append(dict(M=m,nested_indices=ix.tolist(),nested_scoring_ids=si[ix].tolist(),nested=nested,
            disjoint_scoring_groups=groups,n_disjoint_scoring_groups=len(groups),
            unused_scoring_futures=len(x)%m,disjoint_group_corrected_mean_mse=float(np.mean(losses)),
            groups_share_reference=True,groups_independent_for_inference=False,
            full_pool=m==len(x),new_future_uncertainty_is_zero=False))
    return dict(status='complete',M_available=len(x),L=len(r),dimension=x.shape[1],
        permutation=permutation.tolist(),rows=rows,negative_corrections_retained=True,
        primary_score_policy='unchanged first four registered scoring futures; H2 extras are separate',
        reference_is_truth=False,independent_circuit_count_increased=False,
        dependence='disjoint scoring groups within M, shared finite reference; nested reuse across M')


def paired_regime_future_indices(circuit_ids,n_futures,*,prefix_ids=None,repeats=1000,seed=601):
    """Same future ordinal across regimes; distinct prefixes resample independently.

    Without prefix metadata there is one paired stream group per circuit.
    """
    sid=np.asarray(circuit_ids).astype(str)
    if sid.ndim!=1 or not len(sid)or n_futures<1 or repeats<2:raise ValueError('Invalid paired resampling dimensions')
    prefixes=np.zeros(len(sid),int).astype(str)if prefix_ids is None else np.asarray(prefix_ids).astype(str)
    if prefixes.shape!=sid.shape:raise ValueError('Prefix IDs must align with circuit IDs')
    rng=np.random.default_rng(seed);out=np.empty((repeats,len(sid),n_futures),np.int64)
    for s,p in sorted(set(zip(sid,prefixes))):
        common=rng.integers(0,n_futures,(repeats,n_futures))
        out[:,(sid==s)&(prefixes==p),:]=common[:,None,:]
    return out


def paired_future_bootstrap_losses(per_future_losses,circuit_ids,condition_ids,prefix_ids,
                                   *,repeats=1000,seed=601,confidence=.95):
    """Conditional future-only bootstrap, keeping the observed circuit panel fixed."""
    loss=np.asarray(per_future_losses,np.float64)
    if loss.ndim!=2 or not np.isfinite(loss).all():raise ValueError('Finite context-by-future loss required')
    indices=paired_regime_future_indices(circuit_ids,loss.shape[1],prefix_ids=prefix_ids,repeats=repeats,seed=seed)
    point=aggregate_circuit_losses(loss,circuit_ids,condition_ids,prefix_ids)
    draws=[]
    for ix in indices:
        sampled=loss[np.arange(len(loss))[:,None],ix]
        row=aggregate_circuit_losses(sampled,circuit_ids,condition_ids,prefix_ids)
        draws.append(stable_mean(row['values']))
    alpha=1-confidence
    return dict(status='complete',estimate=stable_mean(point['values']),n_circuits=len(point['values']),
        conditional_mc_standard_error=stable_sd(draws),interval=np.quantile(draws,[alpha/2,1-alpha/2]).tolist(),
        confidence=confidence,repeats=repeats,seed=seed,
        interpretation='fixed-panel conditional future uncertainty, not circuit population or training variation',
        regime_pairing='same future ordinal across regimes within circuit/prefix; independent prefix resampling')


def analyze_panel(panel,predictions,*,selected_context_ids,seed=601,bootstrap=200):
    """Run H2 only on the registry-selected contexts; no outcome-based selection."""
    cids=np.asarray(panel['context_id']).astype(str);sid=np.asarray(panel['structure_id']).astype(str)
    wanted=list(map(str,selected_context_ids))
    if len(set(wanted))!=len(wanted)or not set(wanted)<=set(cids):raise ValueError('Missing/duplicate registered H2 contexts')
    p=np.asarray(predictions,np.float64)
    if p.ndim==3:
        if p.shape[0]!=5:raise ValueError('Exactly five fitted neural repeats required')
        # H2 conditional-mean precision does not need predictions. For per-model
        # risks report all seed members independently, never score an ensemble.
        members=p
    elif p.ndim==2:members=p[None]
    else:raise ValueError('Prediction shape must be [context,d] or [5,context,d]')
    records=[]
    if members.shape[1:]!=(len(cids),np.asarray(panel['Y']).shape[-1])or not np.isfinite(members).all():
        raise ValueError('Prediction contexts/coordinates must align with the H2 panel')
    if len(set(cids))!=len(cids):raise ValueError('Duplicate panel context IDs')
    circuits=sorted(set(sid[np.isin(cids,wanted)]))
    stream_seed={s:seed+i for i,s in enumerate(circuits)}
    for cid in wanted:
        k=int(np.flatnonzero(cids==cid)[0]);mask=np.asarray(panel['future_mask'][k],bool);rmask=np.asarray(panel['reference_mask'][k],bool)
        x=np.asarray(panel['Y'][k,mask],np.float64);r=np.asarray(panel['reference_Y'][k,rmask],np.float64)
        if len(x)!=32:raise ValueError('H2 requires complete M32 scoring pool')
        score_ids=np.asarray(panel['future_id'][k,mask]).astype(str)
        ref_ids=np.asarray(panel['reference_future_id'][k,rmask]).astype(str)
        def ordered(tags,pool,count):
            matches=[re.search(r'(?:^|-)'+pool+r'-(\d+)$',tag)for tag in tags]
            if not all(matches):raise ValueError('Registered scoring/reference ordinal IDs required')
            ordinals=[int(m.group(1))for m in matches]
            if sorted(ordinals)!=list(range(count)):raise ValueError('Missing or duplicated registered future ordinals')
            return np.argsort(ordinals)
        sx=ordered(score_ids,'score',32);rx=ordered(ref_ids,'reference',128)
        x=x[sx];r=r[rx];score_ids=score_ids[sx];ref_ids=ref_ids[rx]
        for member,pred in enumerate(members):
            result=analyze_futures(x,r,scoring_ids=score_ids,reference_ids=ref_ids,
                prediction=pred[k],seed=stream_seed[sid[k]],bootstrap=bootstrap)
            records.append(dict(context_id=cid,structure_id=sid[k],fitting_repeat=member,**result))
    return dict(status='complete',n_contexts=len(wanted),n_circuits=len(circuits),records=records,
        primary_future_policy='first four registered scoring futures remain unchanged',
        fitting_repeats_are_independent_circuits=False,
        subset_scope='registry-selected H2 circuits; report differences from entire primary evaluation panel')


def reference_controls(*,repetitions=2000,seed=613):
    """Independent Gaussian and Poisson repeats; synthetic, never neural evidence."""
    rng=np.random.default_rng(seed);out=[];m=4;l=128;dimension=3
    for law in ('Gaussian','Poisson'):
        if law=='Gaussian':
            x=rng.normal(0,2,(repetitions,m,dimension));r=rng.normal(0,2,(repetitions,l,dimension));variance=4.
        else:
            x=rng.poisson(5,(repetitions,m,dimension)).astype(np.float64);r=rng.poisson(5,(repetitions,l,dimension)).astype(np.float64);variance=5.
        raw=((x.mean(1)-r.mean(1))**2).mean(1);correction=r.var(1,ddof=1).mean(1)/l;corrected=raw-correction
        expected=variance/m;mcse=stable_sd(corrected)/np.sqrt(repetitions)
        # Exact marginal intervals for scoring mean under the known control law.
        if law=='Gaussian':
            mean_diff=x.mean(1)-r.mean(1);half=norm.ppf(.975)*np.sqrt(variance*(1/m+1/l))
            covered=(abs(mean_diff)<=half)
            coverage=float(covered.mean());coverage_scope='known-variance Gaussian mean-difference interval, independent pools'
        else:
            sums=x.sum(1);lo=poisson.ppf(.025,m*5);hi=poisson.ppf(.975,m*5)
            coverage=float(((sums>=lo)&(sums<=hi)).mean());coverage_scope='exact known-rate Poisson central count interval; discreteness permits overcoverage'
        # Also exercise the actual sample-variance Welch interval used by H2,
        # distinct from the known-law reference interval reported above.
        xv=x.var(1,ddof=1);rv=r.var(1,ddof=1);se2=xv/m+rv/l
        df=se2**2/((xv/m)**2/(m-1)+(rv/l)**2/(l-1))
        half=t.ppf(.975,df)*np.sqrt(se2)
        welch=float((abs(x.mean(1)-r.mean(1))<=half).mean())
        # Vectorized exact leave-one-out sample variances reproduce the generic
        # jackknife used above; no reference is treated as exact truth.
        centered=r-r.mean(1,keepdims=True);ssd=np.sum(centered**2,axis=1)
        loo=(ssd[:,None,:]-l/(l-1)*centered**2)/(l-2)
        loo_trace=loo.mean(2)
        jack=(l-1)/l*np.sum((loo_trace-loo_trace.mean(1,keepdims=True))**2,axis=1)
        variance_error=(xv.mean(1)-rv.mean(1))**2-jack
        true_variance_mse=(2*variance**2/(m-1)+(variance/m if law=='Poisson'else 0))/dimension
        variance_mcse=stable_sd(variance_error)/np.sqrt(repetitions)
        out.append(dict(law=law,repetitions=repetitions,M=m,L=l,raw_mean=float(raw.mean()),expected_raw=variance*(1/m+1/l),
            corrected_mean=float(corrected.mean()),expected_corrected=expected,monte_carlo_se=float(mcse),
            unbiased_mean_control_pass=bool(abs(corrected.mean()-expected)<=5*mcse),
            negative_estimator_count=int(np.sum(corrected<0)),known_law_interval_coverage=coverage,coverage_scope=coverage_scope,
            actual_welch_interval_coverage=welch,actual_welch_scope='sample variance in both pools, approximate coordinatewise Welch interval',
            corrected_variance_mse=float(variance_error.mean()),true_scoring_variance_mse=true_variance_mse,
            variance_mse_mcse=float(variance_mcse),variance_correction_control_pass=bool(abs(variance_error.mean()-true_variance_mse)<=5*variance_mcse),
            variance_reference_correction='finite-L jackknife correction is approximate; its sampling uncertainty remains'))
    return dict(status='complete',controls=out,scope='cheap independent synthetic validation; no biophysical observations')


def replication(config,root,jobs=None,resume=False,*,bootstrap=200):
    """Resume-safe H2 runner; default job is the frozen H1 primary anchor.

    Additional target jobs are explicit. M32/L128 is required; an engineering
    root lacking those pools reports unavailable rather than inventing evidence.
    """
    from pathlib import Path
    from .artifacts import atomic_json,read_json,file_hash,artifact_record,verify_record,module_hash,canonical_hash,resume_record
    from .derivation import load_panel
    from .models import prediction_record
    from .config import contexts
    root=Path(root);selected=jobs or [config['protocol']['primary_job_id']]
    registered=[c['context_id']for c in contexts(config)if c['reference_futures']>0]
    results=[]
    for job in selected:
        panels,manifest=load_panel(root,job);panel=panels['evaluation']
        model_path=root/'models'/job/'manifest.json';model=read_json(model_path)
        ids=model['selected_member_ids'];predictions=[];input_records=[]
        for member in ids:
            path=root/'models'/job/member/'evaluation.npz'
            record=prediction_record(root,job,path)
            verify_record(root,record);input_records.append(record)
            with np.load(path,allow_pickle=False)as z:
                if not np.array_equal(z['context_id'],panel['context_id']):raise ValueError('H2 prediction/target context mismatch')
                if str(z['target_fingerprint'])!=manifest['target_fingerprint']:raise ValueError('H2 prediction/target risk mismatch')
                predictions.append(np.asarray(z['mean'],np.float64))
        signature=canonical_hash({'model':file_hash(model_path),'target':manifest['target_fingerprint'],
            'target_file':manifest['files']['evaluation']['sha256'],'inputs':input_records,'registered_contexts':registered,
            'analysis':module_hash('replication','statistics'),'bootstrap':bootstrap})
        marker=root/'replication'/job/'summary.json';old=resume_record(marker,signature,resume)
        if old:out=old
        else:
            if not registered or config['forks']['reference_futures']!=128 or config['forks']['high_score_futures']!=32:
                out={'status':'unavailable','reason':'Root lacks the complete frozen H2 M32/L128 allocation','signature':signature,'job_id':job}
            else:
                prediction=predictions[0]if len(predictions)==1 else np.stack(predictions)
                out=analyze_panel(panel,prediction,selected_context_ids=registered,bootstrap=bootstrap)
                out.update(signature=signature,job_id=job,selected_member_ids=ids,input_artifacts=input_records,
                    target_fingerprint=manifest['target_fingerprint'],whole_primary_panel_circuits=len(np.unique(panel['structure_id'])),
                    subset_selection='fixed registry prefix and circuits chosen before outcomes',
                    fitting_seed_scope='members are fitting repeats; shared conditional-mean/reference diagnostics are not additional evidence')
            atomic_json(marker,out)
        results.append({'id':job,'status':out['status'],'path':marker.relative_to(root).as_posix(),'artifact':artifact_record(root,marker)})
    out={'status':'complete'if all(r['status']=='complete'for r in results)else 'unavailable','jobs':results,
         'scientific_question':'H2 finite-future estimation with independent but noisy L128 reference',
         'primary_forecast_policy':'Fixed first four registered scoring futures remain unchanged'}
    atomic_json(root/'replication/manifest.json',out);return out


def synchronized_jackknife(futures,prediction,panel):
    """Conditional MC error: synchronize regimes, independently combine prefixes.

    All conditions share a future ordinal inside one circuit/prefix. Independent
    prefix and circuit stream variances add with squared fixed nesting weights.
    This estimates future Monte Carlo error only, not population heterogeneity.
    """
    y=np.asarray(futures,np.float64);pred=np.asarray(prediction,np.float64)
    sid=np.asarray(panel['structure_id']).astype(str);condition=np.asarray(panel['regime']).astype(str)
    cid=np.asarray(panel['context_id']).astype(str)
    prefix=[]
    for context in cid:
        match=re.fullmatch(r's\d+-p(\d+)-.+',context)
        if match is None:raise ValueError('Registered context prefix required for paired jackknife')
        prefix.append(int(match.group(1)))
    prefix=np.asarray(prefix);mask=np.asarray(panel.get('future_mask',np.ones(y.shape[:2],bool)),bool)
    if y.ndim!=3 or pred.shape!=(len(y),y.shape[-1])or sid.shape!=(len(y),)or condition.shape!=sid.shape or cid.shape!=sid.shape or mask.shape!=y.shape[:2]:
        raise ValueError('Paired jackknife shapes/IDs differ')
    if not np.isfinite(y[mask]).all()or not np.isfinite(pred).all():raise ValueError('Finite active futures/predictions required')
    aggregate_circuit_losses(np.zeros(len(y)),sid,condition,prefix,
        expected_conditions=np.unique(condition),expected_prefixes=np.unique(prefix))
    tags=np.asarray(panel['future_id']).astype(str)
    if tags.shape!=mask.shape:raise ValueError('Future IDs required for synchronized deletion')
    circuits=sorted(set(sid));keys=('mean_risk','state_floor','corrected_excess')
    pooled={k:0. for k in keys};rows=[];available=True
    for s in circuits:
        conds=sorted(set(condition[sid==s]));prefixes=sorted(set(prefix[sid==s]));variance={k:0. for k in keys};groups=[]
        for pr in prefixes:
            take=np.flatnonzero((sid==s)&(prefix==pr));ordered=[];ordinals=[]
            for k in take:
                active=np.flatnonzero(mask[k]);parsed=[re.search(r'(?:^|-)score-(\d+)$',tag)for tag in tags[k,active]]
                if not all(parsed):raise ValueError('Scoring ordinal missing in paired jackknife')
                order=np.array([int(m.group(1))for m in parsed]);sort=np.argsort(order)
                if len(set(order))!=len(order):raise ValueError('Duplicate scoring ordinal in paired jackknife')
                ordinals.append(order[sort]);ordered.append(y[k,active[sort]])
            if any(not np.array_equal(x,ordinals[0])for x in ordinals):raise ValueError('Paired regimes have different active future ordinals')
            yy=np.asarray(ordered);m=yy.shape[1]
            if m<3:
                available=False;groups.append({'prefix':int(pr),'n_futures':m,'status':'unavailable: need >=3 futures for floor/excess jackknife'});continue
            # Every condition contributes 1/C and each of its prefixes 1/P.
            weights=np.asarray([1./(len(conds)*len(set(prefix[(sid==s)&(condition==condition[k])])))for k in take])
            estimates={k:[]for k in keys}
            for j in range(m):
                xx=np.delete(yy,j,axis=1);vv=xx.var(1,ddof=1).mean(1)
                rr=((xx-pred[take,None,:])**2).mean((1,2))
                ee=((xx.mean(1)-pred[take])**2).mean(1)-vv/(m-1)
                for key,value in zip(keys,(rr,vv,ee)):estimates[key].append(float(weights@value))
            group_variance={key:float((m-1)/m*np.sum((np.asarray(values)-np.mean(values))**2))for key,values in estimates.items()}
            for key in keys:variance[key]+=group_variance[key]
            groups.append({'prefix':int(pr),'n_futures':m,'status':'complete','regimes':condition[take].tolist(),
                'future_ordinals':ordinals[0].tolist(),'fixed_context_weights_within_circuit':weights.tolist(),
                'jackknife_variance_contribution':group_variance,'delete_one_weighted_prefix_contributions':estimates})
        for key in keys:pooled[key]+=variance[key]/len(circuits)**2
        complete=all(g['status']=='complete'for g in groups)
        rows.append({'structure_id':s,'n_prefixes':len(prefixes),'prefix_groups':groups,'status':'complete'if complete else 'unavailable',
                     'jackknife_standard_error':{key:float(np.sqrt(value))if complete else None for key,value in variance.items()}})
    return {'status':'complete'if available else 'unavailable','per_circuit':rows,
        'pooled_conditional_mc_se':{key:float(np.sqrt(value))if available else None for key,value in pooled.items()},
        'n_circuits':len(circuits),'weighting':'equal circuits, then fixed equal conditions/prefixes; independent prefix and circuit variances add',
        'interpretation':'Conditional future MC only. Delete matching future ordinal across regimes within circuit/prefix; prefixes are distinct independent stream groups, not one shared circuit draw.'}
