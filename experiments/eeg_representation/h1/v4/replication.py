"""H2 finite independent-future estimation with uncertain reference pools.

A finite reference is never exact truth. Independent reference and scoring IDs
are mandatory. Negative reference-noise corrections remain in every artifact.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.stats import norm,t
from ..replication import finite_m_estimates,check_pool_independence
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json,file_hash,read_json
from .models import load_panel
from .scoring import paired_regime_mc


def jackknife_variance_uncertainty(x):
    x=np.asarray(x,float)
    if x.ndim!=2 or len(x)<3:return None
    values=np.asarray([np.delete(x,i,axis=0).var(0,ddof=1).mean() for i in range(len(x))])
    return float((len(x)-1)/len(x)*np.sum((values-values.mean())**2))


def finite_reference_comparison(scoring,reference,*,scoring_ids,reference_ids,bootstrap=200,seed=0):
    x=np.asarray(scoring,float);r=np.asarray(reference,float)
    if x.ndim!=2 or r.ndim!=2 or x.shape[1]!=r.shape[1] or len(x)<1 or len(r)<2:
        raise ValueError('Scoring M>=1 and independent reference L>=2 with the same dimension required')
    if len(scoring_ids)!=len(x) or len(reference_ids)!=len(r): raise ValueError('Future ID counts do not match pools')
    check_pool_independence(scoring_ids,reference_ids)
    if not np.isfinite(x).all() or not np.isfinite(r).all(): raise ValueError('Finite future arrays required')
    m=len(x);l=len(r);xm=x.mean(0);rm=r.mean(0);rv=r.var(0,ddof=1);vref=float(rv.mean())
    distance=float(np.mean((xm-rm)**2));corrected=distance-vref/l
    result=dict(status='complete',m=m,l=l,dimension=x.shape[1],
        squared_mean_distance_to_finite_reference=distance,reference_mean_noise_correction=vref/l,
        reference_noise_corrected_mean_mse=corrected,
        reference_estimate_variance=vref,expected_distance_using_reference_variance=vref*(1/m+1/l),
        reference_variance_jackknife_sampling_variance=jackknife_variance_uncertainty(r),
        negative_corrections_retained=True,reference_is_exact_truth=False,
        expectation='E||barY_M-barY_ref,L||^2/d = v(C)*(1/M+1/L); corrected mean error estimates v(C)/M',
        interval_scope='conditional future Monte Carlo for one fixed context; not circuit population coverage')
    if m>1:
        xv=x.var(0,ddof=1);vx=float(xv.mean());noise=jackknife_variance_uncertainty(r)
        result.update(scoring_variance=vx,variance_difference_to_finite_reference=vx-vref,
            squared_variance_difference=(vx-vref)**2,
            reference_noise_corrected_variance_mse=None if noise is None else (vx-vref)**2-noise)
        se2=xv/m+rv/l
        dof=np.divide(se2**2,(xv/m)**2/(m-1)+(rv/l)**2/(l-1),out=np.full_like(se2,np.inf),where=se2>0)
        half=t.ppf(.975,dof)*np.sqrt(se2)
        result.update(mean_difference_interval95_low=(xm-rm-half).tolist(),mean_difference_interval95_high=(xm-rm+half).tolist(),
            mean_difference_zero_containment=float(np.mean(np.abs(xm-rm)<=half)),
            interval_method='Welch coordinatewise mean-difference intervals include BOTH M and L variance; approximate for EEG, no joint/exact coverage claim')
    else:result.update(scoring_variance=None,variance_status='unavailable: one future',
        mean_difference_interval_status='unavailable: scoring variance cannot be estimated from M=1')
    if bootstrap and m>1:
        rng=np.random.default_rng(seed);meanvalues=[];variancevalues=[]
        for _ in range(int(bootstrap)):
            xx=x[rng.integers(0,m,m)];rr=r[rng.integers(0,l,l)]
            meanvalues.append(float(((xx.mean(0)-rr.mean(0))**2).mean()-rr.var(0,ddof=1).mean()/l))
            variancevalues.append(float(xx.var(0,ddof=1).mean()-rr.var(0,ddof=1).mean()))
        result['independent_pool_bootstrap_mean_corrected_mse_95']=np.quantile(meanvalues,[.025,.975]).tolist()
        result['independent_pool_bootstrap_variance_difference_95']=np.quantile(variancevalues,[.025,.975]).tolist()
        result['bootstrap_warning']='approximate percentile diagnostic resamples both pools independently; no exact finite-sample EEG coverage claim'
    return result


def replication_curve(scoring,reference=None,*,scoring_ids,reference_ids=(),prediction=None,
                      m_values=(1,2,4,8),repetitions=32,bootstrap=200,seed=0):
    x=np.asarray(scoring,float)
    if x.ndim!=2 or not len(x) or not np.isfinite(x).all():raise ValueError('Finite [M,D] scoring futures required')
    r=None if reference is None or np.asarray(reference).size==0 else np.asarray(reference,float)
    if len(scoring_ids)!=len(x) or len(reference_ids)!=(0 if r is None else len(r)):raise ValueError('Future ID lengths differ')
    check_pool_independence(scoring_ids,reference_ids)
    if repetitions<2:raise ValueError('At least two nested subsamples required')
    rng=np.random.default_rng(seed);permutations=np.asarray([rng.permutation(len(x)) for _ in range(repetitions)])
    arrays={'nested_permutation':permutations};rows=[]
    for m in m_values:
        m=int(m)
        if m<1:raise ValueError('M must be positive')
        if m>len(x):rows.append(dict(m=m,status='unrun',available_futures=len(x)));continue
        # Full-pool order is fixed to avoid machine-rounding masquerading as variation.
        indices=np.tile(np.arange(len(x)),(repetitions,1)) if m==len(x) else permutations[:,:m]
        means=np.asarray([x[ix].mean(0) for ix in indices]);variances=None if m==1 else np.asarray([x[ix].var(0,ddof=1).mean() for ix in indices])
        arrays[f'M{m}_indices']=indices;arrays[f'M{m}_means']=means
        if variances is not None:arrays[f'M{m}_variances']=variances
        row=dict(m=m,status='complete',n_subsamples=repetitions,
            full_pool_subsampling=m==len(x),mean_subsample_spread=float(np.sqrt(np.mean(means.var(0,ddof=1)))),
            variance_mean=None if variances is None else float(variances.mean()),
            variance_subsample_sd=None if variances is None else float(variances.std(ddof=1)),
            mc_uncertainty_is_zero=False,subsample_scope='conditional subsampling spread; at full M the same pool is repeated, not zero Monte Carlo uncertainty')
        if prediction is not None:
            values=[finite_m_estimates(x[ix],prediction) for ix in indices]
            row['model_risk']=float(np.mean([v['model_risk'] for v in values]))
            row['corrected_excess']=None if m==1 else float(np.mean([v['bias_corrected_excess'] for v in values]))
            arrays[f'M{m}_model_risk']=np.asarray([v['model_risk'] for v in values])
            if m>1:arrays[f'M{m}_corrected_excess']=np.asarray([v['bias_corrected_excess'] for v in values])
        if r is not None and len(r)>=2:
            refvar=float(r.var(0,ddof=1).mean());dist=((means-r.mean(0))**2).mean(1)
            arrays[f'M{m}_reference_mean_mse_corrected']=dist-refvar/len(r)
            row.update(reference_status='complete: finite independent uncertain reference',reference_count=len(r),
                mean_mse_against_finite_reference=float(dist.mean()),reference_noise_corrected_mean_mse=float(dist.mean()-refvar/len(r)),
                reference_mean_noise_correction=refvar/len(r),
                expected_squared_mean_distance_using_reference_variance=refvar*(1/m+1/len(r)))
            # One prespecified nested draw gets intervals; repeats do not create subjects.
            row['first_nested_sample_reference_assessment']=finite_reference_comparison(x[indices[0]],r,
                scoring_ids=np.asarray(scoring_ids)[indices[0]],reference_ids=reference_ids,bootstrap=bootstrap,seed=seed+m)
        else:row['reference_status']='unavailable: fewer than two independent reference futures'
        rows.append(row)
    return dict(status='complete',rows=rows,arrays=arrays,
        interpretation='H2 finite futures; finite L is not truth; no one-future impossibility claim',
        scoring_pool_size=len(x),reference_pool_size=0 if r is None else len(r))


def _record(root,path):return dict(path=str(path.relative_to(root)),sha256=file_hash(path),bytes=path.stat().st_size)


def replication(config,root,job_ids=None,resume=False):
    root=Path(root);modelindex=read_json(root/'models/manifest.json');available=[r['id'] for r in modelindex['jobs']]
    jobs=available if job_ids is None else list(job_ids)
    if set(jobs)-set(available):raise ValueError('Replication requires fitted jobs')
    cfg=dict(m_values=[1,2,4,8],repetitions=32,bootstrap=200);cfg.update(config.get('replication',{}));cfg['bootstrap']=cfg.get('bootstrap_samples',cfg['bootstrap']);results=[]
    for job in jobs:
        panel,manifest=load_panel(root,job);models=read_json(root/'models'/job/'manifest.json');ev=panel['evaluation']
        path=root/'models'/job/models['selected_candidate']/'evaluation.npz'
        signature=canonical_hash(dict(config=cfg,input=file_hash(root/'derived'/job/'evaluation.npz'),prediction=file_hash(path),implementation=file_hash(Path(__file__))))
        directory=root/'replication'/job;marker=directory/'summary.json'
        if marker.exists():
            if not resume:raise FileExistsError('Replication exists; use resume')
            old=read_json(marker)
            if old['signature']!=signature:raise ValueError('Replication source/config/input mismatch')
            for f in old['files']:
                if file_hash(root/f['path'])!=f['sha256']:raise ValueError('Replication artifact corrupted')
            results.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=True));continue
        with np.load(path,allow_pickle=False) as f:prediction=f['mean'];pred_ids=f['context_id']
        if not np.array_equal(pred_ids,ev['context_id']):raise ValueError('Replication prediction ID mismatch')
        context_rows=[];files=[]
        for k,cid in enumerate(ev['context_id']):
            mask=ev['future_mask'][k];rmask=ev['reference_mask'][k]
            curve=replication_curve(ev['Y'][k,mask],ev['reference_Y'][k,rmask],
                scoring_ids=ev['future_id'][k,mask],reference_ids=ev['reference_future_id'][k,rmask],
                prediction=prediction[k],m_values=cfg['m_values'],repetitions=int(cfg['repetitions']),bootstrap=int(cfg['bootstrap']),seed=k+2037)
            arrays=curve.pop('arrays');arrays.update(context_id=np.asarray(cid),structure_id=np.asarray(ev['structure_id'][k]),
                scoring_future_id=ev['future_id'][k,mask],reference_future_id=ev['reference_future_id'][k,rmask])
            target=directory/f'context_{k:04d}.npz';atomic_npz(target,**arrays);files.append(_record(root,target))
            context_rows.append(dict(context_id=str(cid),structure_id=int(ev['structure_id'][k]),regime=str(ev['regime'][k]),**curve))
        result=dict(status='complete',signature=signature,job_id=job,contexts=context_rows,files=files,
            paired_regime_full_pool_mc=paired_regime_mc(ev['Y'],prediction,ev['structure_id'],ev['future_mask']),
            n_circuits=len(np.unique(ev['structure_id'])),n_contexts=len(ev['Y']),
            n_independent_reference_contexts=int(np.sum(ev['reference_mask'].sum(1)>=2)),
            pooled_one_future_learning='models fit one observed future per training context; Gaussian pooled conditional variance remains possible under modeling assumptions; no state-variance supervision',
            negative_corrections_retained=True,finite_reference_is_not_truth=True,
            population_scope='circuit is independent unit; prefixes/futures are nested; optimizer seeds do not increase N')
        atomic_json(marker,result);results.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=False))
    result=dict(status='complete',jobs=results,scientific_question='H2 estimation with few independent futures')
    atomic_json(root/'replication/manifest.json',result);return result
