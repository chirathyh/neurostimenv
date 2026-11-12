"""Inspectable matched-context H1 scores; future repeats are not subjects.

Predictions must be frozen before these functions receive scoring outcomes.
Gaussian variances are EEG-conditioned predictive distributions, not estimates
of the full-state fork variance. All vector losses divide by output dimension.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr
from scipy.stats import norm


def _masked_mean(a, mask, axis=1):
    return np.sum(np.where(mask, a, 0.), axis=axis) / np.maximum(mask.sum(axis=axis), 1)


def _jackknife_standard_error(values):
    values = np.asarray(values, dtype=float)
    return float(np.sqrt((len(values)-1)/len(values)*np.sum((values-values.mean())**2))) if len(values)>1 else np.nan


def fork_risk_components(futures, mean, future_mask=None):
    """Per-context risk/floor/excess with finite-M correction, retaining negatives.

    futures: K,M,D, mean: K,D; masks mark genuinely independent future draws.
    sample variance and corrected excess are unavailable at M=1, never zero.
    The direct risk equals floor + corrected excess for the same scoring pool.
    """
    y=np.asarray(futures,dtype=float);mu=np.asarray(mean,dtype=float)
    if y.ndim!=3 or mu.shape!=(y.shape[0],y.shape[2]):
        raise ValueError('Expected futures[K,M,D] and frozen mean[K,D]')
    mask=np.ones(y.shape[:2],bool) if future_mask is None else np.asarray(future_mask,bool)
    if mask.shape!=y.shape[:2] or np.any(mask.sum(1)==0) or not np.isfinite(y[mask]).all() or not np.isfinite(mu).all():
        raise ValueError('Each context requires finite targets and at least one valid future')
    names=['mean_risk','state_floor','pair_half_mse','corrected_excess','uncorrected_mean_error','risk_mc_se','floor_jackknife_mc_se','excess_jackknife_mc_se','loo_oracle_risk','corrected_loo_risk']
    out={name:np.full(len(y),np.nan) for name in names}
    out['n_futures']=mask.sum(1)
    per_future=np.where(mask,((y-mu[:,None])**2).mean(-1),np.nan)
    out['per_future_mse']=per_future
    for k in range(len(y)):
        z=y[k,mask[k]];m=len(z);risk=per_future[k,mask[k]];center=z.mean(0)
        out['mean_risk'][k]=risk.mean();out['uncorrected_mean_error'][k]=((mu[k]-center)**2).mean()
        if m<2:continue
        variance=z.var(axis=0,ddof=1).mean()
        out['state_floor'][k]=variance
        out['corrected_excess'][k]=out['uncorrected_mean_error'][k]-variance/m
        out['risk_mc_se'][k]=risk.std(ddof=1)/np.sqrt(m)
        out['pair_half_mse'][k]=((z[:2*(m//2):2]-z[1:2*(m//2):2])**2).mean()/2
        loo=(z.sum(0)[None]-z)/(m-1)
        out['loo_oracle_risk'][k]=((z-loo)**2).mean()
        out['corrected_loo_risk'][k]=out['loo_oracle_risk'][k]*(m-1)/m
        if m>=3:
            vl=[];el=[]
            for j in range(m):
                v=np.delete(z,j,axis=0);s=v.var(0,ddof=1).mean()
                vl.append(s);el.append(((mu[k]-v.mean(0))**2).mean()-s/(m-1))
            out['floor_jackknife_mc_se'][k]=_jackknife_standard_error(vl)
            out['excess_jackknife_mc_se'][k]=_jackknife_standard_error(el)
    good=out['n_futures']>=2
    if not np.allclose(out['mean_risk'][good],out['state_floor'][good]+out['corrected_excess'][good],rtol=1e-10,atol=1e-12):
        raise AssertionError('Risk decomposition finite-sample identity failed')
    return out


def gaussian_scores(futures,mean,variance,future_mask=None,draws=64,seed=0):
    """Marginal proper scores and a joint diagonal-Gaussian energy score.

    Energy uses independent paired draws: E||X-Y||/sqrt(D) minus
    .5 E||X-X'||/sqrt(D). No paired model-draw MSE is called proper.
    Joint diagonal assumption is explicit; marginal calibration cannot validate
    temporal covariance. Draw arrays permit independent recalculation.
    """
    y=np.asarray(futures,float);mu=np.asarray(mean,float);var=np.asarray(variance,float)
    if y.ndim!=3 or mu.shape!=(len(y),y.shape[-1]) or var.shape!=mu.shape or np.any(var<=0) or not np.isfinite(var).all():
        raise ValueError('Finite positive diagonal predictive variances required')
    if draws<2:raise ValueError('At least two distribution draws required')
    mask=np.ones(y.shape[:2],bool) if future_mask is None else np.asarray(future_mask,bool)
    sd=np.sqrt(var);z=(y-mu[:,None])/sd[:,None]
    phi=np.exp(-z*z/2)/np.sqrt(2*np.pi)
    pit=ndtr(z)
    nll=.5*(np.log(2*np.pi*var[:,None])+z*z)
    crps=sd[:,None]*(z*(2*pit-1)+2*phi-1/np.sqrt(np.pi))
    rng=np.random.default_rng(seed)
    a=mu[None]+rng.normal(size=(draws,*mu.shape))*sd[None]
    b=mu[None]+rng.normal(size=(draws,*mu.shape))*sd[None]
    distance=np.linalg.norm(a-b,axis=-1).mean(0)/np.sqrt(y.shape[-1])
    energy=np.empty(y.shape[:2])
    for k in range(len(y)):
        energy[k]=np.linalg.norm(a[:,k,None]-y[k][None],axis=-1).mean(0)/np.sqrt(y.shape[-1])-.5*distance[k]
    res=dict(gaussian_nll=np.where(mask[...,None],nll,np.nan),marginal_crps=np.where(mask[...,None],crps,np.nan),pit=np.where(mask[...,None],pit,np.nan),energy=np.where(mask,energy,np.nan),draws=a,independent_draws=b,predictive_sd=sd)
    for probability in (.5,.8,.95):
        width=norm.ppf((1+probability)/2)*sd
        res[f'coverage_{int(probability*100)}']=np.where(mask[...,None],np.abs(y-mu[:,None])<=width[:,None],np.nan)
        res[f'interval_width_{int(probability*100)}']=2*width
    return res


def circuit_summary(context_scores,structure_ids,seed=0,bootstrap=2000):
    """Equal contexts within circuits, equal circuits; no clip/future bootstrap."""
    structure_ids=np.asarray(structure_ids).astype(str)
    structures=np.unique(structure_ids)
    rows=[]
    keys=[k for k,v in context_scores.items() if np.asarray(v).shape==(len(structure_ids),) and k!='n_futures']
    for structure in structures:
        ix=structure_ids==structure;row=dict(structure_id=str(structure),n_contexts=int(ix.sum()))
        for key in keys:
            values=np.asarray(context_scores[key])[ix];finite=np.isfinite(values)
            row[key]=float(values[finite].mean()) if finite.any() else None
            row[key+'_available_contexts']=int(finite.sum())
        rows.append(row)
    summary={};rng=np.random.default_rng(seed)
    for key in keys:
        values=np.asarray([r[key] for r in rows if r[key] is not None],float)
        if not len(values):summary[key]=dict(mean=None,n_circuits=0,status='unavailable');continue
        result=dict(mean=float(values.mean()),n_circuits=len(values))
        if len(values)>=3 and bootstrap:
            b=values[rng.integers(0,len(values),(bootstrap,len(values)))].mean(1)
            result['circuit_bootstrap_95']=np.quantile(b,[.025,.975]).tolist()
            result['status']='exploratory circuit bootstrap; simulation circuits are not biological subjects'
        else:result['status']='too few independent circuits for a population precision claim'
        summary[key]=result
    # Primary R_m/R0 is a ratio of equally weighted risks, not an average
    # of conditional ratios. Context ratios remain explicit diagnostics.
    if 'mean_risk' in context_scores and 'r0' in context_scores:
        valid=[r for r in rows if r.get('mean_risk') is not None and r.get('r0') is not None]
        for row in rows:
            denominator=row.get('r0');numerator=row.get('mean_risk')
            row['risk_over_r0']=(numerator/denominator if numerator is not None and denominator is not None and denominator>0 else None)
            row['skill']=None if row['risk_over_r0'] is None else 1-row['risk_over_r0']
        numerator=np.asarray([r['mean_risk'] for r in valid],float);denominator=np.asarray([r['r0'] for r in valid],float)
        ratio=float(numerator.mean()/denominator.mean()) if len(valid) and denominator.mean()>0 else None
        description='ratio of equal-circuit-weighted mean risks; contexts weighted equally within each circuit'
        ratio_result=dict(mean=ratio,n_circuits=len(valid),estimand=description)
        skill_result=dict(mean=None if ratio is None else 1-ratio,n_circuits=len(valid),estimand='1 minus '+description)
        if ratio is not None and len(valid)>=3 and bootstrap:
            ix=rng.integers(0,len(valid),(bootstrap,len(valid)))
            denominators=denominator[ix].mean(1)
            boot=np.divide(numerator[ix].mean(1),denominators,out=np.full(bootstrap,np.nan),where=denominators>0)
            boot=boot[np.isfinite(boot)]
            ratio_result['circuit_bootstrap_95']=np.quantile(boot,[.025,.975]).tolist()
            skill_result['circuit_bootstrap_95']=np.quantile(1-boot,[.025,.975]).tolist()
            ratio_result['status']=skill_result['status']='exploratory paired circuit bootstrap of ratio; simulation circuits are not biological subjects'
        else:ratio_result['status']=skill_result['status']='unavailable denominator or too few independent circuits for population precision'
        summary['risk_over_r0']=ratio_result;summary['skill']=skill_result
        for name in ('context_risk_ratio','context_skill'):
            if name in summary:summary[name]['interpretation']='secondary mean of context-wise ratios; not the population R_m/R0 estimand'
    return dict(summary=summary,per_circuit=rows,weighting='equal contexts within each circuit, then equal circuits; primary risk/R0 uses a ratio of those aggregate risks; optimizer seeds are not subjects')


def score_panel(futures,mean,variance,baseline_mean,structure_ids,*,future_mask=None,
                target_fingerprint,prediction_fingerprint=None,y_scale=1.,y_offset=None,
                draws=64,seed=0,reference_futures=None,reference_mask=None,comparison_fingerprint=None,prediction_comparison_fingerprint=None):
    if prediction_fingerprint is not None and prediction_fingerprint!=target_fingerprint:
        raise ValueError('Target fingerprint mismatch: refuse risk comparison/subtraction')
    if comparison_fingerprint is not None and prediction_comparison_fingerprint is not None and comparison_fingerprint!=prediction_comparison_fingerprint:
        raise ValueError('Comparison context/population fingerprint mismatch')
    if not np.isfinite(y_scale) or y_scale<=0:raise ValueError('Positive affine target scale required')
    y=np.asarray(futures,float);mask=np.ones(y.shape[:2],bool) if future_mask is None else np.asarray(future_mask,bool)
    base=np.broadcast_to(np.asarray(baseline_mean,float),(len(y),y.shape[-1]))
    result=fork_risk_components(y,mean,mask)
    r0=fork_risk_components(y,base,mask)['mean_risk']
    result['r0']=r0
    result['risk_difference_from_r0']=result['mean_risk']-r0
    result['context_risk_ratio']=np.divide(result['mean_risk'],r0,out=np.full(len(y),np.nan),where=r0>0)
    result['context_skill']=1-result['context_risk_ratio']
    good=result['n_futures']>=2
    result['matched_risk_for_floor']=np.where(good,result['mean_risk'],np.nan)
    proper=gaussian_scores(y,mean,variance,mask,draws,seed)
    for key in ('gaussian_nll','marginal_crps','pit','coverage_50','coverage_80','coverage_95'):
        per_future=proper[key].mean(-1)
        result[key]=_masked_mean(per_future,mask)
    result['energy']=_masked_mean(proper['energy'],mask)
    result['predictive_sd']=proper['predictive_sd'].mean(-1)
    result['sharpness_95']=proper['interval_width_95'].mean(-1)
    for key in ('mean_risk','r0','risk_difference_from_r0','state_floor','pair_half_mse','corrected_excess','risk_mc_se','floor_jackknife_mc_se','excess_jackknife_mc_se','loo_oracle_risk','corrected_loo_risk'):
        result[key+'_physical']=result[key]*y_scale**2
    result['gaussian_nll_physical']=result['gaussian_nll']+np.log(y_scale)
    for key in ('marginal_crps','energy','predictive_sd','sharpness_95'):result[key+'_physical']=result[key]*y_scale
    if reference_futures is not None and np.asarray(reference_futures).shape[1]>0:
        ref=np.asarray(reference_futures,float);rm=np.ones(ref.shape[:2],bool) if reference_mask is None else np.asarray(reference_mask,bool)
        if len(ref)!=len(y) or ref.shape[-1]!=y.shape[-1]:raise ValueError('Independent reference pool mismatch')
        available=rm.sum(1)>0
        oracle=np.full((len(y),y.shape[-1]),np.nan)
        oracle[available]=(np.where(rm[...,None],ref,0).sum(1)[available]/rm.sum(1)[available,None])
        oracle_risk=np.full(len(y),np.nan)
        if available.any():oracle_risk[available]=fork_risk_components(y[available],oracle[available],mask[available])['mean_risk']
        result['independent_reference_mean_risk']=oracle_risk
        result['reference_futures']=rm.sum(1)
    context={k:v for k,v in result.items() if np.asarray(v).shape==(len(y),)}
    arrays={'truth':y,'mean':np.asarray(mean),'variance':np.asarray(variance),'baseline_mean':base,'future_mask':mask,'target_fingerprint':np.asarray(target_fingerprint),'y_scale':np.asarray(y_scale),'y_offset':np.zeros(y.shape[-1]) if y_offset is None else np.asarray(y_offset)}
    arrays.update({'context__'+k:v for k,v in context.items()})
    arrays.update({'score__'+k:v for k,v in proper.items()})
    arrays['per_future_mse']=result['per_future_mse']
    offset=np.zeros(y.shape[-1]) if y_offset is None else np.asarray(y_offset)
    arrays.update(truth_physical=y*y_scale+offset,mean_physical=np.asarray(mean)*y_scale+offset,
                  variance_physical=np.asarray(variance)*y_scale**2,
                  draws_physical=proper['draws']*y_scale+offset,
                  independent_draws_physical=proper['independent_draws']*y_scale+offset)
    arrays['score__gaussian_nll_physical']=proper['gaussian_nll']+np.log(y_scale)
    arrays['score__marginal_crps_physical']=proper['marginal_crps']*y_scale
    arrays['score__energy_physical']=proper['energy']*y_scale
    report=circuit_summary(context,structure_ids,seed)
    report.update(target_fingerprint=target_fingerprint,n_contexts=len(y),n_scoring_futures=int(mask.sum()),n_contexts_floor_unavailable=int((~good).sum()),interpretation='state fork floor and combined observation/compression/fitting gap only; no separation in nonlinear biophysical data',distribution='diagonal Gaussian; variance is EEG-conditioned predictive uncertainty, not Sigma_C',energy_definition='Euclidean distance divided by sqrt(output dimension); independent draw-pair correction',mc_precision='future-level standard error and delete-one jackknife estimates; separate from circuit bootstrap',negative_excess_retained=True)
    return report,arrays
