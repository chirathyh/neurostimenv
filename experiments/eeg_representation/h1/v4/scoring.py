"""Matched-contract risk, marginal calibration, and circuit-level inference.

The state-conditioned floor is a panel reference; it is not an EEG Bayes floor.
Paired-regime future Monte Carlo uncertainty is reported separately from population
uncertainty. Neither coordinate count nor optimizer repetitions increases N.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from scipy.stats import norm
from ..artifacts import atomic_npz
from ...storage import atomic_json,file_hash,read_json
from ..targets import canonical_hash
from ..scoring import fork_risk_components,circuit_summary
from .models import load_panel,circuit_weights


def assert_risk_contract(target,prediction):
    def canonical_names(value):
        result=dict(value)
        for old,new in [('targetphysicalhash','target_physical_hash'),('acquisitionhash','acquisition_hash')]:
            if old in result:
                if new in result and result[new]!=result[old]:raise ValueError('Conflicting risk-contract aliases')
                result[new]=result.pop(old)
        return result
    target=canonical_names(target);prediction=canonical_names(prediction)
    required=('target_physical_hash','population_id','acquisition_hash','normalization_id','units','dimension','support')
    for key in required:
        if key not in target or key not in prediction: raise ValueError('Risk contract missing '+key)
    if canonical_hash(target)!=canonical_hash(prediction):
        raise ValueError('Incompatible target/population/units/support/conditioning: refuse risk subtraction')


def gaussian_diagnostics(futures,mean,variance,mask,*,y_scale=1.):
    y=np.asarray(futures,float);mean=np.asarray(mean,float);var=np.asarray(variance,float);mask=np.asarray(mask,bool)
    if y.ndim!=3 or mean.shape!=(len(y),y.shape[-1]) or var.shape!=mean.shape:
        raise ValueError('Gaussian mean/variance shape mismatch')
    if np.any(var<=0) or not np.isfinite(var).all() or not np.isfinite(y[mask]).all():
        raise ValueError('Finite targets and positive variances required')
    if not np.isfinite(y_scale) or y_scale<=0: raise ValueError('Positive physical affine scale required')
    sd=np.sqrt(var); z=(y-mean[:,None])/sd[:,None]; pit=ndtr(z)
    nll=.5*(np.log(2*np.pi*var[:,None])+z*z)
    crps=sd[:,None]*(z*(2*pit-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    arrays=dict(pit=np.where(mask[...,None],pit,np.nan),standardized_residual=np.where(mask[...,None],z,np.nan),
        marginal_nll=np.where(mask[...,None],nll,np.nan),marginal_crps=np.where(mask[...,None],crps,np.nan),
        marginal_nll_physical=np.where(mask[...,None],nll+np.log(y_scale),np.nan),
        joint_diagonal_nll=np.where(mask,nll.sum(-1),np.nan),
        joint_diagonal_nll_physical=np.where(mask,nll.sum(-1)+y.shape[-1]*np.log(y_scale),np.nan),
        marginal_crps_physical=np.where(mask[...,None],crps*y_scale,np.nan))
    def average(a):return np.where(mask[...,None],a,0).sum((1,2))/(mask.sum(1)*y.shape[-1])
    context=dict(gaussian_nll=average(nll),gaussian_nll_physical=average(nll+np.log(y_scale)),
                 marginal_crps=average(crps),marginal_crps_physical=average(crps*y_scale))
    for level in (50,80,95):
        half=norm.ppf((1+level/100)/2)*sd
        coverage=np.abs(y-mean[:,None])<=half[:,None]
        arrays[f'coverage_{level}']=np.where(mask[...,None],coverage,np.nan)
        arrays[f'interval_width_{level}']=2*half
        context[f'coverage_{level}']=average(coverage)
        context[f'width_{level}']=(2*half).mean(-1)
        context[f'width_{level}_physical']=context[f'width_{level}']*y_scale
    # Lag products describe remaining target-coordinate dependence, not joint calibration.
    for lag in (1,2,4,8):
        if y.shape[-1]>lag:
            product=z[...,:-lag]*z[...,lag:]
            context[f'residual_coordinate_lag{lag}_product']=np.where(mask[...,None],product,0).sum((1,2))/(mask.sum(1)*(y.shape[-1]-lag))
    arrays['pit_bin_edges']=np.linspace(0,1,11)
    arrays['pit_histogram']=np.histogram(pit[np.broadcast_to(mask[...,None],pit.shape)],bins=arrays['pit_bin_edges'])[0]
    return context,arrays


def paired_regime_mc(futures,prediction,structure_ids,future_mask=None):
    """Delete a matching future index across ALL regimes/contexts of a circuit.

    Futures are aligned by their recorded index. Different circuit streams are
    independent. Unequal M across paired contexts fails rather than silently
    treating regimes as independent. M<3 has no floor/excess jackknife SE.
    """
    y=np.asarray(futures,float);p=np.asarray(prediction,float);ids=np.asarray(structure_ids)
    mask=np.ones(y.shape[:2],bool) if future_mask is None else np.asarray(future_mask,bool)
    rows=[];structures=np.unique(ids);totals={k:0. for k in ('mean_risk','state_floor','corrected_excess')};available=True
    for circuit in structures:
        take=ids==circuit; yy=y[take]; pp=p[take]; mm=mask[take]
        if not np.all(mm==mm[0]): raise ValueError('Paired regime jackknife requires aligned scoring pools within circuit')
        active=np.flatnonzero(mm[0]); yy=yy[:,active];m=len(active)
        if m<3:
            available=False;rows.append(dict(structure_id=str(circuit),n_futures=m,status='unavailable: need >=3 paired futures'));continue
        estimates={k:[] for k in totals}
        for j in range(m):
            scores=fork_risk_components(np.delete(yy,j,axis=1),pp)
            for key in estimates:estimates[key].append(float(scores[key].mean()))
        se={key:float(np.sqrt((m-1)/m*np.sum((np.asarray(v)-np.mean(v))**2))) for key,v in estimates.items()}
        for key in totals:totals[key]+=se[key]**2/len(structures)**2
        rows.append(dict(structure_id=str(circuit),n_futures=m,status='complete',jackknife_standard_error=se,
            delete_one_estimates=estimates))
    return dict(status='complete' if available else 'unavailable',per_circuit=rows,
        pooled_conditional_mc_se={k:np.sqrt(v) if available else None for k,v in totals.items()},
        interpretation='conditional future Monte Carlo; synchronized delete-one index across regimes within circuit, then independent circuit-stream variances; not circuit population uncertainty')


def hierarchical_circuit_inference(per_future_values,structure_ids,context_ids,*,bootstrap=1000,seed=0):
    """Nested circuit/context/future bootstrap with common draws across regimes.

    Prefix IDs are obtained from context IDs without the final regime token.
    The top-level independent sampling unit is always the circuit. This optional
    bootstrap combines panel and future variation and is labeled separately from
    fixed-context Monte Carlo precision. Rank/noise limits are not power claims.
    """
    a=np.asarray(per_future_values,float);sid=np.asarray(structure_ids);ctx=np.asarray(context_ids).astype(str)
    if a.ndim!=2 or len(a)!=len(sid) or len(a)!=len(ctx) or not np.isfinite(a).all():
        raise ValueError('Expected finite context-by-future values and IDs')
    circuits=np.unique(sid);point=float(np.mean([a[sid==s].mean() for s in circuits]))
    if len(circuits)<2:return dict(status='unavailable',mean=point,n_circuits=len(circuits),reason='fewer than two independent circuits')
    rng=np.random.default_rng(seed);sample=[]
    groups={s:{} for s in circuits}
    for i,(s,c) in enumerate(zip(sid,ctx)):
        prefix=c.rsplit('-R',1)[0] if '-R' in c else c
        groups[s].setdefault(prefix,[]).append(i)
    for _ in range(int(bootstrap)):
        circuit_scores=[]
        for s in rng.choice(circuits,len(circuits),replace=True):
            prefixes=list(groups[s]);future_ix=rng.integers(0,a.shape[1],a.shape[1]);prefix_scores=[]
            for index in rng.integers(0,len(prefixes),len(prefixes)):
                rows=groups[s][prefixes[index]];prefix_scores.append(a[rows][:,future_ix].mean())
            circuit_scores.append(np.mean(prefix_scores))
        sample.append(np.mean(circuit_scores))
    return dict(status='complete',mean=point,n_circuits=len(circuits),bootstrap_95=np.quantile(sample,[.025,.975]).tolist(),
        method='resample circuits, nested prefixes; synchronized future indices across all regimes/prefixes in each sampled circuit',
        interpretation='exploratory hierarchical panel uncertainty; optimizer seeds never independent units')


def _record(root,path):return dict(path=path.relative_to(root).as_posix(),sha256=file_hash(path),bytes=path.stat().st_size)


def score(config,root,job_ids=None,resume=False):
    root=Path(root);index=read_json(root/'models/manifest.json');available=[x['id'] for x in index['jobs']]
    jobs=available if job_ids is None else list(job_ids)
    if set(jobs)-set(available):raise ValueError('Model job not fitted')
    results=[]
    for job in jobs:
        panel,manifest=load_panel(root,job);models=read_json(root/'models'/job/'manifest.json')
        assert_risk_contract(manifest['risk_contract'],models['risk_contract'])
        signature=canonical_hash(dict(model_manifest=file_hash(root/'models'/job/'manifest.json'),
            panel=file_hash(root/'derived'/job/'evaluation.npz'),source=file_hash(Path(__file__))))
        directory=root/'scores'/job;marker=directory/'summary.json'
        if marker.exists():
            if not resume: raise FileExistsError('Scores exist; use resume')
            old=read_json(marker)
            if old['signature']!=signature:raise ValueError('Score resume inputs/source mismatch')
            for record in old['files']:
                if file_hash(root/record['path'])!=record['sha256']:raise ValueError('Score artifact checksum mismatch')
            results.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=True));continue
        ev=panel['evaluation'];y=ev['Y'];mask=ev['future_mask'];scale=manifest['normalization']['y_scale'];rows=[];files=[]
        model_file_hashes={x['path']:x['sha256'] for x in models['files']}
        for candidate in models['candidates']:
            if candidate['status']!='complete':continue
            name=candidate['id'];path=root/'models'/job/name/'evaluation.npz'
            if file_hash(path)!=model_file_hashes[str(path.relative_to(root))]:raise ValueError('Prediction artifact checksum mismatch')
            with np.load(path,allow_pickle=False) as f:pred={k:f[k] for k in f.files}
            assert_risk_contract(manifest['risk_contract'],json.loads(str(pred['risk_contract_json'])))
            if not np.array_equal(pred['context_id'],ev['context_id']):raise ValueError('Prediction contexts/order mismatch')
            components=fork_risk_components(y,pred['mean'],mask);base=np.broadcast_to(pred['baseline_mean'],pred['mean'].shape)
            r0=fork_risk_components(y,base,mask)['mean_risk'];components['r0']=r0
            for key in ('mean_risk','state_floor','corrected_excess','r0'):components[key+'_physical']=components[key]*scale**2
            ratio=float(circuit_weights(ev['structure_id'])@components['mean_risk']/(circuit_weights(ev['structure_id'])@r0))
            for calibration in ('uncalibrated','calibrated','matched_calibrated'):
                context,arrays=gaussian_diagnostics(y,pred['mean'],pred['variance_'+calibration],mask,y_scale=scale)
                context.update({k:v for k,v in components.items() if np.asarray(v).shape==(len(y),)})
                report=circuit_summary(context,ev['structure_id'],bootstrap=int(config.get('scoring',{}).get('circuit_bootstrap',1000)))
                report.update(candidate=name,calibration=calibration,mean_risk_over_mean_baseline=ratio,
                    skill=1-ratio,selected_on_development=name==models['selected_candidate'],
                    state_reference_interpretation='physical-state conditional variance in this panel, not EEG-history Bayes floor; combined gaps only',
                    uncertainty_conditioning='Gaussian variance predicts Y|observed history; state floor conditions on full causal state',
                    density_units=manifest['risk_contract']['units'],physical_marginal_nll_jacobian='add log(y_scale); joint diagonal adds dimension*log(y_scale)',
                    calibration_scope='marginal diagonal Gaussian calibration; not joint calibration',
                    variance_family=({'uncalibrated':'raw learned heteroscedastic diagonal for Gaussian MLP; training-residual constant diagonal for MSE/baselines, except unit-variance reference',
                        'calibrated':'development-fitted coordinatewise multiplier of raw variance; equal calibration access but unequal final heteroscedastic expressivity',
                        'matched_calibrated':'development-fitted context-independent diagonal variance for EVERY mean; exactly matched family and access'})[calibration],
                    primary_matched_variance_comparison=calibration=='matched_calibrated',
                    coordinate_temporal_scope='lag diagnostics refer to target coordinates; Fourier/scalar coordinates are not a time axis')
                arrays.update({f'context__{k}':v for k,v in context.items()});arrays.update(context_id=ev['context_id'],structure_id=ev['structure_id'],
                    future_id=ev['future_id'],future_mask=mask,mean=pred['mean'],variance=pred['variance_'+calibration],truth=y,
                    target_fingerprint=np.asarray(manifest['target_fingerprint']))
                numeric=directory/f'{name}__{calibration}.npz';atomic_npz(numeric,**arrays);files.append(_record(root,numeric));rows.append(report)
            paired=paired_regime_mc(y,pred['mean'],ev['structure_id'],mask)
            reportpath=directory/f'{name}__conditional_mc.json';atomic_json(reportpath,paired);files.append(_record(root,reportpath))
        selected=next(c for c in models['candidates'] if c['id']==models['selected_candidate'])
        selectedfile=root/'models'/job/selected['id']/'evaluation.npz'
        with np.load(selectedfile,allow_pickle=False) as f: mean=f['mean']
        perfuture=((y-mean[:,None])**2).mean(-1)
        hierarchical=hierarchical_circuit_inference(perfuture,ev['structure_id'],ev['context_id'],
            bootstrap=int(config.get('scoring',{}).get('hierarchical_bootstrap',1000))) if mask.all() else dict(status='unavailable',reason='ragged future pools')
        report=dict(status='complete',signature=signature,job_id=job,target_fingerprint=manifest['target_fingerprint'],
            risk_contract=manifest['risk_contract'],selected_candidate=models['selected_candidate'],rows=rows,files=files,
            hierarchical_selected_risk=hierarchical,negative_excess_retained=True,
            scientific_question='H1 prediction-risk accounting and H4 target/prediction scale',
            warning='Risk ratios are within the SAME target/operator/population. Raw standardized MSE is not compared across targets.')
        atomic_json(marker,report);results.append(dict(id=job,status='complete',path=str(marker.relative_to(root)),resumed=False))
    result=dict(status='complete',jobs=results);atomic_json(root/'scores/manifest.json',result);return result
