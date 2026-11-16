"""Outcome-blind v6 precision certificate from verified inspected evidence.

The historical full numeric archive is planning evidence, never fresh test data.
Missing Phase-0 full-history qualification keeps the certificate provisional.
Gaussian upper-SD bounds are assumptions, accompanied by skew/outlier controls.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import io
import json
import zipfile
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, t, norm, binomtest
from .statistics import stable_mean,stable_sd,family_a,family_b,ratio_of_means_interval,select_primary_futures

HISTORICAL_ARCHIVE_SHA256='6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f'
TARGET_HALF_WIDTHS={'B_percentage_points':.5,'A_task_bits':.10,'A_reconstruction_nmse':.01}


class NumericArchive:
    """Read deduplicated lossless arrays without extraction or model unpickling."""
    def __init__(self,path,expected_sha256=HISTORICAL_ARCHIVE_SHA256):
        self.path=Path(path);h=hashlib.sha256()
        with self.path.open('rb')as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        self.sha256=h.hexdigest()
        if expected_sha256 is not None and self.sha256!=expected_sha256:
            raise ValueError('Historical source archive hash mismatch')
        self.z=zipfile.ZipFile(self.path);self.manifest=self.json('EXPORT_MANIFEST.json')
        self.mapping={r['original_path']:r['descriptor_path']for r in self.manifest['transformed_numeric']}
    def close(self):self.z.close()
    def json(self,path):return json.loads(self.z.read(path))
    def arrays(self,path,keys=None):
        desc=self.json(self.mapping[path]);out={}
        for record in desc['arrays']:
            if keys is None or record['key']in keys:
                out[record['key']]=np.load(io.BytesIO(self.z.read(record['array_path'])),allow_pickle=False)
        return out


def _group(x,ids):
    sid=np.asarray(ids).astype(str);a=np.asarray(x,np.float64)
    return np.asarray([stable_mean(a[sid==s])for s in sorted(set(sid))],np.float64)


def upper_sd(sd,n,*,confidence=.95):
    if sd<0 or n<2 or not 0<confidence<1:raise ValueError('Invalid SD confidence bound inputs')
    inflation=np.sqrt((n-1)/chi2.ppf(1-confidence,n-1))
    return {'upper_sd':float(sd*inflation),'inflation':float(inflation),'confidence':confidence,
            'assumption':'one-sided chi-square upper SD under independent Gaussian circuit effects; not distribution-free'}


def planned_half_width(sd,N,*,confidence=.975):
    if N<3 or sd<0:raise ValueError('N>=3 and nonnegative SD required')
    return float(t.ppf((1+confidence)/2,N-1)*sd/np.sqrt(N))


def required_n(sd,half_width,*,confidence=.975,max_n=1000000):
    if half_width<=0:raise ValueError('Positive precision target required')
    if planned_half_width(sd,max_n,confidence=confidence)>half_width:return None
    lo,hi=3,max_n
    while lo<hi:
        mid=(lo+hi)//2
        if planned_half_width(sd,mid,confidence=confidence)<=half_width:hi=mid
        else:lo=mid+1
    return lo


def _component(name,values,half_width,*,family,estimand,source):
    x=np.asarray(values,np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Per-circuit precision values required')
    sd=stable_sd(x);bound=upper_sd(sd,len(x))
    return dict(name=name,family=family,estimand=estimand,source=source,n_pilot_circuits=len(x),
        circuit_values=x.tolist(),sample_sd=sd,upper_sd=bound['upper_sd'],upper_sd_inflation=bound['inflation'],
        upper_sd_assumption=bound['assumption'],target_half_width=half_width,
        standardized_upper_sd=bound['upper_sd']/half_width)


def _risk_component(name,model,baseline,source):
    out=family_b(model,baseline)
    return _component(name,out['influence_values'],.5,family='B',
        estimand='percentage-point influence of ratio of equal-circuit model/baseline risks',source=source)


def archive_components(archive):
    """Recompute old A1 risks and BOTH newly-primary Feature18/PCA components."""
    e=archive
    job='A1__T0_l2_d32';manifest=e.json(f'evidence/models/{job}/manifest.json');selected=manifest['selected_candidate']
    ev=e.arrays(f'evidence/derived/{job}/evaluation.npz',keys=['Y','structure_id','future_id'])
    p=e.arrays(f'evidence/models/{job}/{selected}/evaluation.npz',keys=['mean','baseline_mean'])
    y=np.asarray(ev['Y'],np.float64);pred=np.asarray(p['mean'],np.float64);base=np.asarray(p['baseline_mean'],np.float64)
    sid=ev['structure_id'];components=[]
    primary=select_primary_futures(ev)['Y']
    if y.shape[1]!=8:raise ValueError('Historical reference requires the complete frozen eight-future panel')
    for count in (8,4):
        scored=y if count==8 else primary
        losses=((scored-pred[:,None])**2).mean((1,2));baseloss=((scored-base)**2).mean((1,2))
        components.append(_risk_component(f'historical_A1_{count}_futures',_group(losses,sid),_group(baseloss,sid),e.sha256))
    prefix='evidence/retention/banks/';variant='d8_B32_quantile_005_995'
    carriers={};reconstruction={}
    for bank in ('feature18','pca'):
        a=e.arrays(f'{prefix}{bank}/{variant}/carrier/labels_all/predictions.npz')
        logits=np.asarray(a['evaluation_selected_prediction'],np.float64);classes=np.asarray(a['class_order']);truth=np.asarray(a['evaluation_target'])
        matches=truth[:,None]==classes[None,:]
        if not np.all(matches.sum(1)==1):raise ValueError('Carrier class order/coverage mismatch')
        loss=(logsumexp(logits,axis=1)-logits[np.arange(len(logits)),matches.argmax(1)])/np.log(2.)
        if not np.allclose(loss,a['evaluation_selected_context_loss'],rtol=1e-10,atol=1e-12):raise ValueError('Carrier saved loss reproduction failed')
        carriers[bank]=(a,_group(loss,a['evaluation_structure_id']))
        reconstruction[bank]=e.arrays(f'{prefix}{bank}/{variant}/waveform_reconstruction/labels_all/predictions.npz')
    a,feature=carriers['feature18'];b,pca=carriers['pca']
    if not np.array_equal(a['evaluation_context_id'],b['evaluation_context_id']):raise ValueError('Carrier pairing mismatch')
    task=pca-feature
    components.append(_component('new_primary_PCA_minus_feature18_task',task,.10,family='A',estimand='carrier CE bits, PCA minus feature18',source=e.sha256))
    panel=e.arrays('evidence/derived/A0__T0_l10_d128/evaluation.npz',keys=['X','structure_id','context_id'])
    train=e.arrays('evidence/derived/A0__T0_l10_d128/train.npz',keys=['X'])
    x=np.asarray(panel['X'],np.float64);trainmean=np.asarray(train['X'],np.float64).mean(0)
    basemse=_group(((x-trainmean)**2).mean(1),panel['structure_id']);risk={}
    for bank,a in reconstruction.items():
        if not np.array_equal(a['evaluation_context_id'],panel['context_id']):raise ValueError('Reconstruction pairing mismatch')
        losses=((np.asarray(a['evaluation_selected_prediction'],np.float64)-x)**2).mean(1)
        if not np.allclose(losses,a['evaluation_selected_context_loss'],rtol=1e-6,atol=1e-8):raise ValueError('Reconstruction saved loss reproduction failed')
        risk[bank]=_group(losses,panel['structure_id'])
    rec=ratio_of_means_interval(risk['feature18']-risk['pca'],basemse)
    components.append(_component('new_primary_feature18_minus_PCA_reconstruction',rec['influence_values'],.01,family='A',
        estimand='NMSE difference ratio influence including shared baseline denominator uncertainty',source=e.sha256))
    # Previously primary forecast contrast remains a secondary design comparison.
    forecasts=[]
    for seed in range(3):
        a=e.arrays(f'{prefix}forecast__seed_{seed}/{variant}/carrier/labels_all/predictions.npz')
        logits=np.asarray(a['evaluation_selected_prediction'],np.float64);classes=np.asarray(a['class_order']);truth=np.asarray(a['evaluation_target'])
        index=(truth[:,None]==classes).argmax(1)
        losses=(logsumexp(logits,axis=1)-logits[np.arange(len(logits)),index])/np.log(2.)
        forecasts.append(_group(losses,a['evaluation_structure_id']))
    historical=np.mean(forecasts,axis=0)-feature
    components.append(_component('historical_forecast_minus_feature18_secondary',historical,.10,family='secondary',
        estimand='seed-average carrier CE difference bits, secondary only',source=e.sha256))
    reproduction=dict(source_archive_sha256=e.sha256,selected_historical_A1_candidate=selected,
        historical_forecast_feature_difference_bits=stable_mean(historical),new_PCA_task_difference_bits=stable_mean(task),
        new_PCA_reconstruction_difference_nmse=rec['estimate'],
        reconstruction_denominator=stable_mean(basemse),
        reconstruction_denominator_policy='same fitted training-mean history predictor; equal-circuit evaluation risk, influence includes denominator uncertainty',
        historical_A1_ratio_influence_sd=components[0]['sample_sd'],
        legacy_three_seeds_scope='historical secondary evidence only; v6 neural primary requires five fitting repeats')
    return components,reproduction


def _standard_noise(rng,law,shape):
    if law=='Gaussian':return rng.normal(size=shape)
    if law=='skew_exponential':return rng.exponential(size=shape)-1.
    if law=='rare_outlier':
        # Standardized Bernoulli rare positive jump plus Gaussian component;
        # known zero mean and unit variance, with high skew/kurtosis.
        p=.01;jump=12.;variance=1.+jump**2*p*(1-p)
        return (rng.normal(size=shape)+jump*(rng.binomial(1,p,size=shape)-p))/np.sqrt(variance)
    raise ValueError(law)


def _mc_interval(success,n):
    c=binomtest(int(success),int(n)).proportion_ci(confidence_level=.95,method='exact')
    return [float(c.low),float(c.high)]


def synthetic_controls(*,N=192,repetitions=2000,seed=619,sigma_b=2.831,sigma_task=.622,sigma_reconstruction=.05):
    """Independent synthetic circuit experiments, Gaussian/skew/outlier stresses.

    Expected effects in power cases are explicitly hypothetical, not estimates of
    biological efficacy. Nominal large-N coverage is tested, not assumed exact
    under skew. Family-A boundary scenarios include only one null component.
    """
    if N<4 or repetitions<100:raise ValueError('Meaningful controls require N>=4 and >=100 independent repeats')
    rng=np.random.default_rng(seed);rows=[];warnings=[];q=t.ppf(.9875,N-1);qa=t.ppf(.975,N-1)
    for law in ('Gaussian','skew_exponential','rare_outlier'):
        z=_standard_noise(rng,law,(repetitions,N));base=np.exp(.2*rng.normal(size=(repetitions,N))-.02)
        for true_g in (0.,2.,-2.):
            model=base*(1-true_g/100)-sigma_b*z/100
            # Extreme requested SD can exceed this additive control's useful
            # range. Preserve the failed control and block freeze; never clip or
            # reject individual experiments and thereby change the assumed law.
            if np.any(model<0):
                rows.append(dict(family='B',law=law,true_G_pp=true_g,status='unavailable',
                    reason='additive synthetic risk construction produced negative risks at this scale',
                    negative_risk_count=int(np.sum(model<0)),minimum_risk=float(model.min())))
                warnings.append(f'{law}: B {true_g}pp synthetic risk construction invalid at the supplied SD; coverage unavailable')
                continue
            bm=base.mean(1);ratio=model.mean(1)/bm;estimate=100*(1-ratio)
            influence=-100*(model-ratio[:,None]*base)/bm[:,None]
            se=influence.std(1,ddof=1)/np.sqrt(N);lo=estimate-q*se;hi=estimate+q*se
            hits=(lo<=true_g)&(hi>=true_g);equiv=(lo>=-1)&(hi<=1)
            improving=lo>1;worsening=hi<-1
            rows.append(dict(family='B',law=law,true_G_pp=true_g,status='complete',coverage=float(hits.mean()),
                coverage_mc95=_mc_interval(hits.sum(),repetitions),equivalence_rate=float(equiv.mean()),
                meaningful_improvement_rate=float(improving.mean()),meaningful_worsening_rate=float(worsening.mean()),
                inconclusive_rate=float((~(equiv|improving|worsening)).mean()),
                empirical_sd_pp=float(estimate.std(ddof=1)),mean_reported_se_pp=float(se.mean())))
        z2=_standard_noise(rng,law,(repetitions,N))
        for task_effect,rec_effect,label in [(.20,.04,'task boundary null'),(.40,.02,'reconstruction boundary null'),
                                              (.20,.02,'both boundary null'),(.30,.03,'hypothetical practical alternative')]:
            task=task_effect+sigma_task*z;rec=rec_effect*base+sigma_reconstruction*z2
            tm=task.mean(1);bm=base.mean(1);rm=rec.mean(1)/bm;ts=task.std(1,ddof=1)/np.sqrt(N)
            rec_influence=(rec-rm[:,None]*base)/bm[:,None]
            rs=rec_influence.std(1,ddof=1)/np.sqrt(N)
            lower_t=tm-qa*ts;lower_r=rm-qa*rs
            tp=t.sf((tm-.20)/ts,N-1);rp=t.sf((rm-.02)/rs,N-1)
            passes=(np.maximum(tp,rp)<=.025)&(lower_t>.20)&(lower_r>.02)
            ct=lower_t<=task_effect;cr=lower_r<=rec_effect
            rows.append(dict(family='A',law=law,scenario=label,true_task_bits=task_effect,true_reconstruction_nmse=rec_effect,
                conjunction_rejection_rate=float(passes.mean()),rejection_mc95=_mc_interval(passes.sum(),repetitions),
                task_one_sided_coverage=float(ct.mean()),reconstruction_one_sided_coverage=float(cr.mean()),
                reconstruction_estimator='ratio of paired raw MSE differences to estimated mean baseline; denominator influence included',
                interpretation='type-I diagnostic'if 'null'in label else 'power at hypothetical fixed effects'))
    for row in rows:
        if row['family']=='B'and row.get('status')=='complete'and row['coverage']<.95:
            warnings.append(f"{row['law']}: B {row['true_G_pp']}pp coverage below .95; asymptotic interval needs caution")
        if row['family']=='A'and 'null'in row['scenario']and row['conjunction_rejection_rate']>.04:
            warnings.append(f"{row['law']}: A {row['scenario']} rejection above .04")
    return dict(status='complete'if all(r.get('status','complete')=='complete'for r in rows)else 'incomplete',
        scope='independent cheap synthetic experiments, not biophysical outcomes',N=N,
        repetitions=repetitions,seed=seed,assumed_sd=dict(B_pp=sigma_b,A_task_bits=sigma_task,A_reconstruction_nmse=sigma_reconstruction),
        results=rows,warnings=warnings,universal_coverage_claim=False,
        nominal_probability_note='t intervals exact only for Gaussian paired mean effects; ratio influence is asymptotic even there')


def build_precision_certificate(components,*,N=192,phase0_complete=False,phase0_reasons=(),
                                simulation_repetitions=2000):
    if N<3 or not components:raise ValueError('Nonempty evidence and N>=3 required')
    rows=[]
    for comp in components:
        c=dict(comp);c.update(planned_N=int(N),planned_t_half_width=planned_half_width(c['upper_sd'],N),
            required_N=required_n(c['upper_sd'],c['target_half_width']))
        c['precision_sufficient']=c['planned_t_half_width']<=c['target_half_width']
        c['required_for_primary_precision']=c['family']in ('A','B')
        rows.append(c)
    required=[r for r in rows if r['required_for_primary_precision']]
    worst=max(required,key=lambda x:x['standardized_upper_sd'])
    sufficient=all(r['precision_sufficient']for r in required)
    sigma_b=max(r['upper_sd']for r in rows if r['family']=='B')
    sigma_task=max(r['upper_sd']for r in rows if r['family']=='A'and 'task'in r['name'])
    sigma_rec=max(r['upper_sd']for r in rows if r['family']=='A'and 'reconstruction'in r['name'])
    controls=synthetic_controls(N=N,repetitions=simulation_repetitions,sigma_b=sigma_b,sigma_task=sigma_task,sigma_reconstruction=sigma_rec)
    flags=list(phase0_reasons)
    if not phase0_complete:flags.append('Qualified Phase-0 full-history A1 learner uncertainty unavailable; historical tail-model evidence cannot freeze the new learner')
    if not sufficient:flags.append('Default N does not meet every frozen precision target under conservative pilot SD assumptions')
    if controls['warnings']:flags.append('Synthetic stress diagnostic has coverage/type-I warnings; review before freeze')
    eligible=bool(phase0_complete and not phase0_reasons and sufficient and not controls['warnings'])
    return dict(status='complete'if eligible else 'provisional',eligible_to_freeze=eligible,planned_N=int(N),
        default_N=192,precision_sufficient=sufficient,phase0_complete=phase0_complete,flags=flags,
        worst_standardized_uncertainty_component=worst['name'],components=rows,
        minimum_N_for_all_targets=max(r['required_N']or 1000001 for r in required),
        secondary_precision_policy='Historical forecast/Feature18 remains descriptive, with width reported; it does not hard-gate the two new primary families',
        t_quantile_policy='97.5% two-sided (.9875) for every planning width, conservative relative to A one-sided .975',
        planning_assumptions='upper-SD inflation assumes Gaussian pilot circuits; not a precision/power guarantee for new learners, energy targets or humans',
        fresh_test_policy='Select N before generating/unblinding test outcomes; never increase based on observed new p-value',
        energy_power_status='unavailable: no hypothetical energy effect is powered from signed-waveform evidence',
        controls=controls)


def certificate_from_archive(archive,*,N=192,phase0_comparisons=None,simulation_repetitions=2000,
                             expected_sha256=HISTORICAL_ARCHIVE_SHA256):
    e=NumericArchive(archive,expected_sha256=expected_sha256)
    try:components,reproduction=archive_components(e)
    finally:e.close()
    qualified=False;reasons=[]
    for comp in phase0_comparisons or []:
        if comp.get('status')!='complete':reasons.append(f"Phase-0 {comp.get('name','unnamed')} is incomplete");continue
        if comp.get('selection_split')!='development':raise ValueError('Phase-0 learner was not selected on development only')
        cid=comp.get('circuit_ids',[]);a=comp['model_circuit_risks'];b=comp['baseline_circuit_risks']
        if len(cid)!=len(b)or len(set(map(str,cid)))!=len(cid):raise ValueError('Phase-0 circuit alignment/count required')
        aa=np.asarray(a,np.float64)
        if aa.ndim==2:
            if aa.shape!=(5,len(cid)):raise ValueError('Five fitting repeats must align with circuit losses')
            aa=aa.mean(0)
        if aa.shape!=(len(cid),):raise ValueError('Phase-0 risk dimensions differ')
        components.append(_risk_component('phase0_'+comp['name'],aa,b,comp.get('source_hash','declared phase0 evidence')))
        if comp.get('is_full_history')is True and comp.get('target')=='A1__T0_l2_d32':qualified=True
    out=build_precision_certificate(components,N=N,phase0_complete=qualified,phase0_reasons=reasons,
                                    simulation_repetitions=simulation_repetitions)
    out['historical_reproduction']=reproduction
    return out


def certificate(config,root,source,phase0_root=None):
    """Bind precision inputs to the complete current Phase-0 qualification.

    A numerical full-history risk row alone cannot certify the complete learner
    grid, positive control, acquisition tests and frozen evidence dependencies.
    """
    phase0=None
    if phase0_root is not None:
        path=Path(phase0_root)/'precision_inputs.json'
        if path.exists():
            raw=path.read_bytes();phase0=json.loads(raw).get('comparisons',[])
            source_hash=hashlib.sha256(raw).hexdigest()
            phase0=[{**item,'source_hash':source_hash}for item in phase0]
    cfg=config.get('precision',{});protocol=config.get('protocol',{})
    for key,expected in [('forecast_halfwidth_pp',.5),('carrier_halfwidth_bits',.10),('reconstruction_halfwidth_nmse',.01)]:
        if key in protocol and float(protocol[key])!=expected:
            raise ValueError(f'Precision target {key} differs from the frozen v6 planning contract')
    if Path(source).is_dir():
        raise ValueError('Precision recomputation needs the verified full standard numeric source ZIP; pass --source-archive, not the native directory')
    out=certificate_from_archive(source,N=int(cfg.get('N',protocol.get('test_circuits',config.get('evaluation_circuits',192)))),
        phase0_comparisons=phase0,simulation_repetitions=int(cfg.get('simulation_repetitions',2000)))
    from .planning import qualify_phase0
    if phase0_root is None:
        qualification=dict(status='incomplete',passed=False,missing=['Phase-0 root not supplied'])
    else:
        try:qualification=qualify_phase0(config,phase0_root)
        except (OSError,ValueError,KeyError) as exc:
            qualification=dict(status='incomplete',passed=False,root=str(Path(phase0_root).resolve()),
                               missing=['Phase-0 artifact verification failed: '+str(exc)])
    out['phase0_qualification']=qualification
    out['phase0_evidence_hash']=qualification.get('evidence_hash')
    if not qualification.get('passed'):
        out['phase0_complete']=False
        out['eligible_to_freeze']=False
        out['status']='provisional'
        out['flags'].extend('Incomplete Phase-0 qualification: '+reason for reason in qualification['missing'])
    root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/'precision_certificate.json';tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    return out


def propose_design(config,certificate):
    """Return a registered minimum-N candidate before any fresh test outcomes.

    This proposes a new config/root only. It never freezes, generates data, edits
    the default N, weakens precision or overrides the measured resource caps.
    """
    from copy import deepcopy
    from .config import resolve,counts,registry_check
    from ..artifacts import ROOT
    from ..targets import canonical_hash
    if config.get('claim_status')!='prospective_confirmation':
        raise ValueError('A candidate confirmation protocol is required')
    original=int(config['protocol']['test_circuits']);required=int(certificate['minimum_N_for_all_targets'])
    proposed=max(original,required)
    registry_path=ROOT/'configs/eeg_representation/v6/seed_registry.json'
    registry=json.loads(registry_path.read_text())
    ranges=[registry['ranges']['fresh_evaluation']]
    extension=registry['ranges'].get('fresh_evaluation_pretest_precision_extension')
    if extension is not None:ranges.append(extension)
    registered_ids=sorted({i for lo,hi in ranges for i in range(lo,hi+1)})
    if proposed>len(registered_ids):
        return {'status':'unavailable','proposed_N':proposed,'default_N':192,'previous_N':original,
            'reason':'Required N exceeds the registered fresh evaluation range; reserve unused IDs and revise the bounded plan before proceeding',
            'execution_authorized':False,'precision_certificate_hash':canonical_hash(certificate)}
    candidate=deepcopy(config);candidate['protocol']['test_circuits']=proposed
    candidate['data']['structures']['evaluation']=registered_ids[:proposed]
    candidate=resolve(candidate);checked=registry_check(candidate)
    return {'status':'proposed'if proposed!=original else 'unchanged_candidate','proposed_N':proposed,
        'default_N':192,'previous_N':original,'resolved_config':candidate,'config_hash':canonical_hash(candidate),
        'counts':counts(candidate),'registry':checked,'precision_certificate_hash':canonical_hash(certificate),
        'driver':certificate['worst_standardized_uncertainty_component'],'execution_authorized':False,
        'phase0_complete':bool(certificate['phase0_complete']),
        'limitations':['A new output root, recomputed certificate, numerical qualification and measured resource plan are required.',
                       'If Phase 0 is incomplete this is a provisional count only; its new full-history uncertainty can increase N.',
                       'No circuit outcome, direction or p-value from fresh test data was used.']}
