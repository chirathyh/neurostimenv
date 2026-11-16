"""Fixed-population information/error accounting; fitted errors are not Bayes floors.

All empirical primary-panel rows use the same first four registered futures.
Circuit sampling intervals and conditional-future Monte Carlo diagnostics are
reported separately. Fitting-repeat losses, not predictions, are averaged.
"""
from pathlib import Path
import itertools
import json
import re
import numpy as np
from scipy.stats import norm
from .artifacts import atomic_json,atomic_npz,read_json,artifact_record,verify_record,canonical_hash,module_hash,resume_record
from .derivation import load_panel
from .models import prediction_record
from .statistics import aggregate_circuit_losses,select_primary_futures,ratio_of_means_interval,paired_ratio_difference
from ..v4.scoring import assert_risk_contract

KEYS=('model','baseline','floor','excess')


def _metadata(panel):
    sid=np.asarray(panel['structure_id']).astype(str);reg=np.asarray(panel['regime']).astype(str)
    cid=np.asarray(panel['context_id']).astype(str);prefix=[]
    if len(set(cid))!=len(cid):raise ValueError('Duplicate context IDs')
    for tag in cid:
        m=re.fullmatch(r's\d+-p(\d+)-.+',tag)
        if m is None:raise ValueError('Registered context prefix required')
        prefix.append(int(m.group(1)))
    prefix=np.asarray(prefix)
    aggregate_circuit_losses(np.zeros(len(cid)),sid,reg,prefix,
        expected_conditions=np.unique(reg),expected_prefixes=np.unique(prefix))
    return sid,reg,cid,prefix


def _aggregate(values,panel):
    sid,reg,_,prefix=_metadata(panel)
    return aggregate_circuit_losses(values,sid,reg,prefix)['values']


def _components(y,means,baseline):
    """Physical MSE; an unbiased floor requires at least two futures."""
    y=np.asarray(y,np.float64);means=np.asarray(means,np.float64);baseline=np.asarray(baseline,np.float64)
    if means.ndim==2:means=means[None]
    if y.ndim!=3 or means.ndim!=3 or means.shape[1:]!=(y.shape[0],y.shape[2]):raise ValueError('Target/prediction shape mismatch')
    baseline=np.broadcast_to(baseline,(len(y),y.shape[-1]))
    if not all(np.isfinite(x).all() for x in (y,means,baseline)):raise ValueError('Finite physical arrays required')
    risks=np.mean((y[None]-means[:,:,None,:])**2,axis=(0,2,3))
    base=np.mean((y-baseline[:,None,:])**2,axis=(1,2))
    if y.shape[1]<2:return dict(model=risks,baseline=base,floor=None,excess=None)
    floor=np.var(y,axis=1,ddof=1).mean(1)
    excess=np.mean((means-y.mean(1)[None])**2,axis=(0,2))-floor/y.shape[1]
    # Centering a large log-energy/voltage offset incurs absolute roundoff at
    # the original coordinate scale, even when residual MSE is very small.
    arithmetic_scale=max(float(np.max(np.abs(y)))**2,float(np.max(np.abs(means)))**2,float(np.max(risks)),1e-280)
    if not np.allclose(risks-floor,excess,rtol=1e-10,atol=128*np.finfo(float).eps*arithmetic_scale):
        raise ValueError('Finite-M risk identity failed')
    return dict(model=risks,baseline=base,floor=floor,excess=excess)


def future_jackknife(y,means,baseline,panel):
    """Joint R,B,V,E covariance from synchronized deletion within circuit/prefix.

    Ratios use gradients of the same joint estimates. Adding this MC variance
    again to the circuit variance would double-count observed future variation.
    """
    y=np.asarray(y,np.float64);means=np.asarray(means,np.float64)
    if means.ndim==2:means=means[None]
    baseline=np.broadcast_to(baseline,(len(y),y.shape[-1]));sid,reg,cid,prefix=_metadata(panel)
    m=y.shape[1];tags=np.asarray(panel['future_id']).astype(str)
    if tags.shape!=y.shape[:2]:raise ValueError('Future IDs must match targets')
    ordinals=[]
    for row in tags:
        matches=[re.search(r'(?:^|-)score-(\d+)$',x) for x in row]
        if not all(matches):raise ValueError('Registered scoring IDs required')
        numbers=[int(x.group(1)) for x in matches]
        if len(set(numbers))!=m:raise ValueError('Duplicate future ordinal')
        ordinals.append(numbers)
    if any(x!=ordinals[0] for x in ordinals):raise ValueError('Synchronized future order differs')
    if m<3:return {'status':'unavailable','reason':'At least three futures required for floor/excess delete-one variance','groups':[]}
    circuits=np.unique(sid);cov=np.zeros((4,4));groups=[]
    for circuit in circuits:
        regimes=np.unique(reg[sid==circuit])
        for pr in np.unique(prefix[sid==circuit]):
            take=np.flatnonzero((sid==circuit)&(prefix==pr))
            weights=np.asarray([1/(len(regimes)*np.sum((sid==circuit)&(reg==reg[k]))) for k in take])
            loo=[]
            for j in range(m):
                comp=_components(np.delete(y[take],j,axis=1),means[:,take],baseline[take])
                loo.append([float(weights@comp[key]) for key in KEYS])
            loo=np.asarray(loo);centered=loo-loo.mean(0);vc=(m-1)/m*centered.T@centered
            cov+=vc/len(circuits)**2
            groups.append(dict(structure_id=str(circuit),prefix=int(pr),future_ordinals=ordinals[0],
                delete_one_contributions=loo.tolist(),covariance=vc.tolist()))
    return {'status':'complete','metric_order':list(KEYS),'pooled_covariance':cov.tolist(),
        'pooled_standard_error':dict(zip(KEYS,np.sqrt(np.diag(cov)).tolist())), 'groups':groups,
        'scope':'Conditional Monte Carlo on fixed circuits; matching future ordinal deleted across regimes; independent prefixes/circuits combined with squared weights. Separate from circuit sampling interval; do not add twice.'}


def _interval(a,b,confidence=.975,multiplier=1):
    if len(a)<2:return dict(status='unavailable',reason='At least two independent circuits required')
    return dict(status='estimated',**ratio_of_means_interval(a,b,confidence=confidence,multiplier=multiplier))


def summarize(y,means,baseline,panel,*,method_id,status='measured',information_available='',method='',source=(),confidence=.975):
    """Public numeric API for meaningful tests and subset-specific diagnostics."""
    y=np.asarray(y,np.float64);comp=_components(y,means,baseline)
    circuits={k:None if v is None else _aggregate(v,panel) for k,v in comp.items()}
    c=circuits;den=float(c['baseline'].mean())
    if den<=0:raise ValueError('Positive target-specific training-mean reference required')
    sid,reg,cid,prefix=_metadata(panel);m=y.shape[1];r=float(c['model'].mean())
    record={'id':method_id,'status':status,'information_available':information_available,'method':method,
        'physical_mse':r,'reference_mse':den,'risk_ratio':r/den,'error_baseline100':100*r/den,'relative_mse_reduction_pp':100*(1-r/den),
        'circuit_ids':np.unique(sid).tolist(),'context_ids':cid.tolist(),'future_ids':np.asarray(panel['future_id']).astype(str).tolist(),
        'n_circuits':len(np.unique(sid)),'n_contexts':len(y),'n_futures_per_context':m,'n_scoring_episodes':len(y)*m,
        'fitting_repeats_averaged':1 if np.asarray(means).ndim==2 else len(means),'sources':list(source),
        'weighting':'equal circuits, regimes, prefixes, futures; average fitting-repeat risks, not ensemble predictions',
        'physical_mse_interval':_interval(c['model'],np.ones_like(c['model']),confidence),
        'error_baseline100_interval':_interval(c['model'],c['baseline'],confidence,100),
        'circuit_components':{k:None if v is None else v.tolist() for k,v in c.items()},
        'conditional_mc':future_jackknife(y,means,baseline,panel),'negative_corrected_context_excess':None}
    ri=record['error_baseline100_interval']
    record['relative_mse_reduction_pp_interval']=({'status':'estimated','lower':100-ri['upper'],'upper':100-ri['lower'],'confidence':confidence} if ri['status']=='estimated' else ri)
    if c['floor'] is None:
        record.update(state_variance=None,combined_unresolved_excess=None,state_variance_status='unavailable at M=1; pooled variance models are separate fitted assumptions')
    else:
        v=float(c['floor'].mean());e=float(c['excess'].mean())
        record.update(state_variance=v,combined_unresolved_excess=e,
            state_variance_interval=_interval(c['floor'],np.ones_like(c['floor']),confidence),
            state_variance_baseline100_interval=_interval(c['floor'],c['baseline'],confidence,100),
            combined_unresolved_excess_interval=_interval(c['excess'],np.ones_like(c['excess']),confidence),
            combined_unresolved_excess_baseline100_interval=_interval(c['excess'],c['baseline'],confidence,100),
            negative_corrected_context_excess=int(np.sum(comp['excess']<0)),finite_m_identity_residual=r-v-e,
            excess_interpretation='Hidden-state ambiguity, any compression loss and imperfect fitting remain combined; a fitted-method difference does not identify pure information loss.')
    mc=record['conditional_mc']
    if mc['status']=='complete':
        covariance=np.asarray(mc['pooled_covariance']);gradient=np.array([100/den,-100*r/den**2,0,0]);se=float(np.sqrt(max(0,gradient@covariance@gradient)))
        mc['error_baseline100_standard_error']=se
        mc['error_baseline100_normal_interval']=[100*r/den-norm.ppf((1+confidence)/2)*se,100*r/den+norm.ppf((1+confidence)/2)*se]
    return record,comp


def _load_predictions(root,job,member_ids,panel,manifest):
    predictions=[];records=[];base=None
    for member in member_ids:
        path=root/'models'/job/member/'evaluation.npz';record=prediction_record(root,job,path);verify_record(root,record)
        with np.load(path,allow_pickle=False) as z:
            if not np.array_equal(z['context_id'].astype(str),panel['context_id'].astype(str)):raise ValueError('Information prediction context mismatch')
            assert_risk_contract(manifest['risk_contract'],json.loads(str(z['risk_contract_json'])))
            if str(z['target_fingerprint'])!=manifest['target_fingerprint']:raise ValueError('Information target fingerprint mismatch')
            predictions.append(np.asarray(z['mean'],np.float64));this=np.asarray(z['baseline_mean'],np.float64)
            if base is not None and not np.array_equal(base,this):raise ValueError('Predictors use different training-mean references')
            base=this
        records.append(record)
    n=manifest['normalization'];return np.asarray(predictions)*n['y_scale']+np.asarray(n['y_offset']),base*n['y_scale']+np.asarray(n['y_offset']),records


def _policies(fit):
    records=fit['candidates'];lookup={r['id']:r for r in records};selected=fit['selected_member_ids'];out=[]
    def add(name,members,history,family,display=True):
        out.append(dict(id=name,members=members,history_samples=history,family=family,main_table=display,
            development_mse=float(np.mean([lookup[n]['development_mse'] for n in members])),selection_rule='minimum equal-circuit development MSE; frozen candidate family/grid and mean fallback',
            selected_mean_fallback=members==['train_mean']))
    full_history=int(fit.get('full_history_sample_count',2000))
    add('eeg_full_selected',selected,full_history,lookup[selected[0]]['kind'])
    short=[dict(id=r['id'],members=[r['id']],development_mse=r['development_mse']) for r in records if r['kind']=='ridge' and r.get('history_samples')==64]
    neural=[r for r in records if r['kind']=='neural' and r.get('history_samples')==64]
    if neural:short.append(dict(id='tcn_short_repeat_policy',members=[r['id'] for r in neural],development_mse=float(np.mean([r['development_mse'] for r in neural]))))
    fallback=dict(id='train_mean',members=['train_mean'],development_mse=lookup['train_mean']['development_mse'])
    if short:
        best=min(short+[fallback],key=lambda x:(x['development_mse'],x['id']));add('eeg_short_selected',best['members'],64,lookup[best['members'][0]]['kind'])
    for history,label,policies in [(full_history,'full',fit['policies']),(64,'short',short)]:
        for family in sorted({lookup[p['members'][0]]['kind'] for p in policies if p['members']!=['train_mean']}):
            candidates=[p for p in policies if lookup[p['members'][0]]['kind']==family]+[fallback]
            best=min(candidates,key=lambda x:(x['development_mse'],x['id']))
            add(f'eeg_{label}_{family}_policy',best['members'],history,family,False)
    return out


def _unknown(identifier,info,method,reason,status='unavailable'):
    return dict(id=identifier,status=status,information_available=info,method=method,reason=reason,
        physical_mse=None,reference_mse=None,risk_ratio=None,error_baseline100=None,relative_mse_reduction_pp=None)


def paired_contrast(a,b,*,label,confidence=.975):
    """Same circuits and fitting policy; target-specific baseline denominator each side."""
    for key in ('circuit_ids','context_ids','future_ids'):
        if a[key]!=b[key]:raise ValueError('Paired contrast requires identical '+key)
    ac=a['circuit_components'];bc=b['circuit_components']
    result=paired_ratio_difference(ac['model'],ac['baseline'],bc['model'],bc['baseline'],confidence=confidence)
    result.pop('within_tolerance',None);result.pop('equivalence_margin_pp',None)
    physical=np.asarray(ac['model'])-np.asarray(bc['model'])
    out=dict(label=label,status='estimated',contrast='G(first) minus G(second), percentage points; ratio of equal-circuit risks',
        gain_difference=result,physical_mse_difference=_interval(physical,np.ones(len(physical)),confidence),
        source_rows=[a['id'],b['id']],n_circuits=a['n_circuits'],n_contexts=a['n_contexts'],futures_per_context=a['n_futures_per_context'],
        selected_families=[a.get('family'),b.get('family')],selected_family_changed=a.get('family')!=b.get('family'),
        inferential_status='secondary/descriptive; no new confirmatory multiplicity allocation',
        interpretation='Matched policy comparison with each target-specific denominator; different targets remain different prediction problems.')
    ma=a['conditional_mc'];mb=b['conditional_mc']
    if ma['status']==mb['status']=='complete':
        ga={(r['structure_id'],r['prefix']):r for r in ma['groups']};gb={(r['structure_id'],r['prefix']):r for r in mb['groups']}
        if set(ga)!=set(gb):raise ValueError('Paired MC stream groups differ')
        ra=a['physical_mse'];ba=a['reference_mse'];rb=b['physical_mse'];bb=b['reference_mse'];variance=0
        for key in ga:
            x=np.asarray(ga[key]['delete_one_contributions']);y=np.asarray(gb[key]['delete_one_contributions'])
            if x.shape!=y.shape or ga[key]['future_ordinals']!=gb[key]['future_ordinals']:raise ValueError('Paired MC future order mismatch')
            influence=-100*(x[:,0]-ra/ba*x[:,1])/ba+100*(y[:,0]-rb/bb*y[:,1])/bb
            variance+=(len(x)-1)/len(x)*np.sum((influence-influence.mean())**2)/a['n_circuits']**2
        out['conditional_mc_gain_difference_se']=float(np.sqrt(max(0,variance)))
    else:out['conditional_mc_status']='unavailable'
    return out


def run(config,root,jobs=None,resume=False):
    root=Path(root);index=read_json(root/'derived/manifest.json');available={r.get('id',r.get('job_id')):r for r in index['jobs']}
    ids=list(jobs or config['selected_job_ids']);summaries=[];auxpath=root/'information_models/manifest.json'
    auxiliary=read_json(auxpath) if auxpath.exists() else None
    for job in ids:
        if job not in available or not (root/'derived'/job/'evaluation.npz').exists():
            summaries.append(dict(job_id=job,status='not yet run',reason='Common-population evaluation targets unavailable'));continue
        panel,manifest=load_panel(root,job);ev=select_primary_futures(panel['evaluation']);y=np.asarray(ev['Y_physical'],np.float64)
        fitpath=root/'models'/job/'manifest.json'
        if not fitpath.exists():
            summaries.append(dict(job_id=job,status='not yet run',reason='Training/development-frozen models unavailable'));continue
        fit=read_json(fitpath);assert_risk_contract(manifest['risk_contract'],fit['risk_contract'])
        if fit['selection_split']!='development':raise ValueError('Information model selection must use development only')
        policies=_policies(fit);loaded={};sources=[]
        for policy in policies+[dict(id='train_mean',members=['train_mean'])]:
            prediction,baseline,records=_load_predictions(root,job,policy['members'],ev,manifest)
            loaded[policy['id']]=(prediction,baseline,records);sources+=records
        auxiliary_rows=[]
        if auxiliary:
            for r in auxiliary.get('rows',[]):
                if r.get('job_id')==job:auxiliary_rows.append(dict(r,id=r.get('id',r.get('method','unnamed_readout'))))
            for r in auxiliary.get('jobs',[]):
                if r.get('id',r.get('job_id'))==job:auxiliary_rows.extend(r.get('rows',[]))
        sig=canonical_hash(dict(job=job,target=manifest['target_fingerprint'],evaluation=manifest['files']['evaluation'],fit=artifact_record(root,fitpath),
            sources=sources,auxiliary=artifact_record(root,auxpath) if auxiliary else None,analysis=module_hash('information','statistics'),first_scoring_futures=4))
        marker=root/'information'/job/'summary.json';old=resume_record(marker,sig,resume)
        if old:
            for r in old['files']:verify_record(root,r)
            summaries.append(old);continue
        basepred,base,baserecords=loaded['train_mean'];baseline_row,basecomp=summarize(y,basepred,base,ev,method_id='train_mean',
            information_available='No current recording',method='Training-mean predictor',source=[manifest['files']['evaluation']]+baserecords)
        rows=[dict(id='complete_state_exact_future_theory',status='theoretical',information_available='Complete present causal state, parameters and exact realised future external inputs',
                   method='Deterministic-system ideal',physical_mse=0.,reference_mse=baseline_row['reference_mse'],risk_ratio=0.,error_baseline100=0.,relative_mse_reduction_pp=100.,
                   reason='Theoretical zero conditional on truly complete state/parameters/forcing; not an observed simulation or predictor result'),
              _unknown('deterministic_replay','Complete state and exact future external events','Measured full-prefix deterministic replay',
                       'Replay gate uses separate registered circuits; see replay/summary.json. No measured replay error is inserted into this population.'),
              _unknown('state_features','Present-state features','Fitted partial-state-feature predictor','Verified split-aligned partial-state predictions unavailable'),
              _unknown('eeg_ideal','Specified observed EEG history','Best possible EEG-history predictor','EEG Bayes risk remains unidentified by fitted learner errors'),
              _unknown('compressed_future_readout','Compressed EEG representation','Matched d8/B32 future-target readout','No same-acquisition common-population code predictions available; planned code comparison is A0 only')]
        if basecomp['floor'] is not None:
            floor_row=dict(baseline_row,id='state_variance',status='estimated reference',information_available='Complete present causal state, future external inputs unknown',
                method='Unbiased within-context variance from matched-state independent futures',physical_mse=baseline_row['state_variance'],
                risk_ratio=baseline_row['state_variance']/baseline_row['reference_mse'],
                error_baseline100=100*baseline_row['state_variance']/baseline_row['reference_mse'],
                relative_mse_reduction_pp=100*(1-baseline_row['state_variance']/baseline_row['reference_mse']),
                physical_mse_interval=baseline_row['state_variance_interval'],error_baseline100_interval=baseline_row['state_variance_baseline100_interval'],
                combined_unresolved_excess=None,reference_interpretation='Estimated state-conditioned variance, not fitted hidden-state performance or identified EEG-history floor')
            fi=floor_row['error_baseline100_interval'];floor_row['relative_mse_reduction_pp_interval']=dict(status='estimated',lower=100-fi['upper'],upper=100-fi['lower'],confidence=fi['confidence']) if fi['status']=='estimated' else fi
            floor_row['conditional_mc']={'status':baseline_row['conditional_mc']['status'],'floor_standard_error':baseline_row['conditional_mc'].get('pooled_standard_error',{}).get('floor'),
                'source':'train_mean row joint R,B,V,E jackknife; same floor estimate for every method'}
            # This reference is an estimated variance, not the baseline predictor;
            # do not retain the baseline's unrelated excess-risk intervals.
            for key in ('combined_unresolved_excess_interval','combined_unresolved_excess_baseline100_interval','negative_corrected_context_excess','finite_m_identity_residual'):
                floor_row[key]=None
            floor_row.pop('circuit_components',None);rows.insert(2,floor_row)
        else:rows.insert(2,_unknown('state_variance','Complete state, unknown future inputs','Within-context sample variance','Unavailable at M=1'))
        context_arrays={};method_ids=[]
        for policy in policies:
            pred,base,records=loaded[policy['id']]
            row,comp=summarize(y,pred,base,ev,method_id=policy['id'],information_available=f'EEG history, {policy["history_samples"]} acquired samples',
                method='Development-selected '+policy['family']+' policy',source=[manifest['files']['evaluation']]+records)
            row.update(policy);rows.append(row);method_ids.append(row['id'])
            for key,value in comp.items():context_arrays[row['id']+'__'+key]=value
        for aux in auxiliary_rows:
            if aux.get('information_condition')=='partial present-state features':rows=[r for r in rows if r['id']!='state_features']
            if aux.get('status') not in ('complete','measured'):
                rows.append(_unknown(aux.get('id','unsupported_readout'),aux.get('information_condition',''),aux.get('method','Fitted readout'),aux.get('reason','Not yet run'),aux.get('status','unavailable')));continue
            path=root/aux['prediction_path'];record=next((r for r in auxiliary.get('files',[]) if r['path']==aux['prediction_path']),None)
            if record is None:record=next((r for r in aux.get('files',[]) if r['path']==aux['prediction_path']),None)
            if record is None:raise ValueError('Information readout missing prediction checksum')
            verify_record(root,record)
            with np.load(path,allow_pickle=False) as z:
                key='evaluation_prediction' if 'evaluation_prediction' in z else 'mean'
                idkey='evaluation_context_id' if 'evaluation_context_id' in z else 'context_id'
                if not np.array_equal(z[idkey].astype(str),ev['context_id'].astype(str)):raise ValueError('Readout contexts differ from common population')
                assert_risk_contract(manifest['risk_contract'],json.loads(str(z['risk_contract_json'])))
                if str(z['target_fingerprint'])!=manifest['target_fingerprint']:raise ValueError('Readout target differs')
                means=np.asarray(z[key],np.float64)*manifest['normalization']['y_scale']+np.asarray(manifest['normalization']['y_offset'])
            if aux['information_condition']=='compressed EEG representation':rows=[r for r in rows if r['id']!='compressed_future_readout']
            weights=next((r for r in auxiliary.get('files',[]) if r['path']==aux.get('weights_path')),None)
            if weights is None:raise ValueError('Readout weights lack source checksum')
            verify_record(root,weights)
            with np.load(root/weights['path'],allow_pickle=False) as z:
                expected=z['baseline']*manifest['normalization']['y_scale']+np.asarray(manifest['normalization']['y_offset'])
                if not np.allclose(expected,base,rtol=1e-12,atol=0):raise ValueError('Readout training-mean reference differs')
            row,comp=summarize(y,means,base,ev,method_id=aux['id'],information_available=aux['information_condition'],method=aux.get('method_description',aux['id']),source=[manifest['files']['evaluation'],record,weights])
            row.update(history_samples=aux.get('history_samples'),family=aux.get('family',aux['id']),selection_rule=aux.get('selection'),main_table=True,
                capacity=dict(input_coordinates=aux.get('input_coordinates'),decoder_family=aux.get('decoder_family'),candidates=aux.get('candidates')),code_budget=aux.get('code_budget'),
                representation=aux.get('representation'),variant=aux.get('variant'),optimizer_repeat_scope='Individual frozen bank fit; optimization seeds do not increase circuit N')
            if aux['information_condition']=='partial present-state features':rows=[r for r in rows if r['id']!='state_features']
            rows.append(row);method_ids.append(row['id'])
            for key,value in comp.items():context_arrays[row['id']+'__'+key]=value
        rows.append(baseline_row);method_ids.append('train_mean')
        for key,value in basecomp.items():context_arrays['train_mean__'+key]=value
        arrays=root/'information'/job/'loss_components.npz'
        atomic_npz(arrays,context_id=ev['context_id'],structure_id=ev['structure_id'],regime=ev['regime'],future_id=ev['future_id'],
            method_ids=np.asarray(method_ids),**{k:v for k,v in context_arrays.items() if v is not None})
        summary=dict(status='complete',signature=sig,job_id=job,target_spec=manifest['target_spec'],target_fingerprint=manifest['target_fingerprint'],risk_contract=manifest['risk_contract'],
            acquisition=job.split('__')[0],physical_mse_units=manifest['target_spec'].get('risk_units',manifest['risk_contract']['units']+' squared'),
            population=dict(circuit_ids=baseline_row['circuit_ids'],context_ids=baseline_row['context_ids'],n_circuits=baseline_row['n_circuits'],n_contexts=baseline_row['n_contexts'],
                first_scoring_futures=4,n_scoring_episodes=baseline_row['n_scoring_episodes'],claim_status=config['claim_status']),
            rows=rows,files=[artifact_record(root,arrays)],input_artifacts=[manifest['files']['evaluation'],artifact_record(root,fitpath)],
            ideal_components_identified=False,baseline100_units='Presentation scaling only; physical MSE stored separately',
            inference='Circuit t/influence intervals contain observed finite-future noise; conditional MC is reported separately, not added again. Secondary/descriptive, primary family decisions unchanged.')
        atomic_json(marker,summary);summaries.append(summary)
    contrasts=[]
    for a,b in itertools.combinations([s for s in summaries if s['status']=='complete'],2):
        ta,tb=a['target_spec'],b['target_spec'];fa=ta.get('family');fb=tb.get('family')
        if fa!=fb or ta.get('band')!=tb.get('band'):continue
        ha=ta.get('target_start_offset_ms',ta.get('lead_ms'));hb=tb.get('target_start_offset_ms',tb.get('lead_ms'))
        wa=ta.get('duration_ms');wb=tb.get('duration_ms');sameacq=a['acquisition']==b['acquisition']
        kind='gap' if sameacq and wa==wb and ha!=hb else 'duration' if sameacq and ha==hb and wa!=wb else 'acquisition' if ha==hb and wa==wb and not sameacq else None
        if kind is None:continue
        ar={r['id']:r for r in a['rows'] if 'circuit_components' in r};br={r['id']:r for r in b['rows'] if 'circuit_components' in r}
        for name in sorted(set(ar)&set(br)):
            if name=='train_mean':continue
            row=paired_contrast(ar[name],br[name],label=kind+' comparison');row.update(job_ids=[a['job_id'],b['job_id']],comparison_axis=kind,method_id=name)
            row['held_fixed']=dict(history_samples=ar[name].get('history_samples'),policy=ar[name].get('selection_rule'),acquisition=a['acquisition'] if sameacq else 'varied',family=ar[name].get('family') if ar[name].get('family')==br[name].get('family') else 'overall policy; winner family changed')
            contrasts.append(row)
    # History contrasts use the same physical target and acquisition.
    for s in summaries:
        if s['status']!='complete':continue
        lookup={r['id']:r for r in s['rows']}
        for short,full in [('eeg_short_selected','eeg_full_selected'),('eeg_short_ridge_policy','eeg_full_ridge_policy'),('eeg_short_neural_policy','eeg_full_neural_policy')]:
            if short in lookup and full in lookup:
                row=paired_contrast(lookup[full],lookup[short],label='Four-second minus 128-ms history');row.update(job_ids=[s['job_id'],s['job_id']],comparison_axis='history',method_id=full+' vs '+short);contrasts.append(row)
    cp=root/'information/paired_contrasts.json';atomic_json(cp,dict(status='complete',rows=contrasts,scope='Paired secondary/descriptive contrasts; no independent-sample tests or overlap-of-CI inference'))
    result=dict(status='complete' if all(s['status']=='complete' for s in summaries) else 'partial',jobs=[dict(id=s['job_id'],status=s['status'],path='information/'+s['job_id']+'/summary.json' if s['status']=='complete' else None,reason=s.get('reason')) for s in summaries],
        paired_contrasts=artifact_record(root,cp),meaning='Theoretical identity; only state variance and fitted risks estimated, observation/compression/fitting gaps not separately identified.')
    atomic_json(root/'information/manifest.json',result);return result
