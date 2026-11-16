"""Coupled integration-step qualification with uncertainty and no fake convergence."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
from scipy.stats import t as student_t
from ...storage import atomic_json, file_hash
from ...forks import _read_fork
from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ..v4.targets import TargetSpec, extract_target
from ..v4.measurement import causal_history
from .generation import execute_panel, row_path
from .simulator import verify_coupling


def _interval(estimate,influence,confidence):
    influence=np.asarray(influence,dtype=np.float64)
    if len(influence)<2:return dict(estimate=float(estimate),interval=None,status='unavailable')
    se=float(influence.std(ddof=1)/np.sqrt(len(influence)))
    half=float(student_t.ppf((1+confidence)/2,len(influence)-1)*se)
    return dict(estimate=float(estimate),interval=[float(estimate-half),float(estimate+half)],
                standard_error=se,circuits=len(influence),df=len(influence)-1,confidence=confidence)


def _moments(y,predictions,baseline):
    # C,R,M,D. Seeds average member-specific risk, never average predictions first.
    floor=np.var(y,axis=2,ddof=1).mean(axis=(1,2))
    result=dict(floor=floor,mean=np.mean(y,axis=(1,2,3)),mean_square=np.mean(y*y,axis=(1,2,3)))
    if predictions is not None:
        pred=np.asarray(predictions,dtype=np.float64)
        if pred.ndim==3:pred=pred[None]
        if pred.shape[1:]!=y.shape[:2]+y.shape[3:]:raise ValueError('Prediction seed/circuit/regime/coordinate axes mismatch')
        if not np.isfinite(pred).all():raise ValueError('Nonfinite predictions')
        b=np.asarray(baseline,dtype=np.float64)
        result['risk']=np.mean((pred[:,:,:,None,:]-y[None])**2,axis=(0,2,3,4))
        result['baseline']=np.mean((y-b)**2,axis=(1,2,3))
    return result


def _pair_estimands(a,b):
    denominator=b['floor'].mean()
    if not denominator>0:raise ValueError('Relative variance convergence needs a positive fine-grid floor')
    delta=a['floor']-b['floor'];ratio=delta.mean()/denominator
    outputs={'floor_relative_change':(ratio,(delta-ratio*b['floor'])/denominator)}
    if 'risk' in a and 'risk' in b:
        if min(a['baseline'].mean(),b['baseline'].mean())<=0:raise ValueError('Risk ratio needs positive baseline')
        ra=a['risk'].mean()/a['baseline'].mean();rb=b['risk'].mean()/b['baseline'].mean()
        ga=100*(1-ra);gb=100*(1-rb)
        influence=-100*((a['risk']-ra*a['baseline'])/a['baseline'].mean()-(b['risk']-rb*b['baseline'])/b['baseline'].mean())
        outputs['gain_difference_pp']=(ga-gb,influence)
    return outputs


def evaluate_paired(y_by_dt,predictions_by_dt=None,baseline=None,*,confidence=.975,
                    gain_tolerance_pp=.1,variance_relative_tolerance=.02):
    """Paired circuit CIs plus synchronized delete-one-future conditional MC CIs.

    y has circuit, regime, future, coordinate axes, with equal condition weights.
    The CI machinery is approximate and the result fails closed when precision is
    inadequate. Individual context futures never become circuit sample size.
    """
    keys=sorted(y_by_dt,key=float,reverse=True)
    if len(keys)<2:raise ValueError('At least two registered dt levels required')
    values={k:np.asarray(y_by_dt[k],dtype=np.float64) for k in keys}
    shape=values[keys[0]].shape
    if len(shape)!=4 or shape[0]<2 or shape[2]<3 or any(a.shape!=shape or not np.isfinite(a).all() for a in values.values()):
        raise ValueError('Need matched finite circuit/regime/future/coordinate arrays, C>=2 and M>=3')
    moments={k:_moments(values[k],None if predictions_by_dt is None else predictions_by_dt[k],baseline) for k in keys}
    pairs=[]
    for ai in range(len(keys)-1):
        for bi in range(ai+1,len(keys)):
            a,b=keys[ai],keys[bi];estimates=_pair_estimands(moments[a],moments[b]);checks={}
            deletions={name:[] for name in estimates}
            for c in range(shape[0]):
                per={name:[] for name in estimates}
                for future in range(shape[2]):
                    ma={k:v.copy() for k,v in moments[a].items()};mb={k:v.copy() for k,v in moments[b].items()}
                    for key,target in ((a,ma),(b,mb)):
                        p=None if predictions_by_dt is None else np.asarray(predictions_by_dt[key])
                        if p is not None:
                            if p.ndim==3:p=p[None]
                            p=p[:,c:c+1]
                        one=_moments(np.delete(values[key][c:c+1],future,axis=2),p,baseline)
                        for name,v in one.items():target[name][c]=v[0]
                    result=_pair_estimands(ma,mb)
                    for name,(estimate,_) in result.items():per[name].append(estimate)
                for name in estimates:deletions[name].append(per[name])
            for name,(estimate,influence) in estimates.items():
                circuit=_interval(estimate,influence,confidence)
                leave=np.asarray(deletions[name]);mcvar=(shape[2]-1)/shape[2]*np.sum((leave-leave.mean(1,keepdims=True))**2)
                mcse=float(np.sqrt(mcvar));half=float(student_t.ppf((1+confidence)/2,shape[0]*(shape[2]-1))*mcse)
                mc=[float(estimate-half),float(estimate+half)]
                tol=gain_tolerance_pp if name=='gain_difference_pp' else variance_relative_tolerance
                passes=all(-tol<=lo and hi<=tol for lo,hi in (circuit['interval'],mc))
                checks[name]=dict(**circuit,conditional_mc_interval=mc,conditional_mc_standard_error=mcse,
                    tolerance=[-tol,tol],passed=bool(passes),decision='within_tolerance' if passes else 'inconclusive_or_outside_tolerance')
            if 'gain_difference_pp' not in checks:
                checks['gain_difference_pp']=dict(status='unavailable',passed=False,
                    reason='Frozen predictor and training-mean baseline absent; floor convergence is not risk convergence')
            pairs.append(dict(coarse_dt_ms=float(a),fine_dt_ms=float(b),checks=checks,
                              passed=all(v['passed'] for v in checks.values())))
    passed=all(p['passed'] for p in pairs)
    return dict(status='complete',passed=passed,pairs=pairs,
        circuits=shape[0],regimes=shape[1],futures=shape[2],coordinates=shape[3],
        physical_moments={str(k):{name:v.tolist() for name,v in row.items()} for k,row in moments.items()},
        inference='paired circuit t/influence CI and separate conditional future jackknife CI; both must lie inside tolerance; no nonsignificance-as-convergence',
        future_pairing='same delete-one index across regimes within each circuit and across integration levels',
        uncertainty_caveat='Small-C t/influence and finite-M jackknife approximations, not exact coverage guarantees; no variance components silently added/double-counted',
        decision='qualified' if passed else 'inconclusive_or_failed; no core execution authorized')


def panel_config(c,dt):
    q=deepcopy(c);q['namespace']=c['namespace']+'-numerical-common-forcing';q['simulation']['native_dt_ms']=float(dt)
    q['claim_status']='numerical_only';q['reuse']['enabled']=False
    q['data']['structures']={s:[] for s in ('train','development','calibration','evaluation')}
    q['data']['structures']['evaluation']=list(c['numerical']['circuit_ids'])
    q['data']['prefixes_per_regime']={s:1 for s in q['data']['structures']}
    q['forks'].update(score_futures=c['numerical']['futures'],high_score_futures=c['numerical']['futures'],reference_futures=0,reference_circuits=[])
    return q


def run(c,root,resume=False,*,execute=False,qualification_root=None):
    from .planning import require_execution
    from .config import contexts
    from .models import predict_from_artifact
    require_execution(c,root,execute,numerical=True)
    if c['backend']!='neuron':raise ValueError('Biophysical numerical qualification cannot use analytical replacements')
    if c['numerical']['dts_ms']!=[.0625,.03125,.015625] or c['numerical']['output_dt_ms']!=.0625 or c['numerical']['futures']!=16 or len(c['numerical']['circuit_ids'])!=4 or len(c['data']['regimes'])!=4:
        raise ValueError('Numerical gate requires the registered4x4x16x3 panel and common16kHz output grid')
    root=Path(root);dest=root/'numerical';dest.mkdir(parents=True,exist_ok=True)
    if qualification_root is None:
        plan=json.loads((root/'plan.json').read_text());qualification_root=plan.get('phase0_root') or plan.get('qualification_root')
    if not qualification_root:raise ValueError('Numerical risk qualification requires a frozen phase-0 predictor')
    qualification_root=Path(qualification_root).resolve()
    # Validate the frozen physical-input path before any costly generation and bind
    # exact weights/normalization, rather than accepting a later model refit.
    preflight=predict_from_artifact(qualification_root,'A1__T0_l2_d32',X_physical=np.zeros((1,2000)))
    from ..v4.measurement import metadata
    risk=preflight['risk_contract'];expected=TargetSpec(lead_ms=2,duration_ms=32)
    if (risk.get('target_physical_hash')!=expected.fingerprint or risk.get('acquisition_hash')!=metadata('A1')['operator_hash']
        or risk.get('dimension')!=16 or risk.get('units')!='V' or risk.get('support')!=[6002.,6034.]):
        raise ValueError('Frozen numerical predictor has incompatible physical risk contract')
    qpaths=[qualification_root/'derived/A1__T0_l2_d32/manifest.json',
            qualification_root/'models/A1__T0_l2_d32/manifest.json',
            qualification_root/'models/A1__T0_l2_d32/fit_identity.json']
    fit=json.loads(qpaths[1].read_text())
    qpaths += [qualification_root/r['state_file'] for r in fit['candidates'] if r['id'] in preflight['member_ids']]
    predictor_identity=dict(files={str(p.relative_to(qualification_root)):file_hash(p) for p in qpaths},
        members=preflight['member_ids'],target_fingerprint=preflight['target_fingerprint'],risk_contract=preflight['risk_contract'])
    guard=dest/'predictor_identity.json'
    if guard.exists() and json.loads(guard.read_text())!=predictor_identity:raise ValueError('Frozen numerical predictor/normalization changed')
    atomic_json(guard,predictor_identity)
    ids=list(c['numerical']['circuit_ids']);regimes=[r['name'] for r in c['data']['regimes']];m=c['numerical']['futures']
    specs=[TargetSpec(lead_ms=2,duration_ms=32),TargetSpec(lead_ms=10,duration_ms=128),TargetSpec(family='T3',lead_ms=10,duration_ms=1000)]
    trialsets={};results=[]
    for dt in c['numerical']['dts_ms']:
        q=panel_config(c,dt);directory=dest/('dt_'+str(dt).replace('.','p'))
        results.append(execute_panel(q,directory,contexts(q),resume))
        manifest=json.loads((directory/'native/manifest.json').read_text())
        trialsets[dt]={r['trial_id']:r for r in manifest['trials']}
        for r in trialsets[dt].values():r['_directory']=str(directory)
    arrays={s.id:{} for s in specs};histories={};coupling=[]
    dts=list(c['numerical']['dts_ms']);ordered=contexts(panel_config(c,dts[0]))
    for dt in dts:
        for spec in specs:arrays[spec.id][dt]=np.empty((len(ids),len(regimes),m,spec.dimension),dtype=np.float64)
        histories[dt]=[]
    for context in ordered:
        ci=ids.index(context['structure_id']);ri=regimes.index(context['regime']['name'])
        for future in range(m):
            tid=context['context_id']+f'-score-{future:04d}'
            trials=[_read_fork(row_path(trialsets[dt][tid]['_directory'],trialsets[dt][tid])) for dt in dts]
            audit=verify_coupling(trials);coupling.append(dict(trial_id=tid,**audit))
            for dt,trial in zip(dts,trials):
                for spec in specs:arrays[spec.id][dt][ci,ri,future]=extract_target(trial['eeg_v'],trial['time_ms'],spec)
                if future==0:
                    x,_,_=causal_history(trial['eeg_v'],trial['time_ms'],cutoff_ms=c['cutoff_ms'],history_s=4.,acquisition_id='A1')
                    histories[dt].append(x)
    atomic_json(dest/'coupled_forcing.json',dict(status='complete',passed=True,trials=coupling,
        claim='Exact event schedules, phase paths, physical parameters and common output grid; solver trajectories need not match'))
    prediction={};baseline=None;prediction_records={}
    for dt in dts:
        prediction_records[dt]=predict_from_artifact(qualification_root,'A1__T0_l2_d32',X_physical=np.asarray(histories[dt]))
        pred=prediction_records[dt];prediction[dt]=pred['mean_physical'].reshape(-1,len(ids),len(regimes),16)
        if baseline is not None and not np.array_equal(baseline,pred['baseline_mean_physical']):raise ValueError('Numerical baselines changed across dt')
        baseline=np.asarray(pred['baseline_mean_physical'],dtype=np.float64)
    reports={}
    for spec in specs:
        report=evaluate_paired(arrays[spec.id],prediction if spec.id=='T0_l2_d32' else None,baseline,
            gain_tolerance_pp=c['numerical']['gain_tolerance_pp'],variance_relative_tolerance=c['numerical']['variance_relative_tolerance'])
        if spec.id!='T0_l2_d32':
            for pair in report['pairs']:
                pair['checks'].pop('gain_difference_pp');pair['passed']=pair['checks']['floor_relative_change']['passed']
            report['passed']=all(p['passed'] for p in report['pairs'])
            report['decision']='qualified_floor_only' if report['passed'] else 'inconclusive_or_failed_floor'
            report['risk_scope']='No learner-risk convergence claim for this secondary anchor'
        reports[spec.id]=report
    payload={}
    for spec in specs:
        for dt in dts:payload[spec.id+'__dt_'+str(dt)]=arrays[spec.id][dt]
    for dt in dts:payload['predictions_dt_'+str(dt)]=prediction[dt];payload['history_A1_dt_'+str(dt)]=np.asarray(histories[dt])
    payload.update(circuit_ids=np.asarray(ids),regime_names=np.asarray(regimes),future_indices=np.arange(m),integration_dts_ms=np.asarray(dts))
    payload['baseline_physical']=baseline;atomic_npz(dest/'numerical_arrays.npz',**payload)
    passed=all(x['passed'] for x in reports.values())
    result=dict(status='complete',passed=passed,reports=reports,
        episodes=sum(r['continuations'] for r in results),generation=results,qualification_root=str(Path(qualification_root).resolve()),
        risk_contracts={spec.id:dict(target=spec.to_dict(),circuit_ids=ids,regimes=regimes,futures=m,units='V^2',population='registered numerical panel; physical risks without target-specific standardization') for spec in specs},
        predictor_identity=predictor_identity,
        predictor_member_ids={str(dt):prediction_records[dt]['member_ids'] for dt in dts},
        forcing_verified=True,coupled_forcing_sha256=file_hash(dest/'coupled_forcing.json'),output_grid_hz=16000.,same_native_fourier_normalization=True,
        scientific_decision='qualified' if passed else 'inconclusive_or_failed; preserve all data and block core generation',
        config_hash=canonical_hash(c),arrays_sha256=file_hash(dest/'numerical_arrays.npz'))
    atomic_json(dest/'summary.json',result)
    return result
