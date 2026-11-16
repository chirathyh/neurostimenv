"""Bounded engineering checks and measured fitting/calibration throughput."""
from pathlib import Path
import time,platform
import numpy as np
from .artifacts import atomic_json,read_json,file_hash,canonical_hash,environment,module_hash
from .models import settings,known_learnable_control,train_tcn,matched_variance,configure_torch
from .targets import energy_from_coefficients,gaussian_expected_energy
from ..v4.measurement import qualify
from ..v4.geometry import geometry_controls
from .replication import reference_controls
from .precision import synthetic_controls


def target_controls():
    from ..v4.targets import TargetSpec,extract_target
    from ..v4.measurement import causal_history
    dt=.0625;t=np.arange(112161,dtype=np.float64)*dt
    y=2e-6*np.cos(2*np.pi*10*t/1000)+3e-6*np.sin(2*np.pi*25*t/1000)
    target=TargetSpec(family='T3',lead_ms=10.,duration_ms=1000.)
    a=extract_target(y,t,target);changed=y.copy();changed[t<=6000]+=1
    b=extract_target(changed,t,target)
    freq=np.repeat(np.arange(4.,31.),2);comp=np.tile(['cos','sin'],27)
    energy=energy_from_coefficients(a,freq,comp,'alpha')[0]
    rows={}
    for acquisition in ('A0','A1','A2'):
        after=y.copy();after[t>6000]+=100
        x,_,_=causal_history(y,t,cutoff_ms=6000,history_s=4,acquisition_id=acquisition)
        xx,_,_=causal_history(after,t,cutoff_ms=6000,history_s=4,acquisition_id=acquisition)
        rows[acquisition]=bool(np.array_equal(x,xx))
    total=sum(energy_from_coefficients(a,freq,comp,k)[0] for k in ('alpha','beta_low','beta_high'))
    gaussian=gaussian_expected_energy(np.zeros_like(a),np.ones_like(a),freq,comp,'alpha')[0]
    checks={'future_only_past_invariance':bool(np.array_equal(a,b)),'input_postcutoff_invariance':all(rows.values()),
        'alpha_energy_physical_normalization':bool(np.isclose(energy,2e-12,rtol=1e-10,atol=0)),
        'nonoverlapping_bands':bool(np.isclose(total,6.5e-12,rtol=1e-10,atol=0)),
        'gaussian_variance_contribution':bool(np.isclose(gaussian,8/16000))}
    return {'passed':all(checks.values()),'checks':checks,'acquisition_causality':rows,'units':'V^2 energy; V^4 power MSE','floor_v2':1e-40}


def run(c,root,device='cpu',repetitions=2000):
    parts={'measurement':qualify(root),'future_only_energy':target_controls(),
           'same_neural_path':known_learnable_control(settings(c),device),
           'known_geometry':geometry_controls(),'finite_reference':reference_controls(repetitions=repetitions),
           'statistical_coverage':synthetic_controls(N=c['protocol']['test_circuits'],repetitions=repetitions)}
    passed=all(v.get('passed',True) for v in parts.values())
    passed &= not parts['statistical_coverage']['warnings'] and parts['statistical_coverage']['status']=='complete'
    passed &= all(r['unbiased_mean_control_pass'] for r in parts['finite_reference']['controls'])
    result={'status':'complete','passed':bool(passed),'parts':parts,'scope':'Synthetic/measurement engineering checks; no new full-network observations'}
    atomic_json(Path(root)/'controls.json',result)
    if not passed:raise RuntimeError('Engineering control failed; inspect controls.json')
    return result


def throughput(c,root,device='cpu',resume=False):
    """Same TCN path, actual 2000 coordinates; bounded timing, no scientific fit."""
    cfg=settings(c);hardware={}
    torch,_=configure_torch(device,cfg['dtype'],cfg['threads'])
    if device.startswith('cuda'):
        index=torch.cuda.current_device();properties=torch.cuda.get_device_properties(index)
        hardware={'device_name':properties.name,'total_memory_bytes':properties.total_memory,'compute_capability':[properties.major,properties.minor],'device_index':index}
    signature=canonical_hash({'settings':cfg,'environment':environment(),'device':device,'source':file_hash(Path(__file__)),'models_source':file_hash(Path(__file__).with_name('models.py')),'hardware':hardware})
    dest=Path(root)/'fit_throughput.json'
    if dest.exists():
        old=read_json(dest)
        if not resume or old.get('signature')!=signature:raise ValueError('Throughput environment/source/device changed')
        return old
    configure_torch(device,cfg['dtype'],cfg['threads']);rng=np.random.default_rng(61)
    def panel(n,nc):
        x=rng.normal(size=(n,2000));y=rng.normal(size=(n,1,54))
        return dict(X=x,Y=y,future_mask=np.ones((n,1),bool),structure_id=np.arange(n)%nc,context_id=np.array([f'perf-{i}-R0' for i in range(n)]))
    tr=panel(256,8);dv=panel(64,4);opt={**cfg,'max_steps':50,'patience':50,'eval_every':50}
    started=time.perf_counter();_,_,account=train_tcn(tr,dv,opt,history_samples=2000,seed=19,device=device)
    elapsed=time.perf_counter()-started
    primary_fits=2*len(cfg['neural_seeds']) if cfg['neural_enabled'] else 0
    sensitivity_fits=len(cfg['neural_seeds']) if primary_fits and cfg['fp64_sensitivity'] and cfg['dtype']=='float32' else 0
    fp64_elapsed=0.
    if sensitivity_fits:
        started=time.perf_counter()
        train_tcn(tr,dv,{**opt,'dtype':'float64'},history_samples=2000,seed=19,device=device)
        fp64_elapsed=time.perf_counter()-started
    base_hours=len(c['selected_job_ids'])*primary_fits*cfg['max_steps']*elapsed/50/3600
    sensitivity_hours=len(c['selected_job_ids'])*sensitivity_fits*cfg['max_steps']*fp64_elapsed/50/3600
    cal=panel(192,24);cal['Y']=np.repeat(cal['Y'],4,axis=1);cal['future_mask']=np.ones((192,4),bool)
    tick=time.perf_counter()
    for _ in range(30):matched_variance(cal,np.zeros((192,54)),cfg['variance_floor'])
    calibration=(time.perf_counter()-tick)/30
    result={'status':'complete','signature':signature,'host':platform.node(),'environment':environment(),'device':device,'dtype':cfg['dtype'],'hardware':hardware,'settings':cfg,
        'training_seconds':elapsed,'updates':account.get('updates_completed',account.get('steps_completed',50)),
        'seconds_per_50_updates':elapsed,'calibration_seconds_per_candidate':calibration,
        'example_training_contexts':256,'example_development_contexts':64,'input_samples':2000,'output_coordinates':54,
        'maximum_tcn_updates_per_job':primary_fits*cfg['max_steps'],'jobs':len(c['selected_job_ids']),
        'seconds_per_50_fp64_updates':fp64_elapsed if sensitivity_fits else None,
        'maximum_selected_fp64_updates_per_job':sensitivity_fits*cfg['max_steps'],
        'rough_base_tcn_max_hours':base_hours,'rough_selected_fp64_max_hours':sensitivity_hours,
        'rough_tcn_max_hours':base_hours+sensitivity_hours,
        'scope':'Measured synthetic throughput on same neural/calibration code, including up to five selected fp64 sensitivity fits per job; maximum-update estimate is not a full wall-time bound; it excludes larger training/development passes, ridge, retention and I/O. Actual learning early-stops on development.',
        'generation_device':'CPU only; this GPU/CPU fit timing is separate from NEURON','accounting':account}
    atomic_json(dest,result);return result
