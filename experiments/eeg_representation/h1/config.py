"""Explicit executed jobs, disjoint circuits and predeclared H1 cost gates."""
from copy import deepcopy
from pathlib import Path
import math
import yaml
from . import SCHEMA_VERSION
from .targets import TargetSpec,canonical_hash,aligned
from .acquisition import metadata

REGIMES=[{'name':'R0','frequency_hz':0.,'modulation_depth':0.},
 {'name':'Ralpha','frequency_hz':10.,'modulation_depth':.04},
 {'name':'Rbeta_low','frequency_hz':16.,'modulation_depth':.04},
 {'name':'Rbeta_high','frequency_hz':24.,'modulation_depth':.04}]
DEFAULTS={'schema_version':SCHEMA_VERSION,'name':'h1_development','claim_status':'development','backend':'neuron',
 'namespace':'eegrepr-h1-information-v2-development','simulation':{'native_dt_ms':.0625,'burn_in_s':2.,'celsius':6.3,'population_e':32,'population_i':8,'phase_grid_ms':1.},
 'data':{'structures':{'train':[35000,35001,35002,35003],'development':[35004,35005],'evaluation':[35006,35007,35008,35009]},
  'prefixes_per_regime':{'train':16,'development':8,'evaluation':1},'regimes':REGIMES,'workers':1},
 'forks':{'score_futures':16,'reference_futures':0,'high_reference_futures':0,'high_score_futures':0,'high_contexts':[]},
 'jobs':[{'id':'primary','history_s':4.,'lead_ms':10.,'duration_ms':128.,'operator':'bin_average_waveform','output_grid_hz':500.,'input_rate_hz':500.,'acquisition_bits':None,'snr_db':None}],
 'acquisition':{'passband_hz':[.5,30.],'fir_taps':2049},
 'analytical':{'kind':'oscillator','amplitude_v':1e-6,'damping_per_s':3.},
 'benchmark':{'models':['train_mean','train_mean_unit_variance','persistence','ridge_wiener','harmonic','state_space','neural_mse','neural_gaussian'],
  'seeds':[0,1],'ridge_alphas':[.01,.1,1.],'lags':[16,64],'state_ranks':[2,4,8],
  'harmonic_frequencies_hz':[4+i*.5 for i in range(53)],'harmonic_history_fractions':[.5,.75,1.],'max_candidates':3,
  'steps':500,'batch_size':16,'learning_rates':[.0003],'weight_decay':.01,'draws':64,
  'diagnostic_steps':100,'diagnostic_examples':4,'representation_steps':200,'parameter_tolerance':.15,'threads':1,
  'embedding_dim':128,'layers':4,'heads':4,'patch_samples':32},
 'replication':{'m_values':[1,2,4,8,16,32,64,128],'repetitions':32,'bootstrap_samples':64,'seed':0,
  'pooled_one_future':True,'ridge':1.,'fixed_budget':256,'fixed_budget_m':[1,2,4,8,16],'prefixes_per_stratum':16},
 'compression':{'dimensions':[8,32],'code_bits':[32,128],'representations':['feature18','pca','untrained','masked_mse','forecast'],
  'quantile_range':[.005,.995],'max_scalar_bits':16,'readouts':['linear','flexible'],'label_budgets':[8,'all'],
  'regularization':[.01,1.,100.],'flexible_features':64,'seed':7301},
 'scale':{},'cost_gate':{'required':False},
 'fixed_budget_extension':{'enabled':False,'budget':256,'m_values':[1,2,4,8,16],'prefixes_per_stratum':16}}


def merge(a,b):
    out=deepcopy(a)
    for k,v in b.items():
        if k not in a:raise ValueError(f'Unknown H1 config key {k}')
        out[k]=merge(a[k],v) if isinstance(a[k],dict) and a[k] and isinstance(v,dict) else deepcopy(v)
    return out


def resolve(raw):
    c=merge(DEFAULTS,raw)
    if c['schema_version']!=SCHEMA_VERSION:raise ValueError('Unknown H1 schema')
    if c['backend'] not in ('analytical','neuron'):raise ValueError('Unknown backend')
    if c['claim_status'] not in ('engineering','development','planned_confirmation'):raise ValueError('Invalid claim status')
    s=c['simulation']
    if s['native_dt_ms']!=.0625 or s['celsius']!=6.3:raise ValueError('Inherited native integration and HH temperature remain frozen')
    if c['claim_status']!='engineering' and (s['population_e'],s['population_i'])!=(32,8):raise ValueError('Reduced networks are engineering controls only')
    if c['data']['workers']!=1:raise ValueError('Current H1 coordinator is serial; do not claim parallel wall-time speedup')
    ids=[v for values in c['data']['structures'].values() for v in values]
    if len(ids)!=len(set(ids)) or any(type(v)!=int or not 1<=v<=429495 for v in ids):raise ValueError('Disjoint valid circuit IDs required')
    if set(ids)&set(range(32000,32008)):raise ValueError('Inspected old pilot circuits cannot be new H1 development/confirmation circuits')
    for split in ('train','development','evaluation'):
        if type(c['data']['prefixes_per_regime'][split])!=int or c['data']['prefixes_per_regime'][split]<0:raise ValueError('Nonnegative independent prefix counts required')
    if not c['data']['structures']['evaluation']:raise ValueError('Need explicit evaluation circuits')
    if not c['jobs'] or len({j['id'] for j in c['jobs']})!=len(c['jobs']):raise ValueError('Unique explicit jobs required')
    cutoff=cutoff_ms(c)
    import re
    for j in c['jobs']:
        if not re.fullmatch('[a-zA-Z0-9_-]+',j['id']):raise ValueError('Unsafe job ID')
        for name in ('history_s','lead_ms','duration_ms','operator','output_grid_hz','input_rate_hz','acquisition_bits','snr_db'):
            if name not in j:raise ValueError(f'Missing explicit job setting {name}')
        acquisition=metadata(j['input_rate_hz'],s['native_dt_ms'],c['acquisition']['passband_hz'],c['acquisition']['fir_taps'])
        TargetSpec(operator=j['operator'],cutoff_ms=cutoff,history_s=j['history_s'],lead_ms=j['lead_ms'],duration_ms=j['duration_ms'],output_grid_hz=j['output_grid_hz'],input_acquisition=acquisition)
        aligned(j['history_s']*1000,1000/j['input_rate_hz'],'history on input grid')
        aligned(cutoff,1000/j['input_rate_hz'],'cutoff on input grid')
        if j['acquisition_bits'] not in (None,4,8,12,16):raise ValueError('Unsupported acquisition bits')
        if j['snr_db'] is not None and j['snr_db'] not in (20,10,0):raise ValueError('Only frozen20/10/0dB input noise sensitivity is supported')
        if s['burn_in_s']*1000<=acquisition['support_ms']:raise ValueError('Burn-in must exceed the causal FIR support')
    if c['forks']['score_futures']<2:raise ValueError('Need at least two futures for state-floor benchmark')
    for key in ('reference_futures','high_reference_futures','high_score_futures'):
        if type(c['forks'][key])!=int or c['forks'][key]<0:raise ValueError('Invalid fork count')
    regs=c['data']['regimes']
    if len({r['name'] for r in regs})!=len(regs):raise ValueError('Duplicate regime names')
    for r in regs:
        if not re.fullmatch('[a-zA-Z0-9_-]+',r['name']):raise ValueError('Unsafe regime name')
        if not 0<=r.get('modulation_depth',0)<=1 or r.get('diffusion',0)<0:raise ValueError('Invalid drive')
        if r.get('modulation_depth',0)==0 and (r.get('diffusion',0)>0 or r.get('burst',False)):raise ValueError('Homogeneous regime cannot carry meaningless diffusion/burst contrast')
    ext=c['fixed_budget_extension']
    if ext['enabled']:
        strata=len(c['data']['structures']['train'])*len(regs)
        if not strata or any(type(m)!=int or m<1 or ext['budget']%(m*strata) for m in ext['m_values']):raise ValueError('Fixed-budget allocations require exact circuit/regime balance')
        if c['data']['prefixes_per_regime']['train']!=ext['budget']//min(ext['m_values'])//strata:raise ValueError('Fixed-budget prefix pool must match the largest nested allocation')
    if c['claim_status']=='planned_confirmation'  and len(c['data']['structures']['evaluation'])<12:raise ValueError('Provisional confirmation plan requires at least12 new circuits; no power promise')
    return c


def load_config(path):
    return resolve(yaml.safe_load(Path(path).read_text()))


def cutoff_ms(c):return 1000*(c['simulation']['burn_in_s']+max(j['history_s'] for j in c['jobs']))


def stop_ms(c):
    # End after the last required native future sample, rounded UP only to simulator phase grid.
    exact=cutoff_ms(c)+max(j['lead_ms']+j['duration_ms'] for j in c['jobs'])
    grid=c['simulation']['phase_grid_ms']
    return math.ceil(exact/grid-1e-10)*grid


def context_specs(c):
    result=[];ns=c['namespace'];nreg=len(c['data']['regimes'])
    for split,structures in c['data']['structures'].items():
        for si,sid in enumerate(structures):
            for ri,regime in enumerate(c['data']['regimes']):
                for prefix in range(c['data']['prefixes_per_regime'][split]):
                    cid=f's{sid}-p{prefix:04d}-{regime["name"]}'
                    high=cid in c['forks']['high_contexts']
                    score=(c['forks']['high_score_futures'] if high else c['forks']['score_futures']) if split=='evaluation' else 1
                    if split=='train' and c['fixed_budget_extension']['enabled']:
                        ext=c['fixed_budget_extension'];strata=len(structures)*nreg
                        score=max(m for m in ext['m_values'] if prefix < ext['budget']//m//strata)
                    ref=(c['forks']['high_reference_futures'] if high else c['forks']['reference_futures']) if split=='evaluation' else 0
                    result.append({'context_id':cid,'structure_id':sid,'prefix':prefix,'split':split,'regime':regime,'score_futures':score,'reference_futures':ref})
    high=set(c['forks']['high_contexts'])
    if high-set(x['context_id'] for x in result if x['split']=='evaluation'):raise ValueError('Preselected high-M contexts absent from panel')
    return result
