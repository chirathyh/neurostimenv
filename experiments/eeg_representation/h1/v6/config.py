"""Versioned v6 design. New test IDs never include inspected v4 test circuits."""
from copy import deepcopy
from pathlib import Path
import json
import yaml
from ..config import REGIMES
from ..targets import canonical_hash
from ..artifacts import ROOT

SCHEMA='eegrepr-evaluation-v6.0'
ANCHORS=['T0_l2_d32','T0_l10_d128','T3_l10_d1000']
ENERGIES=['T4_'+kind+'_'+band+'_l10_d1000' for kind in ('energy','log_energy') for band in ('alpha','beta_low','beta_high')]
DEFAULTS={
 'schema_version':SCHEMA,'name':'confirmation','claim_status':'prospective_confirmation','backend':'neuron',
 'namespace':'eegrepr-evaluation-v6-confirmation-20261116','history_s':4.,'cutoff_ms':6000.,'stop_ms':7010.,
 'simulation':{'native_dt_ms':.0625,'burn_in_s':2.,'celsius':6.3,'population_e':32,'population_i':8,'phase_grid_ms':1.,'external_synapses_per_cell':64},
 'data':{'structures':{'train':list(range(36000,36008))+list(range(37000,37024)),
                     'development':list(range(36008,36012))+list(range(37024,37028)),
                     'calibration':list(range(37028,37052)),'evaluation':list(range(37052,37244))},
         'prefixes_per_regime':{'train':8,'development':4,'calibration':2,'evaluation':2},'regimes':deepcopy(REGIMES),'workers':1},
 'forks':{'score_futures':4,'calibration_futures':4,'high_score_futures':32,'reference_futures':128,
          'reference_circuits':list(range(37052,37060)),'reference_regimes':[r['name'] for r in REGIMES],'reference_prefix':0},
 'reuse':{'enabled':True,'circuit_ids':list(range(36000,36012)),
         'source_archive_sha256':'6818befa7ca70df3a0272bf69328f9e2d2316817493e49a13bac6e4b74a5817f','expected_episodes':320},
 'acquisition_ids':['A0','A1','A2'],
 'selected_job_ids':[a+'__'+t for a in ['A0','A1','A2'] for t in ANCHORS+ENERGIES],
 'models':{},'retention':{},'information_models':{},
 'extension':{'enabled':False,'version':'immediate-information-1','secondary_only':True},
 'energy':{'native_samples':16000,'log_floor_v2':1e-40,'log_reference_v2':1.,'bands':{'alpha':[8,12],'beta_low':[12,20],'beta_high':[20,30]}},
 'protocol':{'alpha_total':.05,'family_a_alpha':.025,'family_b_alpha':.025,
             'task_margin_bits':.20,'reconstruction_margin_nmse':.02,'forecast_margin_pp':1.,
             'forecast_interval_level':.975,'primary_job_id':'A1__T0_l2_d32',
             'primary_score_futures':4,'test_circuits':192,'forecast_halfwidth_pp':.5,
             'carrier_halfwidth_bits':.10,'reconstruction_halfwidth_nmse':.01,
             'training_variation_included':False,'h5_h6':'deferred'},
 'numerical':{'circuit_ids':list(range(37500,37504)),'dts_ms':[.0625,.03125,.015625],
              'futures':16,'output_dt_ms':.0625,'gain_tolerance_pp':.1,'variance_relative_tolerance':.02},
 'budget':{'max_generation_hours':500.,'max_native_bytes':120000000000,'minimum_free_bytes':250000000000,
           'provisional_seconds':88.71645*7.010/6.138,'provisional_episode_bytes':4750000000/832,
           'require_measured_gate':True},
 'optional':{'common_event_fraction':0.,'phase_diffusion':0.,'instrumentation':False},
 'analytical':{'kind':'oscillator','amplitude_v':1e-6,'damping_per_s':3.},
}


def target_ids(c):
 if not c.get('extension',{}).get('enabled',False):return ANCHORS+ENERGIES
 waveform=[f'T0_l{h}_d{w}' for h in (0,2,10) for w in (32,128)]
 fourier=[f'T3_l{h}_d1000' for h in (0,10)]
 energy=[f'T4_{kind}_{band}_l{h}_d1000' for h in (0,10) for kind in ('energy','log_energy') for band in ('alpha','beta_low','beta_high')]
 return waveform+fourier+energy

def required_job_ids(c):
 return [a+'__'+t for a in c['acquisition_ids'] for t in target_ids(c)]

def merge(base,update):
 out=deepcopy(base)
 for key,value in update.items():
  if key not in base:raise ValueError('Unknown v6 config key '+key)
  if isinstance(value,dict) and isinstance(base[key],dict) and key not in ('models','retention','information_models'):
   out[key]=merge(base[key],value)
  else:out[key]=deepcopy(value)
 return out


def resolve(raw=None):
 c=merge(DEFAULTS,raw or {})
 if c['extension']['version']!='immediate-information-1' or c['extension']['secondary_only'] is not True:raise ValueError('Extension is a fixed secondary/descriptive amendment; primary families unchanged')
 if not isinstance(c['extension']['enabled'],bool):raise ValueError('Extension enabled must be boolean')
 if c['extension']['enabled'] and 'selected_job_ids' not in (raw or {}):c['selected_job_ids']=required_job_ids(c)
 d=c['data'];s=c['simulation'];f=c['forks'];p=c['protocol']
 if c['schema_version']!=SCHEMA:raise ValueError('Unsupported v6 schema')
 if c['claim_status'] not in ('prospective_confirmation','exploratory_existing','engineering','timing_only','numerical_only'):raise ValueError('Unknown claim status')
 if c['backend'] not in ('neuron','analytical') or c['backend']=='analytical' and c['claim_status']!='engineering':raise ValueError('Analytical data are explicit engineering controls only')
 if s['celsius']!=6.3 or s['native_dt_ms'] not in c['numerical']['dts_ms']:raise ValueError('Unsupported physics/integration law')
 if (s['population_e'],s['population_i'],s['external_synapses_per_cell'])!=(32,8,64) and c['claim_status']!='engineering':raise ValueError('Full40cell circuit required')
 if c['history_s']!=4 or c['cutoff_ms']!=6000 or c['stop_ms']!=7010:raise ValueError('Frozen4s history and7.010s episode support required')
 ids=[int(i) for v in d['structures'].values() for i in v]
 if len(ids)!=len(set(ids)):raise ValueError('Circuit split overlap')
 if any(not 1<=i<=429495 for i in ids):raise ValueError('Circuit ID outside seed range')
 if c['energy']['bands']!=DEFAULTS['energy']['bands']:raise ValueError('Energy bands are frozen physical operators')
 for key in ('task_margin_bits','reconstruction_margin_nmse','forecast_margin_pp','forecast_interval_level','primary_job_id','forecast_halfwidth_pp','carrier_halfwidth_bits','reconstruction_halfwidth_nmse'):
  if c['claim_status']!='engineering' and p[key]!=DEFAULTS['protocol'][key]:raise ValueError('Frozen statistical protocol field changed: '+key)
 if c['claim_status']=='prospective_confirmation':
  if c['acquisition_ids']!=DEFAULTS['acquisition_ids'] or c['selected_job_ids']!=required_job_ids(c):raise ValueError('Complete'+str(len(required_job_ids(c)))+'-job fixed acquisition/target grid required')
  if d['regimes']!=REGIMES or d['prefixes_per_regime']!=DEFAULTS['data']['prefixes_per_regime']:raise ValueError('Frozen paired four-regime/prefix allocation changed')
  if len(d['structures']['train'])!=32 or len(d['structures']['development'])!=8:raise ValueError('Frozen32train/8development circuits required')
  if f['calibration_futures']!=4 or len(f['reference_circuits'])!=8 or f['reference_prefix']!=0 or f['reference_regimes']!=DEFAULTS['forks']['reference_regimes']:raise ValueError('Frozen calibration/H2 allocation changed')
  if c['numerical']!=DEFAULTS['numerical']:raise ValueError('Numerical refinement requires a separate bounded protocol; default768episode grid cannot change silently')
  if set(d['structures']['evaluation'])&set(range(36000,36210)):raise ValueError('Fresh test IDs overlap inspected/previouslyreserved v4')
  if len(d['structures']['evaluation'])!=p['test_circuits']:raise ValueError('TestN differs from frozen protocol')
  if len(d['structures']['calibration'])!=24:raise ValueError('Dedicated24calibration circuits required')
  if f['score_futures']!=4 or f['high_score_futures']!=32 or f['reference_futures']!=128:raise ValueError('PrimaryM4/H2M32/L128 frozen')
  if p['primary_score_futures']!=4:raise ValueError('Primary must use identical first4registeredfutures')
 if set(f['reference_circuits'])-set(d['structures']['evaluation']):raise ValueError('Reference circuits absent from evaluation')
 if d['workers']<1 or d['workers']>64:raise ValueError('Invalid bounded worker count')
 if (p['alpha_total'],p['family_a_alpha'],p['family_b_alpha'])!=(.05,.025,.025):raise ValueError('Two-family alpha allocation frozen')
 if any(c['optional'].values()):raise ValueError('Optional biological/instrumented experiments are outside v6 core')
 if c['energy']['native_samples']!=16000 or c['energy']['log_floor_v2']!=1e-40 or c['energy']['log_reference_v2']!=1.:raise ValueError('Fixed physical energy convention required')
 if s['native_dt_ms']!=.0625 and c['reuse']['enabled']:raise ValueError('Coarse v4 reuse incompatible with finer integration; disable reuse and update measured plan')
 # An absent amendment must not change legacy serialized configurations or their
 # immutable generation hashes. Explicit amendment settings remain recorded.
 if not c['extension']['enabled']:
  for key in ('extension','information_models'):
   if key not in (raw or {}):c.pop(key)
 return c


def load_config(path):
 text=Path(path).read_text()
 raw=json.loads(text) if Path(path).suffix=='.json' or text.lstrip().startswith('{') else yaml.safe_load(text)
 return resolve(raw)

def contexts(c):
 rows=[];f=c['forks'];reuse=set(c['reuse']['circuit_ids']) if c['reuse']['enabled'] else set()
 for split,ids in c['data']['structures'].items():
  for sid in ids:
   for regime in c['data']['regimes']:
    for prefix in range(c['data']['prefixes_per_regime'][split]):
     high=split=='evaluation' and sid in f['reference_circuits'] and regime['name'] in f['reference_regimes'] and prefix==f['reference_prefix']
     m=(f['high_score_futures'] if high else f['score_futures']) if split=='evaluation' else f['calibration_futures'] if split=='calibration' else 1
     rows.append({'context_id':f's{sid}-p{prefix:04d}-{regime["name"]}','structure_id':sid,'prefix':prefix,'split':split,
                  'regime':deepcopy(regime),'score_futures':m,'reference_futures':f['reference_futures'] if high else 0,
                  'reuse':sid in reuse,'cutoff_ms':c['cutoff_ms']})
 return rows

def counts(c):
 rows=contexts(c);total=sum(r['score_futures']+r['reference_futures'] for r in rows);reuse=sum(r['score_futures']+r['reference_futures'] for r in rows if r['reuse'])
 return {'contexts':len(rows),'circuits':{s:len(v) for s,v in c['data']['structures'].items()},
         'contexts_by_split':{s:sum(r['split']==s for r in rows) for s in c['data']['structures']},
         'score_by_split':{s:sum(r['score_futures'] for r in rows if r['split']==s) for s in c['data']['structures']},
         'primary_evaluation_episodes':sum(c['forks']['score_futures'] for r in rows if r['split']=='evaluation'),
         'h2_extension_episodes':sum(r['score_futures']-c['forks']['score_futures'] for r in rows if r['split']=='evaluation'),
         'reference_episodes':sum(r['reference_futures'] for r in rows),'total_episodes':total,'reused_episodes':reuse,'new_episodes':total-reuse,
         'numerical_episodes':len(c['numerical']['circuit_ids'])*len(c['data']['regimes'])*c['numerical']['futures']*len(c['numerical']['dts_ms'])}

def registry_check(c):
 p=ROOT/'configs/eeg_representation/v6/seed_registry.json';r=json.loads(p.read_text());allowed=set()
 for low,high in r['ranges'].values():allowed.update(range(low,high+1))
 requested={i for v in c['data']['structures'].values() for i in v}|set(c['numerical']['circuit_ids'])
 permitted_old=set(range(36000,36012)) if c['claim_status']=='prospective_confirmation' else set(range(36000,36020))
 if requested-allowed-permitted_old:raise ValueError('Unregistered circuit IDs: '+str(sorted(requested-allowed-permitted_old)))
 return {'status':'complete','registry_hash':canonical_hash(r),'circuit_ids':sorted(requested),'old_test_excluded':c['claim_status']!='prospective_confirmation' or not bool(set(range(36012,36020))&set(c['data']['structures']['evaluation']))}
