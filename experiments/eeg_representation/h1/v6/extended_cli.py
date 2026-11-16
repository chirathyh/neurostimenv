"""One complete, gated V6 campaign with explicit immediate-information additions."""
from pathlib import Path
from copy import copy,deepcopy
import argparse,json,os,subprocess,sys,tempfile,time
from . import cli
from .config import load_config,resolve
from .artifacts import ROOT,atomic_json,read_json,canonical_hash,file_hash

ALIASES={'preflight':'doctor','dry-run':'plan','complete':'run','resume':'run'}
EXTRA_MODES=('preflight','dry-run','smoke','complete','resume','status','report-only')
SMOKE_MODULES=('tests.test_eegrepr_v6_extended_targets','tests.test_eegrepr_v6_information',
               'tests.test_eegrepr_v6_information_models','tests.test_eegrepr_v6_statistics')


def parser():
 p=cli.parser();p.description=__doc__
 for action in p._actions:
  if action.dest=='mode':action.choices=tuple(dict.fromkeys((*cli.MODES,*EXTRA_MODES)))
 p.set_defaults(config=str(ROOT/'configs/eeg_representation/v6_extended/confirmation.yaml'))
 return p


def existing_root(root):
 """Resolve a saved stage or the actual pretest-selected campaign, never N192 by guess."""
 root=Path(root).expanduser().resolve()
 if (root/'resolved_config.json').is_file():return root,root/'resolved_config.json'
 if (root/'selected-config.json').is_file():
  c=load_config(root/'selected-config.json');core=root/('confirmation-n'+str(c['protocol']['test_circuits']))
  if (core/'resolved_config.json').is_file():
   if canonical_hash(load_config(core/'resolved_config.json'))!=canonical_hash(c):raise ValueError('Core and selected configuration disagree; no report regeneration')
   return core,core/'resolved_config.json'
  # A selected design can be blocked before core initialization. Its inspected
  # Phase0 evidence remains reportable without manufacturing fresh outcomes.
 if (root/'phase0/resolved_config.json').is_file():return root/'phase0',root/'phase0/resolved_config.json'
 raise FileNotFoundError('No initialized stage, selected core, or Phase0 to report: '+str(root))


def status(root):
 """Read-only artifact status. A saved running record does not prove process liveness."""
 root=Path(root).expanduser().resolve();out={'root':str(root),'status':'not_started' if not root.exists() else 'observed',
  'read_only':True,'process_liveness':'not checked; saved progress is not proof that a process is running','campaign':{},'stages':[],'failures':[]}
 if not root.exists():return out
 for name in ('campaign_progress.json','campaign_result.json','campaign_completed.json','selected-config.json'):
  p=root/name
  if not p.is_file():continue
  try:
   r=read_json(p)
   if name=='selected-config.json':r={'test_circuits':r.get('protocol',{}).get('test_circuits'),'config_hash':canonical_hash(r),'extension':r.get('extension')}
   elif name=='campaign_completed.json':
    valid=r.get('receipt_hash')==canonical_hash({k:v for k,v in r.items() if k!='receipt_hash'})
    r={'recorded_status':r.get('status'),'receipt_hash_valid':valid,'export':r.get('export'),'archive_payloads_verified_this_call':False}
   out['campaign'][name]=r
  except (OSError,ValueError) as exc:out['campaign'][name]={'status':'unreadable','error':str(exc)}
 roots=[root]
 for p in (root/'phase0',root/'timing-gate'):
  if p.is_dir():roots.append(p)
 roots.extend(sorted(p for p in root.glob('confirmation-n*') if p.is_dir()))
 inventory={'derived':'derived/manifest.json','fit':'models/manifest.json','calibration':'calibration/manifest.json',
  'information_readouts':'information_models/manifest.json','information':'information/manifest.json','replay':'replay/summary.json',
  'plan':'plan.json','smoke':'smoke_result.json','statistics':'confirmation_statistics.json','report':'report/main_analysis.json'}
 for stage_root in roots:
  row={'root':str(stage_root),'progress':{},'artifacts':{}}
  for p in sorted((stage_root/'progress').glob('*.json')) if (stage_root/'progress').is_dir() else []:
   try:row['progress'][p.stem]=read_json(p)
   except (OSError,ValueError) as exc:row['progress'][p.stem]={'status':'unreadable','error':str(exc)}
  for key,rel in inventory.items():
   p=stage_root/rel
   if not p.is_file():continue
   try:
    r=read_json(p);row['artifacts'][key]={k:r[k] for k in ('status','passed','completed_jobs','requested_jobs','eligible_numerical_execution','eligible_for_generation','blocked_reasons') if k in r}
    if 'unavailable_jobs' in r:row['artifacts'][key]['unavailable_jobs']=len(r['unavailable_jobs'])
   except (OSError,ValueError) as exc:row['artifacts'][key]={'status':'unreadable','error':str(exc)}
  out['stages'].append(row)
  for p in sorted((stage_root/'failures').glob('*.json'))[-5:] if (stage_root/'failures').is_dir() else []:
   try:r=read_json(p);out['failures'].append({'path':str(p),**{k:r[k] for k in ('stage','status','error','type','time_unix') if k in r}})
   except (OSError,ValueError) as exc:out['failures'].append({'path':str(p),'error':str(exc)})
 return out


def smoke(args):
 """Run real scientific engineering controls, never the full NEURON campaign."""
 root=Path(args.root).expanduser().resolve();c=load_config(args.config)
 c=deepcopy(c);c.update(claim_status='engineering',backend='analytical',name='extended_scientific_smoke')
 c['namespace']=c['namespace']+'-smoke';c=resolve(c)
 a=copy(args);a.mode='controls';start=time.monotonic()
 with tempfile.TemporaryDirectory(prefix='eegrepr-v6-extended-smoke-') as tmp:
  path=Path(tmp)/'config.json';atomic_json(path,c);a.config=str(path)
  controls=cli.stage(a)
 command=[sys.executable,'-m','unittest','-v',*SMOKE_MODULES]
 print('Scientific smoke tests: '+' '.join(command),flush=True)
 log=root/'smoke_tests.log'
 with log.open('w') as stream:
  result=subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=False)
 passed=controls.get('passed') is True and result.returncode==0
 value={'status':'complete' if passed else 'failed','passed':passed,'scope':'Synthetic analytic/saved-array controls plus known learnable neural path; no full-network NEURON generation and no study outcome',
  'test_modules':list(SMOKE_MODULES),'test_command':command,'test_exit_code':result.returncode,'test_log':str(log),'test_log_sha256':file_hash(log),
  'controls':str(root/'controls.json'),'controls_sha256':file_hash(root/'controls.json'),'elapsed_seconds':time.monotonic()-start,
  'extension':c['extension'],'config_hash':canonical_hash(c)}
 atomic_json(root/'smoke_result.json',value)
 if not passed:print(log.read_text()[-12000:],file=sys.stderr)
 return value


def dispatch(args):
 original=args.mode
 if original=='status':return status(args.root)
 if original=='report-only':
  root,config=existing_root(args.root);args=copy(args);args.root=str(root);args.config=str(config);args.mode='report';args.resume=True;args.workers=None
 else:
  args=copy(args);args.mode=ALIASES.get(original,original)
  if original=='resume':args.resume=True
 c=load_config(args.config)
 if not c.get('extension',{}).get('enabled'):raise ValueError('The extended wrapper requires extension.enabled=true; original V6 reports remain available through run_h1_h4_v6.sh')
 if original=='smoke':return smoke(args)
 if args.mode=='run':
  value=cli.campaign(args);atomic_json(Path(args.root)/'campaign_result.json',value);return value
 return cli.stage(args)


def main(argv=None):
 os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');args=parser().parse_args(argv)
 try:
  value=dispatch(args);print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False),flush=True)
  return 2 if value.get('status') in ('blocked','failed') else 0
 except (Exception,KeyboardInterrupt) as exc:
  print('V6 EXTENDED ERROR: '+str(exc),file=sys.stderr,flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
