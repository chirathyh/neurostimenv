"""Stage-scoped, atomic provenance; analysis changes never relabel generation."""
from pathlib import Path
import hashlib,json,time,subprocess,platform,traceback
from ..artifacts import ROOT,atomic_json,atomic_npz,read_json,file_hash,artifact_record,verify_record,output_lock
from ..targets import canonical_hash
from ..v4.artifacts import environment
from .config import SCHEMA


def module_hash(*names):
 h=hashlib.sha256()
 for n in sorted(names):
  p=Path(__file__).parent/(n+'.py');h.update(n.encode());h.update(p.read_bytes())
 return h.hexdigest()

def initialize(root,config,resume=False):
 root=Path(root);p=root/'provenance.json';ch=canonical_hash(config)
 if p.exists():
  old=read_json(p)
  if not resume:raise FileExistsError('Existing v6 result root: use --resume')
  if old.get('schema_version')!=SCHEMA or old['config_hash']!=ch:raise ValueError('Immutable result config mismatch; use new root')
  if canonical_hash(read_json(root/'resolved_config.json'))!=ch:raise ValueError('Stored resolved config changed or is missing')
  return old
 if root.exists() and any(x.name!='.eegrepr.lock' for x in root.iterdir()):raise FileExistsError('Nonempty unversioned root refused')
 root.mkdir(parents=True,exist_ok=True)
 def git(*a):return subprocess.check_output(['git',*a],cwd=ROOT,text=True).strip()
 result={'schema_version':SCHEMA,'config_hash':ch,'commit':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),
         'environment':environment(),'created_unix':time.time(),'claim_status':config['claim_status'],
         'provenance_policy':'Stage-specific analysis dependencies; imported native provenance unchanged; no global-source invalidation of native data.'}
 atomic_json(p,result);atomic_json(root/'resolved_config.json',config);return result


def stage_signature(config,modules,inputs=None,extra=None):
 return canonical_hash({'config_hash':canonical_hash(config),'analysis_dependencies':module_hash(*modules),'inputs':inputs or {},'extra':extra or {}})

def resume_record(path,signature,resume):
 p=Path(path)
 if not p.exists():return None
 if not resume:raise FileExistsError(str(p)+' exists; explicit --resume required')
 old=read_json(p)
 if old.get('signature')!=signature:raise ValueError('Stage dependency/device/dtype changed: '+str(p))
 return old


def progress(root,stage,status,**details):
 r={'stage':stage,'status':status,'updated_unix':time.time(),**details}
 atomic_json(Path(root)/'progress'/f'{stage}.json',r);return r


def failure(root,stage,exc):
 p=Path(root)/'failures'/f'{stage}-{time.time_ns()}.json'
 atomic_json(p,{'stage':stage,'status':'failed','error':str(exc),'type':type(exc).__name__,'traceback':traceback.format_exc(),'time_unix':time.time()});return p
