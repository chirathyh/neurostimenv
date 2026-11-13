"""Immutable campaign roots, stage fingerprints and protocol provenance."""
from pathlib import Path
import os,platform,subprocess,time,importlib.metadata
from ..artifacts import source_hash,atomic_json,atomic_npz,file_hash,read_json,output_lock,artifact_record,verify_record as _legacy_verify_record,ROOT
from ..targets import canonical_hash
from . import SCHEMA_VERSION,SCIENTIFIC_NAMES


def environment():
    versions={}
    for p in ['numpy','scipy','torch','scikit-learn','h5py','NEURON','LFPy','PyYAML','matplotlib']:
        try:versions[p]=importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:versions[p]=None
    return {'host':platform.node(),'python':platform.python_version(),'platform':platform.platform(),
            'logical_cpus':os.cpu_count(),'versions':versions,'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def initialize(root,config,resume=False):
    root=Path(root);p=root/'provenance.json';ch=canonical_hash(config);sh=source_hash()
    if p.exists():
        old=read_json(p)
        if not resume:raise FileExistsError('Output exists; pass --resume explicitly')
        if old.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Historical output roots are immutable; select a new v4 root')
        if (old.get('config_hash'),old.get('analysis_source_hash'))!=(ch,sh):raise ValueError('Changed configuration/source: select a new root, never relabel frozen outputs')
        resolved=root/'resolved_config.json'
        if not resolved.is_file() or canonical_hash(read_json(resolved))!=ch:raise ValueError('Stored resolved configuration changed or is missing')
        if old.get('numerical_environment_hash'):
            current=environment()
            fingerprint=canonical_hash({k:current[k] for k in ('python','versions')})
            if old['numerical_environment_hash']!=fingerprint:raise ValueError('Numerical environment changed; use a new analysis root')
        return old
    if root.exists() and any(p.name!='.eegrepr.lock' for p in root.iterdir()):raise FileExistsError('Nonempty unversioned v4 root refused')
    root.mkdir(parents=True,exist_ok=True)
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    env=environment()
    result={'schema_version':SCHEMA_VERSION,'analysis_source_hash':sh,'source_hash':sh,'config_hash':ch,
            'commit':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'dirty':bool(git('status','--porcelain')),
            'environment':env,'numerical_environment_hash':canonical_hash({k:env[k] for k in ('python','versions')}),'scientific_names':SCIENTIFIC_NAMES,'created_unix':time.time(),
            'generation_provenance_policy':'Imported data retain their original provenance; this record describes new analysis only.',
            'claim_status':config['claim_status'],'backend':config['backend']}
    atomic_json(p,result);atomic_json(root/'resolved_config.json',config)
    return result


def stage_done(root,name,files,details):
    root=Path(root);records=[]
    for path in files:
        p=Path(path)
        if not p.is_file():raise FileNotFoundError('Stage output missing: '+str(p))
        if not p.resolve().is_relative_to(root.resolve()):raise ValueError('Stage output escapes result root')
        records.append(artifact_record(root,p))
    result={'schema_version':SCHEMA_VERSION,'stage':name,'status':details.get('status','complete'),
            'analysis_source_hash':source_hash(),'config_hash':read_json(root/'provenance.json')['config_hash'],
            'outputs':records,'details':details}
    atomic_json(root/'v4_stages'/f'{name}.json',result);return result


def resume_stage(root,name,resume=False):
    root=Path(root);path=root/'v4_stages'/f'{name}.json'
    if not path.exists():return None
    if not resume:raise FileExistsError('Stage already exists: '+name+'; pass --resume')
    r=read_json(path)
    if r.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Stage schema mismatch')
    if r['analysis_source_hash']!=source_hash() or r['config_hash']!=read_json(root/'provenance.json')['config_hash']:raise ValueError('Stage fingerprint mismatch')
    for row in r['outputs']:verify_record(root,row)
    return r['details']


def verify_record(root,record):
    path=Path(root)/record['path']
    if not path.resolve().is_relative_to(Path(root).resolve()):raise ValueError('Artifact escapes result root')
    return _legacy_verify_record(root,record)
