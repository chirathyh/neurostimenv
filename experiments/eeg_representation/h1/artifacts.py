"""Atomic, checksum-aware H1 artifacts isolated from frozen legacy result roots."""
from pathlib import Path
import os, sys, json, tempfile, hashlib, subprocess, platform, time
import numpy as np
from ..storage import atomic_json, read_json, file_hash, output_lock, rng_seed, jsonable
from . import SCHEMA_VERSION
from .targets import canonical_hash

ROOT=Path(__file__).resolve().parents[3]


def source_hash():
    paths=list((ROOT/'experiments/eeg_representation').rglob('*.py'))
    paths+=list((ROOT/'setup/circuits/ballnstick').glob('*.py'))+list((ROOT/'setup/circuits/ballnstick').glob('*.hoc'))
    paths+=list((ROOT/'env/models/neuron').glob('*.py'))+[ROOT/'configs/env/ballnstick.yaml']
    h=hashlib.sha256()
    for p in sorted(paths):h.update(str(p.relative_to(ROOT)).encode());h.update(p.read_bytes())
    return h.hexdigest()


def atomic_npz(path, **arrays):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for name,value in arrays.items():
        if np.asarray(value).dtype.hasobject:raise ValueError(f'Object/pickle array forbidden: {name}')
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name,suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return file_hash(path)


def require_compatible(root,config):
    p=read_json(Path(root)/'provenance.json')
    if p.get('schema_version')!=SCHEMA_VERSION:raise ValueError('Legacy/unknown output root is immutable; choose a NEW H1 output directory')
    if p.get('config_hash')!=canonical_hash(config):raise ValueError('Incompatible H1 config; use a fresh immutable output root')
    if p.get('source_hash')!=source_hash():raise ValueError('H1 source changed; use a fresh output root')
    return p


def initialize_output(root,config,resume=False):
    root=Path(root)
    if (root/'provenance.json').exists():
        if not resume:raise FileExistsError('H1 output exists; explicit --resume required')
        return require_compatible(root,config)
    if root.exists() and any(p.name!='.eegrepr.lock' for p in root.iterdir()):raise FileExistsError('Refusing nonempty unversioned output root')
    root.mkdir(parents=True,exist_ok=True)
    def git(*args):
        try:return subprocess.check_output(['git',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        except (OSError,subprocess.CalledProcessError):return 'unavailable'
    import importlib.metadata
    versions={}
    for name in ['numpy','scipy','torch','h5py','scikit-learn','NEURON','LFPy','PyYAML','matplotlib']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    diff=git('diff','HEAD','--');status=git('status','--porcelain')
    p={'schema_version':SCHEMA_VERSION,'config_hash':canonical_hash(config),'source_hash':source_hash(),
       'commit':git('rev-parse','HEAD'),'branch':git('branch','--show-current'),'dirty':bool(status),
       'git_status':status,'dirty_diff_sha256':hashlib.sha256(diff.encode()).hexdigest(),
       'python':platform.python_version(),'host':platform.node(),'machine':platform.machine(),
       'processor':platform.processor(),'logical_cpus':os.cpu_count(),'versions':versions,'created_unix':time.time(),
       'claim_status':config['claim_status'],'backend':config['backend'],
       'code_includes_untracked_h1_files_by_content_hash':True}
    atomic_json(root/'resolved_config.json',config);atomic_json(root/'provenance.json',p)
    (root/'source_dirty.diff').write_text(diff+'\n')
    return p


def artifact_record(root,path):
    path=Path(path)
    return {'path':path.relative_to(root).as_posix(),'sha256':file_hash(path),'bytes':path.stat().st_size}


def verify_record(root,record):
    rel=Path(record['path'])
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe artifact path')
    p=Path(root)/rel
    if not p.is_file() or p.stat().st_size!=record['bytes'] or file_hash(p)!=record['sha256']:
        raise ValueError(f'Missing/corrupt artifact: {p}')
    return p


def stage_complete(root,stage,paths,details=None):
    root=Path(root)
    result={'schema_version':SCHEMA_VERSION,'stage':stage,'status':'complete',
      'config_hash':read_json(root/'provenance.json')['config_hash'],
      'source_hash':source_hash(),'outputs':[artifact_record(root,Path(p)) for p in paths],
      'details':details or {}}
    atomic_json(root/'stages'/f'{stage}.json',result)
    return result


def resume_stage(root,stage,resume):
    p=Path(root)/'stages'/f'{stage}.json'
    if not p.exists():return None
    if not resume:raise FileExistsError(f'{stage} exists; use --resume')
    state=read_json(p)
    if state.get('schema_version')!=SCHEMA_VERSION or state.get('source_hash')!=source_hash():raise ValueError('Stale stage schema/source')
    for row in state['outputs']:verify_record(root,row)
    return state
