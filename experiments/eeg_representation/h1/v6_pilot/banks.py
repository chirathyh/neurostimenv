"""Reuse verified encoder representations; refit pilot projections/ranges on train."""
from pathlib import Path
import json
import importlib.metadata
import numpy as np
from ..artifacts import atomic_json,atomic_npz,file_hash
from ..targets import canonical_hash
from ..v6.evidence import EvidenceSource
from ..v6.information_models import _forecast_bank,DEFAULTS as CODE_DEFAULTS,_digest
from ..v4.retention import projection_fit,project,fit_range_rule,quantization_attribution
from ...training import feature_matrix
from ...storage import jsonable

def _source_identity(source_archive):
    if source_archive is None:return {'kind':'none','forecast_reuse_available':False}
    names=('retention/banks/forecast__seed_0/representations.npz',
           'retention/task_targets.npz','retention/banks/forecast__seed_0/metadata.json')
    with EvidenceSource(source_archive) as source:
        present=[source.exists(n) for n in names]
        if present[0] and not all(present):raise ValueError('Incomplete supplied forecast bank provenance')
        if all(present):
            meta=source.read_json(names[2])
            if meta.get('representation_family')!='forecast' or meta.get('optimization_seed')!=0 or meta.get('optimizer_updates')!=200 or meta.get('native_width')!=128:
                raise ValueError('Supplied forecast bank does not match the declared frozen seed0/200-update/native128 procedure')
        return dict(source=source.provenance(),forecast_reuse_available=all(present),
            records=[source.verify(n) for n in names if source.exists(n)])


def _contract(config,panel,source_archive,device):
    here=Path(__file__);base=here.parents[2]
    deps=[here,base/'training.py',base/'models.py',here.parent.parent/'v4/retention.py',here.parent.parent/'v6/information_models.py']
    versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','torch','scikit-learn')}
    keys=('X','X_physical','Y','Y_physical','future_mask','context_id','structure_id')
    details=dict(schema='eegrepr-pilot-banks-freeze-1',inputs={split:_digest({k:p[k] for k in keys if k in p}) for split,p in panel.items()},
        source=_source_identity(source_archive),device=device,versions=versions,
        settings=dict(forecast_updates=config['forecast_updates'],forecast_seed=config['forecast_seed'],
            projection_dimension=8,total_bits=32,range_rule='quantile_005_995',sample_hz=500.,
            forecast_dtype='float32',derived_dtype='float64',projection_fit='train only'),
        dependency_sha256={str(p.relative_to(base)):file_hash(p) for p in deps})
    return dict(signature=canonical_hash(details),details=details)


def _npz_immutable(path,arrays):
    path=Path(path)
    if path.exists():
        with np.load(path,allow_pickle=False) as f:
            if set(f.files)!=set(arrays) or any(f[k].dtype!=np.asarray(v).dtype or not np.array_equal(f[k],v) for k,v in arrays.items()):
                raise ValueError('Existing bank numeric artifact differs; refusing overwrite: '+str(path))
    else:atomic_npz(path,**arrays)


def _json_immutable(path,value):
    path=Path(path)
    if path.exists():
        if canonical_hash(json.loads(path.read_text()))!=canonical_hash(jsonable(value)):
            raise ValueError('Existing bank metadata differs; refusing overwrite: '+str(path))
    else:atomic_json(path,value)


def _saved(dest,panel,contract):
    manifest=json.loads((dest/'manifest.json').read_text())
    if manifest.get('signature')!=contract['signature']:raise ValueError('Frozen bank inputs/source/settings changed')
    checksum=manifest.pop('manifest_content_sha256',None)
    if checksum!=canonical_hash(manifest):raise ValueError('Frozen bank manifest content changed')
    for name,digest in manifest['files'].items():
        path=dest/name
        if not path.is_file() or file_hash(path)!=digest:raise ValueError('Frozen bank artifact checksum mismatch: '+name)
    out={}
    for name in ('feature18','pca','forecast'):
        with np.load(dest/(name+'.npz'),allow_pickle=False) as f:
            for split,p in panel.items():
                if not np.array_equal(f[split+'_context_id'],p['context_id']):raise ValueError('Frozen bank context order changed')
            for variant,suffix in (('unquantized','_projected'),('B32','_decoded')):
                key=name+'_'+variant
                out[key]=dict(values={split:f[split+suffix].copy() for split in panel},metadata=manifest['methods'][key])
    return out


def build(config,root,panel,source_archive,device='cpu',resume=False):
    root=Path(root);dest=root/'codes';contract=_contract(config,panel,source_archive,device)
    if (dest/'input_contract.json').exists():
        if not resume:raise FileExistsError('Code bank exists; explicit resume required')
        if json.loads((dest/'input_contract.json').read_text())!=contract:
            raise ValueError('Immutable bank input/source/settings contract changed')
    elif dest.exists() and any(dest.iterdir()):
        raise ValueError('Unversioned partial code bank cannot be overwritten')
    else:
        dest.mkdir(parents=True,exist_ok=True);atomic_json(dest/'input_contract.json',contract)
    if (dest/'manifest.json').exists():return _saved(dest,panel,contract)
    raw={};meta={}
    raw['feature18']={s:feature_matrix(p['X'][:,None],sample_hz=500.) for s,p in panel.items()}
    meta['feature18']=dict(method='fixed 18 features',pretraining='none')
    pca=projection_fit(panel['train']['X'],min(128,len(panel['train']['X'])-1,panel['train']['X'].shape[1]))
    raw['pca']={s:project(p['X'],pca) for s,p in panel.items()}
    _npz_immutable(dest/'pca_native.npz',{k:v for k,v in pca.items() if isinstance(v,np.ndarray)})
    meta['pca']=dict(method='train-only PCA',pretraining='current cohort training only')
    imported=False
    if source_archive is not None:
        with EvidenceSource(source_archive) as source:
            name='retention/banks/forecast__seed_0/representations.npz'
            if source.exists(name) and source.exists('retention/task_targets.npz'):
                reps=source.read_npz(name);targets=source.read_npz('retention/task_targets.npz')
                ids={str(cid):(s,i) for s in ('train','development','evaluation') for i,cid in enumerate(reps[s+'_context_id'])}
                values={}
                for split,p in panel.items():
                    rows=[]
                    for i,cid in enumerate(p['context_id']):
                        if str(cid) not in ids:raise ValueError('Reused forecast representation lacks context '+str(cid))
                        original,j=ids[str(cid)]
                        if str(targets[original+'_context_id'][j])!=str(cid) or not np.array_equal(targets[original+'_X_physical'][j],p['X_physical'][i]):
                            raise ValueError('Forecast code inputs/IDs differ from original physical EEG')
                        if not np.array_equal(targets[original+'_Y_physical'][j],p['Y_physical'][i]):
                            raise ValueError('Forecast anchor target/future pairing differs')
                        rows.append(reps[original][j])
                    values[split]=np.asarray(rows,np.float64)
                raw['forecast']=values;imported=True
                meta['forecast']=dict(method='reused forecast seed0 encoder, 200 updates',
                    pretraining='original 256 train contexts/eight train circuits; differs from matched readout cohort',
                    source=source.provenance(),source_records=[source.verify(name),source.verify('retention/task_targets.npz')],
                    metadata=source.read_json('retention/banks/forecast__seed_0/metadata.json'))
    if not imported:
        encoder_dir=dest/'forecast_encoder'
        if encoder_dir.exists() and any(encoder_dir.iterdir()) and not (encoder_dir/'freeze.json').exists():
            raise ValueError('Unfrozen partial forecast encoder cannot be overwritten')
        options={**CODE_DEFAULTS,'forecast_updates':config['forecast_updates'],'forecast_seeds':[0]}
        raw['forecast'],meta['forecast']=_forecast_bank(root,panel,options,0,dest/'forecast_encoder',device,resume)
        meta['forecast']['pretraining']='current cohort training only; one bounded pilot encoder'
    out={}
    for name,values in raw.items():
        projection=projection_fit(values['train'],8);projected={s:project(x,projection) for s,x in values.items()}
        book=fit_range_rule(projected['train'],32,'quantile_005_995');decoded={};arrays={};clips={}
        for split,x in projected.items():
            packed,decoded[split],clips[split]=quantization_attribution(x,book)
            if packed.shape!=(len(x),4):raise ValueError('d8/B32 packing failed')
            arrays.update({split+'_native':values[split],split+'_projected':x,split+'_decoded':decoded[split],split+'_packed':packed,
                split+'_context_id':panel[split]['context_id']})
        _npz_immutable(dest/(name+'.npz'),arrays)
        _npz_immutable(dest/(name+'_projection.npz'),{k:v for k,v in projection.items() if isinstance(v,np.ndarray)})
        _json_immutable(dest/(name+'_codebook.json'),book)
        for variant,v in [('unquantized',projected),('B32',decoded)]:
            out[name+'_'+variant]=dict(values=v,metadata=dict(meta[name],representation=name,variant=variant,
                dimension=8,bits=None if variant=='unquantized' else 32,clipping=clips,
                range_fit='current cohort train only; quantiles .005/.995',projection_fit='current cohort train only',
                input_id=_digest({s+'_X':p['X'] for s,p in panel.items()})))
    files={p.relative_to(dest).as_posix():file_hash(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    manifest=dict(status='complete',signature=contract['signature'],forecast_reused=imported,
        methods={n:v['metadata'] for n,v in out.items()},files=files,
        training_context_ids=panel['train']['context_id'].tolist(),readout_training='current pilot matched/broad train population')
    manifest['manifest_content_sha256']=canonical_hash(manifest)
    _json_immutable(dest/'manifest.json',manifest)
    return out
