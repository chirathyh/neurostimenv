"""Lossless exact-NPY export, using the verified v4 container/mapping contract."""
from pathlib import Path
import hashlib,json,os,tempfile,zipfile,io
from .artifacts import read_json,file_hash,ROOT,module_hash
from .reporting import report
from .evidence import EvidenceSource

def _omit(name,full_native=False):
    p=Path(name)
    if p.suffix.lower() in ('.pkl','.pickle','.pt','.pth'):
        return 'serialized checkpoint omitted; numeric predictions retained; no pickle model loading'
    if full_native:return None
    if p.suffix.lower() in ('.h5','.hdf5'):return 'native data; full-native export includes this'
    if (p.name in ('state.npz','weights.npz','readout_weights.npz','readout.npz','model.npz')
            or p.name.endswith('_model.npz') or 'weights' in p.parts):
        return 'readout/model weights; numeric predictions and model metadata retained'
    return None


def export(config,root,destination,source=None,full_native=False,max_bytes=None,include_imported=False,deduplicate_npz=True):
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(root):raise ValueError('Export destination must be outside evidence root')
    if destination.exists():raise FileExistsError('Export exists; choose a new filename')
    owned_source=source is not None and not isinstance(source,EvidenceSource)
    source=EvidenceSource(source) if owned_source else source
    tmp=None
    try:
        missing=[(name,r) for name,r in source.records.items() if not source.exists(name)] if source else []
        missing_native=[name for name,_ in missing if Path(name).suffix.lower() in ('.h5','.hdf5')]
        if full_native and missing_native:
            raise ValueError(f'Full-native export unavailable: {len(missing_native)} native files are absent from supplied source')
        report(config,root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        omitted=[];included=[];seen=set();transformed=[]
        deduplicate_npz=bool(deduplicate_npz and not full_native)
        fd,tmp=tempfile.mkstemp(prefix='.'+destination.name,dir=destination.parent);os.close(fd)
        source_inside_root=bool(source and source.archive is None and source.path.is_relative_to(root))
        def add(z,name,stream,record):
            if name in seen:raise ValueError('Duplicate export member '+name)
            seen.add(name);h=hashlib.sha256();size=0
            with z.open(name,'w',force_zip64=True) as out:
                for b in iter(lambda:stream.read(1024*1024),b''):
                    out.write(b);h.update(b);size+=len(b)
            if record and (h.hexdigest()!=record['sha256'] or size!=record['bytes']):raise ValueError('Export input changed/corrupted '+name)
            included.append({'path':name,'sha256':h.hexdigest(),'bytes':size})
        def add_payload(z,name,stream,record):
            if not deduplicate_npz or not name.endswith('.npz'):
                return add(z,name,stream,record)
            import numpy as np
            blob=stream.read();digest=hashlib.sha256(blob).hexdigest()
            if record and (digest!=record['sha256'] or len(blob)!=record['bytes']):
                raise ValueError('Export input changed/corrupted '+name)
            arrays=[]
            with zipfile.ZipFile(io.BytesIO(blob)) as original:
                members=original.namelist()
                if len(members)!=len(set(members)):raise ValueError('Duplicate numeric NPZ members: '+name)
                for member in members:
                    from ..v4.restore_export import safe_path
                    safe_path(member)
                    if not member.endswith('.npy'):raise ValueError('Non-NPY member in numeric NPZ: '+name+'/'+member)
                    payload=original.read(member);reader=io.BytesIO(payload)
                    version=np.lib.format.read_magic(reader)
                    if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(reader)
                    elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(reader)
                    elif version==(3,0):shape,fortran,dtype=np.lib.format._read_array_header(reader,version)
                    else:raise ValueError('Unsupported NPY version in '+name)
                    if dtype.hasobject:raise ValueError('Object/pickle numeric evidence is forbidden: '+name+'/'+member)
                    import math
                    expected=math.prod(shape)*dtype.itemsize
                    if len(payload)-reader.tell()!=expected:raise ValueError('NPY shape/data length mismatch: '+name+'/'+member)
                    sha=hashlib.sha256(payload).hexdigest();array_path='numeric_arrays/'+sha+'.npy'
                    if array_path not in seen:add(z,array_path,io.BytesIO(payload),None)
                    arrays.append({'key':member[:-4],'member_name':member,'array_path':array_path,
                                   'sha256':sha,'bytes':len(payload),'dtype':dtype.str,
                                   'shape':list(shape),'fortran_order':bool(fortran),'npy_version':list(version)})
            descriptor={'schema_version':'eegrepr-npz-array-map-1','original_path':name,
                'original_sha256':digest,'original_bytes':len(blob),'arrays':arrays,
                'equivalence':'Exact original NPY member bytes, including headers and array values; reconstructed NPZ ZIP container bytes may differ'}
            descriptor_path='numeric_descriptors/'+hashlib.sha256(name.encode()).hexdigest()+'.json'
            add(z,descriptor_path,io.BytesIO(json.dumps(descriptor,sort_keys=True,indent=2).encode()),None)
            transformed.append({'original_path':name,'original_sha256':digest,'original_bytes':len(blob),
                                'descriptor_path':descriptor_path,'array_members':len(arrays),
                                'status':'lossless array transformation, not omitted data'})
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.name.startswith('.') or '__pycache__' in p.parts:continue
                if p.is_symlink():raise ValueError('Symlink in export evidence root')
                name=p.relative_to(root).as_posix();why=_omit(name,full_native)
                if source and source.archive and p==source.path:
                    why='source archive container omitted; individual source members included under imported/'
                rec={'path':'evidence/'+name,'sha256':file_hash(p),'bytes':p.stat().st_size}
                if why:omitted.append({**rec,'reason':why,'checksum_scope':'verified file bytes'});continue
                with p.open('rb') as stream:add_payload(z,'evidence/'+name,stream,rec)
            if source is not None and not source_inside_root:
                # A generated source already inside this result root is included once.
                # External evidence retains its own original provenance namespace.
                for name in source.list_names():
                    if '__pycache__' in Path(name).parts or Path(name).name.startswith('.'):continue
                    why=_omit(name,full_native);rec=source.records.get(name)
                    original_metadata=name in ('resolved_config.json','provenance.json','native/manifest.json','EVIDENCE_MANIFEST.json','timing_audit.json','native/prefix_validation.json')
                    if not why and not full_native and not include_imported and not original_metadata:
                        why='optional original evidence payload omitted to avoid duplication; all current numeric evidence retained; use --include-imported to include original payload'
                    if why:
                        if rec is None:rec=source.verify(name)
                        omitted.append({**rec,'path':'imported/'+name,'reason':why,
                            'checksum_scope':'original manifest for omitted bytes; not copied into this export'})
                        continue
                    with source.open_binary(name) as stream:add_payload(z,'imported/'+name,stream,rec)
                for name,rec in missing:
                    omitted.append({**rec,'path':'imported/'+name,'reason':'unavailable in supplied source; cannot reconstruct omitted raw bytes',
                                    'checksum_scope':'original manifest; bytes unavailable'})
            for ledger in sorted((root/'imports/sources').glob('*/source_omissions.json')):
                source_ledger=read_json(ledger)
                for row in source_ledger['records']:
                    rec={**row,'path':'original_source_omissions/'+ledger.parent.name+'/'+row['path'],'checksum_scope':'original source archive omission manifest; bytes absent from supplied source'}
                    if full_native and Path(row['path']).suffix.lower() in ('.h5','.hdf5'):
                        raise ValueError('Full-native export unavailable: imported source omits native files; use the actual native result tree')
                    omitted.append(rec)
            native_manifest=root/'generation/native/manifest.json'
            if native_manifest.exists():
                for row in read_json(native_manifest)['trials']:
                    physical=Path(row.get('source_root',str(root/'generation')))/row['path']
                    if physical.resolve().is_relative_to(root):continue
                    rec={'path':'external_native/'+row['trial_id']+'.h5','sha256':row['sha256'],'bytes':row['bytes'],'original_path':str(physical)}
                    if full_native:
                        if not physical.is_file():raise FileNotFoundError('Missing original reused native file '+str(physical))
                        with physical.open('rb') as stream:add(z,rec['path'],stream,rec)
                    else:omitted.append({**rec,'reason':'original reused native retained at original source; compact evidence contains derived numeric arrays','checksum_scope':'original generation manifest'})
            codepaths=list((ROOT/'experiments/eeg_representation').rglob('*.py'))+list((ROOT/'configs/eeg_representation/v6').glob('*'))+list((ROOT/'tests').glob('test_eegrepr_v6*.py'))+[ROOT/'scripts/eegrepr/run_h1_h4_v6.sh']+list((ROOT/'docs/eegrepr_v6').glob('*'))
            codepaths += list((ROOT/'configs/eeg_representation/v6_extended').glob('*'))+list((ROOT/'docs/eegrepr_v6_extended').glob('*'))+[ROOT/'scripts/eegrepr/run_h1_h4_v6_extended.sh']
            # The NEURON implementation closure is necessary for independent review,
            # not just the evaluation package. Preserve these as source, never load.
            codepaths += list((ROOT/'setup/circuits/ballnstick').rglob('*'))+list((ROOT/'env/models/neuron').rglob('*'))
            codepaths += list((ROOT/'env/eeg').glob('*.py'))+list((ROOT/'requirements').glob('*.txt'))
            codepaths += [ROOT/'configs/env/ballnstick.yaml',ROOT/'AGENTS.md',ROOT/'requirements.txt',ROOT/'LICENSE']
            for p in sorted(set(codepaths)):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() not in ('.pyc','.so','.o'):
                    with p.open('rb') as stream:add(z,'implementation/'+p.relative_to(ROOT).as_posix(),stream,None)
            readme=('Read evidence/report/FINDINGS.md first. EXPORT_MANIFEST.json records every payload file and explicitly omitted scientific files. '
                    'The manifest itself is excluded from its own hash list. implementation/docs/eegrepr_v6 contains limitations and commands. '
                    'Imported generation provenance remains original. H5/H6 are deferred.\n\n'
                    'Compact numeric NPZ files may be transformed losslessly into deduplicated numeric_arrays/<sha256>.npy members and numeric_descriptors/*.json mappings. '
                    'EXPORT_MANIFEST.json transformed_numeric lists every original logical NPZ path and original container checksum. No targets, predictions, variances, logits or code arrays are omitted by this transformation.\n'
                    'To verify or restore using standard-library Python only, extract implementation/experiments/eeg_representation/h1/v4/restore_export.py, then run:\n'
                    'python restore_export.py PACKAGE.zip --verify-only\n'
                    'python restore_export.py PACKAGE.zip --destination restored\n'
                    'Every restored NPY member is byte-identical. NPZ container hashes may differ because ZIP metadata/compression may differ; RESTORE_REPORT.json records actual restored hashes and original hashes. '
                    'Original scientific manifests remain unchanged. If container hashes differ, inspect/score the verified arrays directly or use the restoration equivalence report; do not pretend original-container hash checks passed.\n'
                    'Full-native or --plain-npz exports retain original NPZ container bytes.\n').encode()
            add(z,'README.txt',io.BytesIO(readme),None)
            manifest={'schema_version':'eegrepr-v6-export-1','status':'complete','full_native':full_native,
                'included':included,'omitted':omitted,'transformed_numeric':transformed,'manifest_self_excluded':True,
                'numeric_storage':'deduplicated exact NPY members' if deduplicate_npz else 'original NPZ containers',
                'unique_numeric_arrays':sum(r['path'].startswith('numeric_arrays/') for r in included),
                'external_import_payload_included':bool(full_native or include_imported),
                'original_metadata_policy':'resolved config, provenance and source/native manifests retained even when optional original numeric payload is omitted',
                'native_files_included':sum(Path(r['path']).suffix.lower() in ('.h5','.hdf5') for r in included),
                'analysis_export_fingerprint':module_hash('reporting','exporting'),'source':source.provenance() if source is not None else None,
                'source_layout':'already included inside evidence/' if source_inside_root else 'external evidence under imported/',
                'scope':'Numeric evidence, metadata, source and reports; no pickle model loading. Imported generation provenance is not replaced.'}
            z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True))
        size=Path(tmp).stat().st_size
        if max_bytes is not None and size>max_bytes:raise ValueError(f'Export {size} bytes exceeds requested limit {max_bytes}; select a smaller source or omit redundant imported evidence explicitly')
        os.replace(tmp,destination)
        digest=file_hash(destination);destination.with_suffix(destination.suffix+'.sha256').write_text(f'{digest}  {destination.name}\n')
        return {'status':'complete','path':str(destination),'bytes':size,'sha256':digest,
                'included_files':len(included),'omitted_files':len(omitted),'full_native':full_native,
                'native_files_included':manifest['native_files_included'],
                'transformed_npz_files':len(transformed),'unique_numeric_arrays':manifest['unique_numeric_arrays']}
    finally:
        if tmp is not None and Path(tmp).exists():Path(tmp).unlink()
        if owned_source:source.close()
