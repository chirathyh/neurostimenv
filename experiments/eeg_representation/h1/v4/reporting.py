"""Plain-language findings and numeric, checksum-verifiable expert exports."""
from pathlib import Path
import csv,hashlib,json,os,tempfile,zipfile,shutil,io
from .artifacts import atomic_json,read_json,file_hash,source_hash,ROOT
from .evidence import EvidenceSource
from . import SCIENTIFIC_NAMES


def report(config,root):
    root=Path(root);out=root/'report';out.mkdir(exist_ok=True)
    findings={'status':'complete','claim_status':config['claim_status'],'scientific_names':SCIENTIFIC_NAMES,
              'analysis_source_hash':source_hash(),'stages':{},'selected_models':[]}
    locations={'audit':'audit_existing.json','measurement':'measurement/qualification.json','derivation':'derived/manifest.json',
        'models':'models/manifest.json','scores':'scores/manifest.json','H2':'replication/manifest.json',
        'H3':'retention/summary.json','geometry':'geometry/summary.json','plan':'plan.json','protocol':'protocol.json','generation':'generation_result.json'}
    for name,rel in locations.items():
        if (root/rel).exists():
            value=read_json(root/rel);findings['stages'][name]={'status':value.get('status','complete'),'path':rel,'sha256':file_hash(root/rel)}
        else:findings['stages'][name]={'status':'unrun','path':rel}
    if (root/'scores/manifest.json').exists():
        for job in read_json(root/'scores/manifest.json')['jobs']:
            table=read_json(root/job['path'])
            for row in table['rows']:
                if row['selected_on_development']:
                    findings['selected_models'].append({'job_id':job['id'],'candidate':row['candidate'],'calibration':row['calibration'],
                        'skill':row['skill'],'risk_over_training_mean':row['mean_risk_over_mean_baseline']})
    simulation=config.get('simulation',{})
    population=f"{simulation.get('population_e',32)}E/{simulation.get('population_i',8)}I"
    text=['# NeuroStimEnv manuscript v4 evidence brief','',
        f'This campaign uses {config.get("backend","unspecified")} data with a configured {population} toy circuit and canonical 6.3 C settings. Analytical engineering controls are not biophysical evidence. NEURON runs use ideal EEG and deterministic HH somata; none is human calibration or clinical evidence.',
        '',f"Campaign status: {config['claim_status']}. Source fingerprint: `{findings['analysis_source_hash']}`.",'',
        'H1 — prediction-risk accounting. The physical-state conditional variance is a reference for the exact declared target and population. It is not the EEG-history Bayes floor. Risk minus that reference combines observation, representation and fitting gaps; finite-M corrected negative estimates are retained.',
        '', 'H2 — few independent futures. Scoring futures and finite reference futures are disjoint. A finite L reference remains noisy; mean-reference comparisons subtract its estimated variance/L and retain uncertainty. Optimizer repeats and repeated futures are not independent circuits.',
        '', 'H3 — task-relevant compression. Results compare complete encoder/projection/quantizer/decoder pipelines. Native and projected unquantized controls, exact raw reconstruction, train-PCA inverse, clipping and packed payloads separate losses. Cross-entropy is decoder dependent, not total mutual information. Rhythm uses balanced accuracy; carrier is conditional on rhythmic membership.',
        '', 'H4 — target/prediction scale. A0 retains the original imperfect FIR. A1 is completed native 2-ms bins. A2 is a separately named causal Butterworth acquisition. T0 native waveform, T1 future-block centered waveform and T2 mean-square/AC targets remain visible. T3 needs new longer episodes: a future-only one-second 54-coordinate 4–30 Hz Fourier block, available after the entire block, not a 10-ms-ahead sample.',
        '', 'H5/H6 objective and scheduling superiority are deferred. Covariance rank and participation ratios describe declared sample populations, not minimal biological dimension or a fraction of total information.', '', '## Recorded stage status','']
    text += [f"- {name}: {row['status']} (`{row['path']}`)." for name,row in findings['stages'].items()]
    if findings['selected_models']:
        text+=['','## Development-selected models on evaluation circuits','',
               'Selection used development only. Reanalysis of previously inspected circuits is exploratory. Ratios compare each model with its own target-specific training-mean baseline; standardized MSE across different targets is not a common scale.','',
               '| Job | Selected candidate | Evaluation skill |','|---|---|---:|']
        text += [f"| {r['job_id']} | {r['candidate']} | {100*r['skill']:.4f}% |" for r in findings['selected_models'] if r['calibration']=='calibrated']
    if (root/'retention/summary.json').exists():
        contrast=read_json(root/'retention/summary.json').get('matched_code_contrast',{'status':'unavailable'})
        findings['matched_code_contrast']=contrast
        text+=['','## Frozen matched-code contrast','',
               'Status: '+contrast.get('status','unavailable')+'. Optimizer losses are averaged within circuit before inference; optimizer seeds do not increase independent N.',
               '', 'See `retention/matched_contrast.json` for the exact endpoints, circuit effects and uncertainty. This is task-retention attribution, not an H5/H6 objective-superiority result.']
    text+=['','## Next decision','',
        'Review the measurement qualification, candidate-level scores, calibration/PIT, circuit uncertainty, representation attribution and finite-reference limitations before freezing the prospective protocol. Zero or negative skill is a valid result. The new 832-episode design is bounded prospective evaluation, not powered confirmation. Run the eight-episode full-network timing gate on the actual host, save its measured plan, and only then explicitly execute generation. Native generation has no CUDA speedup here.',
        '', 'The compact export retains all current target arrays, candidate predictions, variances, logits, code payloads, IDs, operators, tables and reports. It omits native HDF5 and fitted weights. Repeated numeric arrays are stored once as exact NPY members with per-NPZ mappings and a standard-library restore tool; this is a lossless transformation, not an omission. Original NPZ container hashes remain recorded, while restored ZIP container hashes may differ. Optional duplicate original payload is excluded by default; original configuration, provenance and manifests remain, and every excluded source member has a size, checksum and reason. Use --include-imported for the original numeric payload or --full-native for all available native data and numeric weights. No pickle model loading is needed.',
        '', 'Questions for the expert: Is the declared primary A2/T3 target suitable for the scientific position? Is the matched-code carrier contrast useful after attribution controls? What circuit-level precision and practical effect should a later frozen confirmation target? Which additional biological sensitivity would change the interpretation?']
    (out/'FINDINGS.md').write_text('\n'.join(text)+'\n');atomic_json(out/'findings.json',findings)
    rows=[r for r in findings['selected_models'] if r['calibration']=='calibrated']
    if rows:
        import matplotlib;matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(8,max(3,len(rows)*.5)))
        ax.barh([r['job_id'] for r in rows],[100*r['skill'] for r in rows]);ax.axvline(0,color='black',lw=.8)
        ax.set_xlabel('Evaluation skill versus target-specific train mean (%)');ax.set_title('Development-selected models; exploratory if inspected circuits')
        fig.tight_layout();fig.savefig(out/'selected_skill.png',dpi=180);fig.savefig(out/'selected_skill.pdf');plt.close(fig)
    return findings


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
                    from .restore_export import safe_path
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
            codepaths=list((ROOT/'experiments/eeg_representation').rglob('*.py'))+list((ROOT/'configs/eeg_representation/v4').glob('*'))+list((ROOT/'tests').glob('test_eegrepr_v4*.py'))+[ROOT/'scripts/eegrepr/run_h1_h4_v4.sh']+list((ROOT/'docs/eegrepr_v4').glob('*'))
            # The NEURON implementation closure is necessary for independent review,
            # not just the evaluation package. Preserve these as source, never load.
            codepaths += list((ROOT/'setup/circuits/ballnstick').rglob('*'))+list((ROOT/'env/models/neuron').rglob('*'))
            codepaths += list((ROOT/'env/eeg').glob('*.py'))+list((ROOT/'requirements').glob('*.txt'))
            codepaths += [ROOT/'configs/env/ballnstick.yaml',ROOT/'AGENTS.md',ROOT/'requirements.txt',ROOT/'LICENSE']
            for p in sorted(set(codepaths)):
                if p.is_file() and '__pycache__' not in p.parts and p.suffix.lower() not in ('.pyc','.so','.o'):
                    with p.open('rb') as stream:add(z,'implementation/'+p.relative_to(ROOT).as_posix(),stream,None)
            readme=('Read evidence/report/FINDINGS.md first. EXPORT_MANIFEST.json records every payload file and explicitly omitted scientific files. '
                    'The manifest itself is excluded from its own hash list. implementation/docs/eegrepr_v4 contains limitations and commands. '
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
            manifest={'schema_version':'eegrepr-v4-export-1','status':'complete','full_native':full_native,
                'included':included,'omitted':omitted,'transformed_numeric':transformed,'manifest_self_excluded':True,
                'numeric_storage':'deduplicated exact NPY members' if deduplicate_npz else 'original NPZ containers',
                'unique_numeric_arrays':sum(r['path'].startswith('numeric_arrays/') for r in included),
                'external_import_payload_included':bool(full_native or include_imported),
                'original_metadata_policy':'resolved config, provenance and source/native manifests retained even when optional original numeric payload is omitted',
                'native_files_included':sum(Path(r['path']).suffix.lower() in ('.h5','.hdf5') for r in included),
                'analysis_source_hash':source_hash(),'source':source.provenance() if source is not None else None,
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
