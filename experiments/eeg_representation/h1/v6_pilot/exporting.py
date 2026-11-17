"""Lossless pilot numeric export with explicit native/ordinary-weight omissions."""
from pathlib import Path
import hashlib,io,json,os,tempfile,zipfile
from ..artifacts import ROOT,file_hash,atomic_json
from ..v4.restore_export import verify_archive,safe_path

def export(root,destination):
    root=Path(root).resolve();dest=Path(destination).resolve()
    if dest.is_relative_to(root):raise ValueError('Export must be outside evidence root')
    if dest.exists():raise FileExistsError('Choose a new export filename')
    dest.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=dest.parent,prefix='.'+dest.name);os.close(fd)
    included=[];omitted=[];transformed=[];seen=set()
    def add(z,name,data):
        if name in seen:raise ValueError('Duplicate member '+name)
        seen.add(name);z.writestr(name,data);included.append(dict(path=name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data)))
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for p in sorted(root.rglob('*')):
                if not p.is_file() or p.name.startswith('.') or '__pycache__' in p.parts:continue
                if p.is_symlink():raise ValueError('Symlink in evidence export')
                name='evidence/'+p.relative_to(root).as_posix()
                why=None
                if p.suffix in ('.h5','.hdf5'):why='Native replay/extension files remain in result root; exact targets/state inputs retained.'
                if p.name in ('state.npz','readout_weights.npz') and 'complete_state' not in p.parts:why='Ordinary readout weights omitted; predictions/metadata retained. Complete-state weights are included.'
                if p.suffix in ('.pkl','.pickle','.pt','.pth'):why='Serialized model not loaded or exported; numeric evidence retained.'
                if why:
                    omitted.append(dict(path=name,sha256=file_hash(p),bytes=p.stat().st_size,reason=why));continue
                blob=p.read_bytes()
                if p.suffix!='.npz':add(z,name,blob);continue
                mappings=[];members=set()
                with zipfile.ZipFile(io.BytesIO(blob)) as npz:
                    for member in npz.namelist():
                        safe_path(member)
                        if member in members:raise ValueError('Duplicate NPZ member: '+member)
                        members.add(member)
                        if not member.endswith('.npy'):raise ValueError('Unexpected NPZ member')
                        data=npz.read(member);sha=hashlib.sha256(data).hexdigest();arraypath='numeric_arrays/'+sha+'.npy'
                        # Reject object arrays without unpickling supplied data.
                        import numpy as np
                        array=np.load(io.BytesIO(data),allow_pickle=False)
                        if array.dtype.hasobject:raise ValueError('Object arrays forbidden')
                        if array.dtype.fields is not None:raise ValueError('Structured dtype requires an explicit versioned descriptor; homogeneous numeric/string arrays are supported')
                        if arraypath not in seen:add(z,arraypath,data)
                        mappings.append(dict(key=member[:-4],member_name=member,array_path=arraypath,sha256=sha,bytes=len(data),dtype=str(array.dtype),shape=list(array.shape)))
                descriptor_path='numeric_descriptors/'+hashlib.sha256(name.encode()).hexdigest()+'.json'
                descriptor=dict(schema_version='eegrepr-npz-array-map-1',original_path=name,original_sha256=hashlib.sha256(blob).hexdigest(),original_bytes=len(blob),arrays=mappings)
                add(z,descriptor_path,json.dumps(descriptor,sort_keys=True).encode())
                transformed.append(dict(original_path=name,original_sha256=descriptor['original_sha256'],original_bytes=len(blob),descriptor_path=descriptor_path,array_members=len(mappings)))
            design=root/'design.json'
            if design.exists():
                # Historical recordings are reused by reference and may live outside
                # this result root; their original hashes remain explicit omissions.
                historical=json.loads(design.read_text()).get('historical_rows',[])
                for row in historical:
                    logical='historical_native/'+str(row['trial_id'])+'.h5';safe_path(logical)
                    omitted.append(dict(path=logical,sha256=row['sha256'],bytes=row['bytes'],
                        original_path=row.get('path'),source_root=row.get('source_root'),
                        reason='Reused original native recording remains at its original source; generation provenance is preserved in evidence/design.json.'))
            for pattern in ('experiments/eeg_representation/**/*.py','configs/eeg_representation/v6_reporting_pilot/*','docs/eegrepr_v6_reporting_pilot/*','tests/test_eegrepr_v6_pilot*.py'):
                for p in sorted(ROOT.glob(pattern)):
                    if p.is_file():add(z,'implementation/'+p.relative_to(ROOT).as_posix(),p.read_bytes())
            p=ROOT/'scripts/eegrepr/run_h1_h4_v6_reporting_pilot.sh';add(z,'implementation/'+p.relative_to(ROOT).as_posix(),p.read_bytes())
            manifest=dict(schema_version='eegrepr-pilot-export-1',included=included,omitted=omitted,transformed_numeric=transformed,
                numeric_storage='exact original NPY bytes; NPZ container timestamps/compression may differ after restoration',
                complete_state_weights_included=True,native_files_included=0,
                restoration='implementation/experiments/eeg_representation/h1/v4/restore_export.py')
            z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,sort_keys=True))
        checked=verify_archive(tmp);checked.pop('manifest')
        # Verification precedes publication, so an invalid archive never acquires the final filename.
        os.replace(tmp,dest)
        sha=file_hash(dest);dest.with_suffix(dest.suffix+'.sha256').write_text(sha+'  '+dest.name+'\n')
        return dict(status='complete',path=str(dest),sha256=sha,bytes=dest.stat().st_size,verification=checked)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()
