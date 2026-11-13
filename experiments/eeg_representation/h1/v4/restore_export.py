"""Verify and restore a compact v4 export using only standard-library Python.

Every original NPY member is byte-identical after restoration. NPZ is a ZIP
container: compression/header timestamps can change, so restored NPZ container
hashes are not claimed equal to the preserved original-container SHA256.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def safe_path(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name:
        raise ValueError('Unsafe/noncanonical archive path: '+str(name))
    return name


def _digest(stream):
    h=hashlib.sha256();size=0
    for chunk in iter(lambda:stream.read(1024*1024),b''):
        h.update(chunk);size+=len(chunk)
    return h.hexdigest(),size


def verify_archive(archive):
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP members')
        for name in names:safe_path(name)
        manifest=json.loads(z.read('EXPORT_MANIFEST.json'))
        included=manifest['included'];mapping={r['path']:r for r in included}
        if len(mapping)!=len(included):raise ValueError('Duplicate manifest records')
        if set(names)!={'EXPORT_MANIFEST.json'}|set(mapping):raise ValueError('ZIP/manifest member mismatch')
        for row in included:
            safe_path(row['path'])
            with z.open(row['path']) as f:digest,size=_digest(f)
            if (digest,size)!=(row['sha256'],row['bytes']):raise ValueError('Checksum mismatch: '+row['path'])
        transformed=manifest.get('transformed_numeric',[]);logical=set(mapping)
        for row in transformed:
            original=safe_path(row['original_path'])
            if original in logical:raise ValueError('Conflicting original/transformed logical path')
            logical.add(original)
            descriptor=json.loads(z.read(row['descriptor_path']))
            if descriptor['original_path']!=original or descriptor['original_sha256']!=row['original_sha256']:
                raise ValueError('NPZ descriptor/original contract mismatch')
            members=set()
            for a in descriptor['arrays']:
                safe_path(a['member_name']);safe_path(a['array_path'])
                if a['member_name'] in members:raise ValueError('Duplicate NPZ member')
                members.add(a['member_name'])
                stored=mapping.get(a['array_path'])
                if stored is None or stored['sha256']!=a['sha256'] or stored['bytes']!=a['bytes']:
                    raise ValueError('NPY descriptor checksum mismatch')
        return {'status':'complete','payload_files_verified':len(included),'transformed_npz_files':len(transformed),
                'npy_bytes_exact':True,'npz_container_bytes_exact':False if transformed else True,
                'manifest':manifest}


def restore(archive,destination):
    checked=verify_archive(archive);manifest=checked.pop('manifest')
    destination=Path(destination).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):raise FileExistsError('Restoration destination must be empty')
    destination.mkdir(parents=True,exist_ok=True)
    def output(name):
        path=destination/safe_path(name)
        if not path.resolve().is_relative_to(destination):raise ValueError('Output path escapes destination')
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError('Refusing to replace '+str(path))
        return path
    marker=destination/'.RESTORE_INCOMPLETE';marker.write_text('Restore has not completed.\n')
    restored=[]
    with zipfile.ZipFile(archive) as z:
        for row in manifest['included']:
            name=row['path']
            if name.startswith(('numeric_arrays/','numeric_descriptors/')):continue
            path=output(name)
            with z.open(name) as source,path.open('xb') as target:
                for chunk in iter(lambda:source.read(1024*1024),b''):target.write(chunk)
        for row in manifest.get('transformed_numeric',[]):
            d=json.loads(z.read(row['descriptor_path']));path=output(d['original_path'])
            with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out:
                for a in d['arrays']:
                    with z.open(a['array_path']) as source,out.open(a['member_name'],'w',force_zip64=True) as target:
                        for chunk in iter(lambda:source.read(1024*1024),b''):target.write(chunk)
            with path.open('rb') as f:digest,size=_digest(f)
            restored.append({'path':d['original_path'],'sha256':digest,'bytes':size,
                'original_sha256':d['original_sha256'],'original_bytes':d['original_bytes'],
                'npy_members_exact':True,'container_hash_equal':digest==d['original_sha256']})
    output('EXPORT_MANIFEST.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    result={**checked,'destination':str(destination),'restored_numeric':restored,
            'interpretation':'All array NPY bytes are exact. Original NPZ container hashes are provenance, not restored-container verification hashes; use this report for restored NPZ hashes. No model pickle was loaded.'}
    output('RESTORE_REPORT.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    marker.unlink()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive');parser.add_argument('--destination');parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    if args.verify_only:
        result=verify_archive(args.archive);result.pop('manifest')
    elif args.destination:result=restore(args.archive,args.destination);result.pop('restored_numeric')
    else:parser.error('--destination or --verify-only is required')
    print(json.dumps(result,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
