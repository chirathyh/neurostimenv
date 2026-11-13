#!/usr/bin/env python3
"""Losslessly repack v4 numeric evidence without altering scientific arrays.

Requires NumPy. Reports remain ordinary ZIP members. `unpack` restores every
original member byte for byte; ZIP container bytes/timestamps are not promised.
The optional float predictor stores bitwise XOR residuals, never rounded values.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import lzma
from pathlib import Path, PurePosixPath
import sys
import zipfile
import numpy as np

SCHEMA='eegrepr-lossless-package-1'
MANIFEST='PACKING_MANIFEST.json'
README='PACKING_README.md'
TOOL='pack_v4_evidence.py'
COEFFICIENTS=(4.,-6.,4.,-1.)


def digest(data):return hashlib.sha256(data).hexdigest()
def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def safe(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name:raise ValueError('Unsafe path: '+str(name))
    return name

def header(data):
    f=io.BytesIO(data);version=np.lib.format.read_magic(f)
    shape,fortran,dtype=np.lib.format._read_array_header(f,version)
    if dtype.hasobject:raise ValueError('Object arrays are not supported')
    return f.tell(),shape,fortran,dtype

def prediction_encode(data):
    offset,shape,fortran,dtype=header(data)
    if fortran or len(shape)<2 or shape[-1]<8 or dtype.kind!='f' or dtype.itemsize not in (4,8) or not dtype.isnative:raise ValueError('Unsupported predictor layout')
    a=np.frombuffer(data[offset:],dtype=dtype).reshape(shape)
    if not np.all(np.isfinite(a)):raise ValueError('Nonfinite predictor input')
    pred=np.zeros_like(a);acc=np.zeros_like(a[...,4:],dtype=np.float64)
    # Explicit separate binary64 multiply then add, fixed left-to-right order.
    with np.errstate(over='raise',invalid='raise'):
        for j,c in enumerate(COEFFICIENTS):
            term=c*a[...,3-j:shape[-1]-j-1].astype(np.float64)
            acc=acc+term
        pred[...,4:]=acc.astype(dtype)
    unsigned=np.dtype('u'+str(dtype.itemsize))
    residual=a.view(unsigned)^pred.view(unsigned)
    return data[:offset]+residual.view(np.uint8).reshape(-1,dtype.itemsize).T.tobytes()

def decode(payload,codec):
    if codec=='identity':return payload
    coded=lzma.decompress(payload);offset,shape,fortran,dtype=header(coded)
    size=int(np.prod(shape,dtype=np.int64));body=coded[offset:]
    if len(body)!=size*dtype.itemsize:raise ValueError('Coded NPY length mismatch')
    unshuffled=np.frombuffer(body,dtype=np.uint8).reshape(dtype.itemsize,size).T.copy()
    if codec=='byte-shuffle-xz':return coded[:offset]+unshuffled.tobytes()
    if codec!='float-predictor4-xor-shuffle-xz':raise ValueError('Unknown codec '+codec)
    if fortran or len(shape)<2 or shape[-1]<8 or dtype.kind!='f' or dtype.itemsize not in (4,8) or not dtype.isnative:raise ValueError('Unsupported predictor layout')
    unsigned=np.dtype('u'+str(dtype.itemsize));residual=unshuffled.reshape(-1).view(unsigned).reshape(shape)
    restored=residual.view(dtype).copy()
    with np.errstate(over='raise',invalid='raise'):
        for t in range(4,shape[-1]):
            acc=np.zeros(shape[:-1],dtype=np.float64)
            for j,c in enumerate(COEFFICIENTS):
                term=c*restored[...,t-j-1].astype(np.float64)
                acc=acc+term
            restored[...,t]=(residual[...,t]^acc.astype(dtype).view(unsigned)).view(dtype)
    return coded[:offset]+restored.tobytes()

def encode(data,original_compressed_size):
    # Keep small arrays and unsupported dtypes unchanged. No archive pickle loads.
    if len(data)<65536:return 'identity',data
    offset,shape,fortran,dtype=header(data)
    if dtype.itemsize==0 or dtype.itemsize>16:return 'identity',data
    body=data[offset:];m=np.frombuffer(body,dtype=np.uint8).reshape(-1,dtype.itemsize)
    candidates=[('identity',data,original_compressed_size)]
    shuffled=lzma.compress(data[:offset]+m.T.tobytes(),preset=3)
    candidates.append(('byte-shuffle-xz',shuffled,len(shuffled)))
    try:
        predicted=lzma.compress(prediction_encode(data),preset=3)
        candidates.append(('float-predictor4-xor-shuffle-xz',predicted,len(predicted)))
    except (ValueError,FloatingPointError):pass
    codec,payload,_=min(candidates,key=lambda x:x[2])
    if decode(payload,codec)!=data:raise ValueError('Encoding round-trip failed')
    return codec,payload


def readme():
    return '''# Exact numeric evidence, packaged below the upload limit

All reports, tables, metadata, and source files remain directly readable in this ZIP.
All current numeric arrays are retained. Large NPY members use reversible byte
shuffling or fourth-order floating prediction with **bitwise XOR residuals** and XZ.
No numeric precision is discarded. Every restored member is checked against its
original SHA256. This layer changes packaging only, not scientific source hashes.

The predictor uses binary64 multiplication and addition as separate operations,
coefficients (4, -6, 4, -1) in that order, then conversion to the original float
width. Residuals preserve every differing IEEE bit. Unsupported/nonfinite layouts
fall back to byte shuffling or unchanged storage. Verification fails if the host
cannot reproduce the specified arithmetic; it never silently accepts differences.

Extract `pack_v4_evidence.py` and install NumPy if absent. Then run:

```bash
python pack_v4_evidence.py verify SHARE.zip
python pack_v4_evidence.py unpack SHARE.zip restored-compact.zip
```

`restored-compact.zip` has every member of the original compact export byte for byte;
its ZIP container hash may differ. To reconstruct ordinary analysis NPZ files,
extract `implementation/experiments/eeg_representation/h1/v4/restore_export.py`
from that ZIP, then run the original standard-library helper:

```bash
python restore_export.py restored-compact.zip --verify-only
python restore_export.py restored-compact.zip --destination restored-analysis
```

That second layer restores exact NPY array members from deduplicated storage;
NPZ container compression/timestamps can differ, as explained by EXPORT_MANIFEST.json.
PACKING_MANIFEST.json preserves the source package hash, original member hashes,
codec assignments and physical payload hashes. Omitted native files and readout
weights are documented in the original export manifest; this packaging layer omits
nothing from its input package. Never unpickle supplied model archives to inspect data.
'''


def package(source,destination,cache_directory=None,workers=2,max_bytes=512000000):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    cache=Path(cache_directory).resolve() if cache_directory else None
    if cache:cache.mkdir(parents=True,exist_ok=True)
    destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.partial')
    if tmp.exists():raise FileExistsError(tmp)
    records=[];codec_counts={};source_manifest=None
    with zipfile.ZipFile(source) as original:
        names=original.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate source members')
        if any(n in names for n in (MANIFEST,README,TOOL)):raise ValueError('Source is already packed or has conflicting names')
        for name in names:safe(name)
        export=json.loads(original.read('EXPORT_MANIFEST.json'))
        expected={r['path']:(r['sha256'],r['bytes']) for r in export['included']}
        if set(names)!=set(expected)|{'EXPORT_MANIFEST.json'}:raise ValueError('Source export manifest mismatch')
        infos=original.infolist()
        def transform(info):
            data=original.read(info.filename);rawhash=digest(data)
            if info.filename!='EXPORT_MANIFEST.json' and (rawhash,len(data))!=expected[info.filename]:raise ValueError('Source checksum mismatch: '+info.filename)
            codec,payload='identity',data
            if info.filename.startswith('numeric_arrays/') and info.filename.endswith('.npy'):
                cachemeta=cache/(rawhash+'.json') if cache else None
                if cachemeta is not None and cachemeta.exists():
                    meta=json.loads(cachemeta.read_text());codec=meta['codec'];payload=(cache/(rawhash+'.bin')).read_bytes()
                    if meta['schema']!=SCHEMA or digest(payload)!=meta['payload_sha256'] or digest(decode(payload,codec))!=rawhash:raise ValueError('Invalid compression cache')
                else:
                    codec,payload=encode(data,info.compress_size)
                    if cache is not None and codec!='identity':
                        (cache/(rawhash+'.bin')).write_bytes(payload)
                        cachemeta.write_text(json.dumps({'schema':SCHEMA,'codec':codec,'payload_sha256':digest(payload)}))
            stored=info.filename if codec=='identity' else info.filename+'.encoded.xz'
            return payload,{'source_path':info.filename,'stored_path':stored,'codec':codec,'original_sha256':rawhash,'original_bytes':len(data),'payload_sha256':digest(payload),'payload_bytes':len(payload)}
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out,ThreadPoolExecutor(max_workers=workers) as pool:
                # Bounded batches avoid retaining all payloads in ThreadPoolExecutor.map.
                for start in range(0,len(infos),16):
                    for payload,row in pool.map(transform,infos[start:start+16]):
                        out.writestr(row['stored_path'],payload,compress_type=zipfile.ZIP_DEFLATED if row['codec']=='identity' else zipfile.ZIP_STORED)
                        records.append(row);codec_counts[row['codec']]=codec_counts.get(row['codec'],0)+1
                    if start%1008==0:print(json.dumps({'completed':min(start+16,len(infos)),'total':len(infos),'bytes_so_far':out.fp.tell(),'codecs':codec_counts}),flush=True)
                extras={README:readme().encode(),TOOL:Path(__file__).read_bytes()}
                for name,data in extras.items():out.writestr(name,data)
                manifest={'schema_version':SCHEMA,'source_archive_sha256':file_digest(source),'source_archive_bytes':source.stat().st_size,'interpretation':'Every source member is retained byte exactly after decoding; ZIP container hash need not match. No scientific array values were rounded or removed.','numpy_version_at_pack':np.__version__,'predictor':'Separate binary64 multiply then add in coefficients (4,-6,4,-1) order; cast to original float width; IEEE bitwise XOR residual; last-axis causal prediction.','entries':records,'extra_files':[{'path':name,'sha256':digest(data),'bytes':len(data)} for name,data in extras.items()],'codec_counts':codec_counts}
                out.writestr(MANIFEST,json.dumps(manifest,indent=2,sort_keys=True)+'\n')
            if max_bytes and tmp.stat().st_size>max_bytes:raise ValueError(f'Package exceeds cap: {tmp.stat().st_size} > {max_bytes}; partial retained for inspection')
            tmp.rename(destination)
            checked=verify(destination)
            destination.with_suffix(destination.suffix+'.sha256').write_text(file_digest(destination)+'  '+destination.name+'\n')
            return {'path':str(destination),'bytes':destination.stat().st_size,**checked}
        except BaseException:raise


def _validated_rows(z):
    names=z.namelist()
    if len(names)!=len(set(names)):raise ValueError('Duplicate package members')
    for name in names:safe(name)
    manifest=json.loads(z.read(MANIFEST))
    if manifest['schema_version']!=SCHEMA:raise ValueError('Unsupported schema')
    rows=manifest['entries'];stored={r['stored_path'] for r in rows};source={r['source_path'] for r in rows}
    extras={r['path']:r for r in manifest['extra_files']}
    if len(stored)!=len(rows) or len(source)!=len(rows):raise ValueError('Duplicate mapping')
    if set(names)!=stored|set(extras)|{MANIFEST}:raise ValueError('Package member mismatch')
    for name,row in extras.items():
        data=z.read(name)
        if (digest(data),len(data))!=(row['sha256'],row['bytes']):raise ValueError('Helper checksum mismatch')
    rawexport=next((r for r in rows if r['source_path']=='EXPORT_MANIFEST.json'),None)
    if rawexport is None or rawexport['codec']!='identity':raise ValueError('Missing original export manifest')
    data=z.read(rawexport['stored_path'])
    if digest(data)!=rawexport['original_sha256']:raise ValueError('Original export manifest checksum mismatch')
    original=json.loads(data);expected={r['path']:(r['sha256'],r['bytes']) for r in original['included']}
    if source!=set(expected)|{'EXPORT_MANIFEST.json'}:raise ValueError('Source contract member mismatch')
    for row in rows:
        safe(row['stored_path']);safe(row['source_path'])
        if row['source_path']!='EXPORT_MANIFEST.json' and (row['original_sha256'],row['original_bytes'])!=expected[row['source_path']]:raise ValueError('Original member contract mismatch')
    return manifest,rows

def iter_decoded(z,rows):
    for row in rows:
        payload=z.read(row['stored_path'])
        if (digest(payload),len(payload))!=(row['payload_sha256'],row['payload_bytes']):raise ValueError('Payload checksum mismatch: '+row['stored_path'])
        data=decode(payload,row['codec'])
        if (digest(data),len(data))!=(row['original_sha256'],row['original_bytes']):raise ValueError('Restored checksum mismatch: '+row['source_path'])
        yield row['source_path'],data

def verify(archive):
    with zipfile.ZipFile(archive) as z:
        manifest,rows=_validated_rows(z)
        count=sum(1 for _ in iter_decoded(z,rows))
    return {'status':'complete','original_members_verified':count,'original_member_bytes_exact':True,'zip_container_bytes_exact':False,'codec_counts':manifest['codec_counts']}

def unpack(archive,destination):
    destination=Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    destination.parent.mkdir(parents=True,exist_ok=True);tmp=destination.with_suffix(destination.suffix+'.partial')
    if tmp.exists():raise FileExistsError(tmp)
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as out:
        manifest,rows=_validated_rows(z)
        for name,data in iter_decoded(z,rows):out.writestr(name,data)
    tmp.rename(destination)
    return {'status':'complete','restored_archive':str(destination),'original_members_verified':len(rows),'original_member_bytes_exact':True,'zip_container_bytes_exact':False}

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='mode',required=True)
    a=s.add_parser('pack');a.add_argument('source');a.add_argument('destination');a.add_argument('--cache-directory');a.add_argument('--workers',type=int,default=2);a.add_argument('--max-bytes',type=int,default=512000000)
    a=s.add_parser('verify');a.add_argument('archive')
    a=s.add_parser('unpack');a.add_argument('archive');a.add_argument('destination')
    a=p.parse_args()
    if a.mode=='pack':result=package(a.source,a.destination,a.cache_directory,a.workers,a.max_bytes)
    elif a.mode=='verify':result=verify(a.archive)
    else:result=unpack(a.archive,a.destination)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
