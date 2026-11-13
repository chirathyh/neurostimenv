"""Read-only, checksum-addressed evidence imports; never relabel generation provenance."""
from pathlib import Path, PurePosixPath
import io,json,zipfile,hashlib
import numpy as np
from ..artifacts import atomic_json,file_hash
from ..targets import canonical_hash


def safe(name):
    name=str(name);p=PurePosixPath(name)
    if not name or name in ('.','./') or p.is_absolute() or '..' in p.parts or '\\' in name or p.as_posix()!=name.rstrip('/'):
        raise ValueError('Unsafe/noncanonical evidence member '+name)
    return name


class EvidenceSource:
    def __init__(self,path):
        self.path=Path(path).expanduser().resolve()
        if not self.path.exists():raise FileNotFoundError('Evidence source unavailable: '+str(self.path))
        self.archive=zipfile.ZipFile(self.path) if self.path.is_file() else None
        self.prefix='';self._identity=None
        if self.archive:
            names=self.archive.namelist()
            if len(names)!=len(set(names)):raise ValueError('Duplicate archive member names')
            for n in names:safe(n)
            if 'evidence/resolved_config.json' in names:self.prefix='evidence/'
            elif 'resolved_config.json' not in names:raise ValueError('Archive lacks scientific evidence root')
            self._names={n[len(self.prefix):] for n in names if n.startswith(self.prefix) and not n.endswith('/')}
        else:
            if not (self.path/'resolved_config.json').is_file() and (self.path/'evidence/resolved_config.json').is_file():self.path=self.path/'evidence'
            if not (self.path/'resolved_config.json').is_file():raise ValueError('Directory lacks scientific evidence root')
            self._names={p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()}
        self.records={}
        def register(record):
            name=safe(record['path'])
            if name in self.records and any(self.records[name].get(k)!=record.get(k) for k in ('sha256','bytes')):
                raise ValueError('Conflicting original evidence records: '+name)
            self.records[name]=record
        if self.exists('EVIDENCE_MANIFEST.json'):
            m=self.read_json('EVIDENCE_MANIFEST.json')
            for key in ['files','source_files','generated_files']:
                for r in m.get(key,[]):register(r)
        if self.exists('native/manifest.json'):
            for r in self.read_json('native/manifest.json').get('trials',[]):register(r)
    def list_names(self):return sorted(self._names)
    def exists(self,name):return safe(name) in self._names
    def open_binary(self,name):
        name=safe(name)
        if self.archive:return self.archive.open(self.prefix+name)
        p=(self.path/name).resolve()
        if not p.is_relative_to(self.path):raise ValueError('Evidence symlink escapes root')
        return p.open('rb')
    def read_bytes(self,name):
        with self.open_binary(name) as f:return f.read()
    read=read_bytes
    def read_json(self,name):return json.loads(self.read_bytes(name))
    def read_npz(self,name):
        with np.load(io.BytesIO(self.read_bytes(name)),allow_pickle=False) as a:return {k:a[k] for k in a.files}
    def verify(self,name):
        r=self.records.get(safe(name));h=hashlib.sha256();size=0
        with self.open_binary(name) as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);size+=len(b)
        result={'path':name,'bytes':size,'sha256':h.hexdigest()}
        if r and (result['sha256']!=r['sha256'] or size!=r['bytes']):raise ValueError('Evidence checksum mismatch '+name)
        return result
    def provenance(self):
        if self._identity is None:
            original=self.read_json('provenance.json') if self.exists('provenance.json') else {}
            manifest=self.read_json('native/manifest.json') if self.exists('native/manifest.json') else {}
            # Archives are addressed by every archive byte. Directories also need
            # an inventory digest: a native manifest alone does not bind saved panels.
            directory_inventory=None
            if self.archive is None:
                directory_inventory=[]
                for name in self.list_names():
                    if name in self.records:
                        r=self.records[name]
                        directory_inventory.append({k:r[k] for k in ('path','sha256','bytes')})
                    else:directory_inventory.append(self.verify(name))
            identity={'inventory_sha256':canonical_hash(directory_inventory) if directory_inventory is not None else None,'kind':'zip' if self.archive else 'directory','path':str(self.path),'logical_prefix':self.prefix,
                      'archive_sha256':file_hash(self.path) if self.archive else None,
                      'generation_provenance':original,'generation_source_hash':manifest.get('source_hash',original.get('source_hash')),
                      'generation_config_hash':manifest.get('config_hash',original.get('config_hash')),
                      'manifest_sha256':hashlib.sha256(self.read_bytes('native/manifest.json')).hexdigest() if manifest else None,
                      'native_present':sum(n.startswith('native/') and n.endswith('.h5') for n in self._names),
                      'policy':'Original generation provenance remains original; new derived operators have separate analysis hashes.'}
            identity['content_id']=canonical_hash({k:v for k,v in identity.items() if k not in ('path','logical_prefix')});self._identity=identity
        return dict(self._identity)
    @property
    def identity(self):return self.provenance()
    def close(self):
        if self.archive:self.archive.close()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


def import_evidence(source,root,*,verify_all=False):
    owned=not isinstance(source,EvidenceSource)
    source=EvidenceSource(source) if owned else source
    root=Path(root)
    try:
        if not source.archive and (root.resolve()==source.path or root.resolve().is_relative_to(source.path)):
            raise ValueError('New analysis root must be separate from source evidence')
        records=[];missing=[]
        for name,r in sorted(source.records.items()):
            if not source.exists(name):missing.append({'path':name,'sha256':r['sha256'],'bytes':r['bytes'],'reason':'absent from supplied export'});continue
            if verify_all or not name.endswith(('.h5','.hdf5')):records.append(source.verify(name))
        identity=source.provenance()
        pointer=root/'imports/current.json'
        previous=None
        if pointer.exists():
            current=json.loads(pointer.read_text())
            previous=json.loads((root/safe(current['manifest'])).read_text())
            expected=canonical_hash({k:v for k,v in previous.items() if k!='import_hash'})
            if previous.get('import_hash')!=expected or current.get('import_hash')!=expected:
                raise ValueError('Import manifest/pointer hash mismatch')
            if previous['identity']['content_id']!=identity['content_id']:
                raise ValueError('Imported evidence source changed; use a new analysis root')
        result={'status':'complete','identity':identity,'verified_files':records,
              'unavailable_original_files':missing,'native_verification':'all available native bytes verified' if verify_all else 'native members verified on derivation read',
              'copy_policy':'Read-only referenced import; no native duplication or provenance replacement.'}
        # The content ID is location independent. Keep the first immutable provenance
        # record when the same bytes move; update only the explicit location pointer.
        if previous is not None:
            if previous['unavailable_original_files']!=missing:
                raise ValueError('Imported evidence availability changed')
            old={r['path']:r for r in previous['verified_files']}
            for r in records:
                if r['path'] in old and r!=old[r['path']]:raise ValueError('Previously verified import changed: '+r['path'])
            result=previous
        else:
            result['import_hash']=canonical_hash(result)
            destination=root/'imports'/f"{identity['content_id']}.json"
            if destination.exists() and json.loads(destination.read_text())!=result:raise ValueError('Import identity collision/changed inventory')
            atomic_json(destination,result)
        destination=root/'imports'/f"{identity['content_id']}.json"
        atomic_json(pointer,{'manifest':destination.relative_to(root).as_posix(),'import_hash':result['import_hash'],
                             'source_content_id':identity['content_id'],'source_path':str(source.path)})
        if verify_all:
            atomic_json(root/'imports'/f"{identity['content_id']}-native-verification.json",
                        {'source_content_id':identity['content_id'],'status':'complete','verified_files':records})
        return result
    finally:
        if owned:source.close()
