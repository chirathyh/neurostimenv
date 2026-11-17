"""Read v4 directories and losslessly deduplicated numeric ZIPs without unpickling."""
from pathlib import Path
import io,json,zipfile,hashlib
import numpy as np
from ..v4.evidence import safe
from ..v4.restore_export import verify_archive
from .artifacts import atomic_json,file_hash,canonical_hash

class EvidenceSource:
 def __init__(self,path):
  self.path=Path(path).expanduser().resolve();self.archive=zipfile.ZipFile(self.path) if self.path.is_file() else None
  self._verified={};self.mapping={};self._identity=None;self.records={}
  if self.archive:
   names=self.archive.namelist()
   if len(names)!=len(set(names)):raise ValueError('Duplicate archive paths')
   for name in names:safe(name)
   self.prefix='evidence/' if any('evidence/'+n in names for n in ('resolved_config.json','pilot_config.json')) else ''
   self._names={n[len(self.prefix):] for n in names if n.startswith(self.prefix)}
   if 'EXPORT_MANIFEST.json' in names:
    self.export=json.loads(self.archive.read('EXPORT_MANIFEST.json'))
    self.payload={safe(r['path']):r for r in self.export['included']}
    if len(self.payload)!=len(self.export['included']):raise ValueError('Duplicate export payload records')
    if set(names)!={'EXPORT_MANIFEST.json'}|set(self.payload):raise ValueError('ZIP/export payload inventory mismatch')
    for row in self.export.get('transformed_numeric',[]):
     name=safe(row['original_path'])
     if name.startswith(self.prefix):
      logical=name[len(self.prefix):]
      if logical in self.mapping or name in self.payload:raise ValueError('Duplicate/conflicting original numeric member')
      self.mapping[logical]=row;self._names.add(logical)
   else:self.export=None;self.payload={}
  else:
   if not self.path.is_dir():raise FileNotFoundError(self.path)
   if not any((self.path/n).exists() for n in ('resolved_config.json','pilot_config.json')) and any((self.path/'evidence'/n).exists() for n in ('resolved_config.json','pilot_config.json')):self.path=self.path/'evidence'
   self.prefix='';self.export=None;self.payload={};self._names={p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()}
  # A v4 campaign has native records under generation/, while legacy roots have native/.
  self.aliases={n[len('generation/'):]:n for n in self._names if n.startswith('generation/native/')}
  self._names.update(self.aliases)
  if self.exists('native/manifest.json'):
   self.records={r['path']:r for r in self.read_json('native/manifest.json').get('trials',[])}
 def list_names(self):return sorted(self._names)
 def exists(self,name):return safe(name) in self._names
 def _name(self,name):return self.aliases.get(safe(name),name)
 def _archive_bytes(self,key):
  key=safe(key);data=self.archive.read(key);record=self.payload.get(key)
  if self.export is not None and record is None:raise ValueError('Archive member absent from export inventory '+key)
  if record and (len(data)!=record['bytes'] or hashlib.sha256(data).hexdigest()!=record['sha256']):raise ValueError('Export payload checksum mismatch: '+key)
  return data
 def _bytes(self,name):
  name=self._name(name)
  if self.archive:
   return self._archive_bytes(self.prefix+name)
  p=(self.path/name).resolve()
  if not p.is_relative_to(self.path):raise ValueError('Source symlink escapes root')
  return p.read_bytes()
 def read_bytes(self,name):return self._bytes(name)
 read=read_bytes
 def read_json(self,name):return json.loads(self._bytes(name))
 def open_binary(self,name):return io.BytesIO(self._bytes(name))
 def read_npz(self,name):
  name=self._name(name)
  if name in self.mapping:
   row=self.mapping[name];d=json.loads(self._archive_bytes(row['descriptor_path']));out={};members=set()
   if d['original_path']!=self.prefix+name or d['original_sha256']!=row['original_sha256'] or d['original_bytes']!=row['original_bytes']:raise ValueError('Descriptor original identity mismatch')
   for r in d['arrays']:
    key=r['key'];member=safe(r['member_name'])
    if key in out or member in members:raise ValueError('Duplicate numeric array key/member')
    members.add(member);stored=self.payload.get(safe(r['array_path']))
    if stored is None or any(stored[k]!=r[k] for k in ('sha256','bytes')):raise ValueError('Descriptor array/export inventory mismatch')
    data=self._archive_bytes(r['array_path'])
    if len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('Numeric checksum mismatch')
    a=np.load(io.BytesIO(data),allow_pickle=False)
    if list(a.shape)!=r['shape'] or a.dtype!=np.dtype(r['dtype']):raise ValueError('Numeric descriptor shape/dtype mismatch')
    out[r['key']]=a
   return out
  with np.load(io.BytesIO(self._bytes(name)),allow_pickle=False) as z:return {k:z[k] for k in z.files}
 def verify(self,name):
  if name in self._verified:return self._verified[name]
  real=self._name(name)
  if real in self.mapping:
   self.read_npz(real);row=self.mapping[real];r={'path':name,'sha256':row['original_sha256'],'bytes':row['original_bytes'],'verification':'exact NPY members; original NPZ container provenance'}
  else:
   data=self._bytes(name);r={'path':name,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
   old=self.records.get(name)
   if old and any(r[k]!=old[k] for k in ('sha256','bytes')):raise ValueError('Native source checksum mismatch')
  self._verified[name]=r;return r
 def provenance(self):
  if self._identity is None:
   roots=['provenance.json','resolved_config.json','pilot_provenance.json','pilot_config.json','generation/provenance.json','native/manifest.json','derived/manifest.json']
   records=[self.verify(n) for n in roots if self.exists(n)]
   p={'kind':'zip' if self.archive else 'directory','path':str(self.path),'archive_sha256':file_hash(self.path) if self.archive else None,
      'identity_records':records,'native_present':sum(n.endswith('.h5') for n in self._names if not n.startswith('generation/')),
      'generation_provenance':self.read_json('generation/provenance.json') if self.exists('generation/provenance.json') else self.read_json('provenance.json') if self.exists('provenance.json') else {},
      'policy':'Original generation provenance preserved; analysis and export identities are separate.'}
   p['content_id']=canonical_hash({k:v for k,v in p.items() if k!='path'});self._identity=p
  return self._identity.copy()
 def verify_all(self):
  if self.archive and self.export:
   r=verify_archive(self.path);r.pop('manifest');return r
  for name in self.list_names():
   if not name.startswith('generation/native/'):self.verify(name)
  return {'status':'complete','verified_files':len(self._verified)}
 def close(self):
  if self.archive:self.archive.close()
 def __enter__(self):return self
 def __exit__(self,*a):self.close()

def import_evidence(source,root,verify_all=False):
 identity=source.provenance();dest=Path(root)/'imports';pointer=dest/'current.json'
 if pointer.exists() and json.loads(pointer.read_text())['content_id']!=identity['content_id']:raise ValueError('Immutable import changed; choose a new output root')
 result={'status':'complete','identity':identity,'verification':source.verify_all() if verify_all else 'checksum verification on numerical/native read','raw_reuse':'explicit; original generation never relabeled'}
 # Retain original generation metadata and the omission ledger, not only links
 # to a large local archive that the recipient may not have.
 original=dest/'sources'/identity['content_id'];records=[]
 for name in ('resolved_config.json','provenance.json','pilot_config.json','pilot_provenance.json','generation/provenance.json','generation/resolved_config.json','native/manifest.json'):
  if source.exists(name):
   blob=source.read_bytes(name);target=original/name;target.parent.mkdir(parents=True,exist_ok=True)
   if target.exists() and target.read_bytes()!=blob:raise ValueError('Imported original metadata changed')
   if not target.exists():
    temporary=target.with_suffix(target.suffix+'.tmp');temporary.write_bytes(blob);temporary.replace(target)
   records.append({'path':str(target.relative_to(root)),'original_path':name,'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)})
 omitted=[]
 if source.export:
  for row in source.export.get('omitted',[]):
   name=row['path'];base=Path(name).name
   if Path(name).suffix.lower() in ('.h5','.hdf5','.pt','.pth','.pkl','.pickle') or base in ('state.npz','weights.npz','readout_weights.npz','readout.npz','model.npz') or base.endswith('_model.npz'):
    omitted.append({**row,'reason':'Not present in supplied original source archive; retained original omission/provenance record'})
 atomic_json(original/'source_omissions.json',{'source':identity,'records':omitted})
 result['original_metadata']=records;result['omission_ledger']=str((original/'source_omissions.json').relative_to(root))
 result['import_hash']=canonical_hash(result);atomic_json(dest/(identity['content_id']+'.json'),result)
 atomic_json(pointer,{'content_id':identity['content_id'],'source_path':str(source.path),'manifest':identity['content_id']+'.json'})
 return result
