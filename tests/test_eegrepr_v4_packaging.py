"""Exact packaging checks; no new scientific analysis is performed."""
import importlib.util
import io
import json
import lzma
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np

spec=importlib.util.spec_from_file_location('v4_packaging',Path(__file__).resolve().parents[1]/'scripts/eegrepr/pack_v4_evidence.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)

def npy(a):
    b=io.BytesIO();np.save(b,a,allow_pickle=False);return b.getvalue()

def fixture(path):
    x=np.linspace(0,20,512);a=(np.sin(x)[None,:]+np.arange(64)[:,None]*.01).astype('f4')
    data=npy(a);values={'numeric_arrays/'+p.digest(data)+'.npy':data,'reports/findings.md':b'All negative findings retained.\n'}
    manifest={'included':[{'path':name,'sha256':p.digest(b),'bytes':len(b)} for name,b in values.items()],'transformed_numeric':[]}
    values['EXPORT_MANIFEST.json']=json.dumps(manifest).encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name,b in values.items():z.writestr(name,b)
    return values

class ExactPackaging(unittest.TestCase):
    def test_predictor_ieee_roundtrip(self):
        for dtype in ('f4','f8'):
            rng=np.random.default_rng(8);a=rng.normal(size=(12,64)).astype(dtype);a[0,0]=-0.
            original=npy(a);coded=lzma.compress(p.prediction_encode(original),preset=1)
            self.assertEqual(p.decode(coded,'float-predictor4-xor-shuffle-xz'),original)

    def test_nonfinite_and_endian_fallback_exact(self):
        for a in [np.arange(10000,dtype='>f8'),np.full(10000,np.nan),np.full(10000,np.inf)]:
            original=npy(a);codec,coded=p.encode(original,len(original))
            self.assertNotEqual(codec,'float-predictor4-xor-shuffle-xz')
            self.assertEqual(p.decode(coded,codec),original)

    def test_package_verify_restore_and_accessible_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);values=fixture(root/'source.zip')
            r=p.package(root/'source.zip',root/'packed.zip',root/'cache')
            self.assertEqual(r['original_members_verified'],len(values))
            self.assertGreater(sum(v for k,v in r['codec_counts'].items() if k!='identity'),0)
            with zipfile.ZipFile(root/'packed.zip') as z:self.assertEqual(z.read('reports/findings.md'),values['reports/findings.md'])
            p.unpack(root/'packed.zip',root/'restored.zip')
            with zipfile.ZipFile(root/'restored.zip') as z:
                self.assertEqual(set(z.namelist()),set(values))
                for name,data in values.items():self.assertEqual(z.read(name),data)
            with self.assertRaises(FileExistsError):p.unpack(root/'packed.zip',root/'restored.zip')

    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'source.zip');p.package(root/'source.zip',root/'packed.zip')
            with zipfile.ZipFile(root/'packed.zip') as z,zipfile.ZipFile(root/'bad.zip','w') as out:
                for name in z.namelist():out.writestr(name,b'changed' if name=='reports/findings.md' else z.read(name))
            with self.assertRaisesRegex(ValueError,'checksum'):p.verify(root/'bad.zip')

    def test_source_contract_and_paths_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'source.zip')
            with zipfile.ZipFile(root/'source.zip','a') as z:z.writestr('../outside',b'x')
            with self.assertRaisesRegex(ValueError,'Unsafe'):p.package(root/'source.zip',root/'packed.zip')

    def test_upload_cap_enforced_without_publish(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);fixture(root/'source.zip')
            with self.assertRaisesRegex(ValueError,'exceeds cap'):p.package(root/'source.zip',root/'packed.zip',max_bytes=10)
            self.assertFalse((root/'packed.zip').exists())
            self.assertTrue((root/'packed.zip.partial').exists())

if __name__=='__main__':unittest.main()
