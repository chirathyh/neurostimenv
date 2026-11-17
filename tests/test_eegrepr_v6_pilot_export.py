"""Exact NPY roundtrip and cross-version evidence reader compatibility."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from experiments.eeg_representation.h1.v6_pilot import exporting
from experiments.eeg_representation.h1.v4.restore_export import verify_archive,restore
from experiments.eeg_representation.h1.v6.evidence import EvidenceSource


class PilotExportTests(unittest.TestCase):
    def test_numeric_roundtrip_reader_dtype_shape_and_omission_rules(self):
        with tempfile.TemporaryDirectory() as d:
            base=Path(d);root=base/'run';root.mkdir();(root/'pilot_config.json').write_text('{"schema_version":"eegrepr-v6-reporting-pilot-1"}');(root/'pilot_provenance.json').write_text('{"purpose":"engineering_fixture"}')
            arrays=dict(float64=np.array([1.234567891234567,-0.0,np.nan]),float32=np.array([[3.25]],dtype=np.float32),ids=np.asarray(['circuit-1','μ-context'],dtype='U20'),mask=np.array([True,False]),scalar=np.asarray(3))
            np.savez_compressed(root/'predictions.npz',**arrays);np.savez_compressed(root/'duplicate.npz',**arrays)
            state=root/'matched/learners/job/complete_state';state.mkdir(parents=True);np.savez_compressed(state/'state.npz',weights=np.arange(6,dtype=np.float64))
            ordinary=root/'matched/learners/job/eeg_full';ordinary.mkdir();np.savez_compressed(ordinary/'state.npz',weights=np.ones(2))
            (root/'native.h5').write_bytes(b'opaque engineering native placeholder')
            destination=base/'evidence.zip';result=exporting.export(root,destination)
            self.assertEqual(result['status'],'complete');self.assertTrue(destination.with_suffix('.zip.sha256').exists())
            manifest=verify_archive(destination)['manifest'];paths={x['path'] for x in manifest['omitted']}
            self.assertIn('evidence/native.h5',paths);self.assertIn('evidence/matched/learners/job/eeg_full/state.npz',paths)
            self.assertNotIn('evidence/matched/learners/job/complete_state/state.npz',paths)
            with EvidenceSource(destination) as source:
                restored=source.read_npz('predictions.npz')
                for key,expected in arrays.items():
                    self.assertEqual(expected.dtype,restored[key].dtype);self.assertEqual(expected.shape,restored[key].shape);np.testing.assert_array_equal(expected,restored[key])
                self.assertEqual(source.read_npz('matched/learners/job/complete_state/state.npz')['weights'].shape,(6,))
                self.assertIn('pilot_config.json',[r['path'] for r in source.provenance()['identity_records']])
            restored_root=base/'restored';receipt=restore(destination,restored_root);self.assertTrue(receipt['npy_bytes_exact'])
            with EvidenceSource(restored_root) as source:np.testing.assert_array_equal(source.read_npz('predictions.npz')['float64'],arrays['float64'])
            for name in ('predictions.npz','duplicate.npz'):
                with zipfile.ZipFile(root/name) as before,zipfile.ZipFile(restored_root/'evidence'/name) as after:
                    self.assertEqual(before.namelist(),after.namelist())
                    for member in before.namelist():self.assertEqual(before.read(member),after.read(member))
            logical=[r for r in manifest['transformed_numeric'] if r['original_path'] in ('evidence/predictions.npz','evidence/duplicate.npz')]
            with zipfile.ZipFile(destination) as z:
                descriptors=[json.loads(z.read(x['descriptor_path'])) for x in logical]
                self.assertEqual([x['array_path'] for x in descriptors[0]['arrays']],[x['array_path'] for x in descriptors[1]['arrays']])
                self.assertTrue(all('dtype' in x and 'shape' in x for x in descriptors[0]['arrays']))

    def test_object_array_rejected_without_publishing_invalid_archive(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir();np.savez(root/'unsafe.npz',value=np.asarray([{'not':'numeric'}],dtype=object))
            destination=Path(d)/'bad.zip'
            with self.assertRaises(ValueError):exporting.export(root,destination)
            self.assertFalse(destination.exists())

    def test_output_root_and_existing_destination_are_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'run';root.mkdir()
            with self.assertRaises(ValueError):exporting.export(root,root/'inside.zip')
            destination=Path(d)/'existing.zip';destination.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):exporting.export(root,destination)
            self.assertEqual(destination.read_bytes(),b'preserve')

if __name__=='__main__':unittest.main()
