"""Replay target arithmetic must reproduce frozen production panels exactly.

These fixtures contain synthetic numeric traces only. No simulator is imported or
invoked, and the frozen panel path uses the actual V6 batched energy derivation.
"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from experiments.eeg_representation.h1.artifacts import atomic_npz
from experiments.eeg_representation.h1.v6.derivation import _energy_panel
from experiments.eeg_representation.h1.v6.targets import (
    axes, extract_physical_target, parse_target_id, target_spec,
)
from experiments.eeg_representation.h1.v6_pilot import precision
from experiments.eeg_representation.h1.v6_pilot.config import jobs
from experiments.eeg_representation.storage import atomic_json, file_hash, write_trial


TARGETS = list(dict.fromkeys(job.split('__')[1] for job in jobs()))
POSITIONS = ((4, 1), (17, 1))


def _trials():
    rng = np.random.default_rng(106)
    # Include the final 128 ms of history and all future support through +1010 ms.
    time = 5872. + np.arange(1, 18208 + 1) * .0625
    seconds = (time - 6000.) / 1000.
    result = []
    for index in range(2):
        eeg = ((index + 1) * np.sin(2 * np.pi * 9.7 * seconds + .3)
               + .4 * np.cos(2 * np.pi * 19.3 * seconds + index)
               + .27 * np.sin(2 * np.pi * 27.1 * seconds)
               + .61 * rng.normal(size=len(time)) - 2.) * 1e-9
        result.append(dict(time_ms=time.copy(), eeg_v=eeg[:, None],
            dipole_na_um=np.zeros((len(time), 3)), privileged={},
            metadata=dict(fork=dict(cutoff_ms=6000.),
                          rng_namespaces=dict(structure=index + 1))))
    return result


def _historical_target(trials, target):
    """Use scalar native waveform/FFT extraction, then actual batched T4 derivation.

    The production evaluation shape is 32 contexts by eight futures. Replay
    observations occupy two nonadjacent rows. Other rows supply valid synthetic
    coefficients, and training normalization is fitted solely to training rows.
    """
    family, lead, _ = parse_target_id(target)
    base = f'T3_l{lead:g}_d1000' if family.startswith('T4_') else target
    physical = np.stack([extract_physical_target(t['eeg_v'], t['time_ms'], base)
                         for t in trials])
    dimension = physical.shape[-1]
    manifest = dict(normalization=dict(y_scale=1e-9, y_offset=[0.] * dimension,
                    x_offset=0., x_scale=1e-9, id='synthetic-train-normalization'),
                    target_fingerprint=target, risk_contract=dict(target=target))
    if not family.startswith('T4_'):
        return physical, manifest
    rng = np.random.default_rng(907 + int(lead))
    panel = {}
    for split, count, futures in [('train', 8, 1), ('evaluation', 32, 8)]:
        c = rng.normal(size=(count, futures, 54)) * 1e-8
        if split == 'evaluation':
            for position, value in zip(POSITIONS, physical):
                c[position] = value
        panel[split] = dict(Y_physical=c, Y=c.copy(),
            reference_Y_physical=rng.normal(size=(count, 2, 54)) * 1e-8,
            reference_Y=np.zeros((count, 2, 54)),
            future_mask=np.ones((count, futures), bool),
            reference_mask=np.ones((count, 2), bool),
            structure_id=np.arange(count), **axes(target_spec(base)))
    converted, manifest = _energy_panel(None, 'A1', target, panel, manifest)
    frozen = np.stack([converted['evaluation']['Y_physical'][position]
                       for position in POSITIONS])
    return frozen, manifest


class ReplayTargetRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trials = _trials()
        cls.frozen = {target: _historical_target(cls.trials, target)
                      for target in TARGETS}

    def test_exact_frozen_production_targets_for_all_twenty_targets(self):
        self.assertEqual(len(TARGETS), 20)
        for target in TARGETS:
            with self.subTest(target=target):
                expected, _ = self.frozen[target]
                actual = precision._replay_physical_targets(self.trials, target)
                np.testing.assert_array_equal(actual, expected)
                self.assertEqual(actual.dtype, np.float64)
                # Read order has no relation to circuit/target selection order.
                reverse = precision._replay_physical_targets(self.trials[::-1], target)
                np.testing.assert_array_equal(reverse, expected[::-1])

    def test_scalar_energy_exhibits_the_original_roundoff_bug(self):
        mismatches = []
        for target in TARGETS:
            if not target.startswith('T4_'):
                continue
            expected, _ = self.frozen[target]
            scalar = np.stack([extract_physical_target(t['eeg_v'], t['time_ms'], target)
                               for t in self.trials])
            if not np.array_equal(scalar, expected):
                mismatches.append(target)
                # This diagnostic bounds the reproduced rounding discrepancy;
                # the production acceptance test above remains exact equality.
                units = np.abs(np.spacing(expected))
                self.assertTrue(np.all(np.abs(scalar - expected) <= 4 * units))
        self.assertTrue(mismatches, 'Fixture must expose the scalar/batch reduction difference')
        with self.assertRaises(ValueError):
            precision._replay_physical_targets(self.trials[:1], mismatches[0])

    def test_real_value_and_shape_mismatches_save_exact_diagnostics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job = 'A1__T0_l0_d32'
            for future, frozen, observed in [
                ('changed-value', np.asarray([1., 2.]), np.asarray([1., 3.])),
                ('changed-shape', np.asarray([1., 2.]), np.asarray([1.])),
            ]:
                with self.subTest(future=future):
                    with self.assertRaisesRegex(ValueError, 'Native replay target differs'):
                        precision._verify_replay_target(root, job, 'context', future,
                                                        frozen, observed)
                    directory = root / 'precision/replay_target_failures' / job / future
                    details = json.loads((directory / 'comparison.json').read_text())
                    self.assertEqual(details['status'], 'failed')
                    self.assertEqual(details['frozen_shape'], list(frozen.shape))
                    self.assertEqual(details['native_shape'], list(observed.shape))
                    self.assertEqual(details['arrays']['sha256'],
                                     file_hash(directory / 'comparison.npz'))
                    if future == 'changed-value':
                        self.assertEqual(details['different_coordinates'], 1)
                        self.assertEqual(details['max_absolute_difference'], 1.)
                    else:
                        self.assertIsNone(details['different_coordinates'])
                        self.assertIsNone(details['max_absolute_difference'])
                    with np.load(directory / 'comparison.npz', allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['frozen_physical'], frozen)
                        np.testing.assert_array_equal(saved['native_physical'], observed)

    def test_replay_tables_preserve_id_matching_and_strict_frozen_target_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cids = np.asarray(['s1-p0000-R0', 's2-p0000-Ralpha'])
            tids = np.asarray([cid + '-score-0001' for cid in cids])
            rows, audits = [], []
            for cid, tid, trial in zip(cids, tids, self.trials):
                original = root / 'original' / str(tid) / 'trial.h5'
                replica = root / 'replica' / str(tid) / 'trial.h5'
                for path in (original, replica):
                    write_trial(path, trial, dict(metadata=trial['metadata']))
                rows.append(dict(trial_id=str(tid), context_id=str(cid),
                    structure_id=int(trial['metadata']['rng_namespaces']['structure']),
                    source_root=str(original.parent), path=original.name,
                    bytes=original.stat().st_size, sha256=file_hash(original)))
                audits.append(dict(context_id=str(cid), original_trial_id=str(tid),
                    replica_path=str(replica), replay_native_sha256=file_hash(replica)))
            atomic_json(root / 'design.json', dict(historical_rows=rows))
            atomic_json(root / 'replay_controls.json', dict(audits=audits))
            original_hashes = {str(path): file_hash(path) for path in root.rglob('*.h5')}
            for target in TARGETS:
                with self.subTest(target=target):
                    job = 'A1__' + target
                    y, manifest = self.frozen[target]
                    manifest = deepcopy(manifest)
                    dimension = y.shape[-1]
                    offset = np.asarray(manifest['normalization']['y_offset'])
                    scale = manifest['normalization']['y_scale']
                    # Deliberately reverse panel context order. Future identities
                    # select score-0001; no positional assumption is permitted.
                    evalp = dict(context_id=cids[::-1], structure_id=np.asarray([2, 1]),
                        future_id=np.asarray([[cid + '-score-0000', tid]
                                              for cid, tid in zip(cids[::-1], tids[::-1])]),
                        future_mask=np.ones((2, 2), bool),
                        Y_physical=np.stack([y[::-1], y[::-1]], axis=1))
                    train = dict(context_id=np.asarray(['s3-p0000-R0', 's4-p0000-Ralpha']),
                        structure_id=np.asarray([3, 4]), regime=np.asarray(['R0', 'Ralpha']),
                        Y=np.zeros((2, 1, dimension)), future_mask=np.ones((2, 1), bool))
                    panel = dict(train=train, evaluation=evalp)
                    directory = root / 'matched' / 'learners' / job / 'complete_state'
                    atomic_npz(directory / 'predictions.npz', evaluation_context_id=cids[::-1],
                        evaluation_prediction=(y[::-1] - offset) / scale,
                        evaluation_candidate_prediction=np.zeros_like(y),
                        risk_contract_json=np.asarray(json.dumps(manifest['risk_contract'])))
                    atomic_json(directory / 'selection.json',
                        dict(files={'predictions.npz': file_hash(directory / 'predictions.npz')}))
                    with patch.object(precision, 'load_panel', return_value=(panel, manifest)):
                        result = precision.replay_tables({}, root, [job])
                    replay = next(row for row in result['replay_information']
                                  if row['method'] == 'measured_replay')
                    self.assertEqual(replay['risk_physical'], 0.)
                    self.assertEqual(replay['context_ids'], cids.tolist())
                    self.assertEqual(replay['future_ids'], tids.tolist())
                    with np.load(root / 'precision/replay' / job / 'predictions_and_losses.npz',
                                 allow_pickle=False) as saved:
                        np.testing.assert_array_equal(saved['Y_physical'], y)
                        np.testing.assert_array_equal(saved['replayed_physical'], y)
                    # Even one ULP of actual frozen-target corruption still fails.
                    corrupt = deepcopy(panel)
                    value = corrupt['evaluation']['Y_physical'][0, 1, 0]
                    corrupt['evaluation']['Y_physical'][0, 1, 0] = np.nextafter(value, np.inf)
                    with patch.object(precision, 'load_panel', return_value=(corrupt, manifest)):
                        with self.assertRaisesRegex(ValueError, 'Native replay target differs'):
                            precision.replay_tables({}, root, [job])
            self.assertEqual(original_hashes,
                {str(path): file_hash(path) for path in root.rglob('*.h5')})


if __name__ == '__main__':
    unittest.main()
