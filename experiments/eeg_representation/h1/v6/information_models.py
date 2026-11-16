"""Bounded future-target readouts of matched EEG codes and partial physical state.

These are fitted learners, never information oracles. Every code uses the same A0
four-second histories; forecast encoders are the frozen 200-update procedural
bank with three fitting repeats. Numeric artifacts support staged split arrival.
"""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path
import time

import numpy as np

from ..artifacts import atomic_npz
from ..targets import canonical_hash
from ...storage import atomic_json, file_hash, read_json
from ...training import feature_matrix
from ..v4 import retention as bank_helpers
from . import models
from .evidence import EvidenceSource
from .geometry import mechanism_axes, scaling_rules

DEFAULTS = dict(acquisition='A0', anchor_job='A0__T0_l10_d128', dimension=8,
    code_bits=32, range_rule='quantile_005_995',
    ridge_alphas=[.01, .1, 1., 10., 100., 1000., 10000.],
    forecast_seeds=[0, 1, 2], forecast_updates=200, forecast_enabled=True,
    partial_state_enabled=True, threads=1, dtype='float32')


def settings(config):
    options = copy.deepcopy(DEFAULTS)
    options.update(config.get('information_models', {}))
    if (options['acquisition'] != 'A0' or options['anchor_job'] != DEFAULTS['anchor_job']
            or options['dimension'] != 8 or options['code_bits'] != 32
            or options['range_rule'] != DEFAULTS['range_rule']
            or options['ridge_alphas'] != DEFAULTS['ridge_alphas']):
        raise ValueError('Matched information readouts require A0, d8/B32 and the complete frozen seven-alpha grid')
    if options['dtype'] != 'float32':
        raise ValueError('Legacy forecast representation procedure is explicitly float32 without AMP/TF32')
    if config.get('claim_status') != 'engineering':
        if set(options['forecast_seeds']) != {0, 1, 2} or options['forecast_updates'] != 200:
            raise ValueError('Forecast representation is the unchanged three-seed, 200-update diagnostic')
        if not options['forecast_enabled']:
            raise ValueError('Complete scientific information comparisons require the forecast bank')
    if not 1 <= int(options['forecast_updates']) <= 200:
        raise ValueError('Forecast representation exceeds bounded diagnostic updates')
    return options


def _digest(arrays):
    h = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        h.update(key.encode()); h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _record(root, path):
    path = Path(path)
    return dict(path=str(path.relative_to(root)), sha256=file_hash(path), bytes=path.stat().st_size)


def _verify(root, records):
    for row in records:
        p = Path(root) / row['path']
        if not p.exists() or file_hash(p) != row['sha256']:
            raise ValueError('Information readout artifact checksum mismatch: ' + str(p))


def _numeric(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _split_identity(data, include_targets=False):
    keys = ['X', 'context_id', 'structure_id']
    if include_targets:
        keys += ['Y', 'future_mask', 'future_id']
    return _digest({k: data[k] for k in keys if k in data})


def fit_design(training, weights, *, scale_floor=1e-8, already_scaled=False):
    """Train-only weighted design, reusable across every target sharing a bank."""
    x = np.asarray(training, np.float64); w = np.asarray(weights, np.float64)
    if x.ndim != 2 or len(x) != len(w) or not np.isfinite(x).all() or np.any(w <= 0):
        raise ValueError('Finite nonempty training features and positive weights required')
    w = w / w.sum(); offset = w @ x
    sd = np.sqrt(w @ ((x - offset) ** 2))
    positive = sd[sd > scale_floor]
    floor = max(float(scale_floor), .001 * float(np.median(positive)) if len(positive) else scale_floor)
    scale = np.ones(x.shape[1]) if already_scaled else np.maximum(sd, floor)
    z = (x - offset) / scale
    # n*w retains the ordinary ridge alpha convention when all weights are equal.
    sqrtw = np.sqrt(len(x) * w)
    u, singular, vt = np.linalg.svd(z * sqrtw[:, None], full_matrices=False)
    return dict(offset=offset, scale=scale, training_weights=w, sqrt_weights=sqrtw,
                u=u, singular=singular, vt=vt, active_training_columns=int(np.sum(sd > floor)),
                scale_floor=float(floor), training_examples=len(x), input_coordinates=x.shape[1])


def fit_readout(panel, values, manifest, destination, *, alphas=None, resume=False,
                design=None, information_condition='compressed EEG representation'):
    """Freeze seven ridge candidates plus mean fallback before evaluation access."""
    fit_started = time.perf_counter()
    destination = Path(destination); alphas = DEFAULTS['ridge_alphas'] if alphas is None else list(alphas)
    if alphas != DEFAULTS['ridge_alphas']:
        raise ValueError('The complete seven-alpha readout grid must be retained')
    train = panel['train']; development = panel['development']
    identity = canonical_hash(dict(
        train=_split_identity(train, True), development=_split_identity(development, True),
        features={s: _digest({'features': values[s]}) for s in ('train', 'development')},
        risk_contract=manifest['risk_contract'], alphas=alphas, source=file_hash(Path(__file__))))
    marker = destination / 'selection.json'
    if marker.exists():
        if not resume:
            raise FileExistsError('Use resume for existing frozen readout ' + str(destination))
        selected = read_json(marker)
        if selected['signature'] != identity:
            raise ValueError('Information readout fitting identity changed')
        if file_hash(destination / 'state.npz') != selected['state_sha256']:
            raise ValueError('Frozen information readout weights changed')
        state = _numeric(destination / 'state.npz')
    else:
        design = fit_design(values['train'], models.context_weights(train)) if design is None else design
        y = models.first_targets(train)
        baseline = design['training_weights'] @ y
        projected_y = design['u'].T @ ((y - baseline) * design['sqrt_weights'][:, None])
        state = dict(offset=design['offset'], scale=design['scale'], baseline=baseline)
        candidates = []
        base_predictions = np.broadcast_to(baseline, (len(development['X']), len(baseline)))
        best = models.circuit_loss(development, base_predictions)
        chosen = 'train_mean'
        candidates.append(dict(id='train_mean', alpha=None, development_mse=best,
                               nominal_parameters=len(baseline), effective_input_df=0.))
        for alpha in alphas:
            tag = 'alpha_' + str(float(alpha)).replace('.', 'p')
            coef = design['vt'].T @ ((design['singular'] / (design['singular'] ** 2 + alpha))[:, None] * projected_y)
            state[tag + '__coef'] = coef
            pred = ((np.asarray(values['development'], float) - state['offset']) / state['scale']) @ coef + baseline
            risk = models.circuit_loss(development, pred)
            candidates.append(dict(id=tag, alpha=float(alpha), development_mse=risk,
                nominal_parameters=int(coef.size + len(baseline)),
                effective_input_df=float(np.sum(design['singular'] ** 2 / (design['singular'] ** 2 + alpha)))))
            if risk < best:
                chosen, best = tag, risk
        atomic_npz(destination / 'state.npz', **state)
        selected = dict(status='frozen', signature=identity, selected_id=chosen,
            state_sha256=file_hash(destination / 'state.npz'), candidates=candidates,
            selection='minimum equal-circuit/condition/prefix weighted development MSE including training-mean fallback',
            selection_split='development', evaluation_selection=False, calibration_selection=False,
            fitting_target='first registered training scoring future per context',
            decoder_family='weighted linear ridge; same complete alpha grid for every bank',
            information_condition=information_condition, input_coordinates=int(values['train'].shape[1]),
            training_contexts=len(train['context_id']), training_circuits=len(np.unique(train['structure_id'])),
            target_fingerprint=manifest['target_fingerprint'], risk_contract=manifest['risk_contract'],
            fitting_elapsed_seconds=time.perf_counter() - fit_started)
        atomic_json(marker, selected)
    # Selection and weights are immutable before any new evaluation feature access.
    previous = read_json(destination / 'inference.json') if (destination / 'inference.json').exists() else None
    if previous:
        if not resume: raise FileExistsError('Inference already exists; explicit resume required')
        if previous['selection_sha256'] != file_hash(marker): raise ValueError('Frozen readout selection changed')
        for filename, digest in previous['file_hashes'].items():
            if file_hash(destination / filename) != digest: raise ValueError('Readout predictions changed on resume')
    arrays = {}; split_inputs = {}; counts = {}
    for split in panel:
        if split not in values:
            continue
        data = panel[split]; z = (np.asarray(values[split], float) - state['offset']) / state['scale']
        split_inputs[split] = _digest(dict(features=np.asarray(values[split]), context_id=data['context_id'], structure_id=data['structure_id']))
        if previous and split in previous['split_inputs'] and previous['split_inputs'][split] != split_inputs[split]:
            raise ValueError('Previously inferred information features or IDs changed: ' + split)
        base = np.broadcast_to(state['baseline'], (len(z), len(state['baseline']))).copy()
        prediction = base
        arrays[split + '__train_mean'] = base
        for row in selected['candidates']:
            tag = row['id']
            if tag == 'train_mean': continue
            pred = z @ state[tag + '__coef'] + state['baseline']
            arrays[split + '__' + tag] = pred.astype(np.float64)
            if tag == selected['selected_id']: prediction = pred
        arrays[split + '_prediction'] = np.asarray(prediction, np.float64)
        arrays[split + '_context_id'] = np.asarray(data['context_id'])
        arrays[split + '_structure_id'] = np.asarray(data['structure_id'])
        counts[split] = dict(contexts=len(z), circuits=len(np.unique(data['structure_id'])),
            available_futures=int(np.asarray(data['future_mask']).sum()),
            scoring_futures_policy='consumer must use first four registered scoring futures; all original targets retained')
    arrays.update(target_fingerprint=np.asarray(manifest['target_fingerprint']),
                  risk_contract_json=np.asarray(json.dumps(manifest['risk_contract'], sort_keys=True)))
    if previous and set(previous['split_inputs']) - set(split_inputs):
        raise ValueError('Cannot remove a previously inferred split')
    if not previous or previous['split_inputs'] != split_inputs:
        atomic_npz(destination / 'predictions.npz', **arrays)
        atomic_json(destination / 'inference.json', dict(status='complete', split_inputs=split_inputs,
            selection_sha256=file_hash(marker), file_hashes={'predictions.npz': file_hash(destination / 'predictions.npz')},
            counts=counts, frozen_weights_reused=previous is not None))
    return dict(status='complete', **{k: selected[k] for k in ('selected_id', 'candidates', 'selection',
        'decoder_family', 'input_coordinates', 'target_fingerprint', 'risk_contract', 'information_condition')},
        counts=counts, numeric_dtype='float64 targets, ridge weights, predictions and statistics',
        fitting_limit='Fitted readout errors do not identify exact compression or observation loss')


def _reuse_forecast(root, panel, name):
    """Reuse only a checksum-covered retention bank with identical fitting arrays."""
    cache_path = root / 'retention/legacy_bank_cache.json'
    target_path = root / 'retention/task_targets.npz'
    path = root / 'retention/banks' / (name + '_model.npz')
    if not (cache_path.exists() and target_path.exists() and path.exists()): return None
    cache = read_json(cache_path)
    model_record = next((r for r in cache.get('files', []) if r['path'] == str(path.relative_to(root / 'retention'))), None)
    if model_record is None or file_hash(path) != model_record['sha256']:
        raise ValueError('Existing retention forecast weights lack matching cache provenance')
    saved = _numeric(target_path)
    for split in ('train', 'development'):
        for key in ('X', 'Y', 'context_id', 'future_mask', 'structure_id'):
            if not np.array_equal(saved[split + '_' + key], panel[split][key]):
                return None
    bank_name = 'legacy200_' + name
    metadata = cache.get('banks', {}).get(bank_name, {}).get('metadata')
    if not metadata or metadata.get('optimizer_updates') != 200: return None
    return _numeric(path), metadata, _record(root, path)


def _forecast_bank(root, panel, options, seed, destination, device, resume):
    from ...models import make_encoder
    torch, dtype = models.configure_torch(device, 'float32', options['threads'])
    legacy = bank_helpers.DEFAULTS; name = 'forecast__seed_' + str(seed)
    marker = destination / 'freeze.json'
    signature = canonical_hash(dict(train=_split_identity(panel['train'], True),
        procedure=dict(encoder=legacy['encoder_config'], steps=options['forecast_updates'], seed=seed,
                       batch=legacy['batch_size'], learning_rate=legacy['learning_rate'], weight_decay=legacy['weight_decay']),
        device=device, dtype='float32', model_source=file_hash(Path(__file__).parents[2] / 'models.py'),
        implementation=file_hash(Path(__file__))))
    if marker.exists():
        if not resume: raise FileExistsError('Forecast bank exists; explicit resume required')
        frozen = read_json(marker)
        if frozen['signature'] != signature or file_hash(destination / 'state.npz') != frozen['state_sha256']:
            raise ValueError('Forecast information bank freeze changed')
        state = _numeric(destination / 'state.npz'); metadata = frozen['metadata']
    else:
        reused = _reuse_forecast(root, panel, name) if options['forecast_updates'] == 200 else None
        if reused:
            state, metadata, source = reused
            metadata = dict(metadata, reuse_source=source, reused=True)
        else:
            torch.manual_seed(int(seed)); encoder = make_encoder(legacy['encoder_config']).to(device=device, dtype=dtype)
            y = models.first_targets(panel['train']); model = torch.nn.Sequential(encoder, torch.nn.Linear(128, y.shape[1])).to(device)
            x = torch.as_tensor(panel['train']['X'][:, None], dtype=dtype, device=device)
            y = torch.as_tensor(y, dtype=dtype, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=legacy['learning_rate'], weight_decay=legacy['weight_decay'])
            rng = np.random.default_rng(seed); batch = min(legacy['batch_size'], len(x)); losses = []; start = time.perf_counter()
            for _ in range(int(options['forecast_updates'])):
                idx = rng.choice(len(x), batch, replace=False)
                optimizer.zero_grad(set_to_none=True); loss = (model(x[idx]) - y[idx]).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite frozen-procedure forecast bank loss')
                loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            state = {'tensor__' + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
            metadata = dict(representation_family='forecast', optimization_seed=int(seed), optimizer_updates=int(options['forecast_updates']),
                fixed_training_steps=True, checkpoint_selection='frozen final update; no evaluation use',
                parameter_count=sum(v.numel() for v in model.parameters()), loss_history=losses,
                encoder_config=legacy['encoder_config'], batch_size=batch, forward_examples=batch * options['forecast_updates'],
                forecast_supervision='first scoring future of A0 legacy long-waveform training target only',
                elapsed_seconds=time.perf_counter() - start, reused=False)
        metadata.update(scope='secondary legacy200 procedural representation; not objective-superiority evidence',
            dtype='float32', amp=False, tf32=False, device=device,
            optimizer_repeats='fitting repeats, not independent circuits', consumed_history_samples=int(panel['train']['X'].shape[1]))
        atomic_npz(destination / 'state.npz', **state)
        frozen = dict(status='frozen', signature=signature, state_sha256=file_hash(destination / 'state.npz'), metadata=metadata)
        atomic_json(marker, frozen)
    # Only encoder tensors are inputs to ordinary inference; no future information.
    encoder = make_encoder(legacy['encoder_config']).to(device=device, dtype=dtype)
    encoder.load_state_dict({k.removeprefix('tensor__0.'): torch.as_tensor(v) for k, v in state.items() if k.startswith('tensor__0.')})
    values = {s: bank_helpers._encode_tokens(encoder, panel[s]['X'], device).mean(1) for s in panel}
    return values, metadata


def _code_banks(config, root, panel, options, device, resume):
    dest = root / 'information_models/banks'; values = {}; metadata = {}
    if config.get('claim_status') != 'engineering' and panel['train']['X'].shape[1] != 2000:
        raise ValueError('Information code banks must consume the genuine four-second history')
    values['feature18'] = {s: feature_matrix(panel[s]['X'][:, None], sample_hz=500.) for s in panel}
    metadata['feature18'] = dict(native_width=18, method='18 fixed full-history EEG features', fit_split='train')
    pca_path = dest / 'pca/native_projection.npz'; pca_guard = dest / 'pca/native_projection.json'
    train_hash = _digest(dict(X=panel['train']['X'], context_id=panel['train']['context_id']))
    if pca_path.exists():
        frozen = read_json(pca_guard)
        if not resume or frozen['training_hash'] != train_hash or file_hash(pca_path) != frozen['sha256']:
            raise ValueError('Train-PCA information bank changed')
        pca = _numeric(pca_path)
    else:
        pca = bank_helpers.projection_fit(panel['train']['X'], min(128, len(panel['train']['X']) - 1, panel['train']['X'].shape[1]))
        atomic_npz(pca_path, **{k: v for k, v in pca.items() if isinstance(v, np.ndarray)})
        atomic_json(pca_guard, dict(training_hash=train_hash, sha256=file_hash(pca_path), fit_split='train'))
    values['pca'] = {s: bank_helpers.project(panel[s]['X'], pca) for s in panel}
    metadata['pca'] = dict(native_width=len(pca['components']), method='train-only PCA on same A0 full history', fit_split='train')
    if options['forecast_enabled']:
        for seed in options['forecast_seeds']:
            name = 'legacy200_forecast__seed_' + str(seed)
            values[name], metadata[name] = _forecast_bank(root, panel, options, seed, dest / name, device, resume)
    variants = {}; unavailable = []
    for name, raw in values.items():
        directory = dest / name; marker = directory / 'code_freeze.json'
        signature = canonical_hash(dict(training_values=_digest(dict(values=raw['train'], ids=panel['train']['context_id'])),
            dimension=8, bits=32, range_rule=options['range_rule'], source=file_hash(Path(bank_helpers.__file__))))
        if marker.exists():
            frozen = read_json(marker)
            if not resume or frozen['signature'] != signature: raise ValueError('Frozen information projection/range changed')
            for filename, digest in frozen['files'].items():
                if file_hash(directory / filename) != digest: raise ValueError('Frozen projection/codebook checksum mismatch')
            projection = _numeric(directory / 'projection.npz'); book = read_json(directory / 'codebook.json')
            for key in ('lower', 'upper', 'constant', 'bits'): book[key] = np.asarray(book[key])
        else:
            try: projection = bank_helpers.projection_fit(raw['train'], 8)
            except ValueError as exc:
                unavailable.append(dict(method=name, status='unavailable', reason=str(exc))); continue
            projected_train = bank_helpers.project(raw['train'], projection)
            book = bank_helpers.fit_range_rule(projected_train, 32, options['range_rule'])
            atomic_npz(directory / 'projection.npz', **{k: v for k, v in projection.items() if isinstance(v, np.ndarray)})
            atomic_json(directory / 'codebook.json', book)
            atomic_json(marker, dict(signature=signature, fit_split='train', files={f: file_hash(directory / f) for f in ('projection.npz', 'codebook.json')}))
        projected = {s: bank_helpers.project(raw[s], projection) for s in panel}
        decoded = {}; codes = {}; clipping = {}
        for split in panel:
            packed, decoded[split], clipping[split] = bank_helpers.quantization_attribution(projected[split], book)
            if packed.shape != (len(projected[split]), 4): raise AssertionError('d8/B32 requires exact four-byte packed payloads')
            codes.update({split + '_packed': packed, split + '_projected': projected[split], split + '_decoded': decoded[split],
                          split + '_context_id': panel[split]['context_id']})
        atomic_npz(directory / 'codes.npz', **codes)
        atomic_json(directory / 'metadata.json', dict(metadata[name], dimension=8, payload_bits=32,
            range_rule=options['range_rule'], clipping=clipping, shared_storage_excluded=True,
            interpretation='Projection, clipping, quantization and fitted readout can each lose task performance'))
        for variant, a in [('d8_unquantized', projected), ('d8_B32_quantile_005_995', decoded)]:
            variants[name + '__' + variant] = dict(values=a, metadata=dict(metadata[name], representation=name, variant=variant,
                dimension=8, code_bits=None if variant == 'd8_unquantized' else 32,
                range_rule=options['range_rule'] if variant != 'd8_unquantized' else None,
                history_samples=int(panel['train']['X'].shape[1]), input_acquisition='A0', information_condition='compressed EEG representation'))
    return variants, unavailable



def verify_state_source(manifest, source, config):
    """IDs alone cannot match a physical snapshot to the EEG recording."""
    owned = isinstance(source, (str, Path)); src = EvidenceSource(source) if owned else source
    try:
        observed = src.provenance()
        expected = [manifest.get('source_provenance', {}), manifest.get('original_generation_provenance', {})]
        content_ids = {v.get('content_id') for v in expected} | {manifest.get('source_content_id')}
        if observed.get('content_id') in content_ids - {None}:
            return dict(status='verified', method='identical content-addressed derived/source evidence', content_id=observed['content_id'])
        expected_native = {r['sha256'] for e in expected for r in e.get('identity_records', []) if r['path'].endswith('native/manifest.json')}
        observed_native = {r['sha256'] for r in observed.get('identity_records', []) if r['path'].endswith('native/manifest.json')}
        if expected_native & observed_native:
            return dict(status='verified', method='identical original native-generation manifest', native_manifest_sha256=sorted(expected_native & observed_native))
        if config.get('claim_status') == 'engineering':
            return dict(status='engineering_fixture', scientific_source_linkage=False)
        raise ValueError('Partial-state source does not match the derived EEG recording provenance; context IDs alone are insufficient')
    finally:
        if owned: src.close()

def partial_state(panel, root, source=None):
    """Fixed 47-per-cell membrane/gate coordinates only; never full hidden state."""
    owned = source is not None and isinstance(source, (str, Path))
    src = EvidenceSource(source) if owned else source
    wanted = {str(cid) for data in panel.values() for cid in data['context_id']}
    by_id = {}; records = []
    try:
        name = 'geometry/privileged_state_geometry.npz'
        if src is not None and src.exists(name):
            record = src.verify(name); a = src.read_npz(name)
            if not {'membrane_gate_state', 'context_id', 'structure_id'} <= set(a):
                return None, dict(status='unavailable', reason='Saved partial state lacks membrane/gate/ID axes')
            for cid, sid, row in zip(a['context_id'], a['structure_id'], a['membrane_gate_state']):
                cid = str(cid)
                if cid in by_id and not np.array_equal(by_id[cid][1], row): raise ValueError('Fork-identical present state differs')
                by_id[cid] = (str(sid), row)
            records = [record]
        elif (src is not None and src.exists('native/manifest.json')) or (Path(root) / 'generation/native/manifest.json').exists():
            import h5py
            from .derivation import NativeComposite
            from ..v4.geometry import _parse_state
            native = src if src is not None and src.exists('native/manifest.json') else NativeComposite(root)
            native_rows = native.records if isinstance(native, EvidenceSource) else native.rows
            try:
                rows = sorted(native_rows.values(), key=lambda r: (r['context_id'], r.get('pool') != 'score', r.get('future_index', 0)))
                for row in rows:
                    cid = str(row['context_id'])
                    if cid not in wanted or cid in by_id: continue
                    handle = native.open_binary(row['path']); blob = handle.getvalue()
                    if hashlib.sha256(blob).hexdigest() != row['sha256']: raise ValueError('Native state snapshot hash mismatch')
                    with h5py.File(handle, 'r') as f:
                        if 'privileged/cutoff_physical_state' not in f or 'privileged/synapse_parameters' not in f: continue
                        meta = json.loads(f.attrs['metadata_json']); meta = meta.get('metadata', meta)
                        cells = sum(meta['population_sizes'].values()); state = f['privileged/cutoff_physical_state'][:]
                        synapses = f['privileged/synapse_parameters'][:]
                        compartments = (len(state) - 2 * len(synapses) - 3 * cells) // 2
                        membrane, _ = _parse_state(state, synapses, cells, compartments)
                        by_id[cid] = (str(row['structure_id']), membrane)
                    records.append(dict(context_id=cid, path=row['path'], sha256=row['sha256'], source_root=row.get('source_root')))
            finally:
                if native is not src: native.close()
        if wanted - set(by_id):
            return None, dict(status='unavailable', reason='Actual aligned membrane/gate snapshot unavailable for all requested contexts',
                missing_contexts=sorted(wanted - set(by_id)), available_contexts=len(wanted & set(by_id)), required_contexts=len(wanted))
        values = {}
        for split, data in panel.items():
            for cid, sid in zip(data['context_id'], data['structure_id']):
                if by_id[str(cid)][0] != str(sid): raise ValueError('State/context circuit identity mismatch')
            values[split] = np.stack([by_id[str(cid)][1] for cid in data['context_id']]).astype(np.float64)
        width = values['train'].shape[1]
        groups, units, _, _ = mechanism_axes(width, width)
        if any(a.shape[1] != width or not np.isfinite(a).all() for a in values.values()): raise ValueError('Partial-state shape or finite-value mismatch')
        scale = scaling_rules(values['train'], groups)
        # Dropping near-constant coordinates is train-only, with physical absolute
        # floors and mechanism-specific relative floors, never epsilon division.
        keep = scale['keep']
        if not keep.any(): return None, dict(status='unavailable', reason='No partial-state coordinates exceed frozen train-only physical scale floors')
        transformed = {s: (a[:, keep] - scale['offset'][keep]) / scale['scale_floor'][keep] for s, a in values.items()}
        return transformed, dict(status='complete', information_condition='partial present-state features',
            method='fitted membrane-voltage/extracellular-voltage/HH-gate predictor; NOT full-state oracle',
            original_coordinates=width, retained_coordinates=int(keep.sum()), rules=scale['rules'], units=units,
            omitted=['all synaptic A/B states', 'drive phase', 'pending recurrent deliveries', 'connectivity/parameters', 'acquisition filter state'],
            forbidden_inputs='future RNG, seeds, paths and event schedules excluded', source_files=records,
            scaling='train-only per-mechanism absolute/relative SD floors; drop below-floor coordinates',
            scale_arrays={k: v for k, v in scale.items() if isinstance(v, np.ndarray)})
    finally:
        if owned: src.close()


def fit(config, root, device='cpu', resume=False, source=None):
    """Fit or append inference; no simulation and no fresh-outcome model selection."""
    root = Path(root); options = settings(config); started = time.perf_counter()
    preexisting_readouts = {str(p.relative_to(root)) for p in (root / 'information_models/readouts').rglob('selection.json')}
    preexisting_encoders = {str(p.relative_to(root)) for p in (root / 'information_models/banks').glob('legacy200_forecast__*/freeze.json')}
    all_jobs = read_json(root / 'derived/manifest.json')['jobs']
    jobs = [j.get('id', j.get('job_id')) for j in all_jobs if j.get('status') == 'complete']
    code_jobs = [j for j in jobs if j.startswith(options['acquisition'] + '__')]
    if options['anchor_job'] not in code_jobs:
        result = dict(status='unavailable', reason='Frozen A0 long-waveform bank training anchor missing', rows=[], files=[])
        atomic_json(root / 'information_models/manifest.json', result); return result
    panel, anchor_manifest = models.load_panel(root, options['anchor_job'])
    bank_identity = canonical_hash(dict(options=options, device=device, source=file_hash(Path(__file__)),
        train=_split_identity(panel['train'], True), development=_split_identity(panel['development'], True)))
    guard = root / 'information_models/identity.json'
    if guard.exists():
        old = read_json(guard)
        if not resume or old['signature'] != bank_identity: raise ValueError('Information bank options/source/train/dev identity changed')
    else:
        atomic_json(guard, dict(signature=bank_identity, options=options, device=device, selection='train/development only'))
    variants, unsupported = _code_banks(config, root, panel, options, device, resume)
    linkage = verify_state_source(anchor_manifest, source, config) if source is not None and options['partial_state_enabled'] else dict(status='bound_to_current_native_composite')
    partial, partial_meta = partial_state(panel, root, source) if options['partial_state_enabled'] else (None, dict(status='unavailable', reason='Explicit engineering setting disabled state diagnostic'))
    partial_meta['source_linkage'] = linkage
    if partial is not None:
        arrays = partial_meta.pop('scale_arrays')
        path = root / 'information_models/partial_state/scaling.npz'
        if path.exists():
            old = _numeric(path)
            if any(not np.array_equal(old[k], v) for k, v in arrays.items()): raise ValueError('Frozen partial-state scaling changed')
        else: atomic_npz(path, **arrays)
        atomic_npz(root / 'information_models/partial_state/features.npz',
            **{s: a for s, a in partial.items()}, **{s + '_context_id': panel[s]['context_id'] for s in partial})
        variants['partial_state_membrane_gates'] = dict(values=partial, metadata=dict(partial_meta,
            history_samples=None, input_acquisition=None, representation='partial_state_membrane_gates', variant='train_scaled_unquantized'))
    atomic_json(root / 'information_models/partial_state/status.json', partial_meta)
    designs = {key: fit_design(v['values']['train'], models.context_weights(panel['train']),
        already_scaled=(key == 'partial_state_membrane_gates')) for key, v in variants.items()}
    rows = []; count = 0
    for job in jobs:
        data, manifest = models.load_panel(root, job)
        for split in data:
            if split not in panel or not np.array_equal(data[split]['context_id'], panel[split]['context_id']):
                raise ValueError('Information comparison target/population context mismatch')
            if job.startswith('A0__') and not np.array_equal(data[split]['X'], panel[split]['X']):
                raise ValueError('Matched code readouts must see identical normalized A0 histories')
        for key, v in variants.items():
            if not job.startswith('A0__') and key != 'partial_state_membrane_gates': continue
            destination = root / 'information_models/readouts' / job / key
            result = fit_readout(data, v['values'], manifest, destination, alphas=options['ridge_alphas'], resume=resume,
                                 design=designs[key], information_condition=v['metadata']['information_condition'])
            row_metadata = dict(v['metadata'])
            if 'method' in row_metadata: row_metadata['method_description'] = row_metadata.pop('method')
            rows.append(dict(result, job_id=job, method=key, **row_metadata,
                prediction_path=str((destination / 'predictions.npz').relative_to(root)),
                weights_path=str((destination / 'state.npz').relative_to(root)),
                selection_path=str((destination / 'selection.json').relative_to(root)),
                inference_path=str((destination / 'inference.json').relative_to(root))))
            count += len(result['candidates'])
    for job in code_jobs:
        rows.extend(dict(r, job_id=job) for r in unsupported)
    if partial is None:
        rows.extend(dict(partial_meta, job_id=job, method='partial_state_membrane_gates', information_condition='partial present-state features') for job in jobs)
    destination = root / 'information_models'; marker = destination / 'manifest.json'
    timing = _record_timing(config, root, panel, device, started, preexisting_readouts, preexisting_encoders, rows)
    result = dict(status='complete', signature=bank_identity, rows=rows, options=options,
        selected_job_ids=jobs, code_target_jobs=code_jobs, candidates_including_mean=count,
        forecast_training='once on legacy A0 long-waveform training target, then frozen across all matched A0 readout targets',
        other_acquisitions='compressed comparison deliberately fixed to A0; ordinary EEG policies still compare A0/A1/A2',
        comparison='additional secondary/descriptive analyses; original primary families and alpha unchanged',
        ideal_limits='Complete-state predictor and EEG-history Bayes floor are not supplied by these fitted models',
        calibration='held-out calibration outcomes never tune these point-forecast readouts',
        elapsed_seconds=time.perf_counter() - started, claim_status=config.get('claim_status'),
        fitting_timing=timing,
        files=[_record(root, p) for p in sorted(destination.rglob('*')) if p.is_file() and p != marker])
    atomic_json(marker, result)
    _freeze(config, root, result)
    return result



def runtime_identity(config, device='cpu'):
    """Same-host fit-cost identity, distinct from portable prediction provenance."""
    from .generation import cpu_topology
    from ..v4.artifacts import environment
    options = settings(config); env = environment()
    hardware = {}
    if device.startswith('cuda'):
        torch, _ = models.configure_torch(device, 'float32', options['threads'])
        d = torch.cuda.get_device_properties(device)
        hardware = dict(name=d.name, total_memory=d.total_memory, capability=[d.major, d.minor])
    return dict(host=platform.node(), platform_machine=platform.machine(), device=device,
        cpu=cpu_topology(), device_hardware=hardware, threads=options['threads'],
        environment={k: env[k] for k in ('python', 'versions')}, options=options,
        source_dependencies={str(p.relative_to(Path(__file__).parents[2])): file_hash(p) for p in (Path(__file__), Path(models.__file__), Path(bank_helpers.__file__),
            Path(__file__).parents[2] / 'models.py', Path(__file__).parents[2] / 'training.py')})


def _record_timing(config, root, panel, device, started, prior_readouts, prior_encoders, rows):
    path = root / 'information_models/timing.json'
    previous = read_json(path) if path.exists() else dict(calls=[])
    readouts = {str(p.relative_to(root)) for p in (root / 'information_models/readouts').rglob('selection.json')}
    encoders = {str(p.relative_to(root)) for p in (root / 'information_models/banks').glob('legacy200_forecast__*/freeze.json')}
    runtime = runtime_identity(config, device); runtime_hash = canonical_hash(runtime)
    call = dict(runtime_hash=runtime_hash, runtime=runtime, elapsed_seconds=time.perf_counter() - started,
        new_readout_fits=sorted(readouts - prior_readouts), new_encoder_fits=sorted(encoders - prior_encoders),
        total_frozen_readouts=len(readouts), total_frozen_encoders=len(encoders),
        training_contexts=len(panel['train']['X']), training_circuits=len(np.unique(panel['train']['structure_id'])),
        development_contexts=len(panel['development']['X']),
        split_context_counts={s: len(p['X']) for s, p in panel.items()},
        complete_target_jobs=sorted({r['job_id'] for r in rows if r['status'] == 'complete'}),
        complete_methods=len([r for r in rows if r['status'] == 'complete']))
    call['includes_new_fitting'] = bool(call['new_readout_fits'] or call['new_encoder_fits'])
    previous['calls'].append(call)
    measured = [a for a in previous['calls'] if a['runtime_hash'] == runtime_hash and a['includes_new_fitting']]
    summary = dict(status='measured' if measured else 'unavailable', runtime_hash=runtime_hash, runtime=runtime,
        measured_fit_elapsed_seconds=sum(a['elapsed_seconds'] for a in measured),
        measured_new_readout_count=sum(len(a['new_readout_fits']) for a in measured),
        measured_new_encoder_count=sum(len(a['new_encoder_fits']) for a in measured),
        covered_target_jobs=sorted(set().union(*(set(a['complete_target_jobs']) for a in measured))) if measured else [],
        training_contexts=call['training_contexts'], training_circuits=call['training_circuits'],
        fit_call_count=len(measured),
        scope='Sum of same-host, same-code invocations that performed new fitting; includes their derivation reads, representation inference, scaling/SVD and writes. Inference-only resumes are separately retained and never replace fitting time.',
        projection_limit='Observed cost is not a wall-time upper bound; increased contexts and state-design SVD have nonlinear costs.',
        separate_retention_overhead='Original retention procedure remains unchanged and may fit another three 200-update forecast encoders; include it in the campaign cost.')
    atomic_json(path, dict(status='complete', calls=previous['calls'], summary=summary))
    return summary

def _freeze(config, root, result):
    root = Path(root); destination = root / 'information_models'
    expected = {bank + '__' + variant for bank in
        ['feature18', 'pca'] + (['legacy200_forecast__seed_' + str(s) for s in result['options']['forecast_seeds']] if result['options']['forecast_enabled'] else [])
        for variant in ('d8_unquantized', 'd8_B32_quantile_005_995')}
    if config.get('claim_status') != 'engineering':
        for job in result['code_target_jobs']:
            actual = {r['method'] for r in result['rows'] if r['job_id'] == job and r['status'] == 'complete'}
            if not expected <= actual:
                raise ValueError('Required information code readouts incomplete for ' + job + ': ' + str(sorted(expected - actual)))
    files = []
    for p in sorted(destination.rglob('*')):
        if not p.is_file(): continue
        if p.name in ('state.npz', 'selection.json', 'freeze.json', 'projection.npz', 'codebook.json',
                      'code_freeze.json', 'native_projection.npz', 'native_projection.json', 'scaling.npz', 'identity.json'):
            if p == destination / 'freeze.json': continue
            files.append(_record(root, p))
    payload = dict(status='frozen', signature=result['signature'], options=result['options'],
        code_target_jobs=result['code_target_jobs'], selected_job_ids=result['selected_job_ids'],
        fitted_methods=[dict(job_id=r['job_id'], method=r['method'], risk_contract=r['risk_contract'])
                       for r in result['rows'] if r['status'] == 'complete'],
        partial_state_status=read_json(destination / 'partial_state/status.json')['status'],
        selection='train/development only; frozen before calibration/test generation and outcomes',
        files=files)
    payload['freeze_hash'] = canonical_hash(payload)
    path = destination / 'freeze.json'
    if path.exists():
        if read_json(path) != payload: raise ValueError('Immutable information model freeze changed after staged arrival')
    else: atomic_json(path, payload)
    return payload


def verify_freeze(config, root):
    root = Path(root); path = root / 'information_models/freeze.json'
    if not path.exists(): raise ValueError('Information-model train/dev freeze is required before fresh evaluation')
    record = read_json(path); body = dict(record); digest = body.pop('freeze_hash')
    if digest != canonical_hash(body) or record['options'] != settings(config):
        raise ValueError('Information model frozen protocol/options mismatch')
    _verify(root, record['files'])
    configured = set(config.get('selected_job_ids', []))
    if configured and set(record['selected_job_ids']) != configured:
        raise ValueError('Information model freeze is missing configured target settings')
    return dict(status='frozen', freeze_hash=digest, sha256=file_hash(path),
        path=str(path.relative_to(root)), code_target_jobs=record['code_target_jobs'],
        fitted_method_count=len(record['fitted_methods']), partial_state_status=record['partial_state_status'])


def cost(config, target_jobs=None):
    """Explicit bounded fitting counts; wall time requires actual-host timing."""
    options = settings(config)
    jobs = config.get('selected_job_ids', []) if target_jobs is None else target_jobs
    a0 = sum(j.startswith('A0__') for j in jobs); banks = 2 + (len(options['forecast_seeds']) if options['forecast_enabled'] else 0)
    return dict(code_target_jobs=a0, code_methods_per_target=2 * banks,
        code_ridge_candidates=a0 * 2 * banks * 7, code_mean_fallbacks=a0 * 2 * banks,
        partial_state_ridge_candidates_if_available=len(jobs) * 7,
        forecast_encoder_fits_if_no_compatible_bank=len(options['forecast_seeds']) if options['forecast_enabled'] else 0,
        forecast_updates_per_fit=options['forecast_updates'], reused_encoder_training_target=options['anchor_job'],
        new_simulation_episodes=0, acquisition='A0 full history', payload_bytes_per_context_per_bank=4,
        shared_weights='outside per-example B32 code budget',
        timing_status='provisional operation counts; actual-host timing required before expensive execution')
