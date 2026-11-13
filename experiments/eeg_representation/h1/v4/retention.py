"""H3 attribution at explicit operational code budgets, never an MI estimator.

Old numeric representation banks are read without pickle. New diagnostic banks
use three optimization repeats; these are fitting repeats, not new circuits.
Everything upstream of a readout is fitted on training histories only (the
forecast encoder additionally uses its first training future). Evaluation labels
never select a representation, projection, code range, or decoder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import warnings

import numpy as np
from scipy.special import logsumexp

from ..benchmark import atomic_npz
from ..compression import allocate_bits, pack_codes, unpack_codes, quantize, fit_quantizer
from ...storage import atomic_json, file_hash
from ...training import feature_matrix

SPLITS = ('train', 'development', 'evaluation')
DEFAULTS = dict(job_id='A0__T0_l10_d128', dimensions=[8, 32], code_bits=[32, 128],
                banks=['feature18', 'pca', 'untrained', 'masked_mse', 'forecast'],
                label_budgets=['all'], regularization=[.01, .1, 1., 10., 100., 1000., 10000.],
                range_rules=['quantile_005_995', 'robust_mad6'],
                tasks=['waveform_reconstruction', 'rhythm', 'carrier'],
                optimization_seeds=[0, 1, 2], representation_steps=200, batch_size=16,
                learning_rate=.001, weight_decay=.01, threads=1, token_bins=4, pooling_banks=['forecast'],
                encoder_config=dict(architecture='transformer', channels=1, embedding_dim=128,
                                    layers=4, heads=4, patch_samples=32))


def _digest_arrays(values):
    h = hashlib.sha256()
    for k, a in sorted(values.items()):
        a = np.ascontiguousarray(a)
        h.update(k.encode()); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=lambda a: np.asarray(a).tolist()).encode()).hexdigest()


def _source(source):
    if source is None or hasattr(source, 'read_npz'):
        return source
    from .evidence import EvidenceSource
    return EvidenceSource(source)


def load_panel(root, job_id, source=None):
    """Load an explicit derived job; a legacy fallback never changes provenance."""
    root = Path(root)
    directory = root / 'derived' / job_id
    if (directory / 'manifest.json').exists() and (root / 'derived/manifest.json').exists():
        from .derivation import load_panel as verified_panel
        panels, manifest = verified_panel(root, job_id)
        return dict(manifest=manifest, **panels)
    if (directory / 'manifest.json').exists():
        manifest = json.loads((directory / 'manifest.json').read_text())
        panel = {'manifest': manifest}
        for s in SPLITS:
            with np.load(directory / (s + '.npz'), allow_pickle=False) as f:
                panel[s] = {k: f[k] for k in f.files}
        return panel
    source = _source(source)
    if source is not None and source.exists('panels/primary/manifest.json') and job_id in ('primary', DEFAULTS['job_id']):
        source.verify('panels/primary/manifest.json')
        for split in SPLITS:
            source.verify('panels/primary/' + split + '.npz')
        return dict(manifest=source.read_json('panels/primary/manifest.json'),
                    **{s: source.read_npz('panels/primary/' + s + '.npz') for s in SPLITS})
    raise FileNotFoundError(f'No derived panel {directory}; derive this job first')


def validate_panel(panel):
    for s in SPLITS:
        p = panel[s]
        if np.asarray(p['X']).ndim != 2 or len(p['X']) != len(p['context_id']):
            raise ValueError('Invalid history/context shape')
        if not np.isfinite(p['X']).all() or len(set(map(str, p['context_id']))) != len(p['X']):
            raise ValueError('Nonfinite history or repeated context ID')
    for a, b in (('train', 'development'), ('train', 'evaluation'), ('development', 'evaluation')):
        for k in ('structure_id', 'context_id'):
            if set(map(str, panel[a][k])) & set(map(str, panel[b][k])):
                raise ValueError(f'{k} overlaps between {a} and {b}')


def balanced_label_subset(labels, circuits, budget='all', seed=0, required_classes=None):
    """Require every observed class-by-circuit cell before reducing labels.

    An absent required cell or too-small budget is explicitly unavailable. The
    round-robin remainder balances cells without fabricating absent examples.
    """
    labels, circuits = np.asarray(labels), np.asarray(circuits)
    if labels.ndim != 1 or len(labels) != len(circuits):
        raise ValueError('One label and circuit ID required per context')
    classes = np.unique(labels) if required_classes is None else np.asarray(required_classes)
    groups = np.unique(circuits)
    rng = np.random.default_rng(seed)
    buckets = [rng.permutation(np.flatnonzero((labels == c) & (circuits == g))).tolist()
               for g in groups for c in classes]
    missing = [(str(g), str(c)) for g in groups for c in classes if not np.any((labels == c) & (circuits == g))]
    if missing:
        return None, dict(status='unavailable', reason='required class-by-circuit coverage absent', missing_cells=missing)
    count = len(labels) if budget in ('all', None) else int(budget)
    if count < len(buckets):
        return None, dict(status='unavailable', reason='label budget cannot cover every required class-by-circuit cell',
                          minimum_budget=len(buckets), requested_budget=count)
    selected = []
    while len(selected) < min(count, len(labels)):
        for bucket in buckets:
            if bucket and len(selected) < min(count, len(labels)):
                selected.append(bucket.pop())
    return np.asarray(sorted(selected)), dict(status='complete', selected_contexts=len(selected),
        selected_circuits=len(groups), classes=classes.tolist(), sampling='round-robin class-by-circuit cells', seed=seed)


def projection_fit(x, dimension):
    x = np.asarray(x, float)
    if dimension > min(x.shape[1], len(x) - 1):
        raise ValueError('dimension exceeds native width or K-1 sample rank ceiling; padding forbidden')
    center = x.mean(0); a = x - center
    # Gram eigendecomposition avoids a huge full-sequence right-singular basis.
    if a.shape[1] > a.shape[0]:
        w, u = np.linalg.eigh(a @ a.T)
        order = np.argsort(w)[::-1]; singular = np.sqrt(np.maximum(w[order], 0))
        rank = int(np.sum(singular > (singular[0] if len(singular) else 1) * 1e-7))
        components = (u[:, order[:dimension]].T @ a) / np.maximum(singular[:dimension, None], np.finfo(float).tiny)
    else:
        _, singular, v = np.linalg.svd(a, full_matrices=False)
        rank = int(np.sum(singular > (singular[0] if len(singular) else 1) * 1e-10))
        components = v[:dimension]
    if dimension > rank:
        raise ValueError(f'dimension exceeds observed numerical rank {rank}; padding forbidden')
    return dict(center=center, components=components, singular_values=singular[:dimension], numerical_rank=rank)


def project(x, p):
    return (np.asarray(x) - p['center']) @ p['components'].T


def fit_range_rule(training, budget_bits, rule):
    z = np.asarray(training, float)
    q = fit_quantizer(z, budget_bits)
    if rule == 'quantile_005_995':
        q['range_rule'] = 'training coordinate quantiles .005/.995, historical primary rule'
    elif rule in ('robust_mad6', 'robust_iqr'):
        median = np.median(z, axis=0)
        mad = (1.4826 * np.median(np.abs(z - median), axis=0) if rule == 'robust_mad6'
               else (np.quantile(z, .75, axis=0) - np.quantile(z, .25, axis=0)) / 1.349)
        # Training-only fixed fallback for a zero MAD, never test-informed.
        scale = np.where(mad > 0, mad, z.std(0))
        q.update(lower=median - 6 * scale, upper=median + 6 * scale,
                 constant=scale == 0, median=median, robust_scale=scale,
                 range_rule=('training median +/- 6 * 1.4826 MAD; zero-MAD fallback train SD' if rule == 'robust_mad6'
                             else 'training median +/- 6 * IQR/1.349; zero-IQR fallback train SD'),
                 quantile_range=None)
    else:
        raise ValueError('Unknown preregistered range rule')
    q['fit_split'] = 'train'
    return q


def quantization_attribution(z, codebook):
    codes, decoded, report = quantize(z, codebook)
    clipped = np.clip(z, codebook['lower'], codebook['upper'])
    report.update(clipping_only_mse=float(np.mean((z - clipped) ** 2)),
                  rounding_after_clipping_mse=float(np.mean((clipped - decoded) ** 2)),
                  clipping_rounding_cross_term=float(2 * np.mean((z - clipped) * (clipped - decoded))),
                  interpretation='coordinate errors in this projected representation; not input information fractions')
    payload = pack_codes(codes, codebook['bits'])
    if not np.array_equal(codes, unpack_codes(payload, codebook['bits'])):
        raise AssertionError('Fixed-payload round trip failed')
    return payload, decoded, report


def token_views(tokens, bins=4):
    tokens = np.asarray(tokens)
    if tokens.ndim != 3 or bins > tokens.shape[1]:
        raise ValueError('Tokens must be examples x token index x embedding coordinate')
    return dict(pooled=tokens.mean(1), full_tokens=tokens.reshape(len(tokens), -1),
                fixed_token_readout=np.concatenate([a.mean(1) for a in np.array_split(tokens, bins, axis=1)], axis=1))


def _encode_tokens(encoder, x, device, batch=16):
    import torch
    encoder.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            a = torch.as_tensor(np.asarray(x[start:start + batch, None]), dtype=torch.float32, device=device)
            chunks.append(encoder.tokens(a).cpu().numpy())
    return np.concatenate(chunks)


def _frozen_banks(panel, source, destination, options, device):
    """Only reuse embeddings if their actual histories and IDs agree exactly."""
    base = 'compression/primary/representations/'
    if source is None or not source.exists(base + 'manifest.json'):
        return None
    if not all(source.exists('panels/primary/' + s + '.npz') for s in SPLITS):
        raise ValueError('Frozen banks lack source histories for alignment verification')
    for s in SPLITS:
        old = source.read_npz('panels/primary/' + s + '.npz')
        if not np.array_equal(old['context_id'], panel[s]['context_id']) or not np.array_equal(old['X'], panel[s]['X']):
            raise ValueError('Frozen bank source histories/IDs differ; do not silently apply old acquisition provenance')
    banks, architecture = {}, []
    for name in options['banks']:
        path = base + name + '.npz'
        if not source.exists(path):
            banks[name] = dict(status='unavailable', reason='frozen bank absent'); continue
        source.verify(path); source.verify(base + name + '.json')
        values = source.read_npz(path); metadata = source.read_json(base + name + '.json')
        for split in SPLITS:
            if not np.array_equal(values[split + '_context_id'], panel[split]['context_id']):
                raise ValueError('Frozen bank context alignment mismatch')
        banks[name] = dict(values={s: values[s] for s in SPLITS}, metadata=dict(
            method='reused frozen numeric bank', original_metadata=metadata,
            imported_sha256=hashlib.sha256(source.read_bytes(path)).hexdigest(), optimization_repeats=1,
            interpretation='historical attribution; single optimizer seed cannot support objective superiority'))
        if name == 'pca' and source.exists(base + 'pca_model.npz'):
            banks[name]['inverse'] = source.read_npz(base + 'pca_model.npz')
        if 'encoder_config' not in metadata:
            continue
        model_path = base + name + '_model.npz'
        if not source.exists(model_path):
            architecture.append(dict(bank=name, status='unavailable', reason='numeric weights absent for token audit')); continue
        import torch
        from ...models import make_encoder
        torch.set_num_threads(int(options['threads']))
        encoder = make_encoder(metadata['encoder_config']).to(device)
        source.verify(model_path)
        arrays = source.read_npz(model_path)
        state = {}
        for k, v in arrays.items():
            if not k.startswith('tensor__'):
                continue
            k = k[len('tensor__'):]
            if name == 'masked_mse':
                if not k.startswith('encoder.'):
                    continue
                k = k[len('encoder.'):]
            state[k] = torch.as_tensor(v)
        encoder.load_state_dict(state, strict=True)
        tokens = {s: _encode_tokens(encoder, panel[s]['X'], device) for s in SPLITS}
        error = max(float(np.max(np.abs(tokens[s].mean(1) - values[s]))) for s in SPLITS)
        relative_rms = max(float(np.sqrt(np.mean((tokens[s].mean(1) - values[s]) ** 2)) / max(np.sqrt(np.mean(values[s] ** 2)), 1e-12)) for s in SPLITS)
        if relative_rms > .01:
            raise ValueError(f'Frozen encoder cross-platform discrepancy exceeds 1% relative RMS guard: {name}, error={error}')
        architecture.append(dict(bank=name, status='complete', token_shape=list(tokens['train'].shape[1:]),
            pooled_shape=[values['train'].shape[1]], fixed_readout_shape=[options['token_bins'] * tokens['train'].shape[2]],
            maximum_pooled_reproduction_error=error, relative_rms_error=relative_rms, numerical_tolerance_relative_rms=.01,
            numerical_note='Frozen saved pooled arrays retained verbatim; token inference from verified numeric weights may differ across CPU/CUDA float32 kernels; maximum difference reported',
            reconstruction_uses='token sequence',
            forecasting_and_classification_use='token mean pooled vector',
            statistical_unit='circuit; token coordinates are not independent replicates'))
        for view in (('pooled', 'full_tokens', 'fixed_token_readout') if name in options['pooling_banks'] else ()):
            banks[name + '__' + view] = dict(values={s: token_views(tokens[s], options['token_bins'])[view] for s in SPLITS},
                metadata=dict(parent_bank=name, view=view, method='fixed deterministic readout of same frozen tokens',
                              matched_final_projection_and_bit_budget=True))
    return banks, architecture


def _new_banks(panel, destination, options, device):
    """Prospective diagnostic repeats with equal data exposures, no leaderboard."""
    import torch
    from ...models import make_encoder, ReconstructionModel, gap_mask, masked_loss
    seeds = list(options['optimization_seeds'])
    if set(options['banks']) & {'masked_mse', 'forecast'} and len(set(seeds)) < 3:
        raise ValueError('Prospective representation diagnostics require >=3 distinct optimization seeds')
    torch.set_num_threads(int(options['threads'])); torch.use_deterministic_algorithms(True)
    values = {s: np.asarray(panel[s]['X'], float) for s in SPLITS}
    fs = float(panel['manifest'].get('input', {}).get('sample_rate_hz', 500.))
    banks = {}
    if 'feature18' in options['banks']:
        banks['feature18'] = dict(values={s: feature_matrix(values[s][:, None], sample_hz=fs) for s in SPLITS},
                                 metadata=dict(method='18 declared EEG features', fit_split='train'))
    if 'pca' in options['banks']:
        dimension = min(128, len(values['train']) - 1, values['train'].shape[1])
        p = projection_fit(values['train'], dimension)
        banks['pca'] = dict(values={s: project(values[s], p) for s in SPLITS},
                            inverse=dict(offset=p['center'], components=p['components']), metadata=dict(method='train-only PCA'))
    architecture = []
    x = torch.as_tensor(panel['train']['X'][:, None], dtype=torch.float32, device=device)
    first = np.argmax(panel['train']['future_mask'], axis=1)
    y = torch.as_tensor(panel['train']['Y'][np.arange(len(first)), first], dtype=torch.float32, device=device)
    for seed in seeds:
        for kind in ('untrained', 'masked_mse', 'forecast'):
            if kind not in options['banks']:
                continue
            torch.manual_seed(seed); encoder = make_encoder(options['encoder_config']).to(device)
            if kind == 'masked_mse':
                model = ReconstructionModel(encoder).to(device)
            elif kind == 'forecast':
                model = torch.nn.Sequential(encoder, torch.nn.Linear(encoder.embedding_dim, y.shape[-1])).to(device)
            else:
                model = encoder
            losses = []; steps = 0 if kind == 'untrained' else int(options['representation_steps'])
            batch = min(int(options['batch_size']), len(x)); rng = np.random.default_rng(seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=float(options['learning_rate']), weight_decay=float(options['weight_decay']))
            for _ in range(steps):
                idx = rng.choice(len(x), batch, replace=False); a = x[idx]
                optimizer.zero_grad(set_to_none=True)
                if kind == 'masked_mse':
                    mask = gap_mask(a, .4, min(64, a.shape[-1] // 2))
                    loss = masked_loss(model(a.masked_fill(mask, 0)), a, mask)
                else:
                    loss = (model(a) - y[idx]).square().mean()
                loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            name = kind + '__seed_' + str(seed)
            tokens = {s: _encode_tokens(encoder, values[s], device) for s in SPLITS}
            metadata = dict(method='prospective diagnostic bank; H5/H6 deferred', representation_family=kind, optimization_seed=seed,
                optimizer_updates=steps, forward_examples=steps * batch, batch_size=batch,
                fixed_training_steps=True, checkpoint_selection='fixed last update; no evaluation use',
                parameter_count=sum(a.numel() for a in model.parameters()), loss_history=losses,
                data_exposure='same context index sequence per seed for masked and forecast; same history tensors',
                forecast_supervision='first training future only', encoder_config=options['encoder_config'])
            model_file = destination / 'banks' / (name + '_model.npz')
            atomic_npz(model_file, **{'tensor__' + k: v.detach().cpu().numpy() for k, v in model.state_dict().items()})
            metadata['model_sha256'] = file_hash(model_file)
            banks[name] = dict(values={s: tokens[s].mean(1) for s in SPLITS}, metadata=metadata)
            for view in (('full_tokens', 'fixed_token_readout') if kind in options['pooling_banks'] else ()):
                banks[name + '__' + view] = dict(values={s: token_views(tokens[s], options['token_bins'])[view] for s in SPLITS},
                                                metadata=dict(parent_bank=name, method=view, matched_final_code_budget=True))
            architecture.append(dict(bank=name, token_shape=list(tokens['train'].shape[1:]),
                                     pooled_shape=[tokens['train'].shape[2]], fixed_readout_shape=[options['token_bins'] * tokens['train'].shape[2]],
                                     reconstruction_uses='token sequence', forecasting_and_classification_use='pooled vector'))
    return banks, architecture


def _equal_circuit(loss, circuits):
    return float(np.mean([np.mean(loss[np.asarray(circuits) == c]) for c in np.unique(circuits)]))


def _classifier_score(y, logits, classes):
    mapping = {c: i for i, c in enumerate(classes.tolist())}
    if any(v not in mapping for v in y):
        raise ValueError('Scoring labels outside saved class order')
    index = np.asarray([mapping[v] for v in y]); logp = logits - logsumexp(logits, axis=1, keepdims=True)
    pred = logits.argmax(1); confusion = np.zeros((len(classes), len(classes)), int)
    np.add.at(confusion, (index, pred), 1)
    recall = np.divide(np.diag(confusion), confusion.sum(1), out=np.full(len(classes), np.nan), where=confusion.sum(1) > 0)
    return -logp[np.arange(len(y)), index] / np.log(2), dict(accuracy=float(np.mean(pred == index)),
        balanced_accuracy=float(np.nanmean(recall)), confusion_matrix=confusion.tolist(), class_order=classes.tolist(),
        class_support=confusion.sum(1).tolist(), balanced_accuracy_scope='classes present in scoring set; required coverage checked separately')


def evaluate_readout(panel, values, task, budget, alphas, destination):
    """Dev-selected linear decoder, with explicit capacity and split predictions."""
    classification = task in ('rhythm', 'carrier')
    key = ('X' if task == 'waveform_reconstruction' else 'Y' if task == 'future_mean' else 'task_' + task)
    if any(key not in panel[s] for s in SPLITS):
        return dict(status='unavailable', reason='task label unavailable', task=task, label_budget=budget)
    y, masks = {}, {}
    for s in SPLITS:
        y[s] = np.asarray(panel[s][key])
        if task == 'future_mean':
            first = np.argmax(panel[s]['future_mask'], axis=1)
            y[s] = y[s][np.arange(len(first)), first]
        if not classification and y[s].ndim == 1:
            y[s] = y[s][:, None]
        masks[s] = np.isfinite(y[s]).all(1) if y[s].ndim > 1 else np.isfinite(y[s])
        if task == 'carrier':
            masks[s] &= y[s] >= 0
        y[s] = y[s][masks[s]]
    if any(not masks[s].any() for s in SPLITS):
        return dict(status='unavailable', reason='empty eligible task population', task=task)
    groups = {s: np.asarray(panel[s]['structure_id'])[masks[s]] for s in SPLITS}
    classes = np.unique(y['train']) if classification else np.asarray([0])
    if classification and any(set(y[s].tolist()) != set(classes.tolist()) for s in SPLITS):
        return dict(status='unavailable', reason='required class coverage differs across splits', task=task)
    indices, sampling = balanced_label_subset(y['train'] if classification else np.zeros(len(y['train'])), groups['train'], budget, required_classes=classes)
    if indices is None:
        return dict(sampling, task=task, label_budget=budget)
    a = {s: np.asarray(values[s], float)[masks[s]] for s in SPLITS}
    offset = a['train'][indices].mean(0); scale = a['train'][indices].std(0); scale = np.where(scale > 1e-12, scale, 1.)
    z = {s: (a[s] - offset) / scale for s in SPLITS}; fitx = z['train'][indices]; fity = y['train'][indices]
    candidates = []; prediction_arrays = {}; model_arrays = dict(offset=offset, scale=scale, label_indices=indices)
    if not classification:
        ycenter = fity.mean(0); xc = fitx.mean(0); u, singular, vt = np.linalg.svd(fitx - xc, full_matrices=False)
    for alpha in alphas:
        alpha = float(alpha); predictions = {}; caught = []
        if classification:
            from sklearn.linear_model import LogisticRegression
            clf = LogisticRegression(C=1 / alpha, solver='lbfgs', max_iter=2000)
            with warnings.catch_warnings(record=True) as ww:
                warnings.simplefilter('always'); clf.fit(fitx, fity)
                caught = [str(w.message) for w in ww]
            if not np.array_equal(clf.classes_, classes):
                raise ValueError('Class order changed during readout fitting')
            for s in SPLITS:
                logits = z[s] @ clf.coef_.T + clf.intercept_
                predictions[s] = np.c_[np.zeros(len(logits)), logits[:, 0]] if logits.shape[1] == 1 else logits
            devloss, _ = _classifier_score(y['development'], predictions['development'], classes)
            arrays = dict(coef=clf.coef_, intercept=clf.intercept_, classes=classes)
            effective = int(np.linalg.matrix_rank(fitx)); parameters = int(clf.coef_.size + clf.intercept_.size)
        else:
            # Save a dual numeric decoder to avoid O(input_width * output_width) storage.
            dual = (u * (1 / (singular ** 2 + alpha))) @ (u.T @ (fity - ycenter))
            for s in SPLITS:
                predictions[s] = ((z[s] - xc) @ (fitx - xc).T) @ dual + ycenter
            if task == 'future_mean':
                dy = panel['development']['Y'][masks['development']]
                dm = panel['development']['future_mask'][masks['development']]
                devloss = np.where(dm, np.mean((predictions['development'][:, None] - dy) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
            else:
                devloss = np.mean((predictions['development'] - y['development']) ** 2, axis=1)
            arrays = dict(fit_basis=fitx - xc, dual_coef=dual, xcenter=xc, intercept=ycenter)
            effective = float(np.sum(singular ** 2 / (singular ** 2 + alpha)))
            parameters = int((fitx.shape[1] + 1) * np.prod(fity.shape[1:]))
        score = _equal_circuit(devloss, groups['development'])
        candidates.append(dict(alpha=alpha, development_equal_circuit_loss=score, warnings=caught,
                               nominal_parameters=parameters, effective_input_df_or_rank=effective))
        tag = str(alpha).replace('.', 'p')
        if classification:
            for s in SPLITS:
                prediction_arrays[s + '__alpha_' + tag] = predictions[s]
        if len(candidates) == 1 or score < selected_score:
            selected_score, selected_alpha, selected, selected_arrays = score, alpha, predictions, arrays
    if not classification:
        # Explicit float32 prediction storage is scored after round trip; physical targets remain exact.
        selected = {s: selected[s].astype(np.float32) for s in SPLITS}
    result = dict(prediction_storage='selected float32 regression predictions for all splits (scored after storage round trip); all candidate classification logits; regression candidate development losses retained',
        target_arrays='retention/task_targets.npz, shared exact target arrays and IDs; no repeated target copies per decoder', status='complete', task=task, label_budget=budget, sampling=sampling,
        selected_alpha=selected_alpha, candidates=candidates, decoder_family='linear logistic' if classification else 'linear ridge',
        selection='equal circuit-weighted development loss; evaluation never selects',
        cross_validation='none; genuinely circuit-disjoint train/development/evaluation; frozen-upstream diagnostic, not end-to-end CV',
        interpretation='decoder-accessible task information; cross entropy is not measured total mutual information',
        label_population='rhythmic contexts only' if task == 'carrier' else 'all eligible contexts')
    for s in SPLITS:
        ids = np.asarray(panel[s]['context_id'])[masks[s]]
        prediction_arrays[s + '_context_id'] = ids; prediction_arrays[s + '_structure_id'] = groups[s]
        if classification:
            prediction_arrays[s + '_target'] = y[s]
        prediction_arrays[s + '_selected_prediction'] = selected[s]
        if classification:
            loss, score = _classifier_score(y[s], selected[s], classes)
            score['cross_entropy_bits'] = _equal_circuit(loss, groups[s])
        else:
            if task == 'future_mean':
                dy = panel[s]['Y'][masks[s]]; dm = panel[s]['future_mask'][masks[s]]
                loss = np.where(dm, np.mean((selected[s].astype(float)[:, None] - dy) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
                baseline = np.where(dm, np.mean((dy - fity.mean(0)) ** 2, axis=-1), 0).sum(1) / dm.sum(1)
                prediction_arrays[s + '_future_mask'] = dm
            else:
                loss = np.mean((selected[s].astype(float) - y[s]) ** 2, axis=1)
                baseline = np.mean((y[s] - fity.mean(0)) ** 2, axis=1)
            score = dict(mse=_equal_circuit(loss, groups[s]), train_mean_baseline_mse=_equal_circuit(baseline, groups[s]))
            if task in ('waveform_reconstruction', 'future_mean') and 'normalization' in panel.get('manifest', {}):
                scale_key = 'x_scale' if task == 'waveform_reconstruction' else 'y_scale'
                target_scale = float(panel.get('manifest', {}).get('normalization', {}).get(scale_key, 1.))
                score['physical_mse'] = score['mse'] * target_scale ** 2
                score['physical_train_mean_baseline_mse'] = score['train_mean_baseline_mse'] * target_scale ** 2
                score['physical_unit'] = 'V^2' if task == 'waveform_reconstruction' else 'squared target units from panel manifest'
            score['normalized_by_train_mean_risk'] = score['mse'] / score['train_mean_baseline_mse'] if score['train_mean_baseline_mse'] > 0 else None
        prediction_arrays[s + '_selected_context_loss'] = loss
        score['per_circuit_loss'] = {str(c): float(np.mean(loss[groups[s] == c])) for c in np.unique(groups[s])}
        score['contexts'] = len(ids); score['circuits'] = len(np.unique(groups[s])); result[s] = score
    if classification:
        prediction_arrays['class_order'] = classes
    atomic_npz(destination / 'predictions.npz', **prediction_arrays)
    atomic_npz(destination / 'readout_weights.npz', **model_arrays, **selected_arrays)
    atomic_json(destination / 'scores.json', result)
    return result


def matched_code_contrast(rows, specification, *, job_id, historical, required_seeds):
    """One frozen complete-pipeline contrast, averaging fits inside each circuit."""
    import itertools
    import re
    spec = dict(specification)
    required = {'banks', 'dimension', 'bits', 'readout', 'label_budget', 'task', 'range_rule', 'pooling', 'job_id'}
    base = dict(specification=spec, protocol_hash=_json_hash(spec),
        interpretation='complete representation/projection/range/quantizer/decoder pipelines; no training-objective superiority claim',
        sampling_unit='circuit; optimizer repeats averaged within circuit before inference',
        claim_status='exploratory inspected historical panel' if historical else 'bounded prospective evaluation, not powered confirmation')
    if required - set(spec):
        return dict(base, status='unavailable', reason='contrast contract incomplete', missing_fields=sorted(required-set(spec)))
    if spec['job_id'] != job_id or spec['readout'] != 'linear' or spec['pooling'] != 'native pooled':
        return dict(base, status='unavailable', reason='requested job/readout/pooling differs from implemented matched endpoint')
    if spec['banks'] != ['feature18', 'forecast']:
        return dict(base, status='unsupported', reason='only the frozen feature18/forecast pipeline contrast is declared')
    selected = {}
    for family in spec['banks']:
        candidates = [r for r in rows if r.get('status') == 'complete'
            and r.get('representation_family', r.get('representation', '').split('__')[0]) == family
            and (r.get('representation') == family or re.fullmatch(re.escape(family) + r'__seed_\d+', r.get('representation', '')))
            and r.get('dimension') == spec['dimension'] and r.get('code_bits') == spec['bits']
            and r.get('range_rule') == spec['range_rule'] and r.get('task') == spec['task']
            and r.get('label_budget') == spec['label_budget']]
        if not candidates:
            return dict(base, status='unavailable', reason='required endpoint unavailable', missing_family=family)
        seeds = [r.get('optimization_seed') for r in candidates]
        if family == 'forecast' and not historical and (set(seeds) != set(required_seeds) or len(seeds) != len(set(seeds))):
            return dict(base, status='unavailable', reason='required forecast optimization repeats absent/duplicated',
                        observed_seeds=seeds, required_seeds=list(required_seeds))
        if family == 'feature18' and len(candidates) != 1:
            return dict(base, status='unavailable', reason='deterministic feature endpoint is ambiguous')
        if spec['task'] == 'carrier' and any(r['evaluation'].get('class_order') != [0, 1, 2] for r in candidates):
            return dict(base, status='unavailable', reason='all three conditional carrier classes are required')
        selected[family] = candidates
    circuits = sorted(selected['feature18'][0]['evaluation']['per_circuit_loss'])
    if any(set(r['evaluation']['per_circuit_loss']) != set(circuits) for c in selected.values() for r in c):
        return dict(base, status='unavailable', reason='contrasts do not share exactly the same evaluation circuits')
    means = {}; repeats = {}
    for family, candidates in selected.items():
        array = np.array([[r['evaluation']['per_circuit_loss'][c] for c in circuits] for r in candidates])
        means[family] = array.mean(0)
        repeats[family] = [dict(representation=r['representation'], optimization_seed=r.get('optimization_seed'),
                               circuit_loss=r['evaluation']['per_circuit_loss']) for r in candidates]
    difference = means['forecast'] - means['feature18']; n = len(circuits); observed = float(difference.mean())
    if n <= 16:
        signed = np.array([np.mean(difference * np.asarray(sign)) for sign in itertools.product((-1., 1.), repeat=n)])
        p = float(np.mean(np.abs(signed) >= abs(observed) - 1e-12))
        signflip = dict(status='complete', method='exact two-sided circuit sign flips under paired exchangeability',
                        permutations=len(signed), p_value=p, minimum_two_sided_p=2./(2**n))
    else:
        signflip = dict(status='unavailable', reason='exact sign-flip enumeration limited to16circuits; no silent asymptotic substitution')
    ci = dict(status='unavailable', reason='fewer than four independent circuits')
    if n >= 4:
        rng = np.random.default_rng(82471); bootstrap = difference[rng.integers(0, n, (10000, n))].mean(1)
        ci = dict(status='complete', lower=float(np.quantile(bootstrap, .025)), upper=float(np.quantile(bootstrap, .975)),
                  method='exploratory 95% percentile circuit bootstrap,10000draws',
                  caveat='few-circuit precision is weak; no power or confirmatory coverage guarantee')
    return dict(base, status='complete', circuit_count=n, optimizer_repeat_counts={k:len(v) for k,v in selected.items()},
        contrast='forecast minus feature18 per-circuit loss; positive favors feature18 pipeline',
        units='bits of decoder cross entropy' if spec['task'] in ('carrier','rhythm') else 'target-specific squared error',
        per_circuit_difference=dict(zip(circuits,map(float,difference))), mean_difference=observed,
        family_mean_loss={k:float(v.mean()) for k,v in means.items()}, optimizer_repeats=repeats,
        signflip=signflip, circuit_interval=ci,
        historical_optimizer_limit='single frozen training repeat; no objective comparison' if historical else None,
        carrier_population='conditional on true rhythmic membership' if spec['task']=='carrier' else None,
        multiplicity='one frozen primary matched-code contrast; other grid points are descriptive sensitivities')


def retention(config, root, source=None, device='cpu', resume=False):
    root = Path(root); source = _source(source); options = dict(DEFAULTS, **config.get('retention', {}))
    panel = load_panel(root, options['job_id'], source); validate_panel(panel)
    module_hash = file_hash(Path(__file__))
    signature = _json_hash(dict(analysis_module_sha256=module_hash, options=options, matched_code_contrast=config.get('protocol', {}).get('matched_code_contrast'), panel=panel['manifest'],
        source_content_id=source.provenance()['content_id'] if source is not None else None,
        inputs={s: _digest_arrays({k: panel[s][k] for k in ('X', 'Y', 'context_id', 'structure_id')}) for s in SPLITS}))
    destination = root / 'retention'; marker = destination / 'summary.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if not resume or previous['signature'] != signature:
            raise ValueError('Retention output exists or resume fingerprint mismatch')
        for row in previous['artifacts']:
            if file_hash(destination / row['path']) != row['sha256']:
                raise ValueError('Retention artifact checksum mismatch')
        return previous
    guard = destination / 'analysis_identity.json'
    if guard.exists() and (not resume or json.loads(guard.read_text())['signature'] != signature):
        raise ValueError('Incompatible/incomplete retention output; use --resume with same contract')
    atomic_json(guard, dict(signature=signature, analysis_module_sha256=module_hash, options=options, panel_manifest=panel['manifest']))
    atomic_npz(destination / 'task_targets.npz', **{s + '_' + k: v for s in SPLITS for k, v in panel[s].items()
               if k in ('X', 'X_physical', 'Y', 'Y_physical', 'future_mask', 'future_id', 'structure_id', 'context_id') or k.startswith('task_')})
    imported = _frozen_banks(panel, source, destination, options, device)
    banks, architecture = imported if imported is not None else _new_banks(panel, destination, options, device)
    direct = {s: np.asarray(panel[s]['X'], float) for s in SPLITS}
    banks['raw_history'] = dict(values=direct, metadata=dict(method='identity history; ridge readout is not an information oracle'))
    rows = []; attribution = []; inverse_rows = []
    for name, bank in banks.items():
        if 'values' not in bank:
            rows.append(dict(representation=name, **bank)); continue
        values = bank['values']; width = values['train'].shape[1]
        directory = destination / 'banks' / name
        atomic_npz(directory / 'representations.npz', **values,
                   **{s + '_context_id': panel[s]['context_id'] for s in SPLITS})
        atomic_json(directory / 'metadata.json', dict(bank['metadata'], native_width=width))
        variants = [('native_unquantized', values, dict(dimension=width, code_bits=None, endpoint='native-width unquantized'))]
        if 'inverse' in bank:
            inv = bank['inverse']
            inverse_rows.append(dict(representation=name, dimension=width, code_bits=None, status='complete',
                endpoint='deterministic native-width train-PCA inverse',
                scores={s: float(np.mean((values[s] @ inv['components'] + inv['offset'] - direct[s]) ** 2)) for s in SPLITS}))
        if name == 'raw_history':
            inverse_rows.append(dict(representation=name, endpoint='exact raw-history identity', mse=0.,
                                     status='complete', interpretation='exact identity is separate from a fitted raw-history decoder'))
            # Native history is an important capacity baseline, not another bit sweep.
        else:
            for d in options['dimensions']:
                try:
                    p = projection_fit(values['train'], int(d))
                except ValueError as exc:
                    rows.append(dict(status='unsupported', representation=name, dimension=d, reason=str(exc))); continue
                latent = {s: project(values[s], p) for s in SPLITS}
                projected = directory / ('d' + str(d)); atomic_npz(projected / 'projection.npz', **{k: v for k, v in p.items() if isinstance(v, np.ndarray)})
                projection_report = {s: dict(native_coordinate_truncation_mse=float(np.mean((values[s] - (latent[s] @ p['components'] + p['center'])) ** 2))) for s in SPLITS}
                variants.append((f'd{d}_unquantized', latent, dict(dimension=d, code_bits=None, endpoint='projected unquantized', projection=projection_report)))
                for rule in options['range_rules']:
                    for bits in options['code_bits']:
                        try:
                            codebook = fit_range_rule(latent['train'], int(bits), rule)
                        except ValueError as exc:
                            rows.append(dict(status='unsupported', representation=name, dimension=d, code_bits=bits, reason=str(exc))); continue
                        decoded = {}; arrays = {}; reports = {}
                        for s in SPLITS:
                            packed, decoded[s], reports[s] = quantization_attribution(latent[s], codebook)
                            arrays.update({s + '_packed': packed, s + '_decoded': decoded[s], s + '_context_id': panel[s]['context_id']})
                        variant = f'd{d}_B{bits}_{rule}'; point = directory / variant
                        atomic_npz(point / 'codes.npz', **arrays)
                        atomic_npz(point / 'codebook.npz', **{k: v for k, v in codebook.items() if isinstance(v, np.ndarray)})
                        atomic_json(point / 'codebook.json', dict(codebook, clipping=reports, shared_storage='projection/codebook/encoder/readout weights excluded from per-example payload'))
                        meta = dict(dimension=d, code_bits=bits, range_rule=rule, endpoint='same projected code quantized', projection=projection_report, quantization=reports)
                        attribution.append(dict(representation=name, **meta))
                        variants.append((variant, decoded, meta))
                        if 'inverse' in bank:
                            inv = bank['inverse']
                            inverse_rows.append(dict(representation=name, dimension=d, code_bits=bits, range_rule=rule,
                                status='complete', endpoint='deterministic train-PCA inverse',
                                scores={s: float(np.mean(((decoded[s] @ p['components'] + p['center']) @ inv['components'] + inv['offset'] - direct[s]) ** 2)) for s in SPLITS}))
                if 'inverse' in bank:
                    inv = bank['inverse']; inverse_rows.append(dict(representation=name, dimension=d, code_bits=None, status='complete',
                        endpoint='deterministic train-PCA inverse', scores={s: float(np.mean(((latent[s] @ p['components'] + p['center']) @ inv['components'] + inv['offset'] - direct[s]) ** 2)) for s in SPLITS}))
        for variant, a, meta in variants:
            for task in options['tasks']:
                for budget in options['label_budgets']:
                    result = evaluate_readout(panel, a, task, budget, options['regularization'], directory / variant / task / ('labels_' + str(budget)))
                    rows.append(dict(result, representation=name, representation_family=name.split('__')[0],
                                     optimization_seed=bank['metadata'].get('optimization_seed', bank['metadata'].get('original_metadata', {}).get('seed')), variant=variant, **meta))
        print('V4 retention bank complete:', name, flush=True)
    contrast = matched_code_contrast(rows, config.get('protocol', {}).get('matched_code_contrast', {}), job_id=options['job_id'], historical=imported is not None, required_seeds=options['optimization_seeds'])
    atomic_json(destination / 'matched_contrast.json', contrast)
    artifacts = [dict(path=str(p.relative_to(destination)), sha256=file_hash(p), bytes=p.stat().st_size)
                 for p in sorted(destination.rglob('*')) if p.is_file() and p != marker]
    summary = dict(schema_version='eegrepr-evaluation-v4-retention-1', status='complete', signature=signature, analysis_module_sha256=module_hash,
        scientific_question='H3 task-relevant compression', options=options, job_id=options['job_id'],
        claim_status='exploratory reanalysis of inspected circuits' if imported else 'bounded prospective diagnostic; not powered confirmation',
        split_counts={s: dict(contexts=len(panel[s]['X']), circuits=len(np.unique(panel[s]['structure_id']))) for s in SPLITS},
        rows=rows, matched_code_contrast=contrast, attribution=attribution, deterministic_reconstruction=inverse_rows, architecture=architecture,
        objective_claims='H5/H6 deferred; attribution compares complete encoder/projection/quantizer/decoder pipelines',
        representation_storage='numeric arrays only; no pickle model loading', artifacts=artifacts)
    atomic_json(marker, summary)
    return summary
