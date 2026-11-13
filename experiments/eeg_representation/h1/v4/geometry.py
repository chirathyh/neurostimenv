"""Declared-population geometry and innovation diagnostics, not universal ID.

Coordinate counts, covariance rank, participation ratios, code rates and nonlinear
manifold dimension are different objects. No privileged array is a learner input.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import numpy as np

from .retention import SPLITS, load_panel, _source, _json_hash, _digest_arrays
from ..benchmark import atomic_npz
from ...storage import atomic_json, file_hash


def covariance_geometry(values, *, scaling='physical', training=None, thresholds=(.8, .9, .95, .99)):
    x = np.asarray(values, float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('Geometry requires a finite samples-by-coordinates matrix')
    n, d = x.shape
    if n < 2:
        return dict(status='unavailable', samples=n, coordinates=d, reason='sample covariance needs K>=2', rank_ceiling=0)
    if scaling == 'train_coordinate_standardized':
        if training is None:
            raise ValueError('Training coordinate reference required')
        train = np.asarray(training, float)
        scale = train.std(0, ddof=1)
        scale = np.where(scale > 0, scale, 1.)
        scaling_diagnostics = dict(minimum_positive_training_sd=float(np.min(scale)), maximum_training_sd=float(np.max(scale)),
            caveat='Sparse nearly inactive training coordinates can dominate pooled standardized covariance; inspect physical companion and per-circuit spectra')
        x = (x - train.mean(0)) / scale
    elif scaling != 'physical':
        raise ValueError('Declare physical or train_coordinate_standardized scaling')
    if scaling == 'physical':
        scaling_diagnostics = dict(caveat='Mixed-unit state spectra depend on coordinate units; EEG coordinates have a common voltage unit')
    centered = x - x.mean(0)
    if d > n:
        eigenvalues = np.linalg.eigvalsh(centered @ centered.T / (n - 1))[::-1]
    else:
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / (n - 1))[::-1]
    # Ordinary sample covariance is PSD; expose tiny round-off before zeroing.
    negative_roundoff = float(min(0., eigenvalues.min(initial=0)))
    tol = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny) * 1e-10
    if np.min(eigenvalues) < -tol:
        raise ValueError('Sample covariance has a materially negative eigenvalue')
    eigenvalues = np.maximum(eigenvalues, 0.)
    total = float(eigenvalues.sum())
    # PR is scale invariant. Normalize before squaring so sparse train-SD
    # scaling cannot overflow even when covariance eigenvalues are finite.
    peak = float(eigenvalues.max(initial=0))
    relative = eigenvalues / peak if peak else np.zeros_like(eigenvalues)
    square = float(np.sum(relative ** 2))
    participation = float(relative.sum() ** 2 / square) if square else 0.
    cumulative = np.cumsum(eigenvalues) / total if total > 0 else np.zeros_like(eigenvalues)
    counts = {str(t): int(np.searchsorted(cumulative, t) + 1) if total > 0 else 0 for t in thresholds}
    return dict(status='complete', samples=n, coordinates=d, rank_ceiling=min(n - 1, d),
        numerical_rank=int(np.sum(eigenvalues > tol)), eigenvalues=eigenvalues.tolist(), covariance_trace=total,
        participation_ratio=participation, variance_threshold_counts=counts,
        scaling=scaling, scaling_diagnostics=scaling_diagnostics, numerical_negative_roundoff=negative_roundoff,
        interpretation='linear covariance geometry at declared sampling/scaling; not intrinsic manifold dimension')


def deduplicate_prefixes(values, ids):
    values, ids = np.asarray(values), np.asarray(ids)
    seen = {}; keep = []
    for i, cid in enumerate(map(str, ids)):
        if cid in seen:
            if not np.array_equal(values[i], values[seen[cid]]):
                raise ValueError('Same prefix ID has different values')
        else:
            seen[cid] = i; keep.append(i)
    return np.asarray(keep, int)


def grouped_geometry(values, circuits, regimes, *, training=None, scaling='physical'):
    x = np.asarray(values); c = np.asarray(circuits); r = np.asarray(regimes)
    output = dict(pooled=covariance_geometry(x, training=training, scaling=scaling), within_circuit=[], within_circuit_regime=[])
    for circuit in np.unique(c):
        mask = c == circuit
        output['within_circuit'].append(dict(circuit=str(circuit), **covariance_geometry(x[mask], training=training, scaling=scaling)))
        for regime in np.unique(r[mask]):
            select = mask & (r == regime)
            output['within_circuit_regime'].append(dict(circuit=str(circuit), regime=str(regime),
                **covariance_geometry(x[select], training=training, scaling=scaling)))
    # Centered within-group residuals distinguish covariance geometry from circuit means.
    residual = x.astype(float).copy()
    for circuit in np.unique(c):
        for regime in np.unique(r[c == circuit]):
            select = (c == circuit) & (r == regime); residual[select] -= residual[select].mean(0)
    output['pooled_group_centered'] = covariance_geometry(residual, training=training, scaling=scaling)
    output['group_centering'] = 'remove each observed circuit-regime mean; no claim of independent residual samples'
    rng = np.random.default_rng(701)
    output['sample_size_sensitivity'] = []
    for n in sorted(set([min(len(x), 16), min(len(x), 32), min(len(x), 64), len(x)])):
        if n < 2:
            continue
        idx = np.sort(rng.choice(len(x), n, replace=False))
        g = covariance_geometry(x[idx], training=training, scaling=scaling)
        output['sample_size_sensitivity'].append({k: g[k] for k in ('samples', 'rank_ceiling', 'participation_ratio', 'variance_threshold_counts')})
    output['uncertainty'] = 'descriptive sample-size sensitivity, not independent-population confidence intervals; circuits remain the statistical unit'
    return output


def innovation_geometry(y, mask=None):
    """Independent future-half cross covariance; never silently PSD project it."""
    y = np.asarray(y, float)
    if y.ndim != 3:
        raise ValueError('Y must have context, future, coordinate axes')
    mask = np.ones(y.shape[:2], bool) if mask is None else np.asarray(mask, bool)
    valid = np.flatnonzero(mask.sum(1) >= 2)
    if len(valid) < 2:
        return dict(status='unavailable', reason='need >=2 contexts with >=2 futures'), {}
    means1, means2, variances, first = [], [], [], []
    half_indices = []
    for i in valid:
        indexes = np.flatnonzero(mask[i]); cut = len(indexes) // 2
        a, b = indexes[:cut], indexes[cut:]
        assert not set(a) & set(b)
        means1.append(y[i, a].mean(0)); means2.append(y[i, b].mean(0)); first.append(y[i, indexes[0]])
        variances.append(np.atleast_2d(np.cov(y[i, indexes], rowvar=False, ddof=1)))
        half_indices.append(dict(context_index=int(i), first_half=a.tolist(), second_half=b.tolist(),
                                 conditional_rank_ceiling=min(len(indexes)-1, y.shape[-1])))
    a, b = np.asarray(means1), np.asarray(means2)
    cross = (a - a.mean(0)).T @ (b - b.mean(0)) / (len(a) - 1)
    symmetric = (cross + cross.T) / 2
    noise = np.mean(variances, axis=0)
    total = np.atleast_2d(np.cov(np.asarray(first), rowvar=False, ddof=1))
    eig = np.linalg.eigvalsh(symmetric)[::-1]
    arrays = dict(cross_covariance=cross, symmetric_cross_covariance=symmetric,
        symmetric_cross_eigenvalues=eig, mean_conditional_covariance=noise,
        one_future_total_covariance=total, first_half_means=a, second_half_means=b,
        context_indices=valid)
    return dict(status='complete', contexts=len(valid), coordinates=y.shape[-1], half_indices=half_indices,
        cross_covariance_rank_ceiling=min(len(valid)-1, y.shape[-1]),
        symmetry_residual_frobenius=float(np.linalg.norm(cross - cross.T)),
        symmetric_cross_eigenvalues=eig.tolist(), negative_eigenvalues=int(np.sum(eig < 0)),
        trace_mean_response_estimate=float(np.trace(cross)), trace_conditional_variance=float(np.trace(noise)),
        trace_total_one_future_estimate=float(np.trace(total)),
        finite_sample_decomposition_residual=float(np.trace(total - symmetric - noise)),
        psd_clipping=False, units='squared declared target units',
        sampling='one first future per context for total covariance; disjoint fixed future halves for response covariance',
        inference='descriptive context-population geometry; regimes share corresponding streams, circuits remain population units',
        limitations='finite noisy cross covariance can be nonsymmetric/indefinite; mean-response geometry omits variance-coded task information'), arrays


def geometry_controls():
    rng = np.random.default_rng(101)
    t = np.arange(1024) * 2 * np.pi / 1024
    circle = np.c_[np.cos(t), np.sin(t)]
    surface = rng.uniform(-1, 1, (2048, 2)); surface = np.c_[surface, np.zeros(len(surface))]
    noisy = np.c_[circle, rng.normal(0, .1, len(circle))]
    poisson = rng.poisson(3., size=(2048, 4)).astype(float)
    values = dict(circle=circle, plane_surface=surface, noisy_circle=noisy, poisson_counts=poisson)
    results = {k: covariance_geometry(v) for k, v in values.items()}
    passed = (results['circle']['numerical_rank'] == 2 and abs(results['circle']['participation_ratio'] - 2) < 1e-8
              and results['plane_surface']['numerical_rank'] == 2 and results['noisy_circle']['numerical_rank'] == 3
              and results['poisson_counts']['numerical_rank'] == 4)
    results['known_intrinsic_dimensions'] = dict(circle=1, plane_surface=2, noisy_circle='3-dimensional full-support noise', poisson_counts='discrete support; no smooth manifold ID assumed')
    results['nonlinear_id'] = dict(status='unavailable', gate_passed=False,
        reason='No nonlinear estimator is qualified for noisy/discrete mixed populations; controls demonstrate linear rank is not intrinsic dimension',
        future_gate='require circle/surface recovery plus sample-size, neighborhood-resolution, noise and Poisson sensitivity before applying any nonlinear ID estimator')
    return dict(status='complete', passed=passed, controls=results)


def _parse_state(state, synapses, cells, compartments):
    """Decode legacy storage order using actual contact location metadata.

    The HH soma has 3 gates, all compartments store v/vext, and each Exp2Syn
    stores A/B. Conductance states sharing compartment/e/kinetics are additive.
    This is bookkeeping/possible redundancy, never a minimal biological DOF.
    """
    if compartments != 22 * cells:
        raise ValueError('Legacy decoding requires audited one-soma/21-apical geometry')
    state = np.asarray(state); cursor = 0; membrane = []; aggregate = {}
    for gid in range(cells):
        for section, count in ((0, 1), (1, 21)):
            for seg in range(count):
                position = (seg + .5) / count
                membrane.extend(state[cursor:cursor + 2]); cursor += 2
                if section == 0:
                    membrane.extend(state[cursor:cursor + 3]); cursor += 3
                rows = synapses[(synapses[:, 0] == gid) & (synapses[:, 1] == section) & np.isclose(synapses[:, 2], position, rtol=0, atol=1e-8)]
                for row in rows:
                    key = tuple(row.tolist())
                    aggregate[key] = aggregate.get(key, np.zeros(2)) + state[cursor:cursor + 2]
                    cursor += 2
    if cursor != len(state):
        raise ValueError('Physical-state decoding does not consume stored coordinates')
    return np.asarray(membrane), aggregate


def event_generator_inventory(contacts, rate_groups, fraction):
    """Count active Poisson generators separately from stored contact streams."""
    if not 0 <= fraction <= 1 or contacts < 1 or rate_groups < 1:
        raise ValueError('Invalid common-event driver inventory')
    private = int(contacts) if fraction < 1 else 0
    common = int(rate_groups) if fraction > 0 else 0
    return dict(stored_contact_event_streams=int(contacts), independent_private_event_generators=private,
                common_event_groups=common, conditional_independent_poisson_generators_total=private + common,
                common_event_fraction=float(fraction),
                generator_scope='active conditional Poisson sources; c=1 retains contact arrays but private processes have zero rate')


def native_inventory(source, panel, destination):
    source = _source(source)
    if source is None or not source.exists('native/manifest.json'):
        return dict(status='unavailable', reason='native archive absent; no invented voltage/current trajectories'), {}
    import h5py
    trials = source.read_json('native/manifest.json')['trials']
    generation_config = source.read_json('resolved_config.json') if source.exists('resolved_config.json') else {}
    driver_config = generation_config.get('v4_driver', generation_config.get('optional', {}))
    common_fraction = float(driver_config.get('common_event_fraction', 0.))
    wanted = {str(cid) for s in SPLITS for cid in panel[s]['context_id']}
    unique = {}
    for row in trials:
        if str(row['context_id']) in wanted and source.exists(row['path']):
            unique.setdefault(str(row['context_id']), row)
    if not unique:
        return dict(status='unavailable', reason='native files omitted from supplied archive'), {}
    states = {}; membranes = []; aggregates = []; ids = []; circuits = []; regimes = []; records = []
    for cid, row in sorted(unique.items()):
        source.verify(row['path'])
        with h5py.File(io.BytesIO(source.read_bytes(row['path'])), 'r') as f:
            metadata = json.loads(f.attrs['metadata_json']); m = metadata.get('metadata', metadata)
            p = f['privileged']; required = ('cutoff_physical_state', 'synapse_parameters', 'external_netcons', 'recurrent_netcons')
            if any(k not in p for k in required):
                continue
            state = p['cutoff_physical_state'][:]; syn = p['synapse_parameters'][:]
            pops = m['population_sizes']; cells = sum(pops.values()); compartments = (len(state) - 2 * len(syn) - 3 * cells) // 2
            if 2 * compartments + 3 * cells + 2 * len(syn) != len(state):
                raise ValueError('State coordinate accounting mismatch')
            membrane, aggregate = _parse_state(state, syn, cells, compartments)
            groups = np.unique(syn, axis=0)
            rates = {k: v.get('baseline_afferent_rate_hz') for k, v in m.get('drive_diagnostics', {}).items() if isinstance(v, dict) and k in ('E', 'I')}
            rate_groups = len(set(v for v in rates.values() if v is not None))
            if common_fraction > 0 and not rate_groups:
                raise ValueError('Common-event inventory lacks observed equal-rate group metadata')
            generators = event_generator_inventory(len(p['external_netcons']), max(1, rate_groups), common_fraction)
            circuit = str(row['structure_id']); states.setdefault(circuit, []).append((cid, state))
            membranes.append(membrane); aggregates.append(aggregate); ids.append(cid); circuits.append(circuit)
            regimes.append(m.get('regime', {}).get('name', cid.split('-')[-1]))
            records.append(dict(context_id=cid, structure_id=circuit, native_sha256=row.get('sha256'),
                cells=cells, population_sizes=pops, compartments=int(compartments),
                compartments_by_mechanism=dict(hh_soma=cells, passive_apical=int(compartments-cells), extracellular=int(compartments)),
                stored_state_coordinates=len(state), membrane_gate_coordinates=len(membrane),
                external_contacts=len(p['external_netcons']), **generators,
                recurrent_contacts=len(p['recurrent_netcons']), total_synapses=len(syn),
                additive_contact_groups=len(groups), possible_Exp2Syn_A_B_aggregation_redundancy=2*(len(syn)-len(groups)),
                synapse_kinetics_tau1_tau2_reversal=np.unique(syn[:, 3:6], axis=0).tolist(),
                causally_pending_recurrent_events=(len(p['pending_recurrent_events']) if 'pending_recurrent_events' in p else len(json.loads(p.attrs['pending_recurrent_events'])) if 'pending_recurrent_events' in p.attrs else None),
                shared_rate_phase_groups=1, shared_rate_phase_active=bool(m.get('regime', {}).get('modulation_depth', 0)),
                aggregation='one net three-component dipole -> one scalp EEG channel', channels=f['eeg_v'].shape[-1],
                native_dt_ms=m.get('native_dt_ms'), celsius=m.get('effective_celsius'),
                per_neuron_voltage_current_trajectories='not stored',
                recurrent_output_statistics='not assumed independent Poisson',
                afferent_mean_rates_per_synapse_hz=rates,
                excluded_from_state_geometry='future RNG state/seeds/schedules/drive paths; state arrays remain privileged audit only'))
    if not records:
        return dict(status='unavailable', reason='native backend lacks biophysical-state instrumentation'), {}
    # Aligned aggregate contact labels provide zero coordinates for absent groups;
    # full original state is pooled only within identical circuit structure.
    union = sorted(set().union(*(set(a) for a in aggregates)))
    combined = np.array([np.r_[membrane, np.concatenate([a.get(k, np.zeros(2)) for k in union])]
                         for membrane, a in zip(membranes, aggregates)])
    lookup = {str(cid): s for s in SPLITS for cid in panel[s]['context_id']}
    train = np.asarray([lookup[cid] == 'train' for cid in ids]); circuits = np.asarray(circuits); regimes = np.asarray(regimes)
    results = dict(status='complete', contexts=len(ids), native_trials_in_source=len(trials),
        deduplication='one state per context ID; fork-identical repeated futures excluded', records=records,
        coordinate_semantics='stored coordinates and possible additive redundancy; not minimal biological state dimension',
        state_geometry_scope='Stored membrane/gate/synapse numeric snapshot only; causally pending variable-length queue and phase are inventoried separately, not claimed to be represented by this covariance matrix',
        full_state_pooled=dict(status='unsupported', reason='recurrent contact layouts differ across circuit seeds; stored-coordinate order is not globally aligned'),
        aggregation_axes=[list(k) for k in union],
        aligned_aggregation_scaling='train-coordinate standardization of v/vext/HH gates and same-location/kinetics/reversal summed A/B states',
        aligned_aggregation=grouped_geometry(combined, circuits, regimes, training=combined[train], scaling='train_coordinate_standardized'),
        aligned_aggregation_physical=grouped_geometry(combined, circuits, regimes, scaling='physical'),
        within_circuit_full_state={})
    for circuit, entries in states.items():
        a = np.stack([v for _, v in entries])
        results['within_circuit_full_state'][circuit] = covariance_geometry(a, scaling='physical')
        results['within_circuit_full_state'][circuit]['scaling_caveat'] = 'mixed physical units; unit choices affect spectrum; inspect aligned train-standardized companion'
    arrays = dict(context_id=np.asarray(ids), structure_id=circuits, regime=regimes,
                  membrane_gate_state=np.asarray(membranes), aligned_aggregate_state=combined,
                  aggregation_axes=np.asarray(union))
    return results, arrays


def geometry(config, root, source=None, resume=False):
    root = Path(root); source = _source(source); opts = dict(config.get('geometry', {}))
    if source is None and (root / 'native/manifest.json').exists():
        source = _source(root)
    job = opts.get('job_id', config.get('retention', {}).get('job_id', 'A0__T0_l10_d128'))
    panel = load_panel(root, job, source)
    module_hash = file_hash(Path(__file__))
    signature = _json_hash(dict(analysis_module_sha256=module_hash, options=opts, manifest=panel['manifest'], source_content_id=source.provenance()['content_id'] if source is not None else None,
        arrays={s: _digest_arrays({k: panel[s][k] for k in ('X', 'Y', 'context_id')}) for s in SPLITS}))
    dest = root / 'geometry'; marker = dest / 'summary.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if not resume or previous['signature'] != signature:
            raise ValueError('Geometry output exists or resume identity mismatch')
        for row in previous['artifacts']:
            if file_hash(dest / row['path']) != row['sha256']:
                raise ValueError('Geometry artifact checksum mismatch')
        return previous
    guard = dest / 'analysis_identity.json'
    if guard.exists() and (not resume or json.loads(guard.read_text())['signature'] != signature):
        raise ValueError('Incompatible/incomplete geometry output')
    atomic_json(guard, dict(signature=signature, analysis_module_sha256=module_hash))
    controls = geometry_controls()
    if not controls['passed']:
        raise AssertionError('Known linear geometry controls failed')
    rows = {}
    circuits = np.concatenate([panel[s]['structure_id'].astype(str) for s in SPLITS])
    regimes = np.concatenate([panel[s].get('regime', panel[s].get('regime_id', np.full(len(panel[s]['X']), 'unspecified'))) for s in SPLITS])
    ids = np.concatenate([panel[s]['context_id'] for s in SPLITS])
    x = np.concatenate([panel[s].get('X_physical', panel[s]['X']) for s in SPLITS])
    keep = deduplicate_prefixes(x, ids)
    train = panel['train'].get('X_physical', panel['train']['X'])
    for scaling in ('physical', 'train_coordinate_standardized'):
        rows['EEG_' + scaling] = grouped_geometry(x[keep], circuits[keep], regimes[keep], training=train, scaling=scaling)
    rows['EEG_population'] = dict(contexts=len(keep), coordinates=x.shape[1],
        temporal_dependence='history coordinates are serially dependent; 2000 coordinates do not mean 2000 independent samples',
        population='one retained history per distinct context, pooled across declared circuit/regime groups')
    # Numeric bank arrays already produced by retention; fall back to imported frozen banks.
    bank_paths = sorted((root / 'retention/banks').glob('*/representations.npz'))
    for path in bank_paths:
        with np.load(path, allow_pickle=False) as f:
            values = {s: f[s] for s in SPLITS}
            if any(not np.array_equal(f[s+'_context_id'], panel[s]['context_id']) for s in SPLITS):
                raise ValueError('Geometry representation-context mismatch')
        rows['representation_' + path.parent.name] = grouped_geometry(np.concatenate([values[s] for s in SPLITS]),
            circuits, regimes, training=values['train'], scaling='train_coordinate_standardized')
    if not bank_paths and source is not None and source.exists('compression/primary/representations/manifest.json'):
        for record in source.read_json('compression/primary/representations/manifest.json')['representations']:
            p = 'compression/primary/representations/' + record['path']
            values = source.read_npz(p)
            if any(not np.array_equal(values[s+'_context_id'], panel[s]['context_id']) for s in SPLITS):
                raise ValueError('Imported geometry representation-context mismatch')
            rows['representation_' + record['name']] = grouped_geometry(np.concatenate([values[s] for s in SPLITS]),
                circuits, regimes, training=values['train'], scaling='train_coordinate_standardized')
    innovations, arrays = innovation_geometry(panel['evaluation'].get('Y_physical', panel['evaluation']['Y']), panel['evaluation']['future_mask'])
    if arrays:
        arrays['context_id'] = panel['evaluation']['context_id'][arrays['context_indices']]
        atomic_npz(dest / 'innovation.npz', **arrays)
    inventory, state_arrays = native_inventory(source, panel, dest)
    inventory['observed_array_shapes'] = {s: {k: list(panel[s][k].shape) for k in ('X', 'Y', 'future_mask')} for s in SPLITS}
    inventory['representation_shapes'] = {k.removeprefix('representation_'): [v['pooled']['samples'], v['pooled']['coordinates']] for k, v in rows.items() if k.startswith('representation_')}
    retention_marker = root / 'retention/summary.json'
    if retention_marker.exists():
        retention_report = json.loads(retention_marker.read_text())
        inventory['actual_token_pooling_shapes'] = retention_report.get('architecture', [])
        inventory['code_shapes'] = sorted({(int(r['dimension']), int(r['code_bits']), int(r['code_bits']) // 8) for r in retention_report.get('rows', []) if r.get('status') == 'complete' and r.get('code_bits') is not None})
    else:
        inventory['actual_token_pooling_shapes'] = {'status': 'unavailable', 'reason': 'run retention numeric encoder audit to observe token/pooled/fixed readout shapes'}
        inventory['code_shapes'] = {'status': 'unavailable', 'reason': 'no executed retention payloads in this root'}
    if state_arrays:
        atomic_npz(dest / 'privileged_state_geometry.npz', **state_arrays)
    atomic_json(dest / 'runtime_inventory.json', inventory)
    atomic_json(dest / 'covariance_geometry.json', rows)
    atomic_json(dest / 'geometry_controls.json', controls)
    atomic_json(dest / 'innovation.json', innovations)
    artifacts = [dict(path=str(p.relative_to(dest)), sha256=file_hash(p), bytes=p.stat().st_size)
                 for p in sorted(dest.rglob('*')) if p.is_file() and p != marker]
    summary = dict(schema_version='eegrepr-evaluation-v4-geometry-1', status='complete', signature=signature, analysis_module_sha256=module_hash,
        job_id=job, controls=controls, runtime_inventory_status=inventory['status'],
        covariance_populations=list(rows), innovation=innovations, artifacts=artifacts,
        nonlinear_id=controls['controls']['nonlinear_id'],
        limitations='Descriptive covariance geometry only. No universal intrinsic dimension, human EEG claim, or information fraction. Native archive has no full neuron voltage/current trajectories.',
        conditioning='Privileged state arrays are audit-only; no future RNG/schedule is used as a feature')
    atomic_json(marker, summary)
    return summary
