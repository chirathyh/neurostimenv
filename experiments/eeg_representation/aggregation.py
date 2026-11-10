"""Aggregate actual transfer artifacts, preserving missing and negative results."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from .statistics import benjamini_hochberg, exact_sign_flip, paired_cluster_interval
from .tasks import TASKS


PROTOCOLS = ('leave-task-label-out', 'leave-generator-family-out')


def _write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name + '.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path, default):
    return json.loads(path.read_text()) if path.exists() else default


def _csv(path, rows, columns):
    content = io.StringIO()
    writer = csv.DictWriter(content, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({key: ('Pending' if value is None else value) for key, value in row.items()})
    _write(path, content.getvalue())


def _recipes(config):
    recipes = []
    for recipe in config['training']['recipes']:
        recipes.append(recipe)
        if recipe == 'random_order':
            distinct_orders = min(config['training'].get('random_orders', 3), math.factorial(len(config['training']['source_tasks'])))
            recipes.extend(f'random_order__order_{i}' for i in range(1, distinct_orders))
    return recipes


def _paired(left, right, bootstrap, seed=0):
    common = sorted(set(left) & set(right))
    if not common:
        return {'status': 'pending: no paired structures', 'estimate': None, 'ci': None, 'p_value': None}
    x = np.array([left[key] - right[key] for key in common])
    result = paired_cluster_interval([left[key] for key in common], [right[key] for key in common], common,
                                     seed=seed, n_bootstrap=bootstrap)
    if len(x) <= 20:
        test = exact_sign_flip(x)
        result.update(p_value=test['p_value'], p_method='exact structure sign flip')
    else:
        rng = np.random.default_rng(seed)
        signs = rng.choice([-1, 1], size=(bootstrap, len(x)))
        count = (np.abs((signs * x).mean(1)) >= abs(x.mean())).sum()
        result.update(p_value=float((count + 1) / (bootstrap + 1)), p_method='Monte Carlo structure sign flip')
    result.update(status='complete', structures=common, missing_left=sorted(set(right) - set(left)),
                  missing_right=sorted(set(left) - set(right)))
    return result


def _comparison_pairs(recipes):
    pairs = []
    reconstructive = ['masked_mse', 'masked_l1', 'spectral_hybrid', 'reconstruction_supervised',
                      'forecast_lowrank', 'forecast_diagonal', 'forecast_ar', 'momentum_contrastive', 'features']
    for preferred in ['joint', 'progressive']:
        for comparator in reconstructive:
            if preferred in recipes and comparator in recipes:
                pairs.append(('H2a', preferred, comparator))
    if 'progressive' in recipes:
        for comparator in recipes:
            if comparator in {'joint', 'reverse', 'random_order'} or comparator.startswith('random_order__order_'):
                pairs.append(('H2b', 'progressive', comparator))
    return pairs


def _exposure_audit(runs, protocol, left, right, budget, seeds):
    checks = []
    for seed in seeds:
        pair = []
        for recipe in (left, right):
            found = [r for r in runs if r.get('protocol') == protocol and r.get('recipe') == recipe
                     and str(r.get('budget')) == budget and r.get('seed') == seed and r.get('status') == 'complete']
            pair.append(found[0] if len(found) == 1 else None)
        if any(r is None for r in pair):
            checks.append({'seed': seed, 'matched': False, 'reason': 'missing or duplicate complete training run'})
            continue
        a, b = pair
        keys = ['training_parent_hash', 'examples_processed', 'optimizer_updates']
        mismatches = [key for key in keys if a.get(key) is None or a.get(key) != b.get(key)]
        aa, ba = a.get('accounting', {}), b.get('accounting', {})
        if not aa.get('label_exposures') or aa.get('label_exposures') != ba.get('label_exposures'):
            mismatches.append('label_exposures')
        checks.append({'seed': seed, 'matched': not mismatches, 'mismatches': mismatches})
    return {'matched': all(row['matched'] for row in checks), 'by_optimizer_seed': checks}


def summarize(config, output_root):
    """Create JSON/CSV/LaTeX/PNG/PDF only from actual result artifacts.

    Smoke outputs remain diagnostic and never receive empirical H2 inference.
    Incomplete tasks, optimizer repeats, or capped budget axes remain Pending.
    """
    root = Path(output_root)
    destination = root / 'summary'
    destination.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest()
    previous = _read(destination / 'summary.json', {})
    if previous and previous.get('config_hash') != digest:
        raise ValueError('Incompatible aggregation configuration; use the original config or a new output root')
    recipes = _recipes(config)
    seeds = list(config['training']['seeds'])
    tasks = list(config['evaluation']['tasks'])
    categorical = [task for task in tasks if TASKS[task]['kind'] == 'classification']
    probes = config['evaluation'].get('probe_types', ['linear', 'nonlinear'])
    budgets = list(config['training']['label_budgets'])
    pretraining_budgets = [str(x) for x in config['training'].get('pretraining_budgets', [None])]
    smoke = config.get('claim_status') == 'smoke'
    bootstrap = int(config['evaluation'].get('bootstrap_samples', 2000))
    runs = []
    for file in sorted((root / 'models').glob('**/run.json')):
        run = _read(file, {})
        run['artifact'] = str(file.relative_to(root))
        runs.append(run)
    missing_jobs, task_rows, primary, comparisons, diagnostics, forecasts = [], [], [], [], [], []
    source_files = []
    for protocol in PROTOCOLS:
        directory = root / 'evaluations' / protocol
        results = _read(directory / 'probe_results.json', {})
        areas = _read(directory / 'transfer_areas.json', [])
        forecasts.extend(dict(row, protocol=protocol) for row in _read(directory / 'forecasts.json', []))
        for name in ('probe_results.json', 'transfer_areas.json', 'forecasts.json'):
            file = directory / name
            if file.exists():
                source_files.append({'path': str(file.relative_to(root)), 'sha256': hashlib.sha256(file.read_bytes()).hexdigest()})
        if results.get('status') == 'functional smoke' and not smoke:
            raise ValueError('Cannot promote functional-smoke evaluation artifacts into scientific aggregation')
        if any(row.get('mock') or row.get('dry_run') for row in results.get('rows', [])):
            raise ValueError('Mock or dry-run artifacts cannot enter scientific aggregation')
        for recipe in recipes:
            for seed in seeds:
                for prebudget in pretraining_budgets:
                    found = [r for r in runs if r.get('protocol') == protocol and r.get('recipe') == recipe
                             and r.get('seed') == seed and str(r.get('budget')) == prebudget and r.get('status') == 'complete']
                    if len(found) != 1:
                        missing_jobs.append({'stage': 'fit', 'protocol': protocol, 'recipe': recipe,
                                             'seed': seed, 'pretraining_budget': prebudget, 'status': 'Pending'})
                    for task in tasks:
                        for probe in probes:
                            for budget in budgets:
                                found_rows = [r for r in results.get('rows', []) if r.get('recipe') == recipe
                                    and r.get('optimizer_seed') == seed and str(r.get('pretraining_budget')) == prebudget
                                    and r.get('task') == task and r.get('probe') == probe and r.get('budget') == budget]
                                if len(found_rows) != 1 or found_rows[0].get('status') != 'complete':
                                    missing_jobs.append({'stage': 'evaluate', 'protocol': protocol, 'recipe': recipe,
                                        'seed': seed, 'pretraining_budget': prebudget, 'task': task, 'probe': probe,
                                        'budget': budget, 'status': 'Pending',
                                        'reason': found_rows[0].get('status') if found_rows else 'missing result'})
        task_maps = {}
        for recipe in recipes:
            for prebudget in pretraining_budgets:
                for probe in probes:
                    per_task = {}
                    for task in categorical:
                        subset = [r for r in areas if r.get('recipe') == recipe and str(r.get('pretraining_budget')) == prebudget
                                  and r.get('probe') == probe and r.get('task') == task and r.get('optimizer_seed') in seeds]
                        complete = (len(subset) == len(seeds) and {r['optimizer_seed'] for r in subset} == set(seeds)
                                    and all(r.get('status') == 'complete' and r.get('budgets') == budgets and r.get('structure_scores') for r in subset))
                        scores = {}
                        if complete:
                            structure_sets = [{str(key) for key in row['structure_scores']} for row in subset]
                            complete = all(values == structure_sets[0] for values in structure_sets)
                            if complete:
                                scores = {s: float(np.mean([r['structure_scores'][s] for r in subset])) for s in sorted(structure_sets[0])}
                                if any(not 0 <= value <= 1 for value in scores.values()):
                                    raise ValueError('Normalized balanced-accuracy area must lie in [0,1]')
                        status = 'functional smoke; excluded from scientific inference' if smoke else ('complete' if complete else 'Pending')
                        record = dict(protocol=protocol, recipe=recipe, pretraining_budget=prebudget, task=task, probe=probe,
                                      status=status, normalized_log_budget_area=float(np.mean(list(scores.values()))) if scores and not smoke else None,
                                      n_structures=len(scores), n_optimizer_seeds=len(subset), budgets=budgets,
                                      structure_scores=scores if not smoke else {})
                        task_rows.append(record)
                        if complete:
                            per_task[task] = scores
                            if not smoke:
                                task_maps[(recipe, prebudget, probe, task)] = scores
                        if smoke and scores:
                            diagnostics.append(dict(record, diagnostic_area=float(np.mean(list(scores.values()))), excluded_from_scientific_aggregation=True))
                    if len(per_task) == len(categorical) and categorical and not smoke:
                        sets = [set(x) for x in per_task.values()]
                        if all(values == sets[0] for values in sets):
                            score = {s: float(np.mean([per_task[task][s] for task in categorical])) for s in sorted(sets[0])}
                            uncertainty = paired_cluster_interval(list(score.values()), [0] * len(score), list(score), n_bootstrap=bootstrap)
                            primary.append(dict(protocol=protocol, recipe=recipe, pretraining_budget=prebudget, probe=probe,
                                status='complete', task_count=len(categorical), structure_scores=score,
                                area=float(np.mean(list(score.values()))), ci=uncertainty['ci'], n_structures=len(score)))
                            continue
                    primary.append(dict(protocol=protocol, recipe=recipe, pretraining_budget=prebudget, probe=probe,
                                        status='functional smoke; excluded' if smoke else 'Pending', area=None, ci=None, structure_scores={}))
        if not smoke:
            for hypothesis, left, right in _comparison_pairs(recipes):
                for prebudget in pretraining_budgets:
                    for probe in probes:
                        base = dict(protocol=protocol, hypothesis=hypothesis, left=left, right=right,
                                    pretraining_budget=prebudget, probe=probe)
                        selected = [[r for r in primary if r['protocol'] == protocol and r['recipe'] == recipe
                                     and r['probe'] == probe and r['pretraining_budget'] == prebudget and r['status'] == 'complete'] for recipe in (left, right)]
                        if all(selected):
                            result = _paired(selected[0][0]['structure_scores'], selected[1][0]['structure_scores'], bootstrap)
                        else:
                            result = {'status': 'Pending', 'estimate': None, 'ci': None, 'p_value': None}
                        exposure = _exposure_audit(runs, protocol, left, right, prebudget, seeds) if hypothesis == 'H2b' else None
                        comparisons.append(dict(base, task='primary_equal_task_area', exposure_audit=exposure,
                            matched_exposure_claim_eligible=(exposure['matched'] if exposure else None), **result))
                        for task in categorical:
                            a, b = task_maps.get((left, prebudget, probe, task)), task_maps.get((right, prebudget, probe, task))
                            result = _paired(a, b, bootstrap) if a and b else {'status': 'Pending', 'estimate': None, 'ci': None, 'p_value': None}
                            comparisons.append(dict(base, task=task, **result))
    secondary = [row for row in comparisons if row['task'] != 'primary_equal_task_area' and row.get('p_value') is not None]
    if secondary:
        for row, adjusted in zip(secondary, benjamini_hochberg([row['p_value'] for row in secondary])):
            row['fdr_bh_p'] = float(adjusted)
            row['multiplicity_family'] = 'all reported secondary task/recipe/probe/protocol contrasts'
    efficiency = []
    for run in runs:
        efficiency.append({key: run.get(key) for key in ('protocol', 'recipe', 'seed', 'budget', 'status', 'parameter_count',
                           'encoder_parameter_count', 'examples_processed', 'optimizer_updates', 'unique_recordings',
                           'elapsed_seconds', 'cpu_seconds', 'peak_process_rss_bytes', 'peak_cuda_memory_bytes', 'device', 'artifact')} |
                          {'accounting': run.get('accounting', {})})
    continuous_rows = []
    retention = []
    for protocol in PROTOCOLS:
        report = _read(root / 'evaluations' / protocol / 'probe_results.json', {})
        for row in report.get('rows', []):
            if row.get('kind') == 'regression' and not row.get('excluded_from_primary'):
                continuous_rows.append(dict(row, protocol=protocol))
    for file in sorted((root / 'models').glob('**/retention.json')):
        retention.append({'artifact': str(file.relative_to(root)), 'values': _read(file, [])})
    payload = {'schema_version': 'eegrepr-aggregate-v1', 'config_hash': digest,
               'status': 'functional smoke; scientific results pending' if smoke else config.get('claim_status', 'pilot'),
               'scientific_aggregation_enabled': not smoke, 'source_artifacts': source_files,
               'independent_unit': 'structure; optimizer repeats averaged before equal-task aggregation',
               'minimum_meaningful_gain': config['evaluation'].get('minimum_gain', 0.02),
               'primary': primary, 'task_areas': task_rows, 'comparisons': comparisons, 'forecasts': forecasts,
               'missing_jobs': missing_jobs, 'training_efficiency': efficiency, 'smoke_diagnostics': diagnostics,
               'continuous_endpoints': continuous_rows, 'old_task_retention': retention,
               'claims': {'H1': 'analytical identity established; circuit prediction frontier pending',
                          'H2a': 'pending scientific matched-budget comparison', 'H2b': 'pending scientific matched-exposure comparison'}}
    _write(destination / 'summary.json', json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + '\n')
    _csv(destination / 'task_areas.csv', task_rows, ['protocol', 'recipe', 'pretraining_budget', 'probe', 'task', 'status', 'normalized_log_budget_area', 'n_structures'])
    _csv(destination / 'primary.csv', primary, ['protocol', 'recipe', 'pretraining_budget', 'probe', 'status', 'area', 'n_structures'])
    _csv(destination / 'missing_jobs.csv', missing_jobs, ['stage', 'protocol', 'recipe', 'seed', 'pretraining_budget', 'task', 'probe', 'budget', 'status', 'reason'])
    _csv(destination / 'training_efficiency.csv', efficiency, ['protocol', 'recipe', 'seed', 'budget', 'status', 'parameter_count', 'examples_processed', 'optimizer_updates', 'unique_recordings', 'elapsed_seconds', 'cpu_seconds', 'peak_process_rss_bytes', 'peak_cuda_memory_bytes', 'device'])
    def escape(value):
        return str(value).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')
    latex = [r'\begin{tabular}{llllr}', r'Protocol & Recipe & Probe & Status & Area \\', r'\hline']
    for row in primary:
        value = 'Pending' if row['area'] is None else f"{row['area']:.4f}"
        latex.append(' & '.join([escape(row['protocol']), escape(row['recipe']), escape(row['probe']), escape(row['status']), value]) + r' \\')
    latex.append(r'\end{tabular}')
    _write(destination / 'transfer_table.tex', '\n'.join(latex) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    actual = [row for row in primary if row['area'] is not None]
    figure, axes = plt.subplots(figsize=(max(7, len(actual) * 0.45), 4.5))
    if actual:
        axes.scatter(range(len(actual)), [r['area'] for r in actual], color='#176b8a')
        for i, row in enumerate(actual):
            if row['ci']:
                axes.vlines(i, *row['ci'], color='#176b8a')
        axes.set_xticks(range(len(actual)), [f"{r['recipe']}\n{r['protocol']}\n{r['probe']}" for r in actual], rotation=35, ha='right')
        axes.set_ylim(0, 1)
        axes.set_ylabel('Equal-task normalized log-budget area')
        axes.set_title(f"{payload['status']}: circuit-weighted transfer")
    else:
        axes.axis('off')
        axes.text(0.5, 0.55, 'Scientific transfer results pending', ha='center', va='center', transform=axes.transAxes, fontsize=15)
        axes.text(0.5, 0.4, 'Functional smokes are excluded; missing or capped budget axes stay Pending.', ha='center', va='center', transform=axes.transAxes, wrap=True)
    figure.tight_layout()
    for suffix in ('png', 'pdf'):
        fd, temporary = tempfile.mkstemp(dir=destination, suffix=f'.{suffix}')
        os.close(fd)
        try:
            figure.savefig(temporary, dpi=160)
            os.replace(temporary, destination / f'transfer_summary.{suffix}')
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    plt.close(figure)
    return payload
