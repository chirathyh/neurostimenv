"""Atomic four-split v6 coordinator with explicit immutable v4 native reuse.

A per-trial record commits after each HDF5 file, allowing orphan recovery. Physics,
seed allocation and actual generation dependencies determine resume identity;
changing fitting/report code does not invalidate recordings. NEURON workers are
spawned processes, never threads, and each replays its complete physical prefix.
"""
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from pathlib import Path
import importlib.metadata
import importlib.util
import json
import multiprocessing
import os
import platform
import resource
import subprocess
import time
import numpy as np
from ...storage import atomic_json, file_hash, write_trial, output_lock
from ...forks import _read_fork, validate_fork_prefixes
from ..targets import canonical_hash
from ..artifacts import rng_seed
from .config import contexts

ROOT=Path(__file__).resolve().parents[4]


def cpu_topology():
    model='unavailable'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):model=line.split(':',1)[1].strip();break
    except OSError:pass
    return dict(cpu_model=model,logical_cpus=os.cpu_count(),architecture=platform.machine(),
                process_cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None)


def physics_source_files():
    paths=[ROOT/'experiments/eeg_representation/simulator.py',ROOT/'configs/env/ballnstick.yaml']
    for folder in ('env/models/neuron','setup/circuits/ballnstick'):
        paths.extend(sorted((ROOT/folder).glob('*.py')))
        paths.extend(sorted((ROOT/folder).glob('*.hoc')))
    return sorted(set(paths))


def generation_contract(c):
    files=physics_source_files()+[Path(__file__).with_name('simulator.py')]
    return dict(backend=c['backend'],simulation=deepcopy(c['simulation']),cutoff_ms=c['cutoff_ms'],
        stop_ms=c['stop_ms'],output_dt_ms=.0625,namespace=c['namespace'],
        regime_law=c['data']['regimes'],optional=c.get('optional',{}),
        physics_source={str(p.relative_to(ROOT)):file_hash(p) for p in files},
        versions={name:importlib.metadata.version(name) if importlib.util.find_spec(module) else None
                  for name,module in [('numpy','numpy'),('NEURON','neuron'),('LFPy','LFPy'),('scipy','scipy'),('mpi4py','mpi4py'),('lfpykit','lfpykit'),('omegaconf','omegaconf')]},
        python_version=platform.python_version(),
        future_seed_convention='regime-paired independent history/score/reference namespaces; no dt in RNG key')


def adapter(c):
    from ...schema import DEFAULTS
    out=deepcopy(DEFAULTS);out['simulation'].update(c['simulation'])
    out['simulation']['phase_grid_dt_ms']=c['simulation']['phase_grid_ms']
    out['simulation']['duration_s']=c['stop_ms']/1000-c['simulation']['burn_in_s']
    out['data']['backend']=c['backend'];out['data']['analytical_kind']=c['analytical']['kind']
    out['analytical']={k:v for k,v in c['analytical'].items() if k!='kind'}
    return out


def trial_spec(c,context,pool,future):
    ns=c['namespace'];sid=int(context['structure_id']);p=int(context['prefix'])
    return dict(trial_id=context['context_id']+f'-{pool}-{future:04d}',history_id=context['context_id'],
        structure_id=str(sid),structure_seed=sid,split=context['split'],regime=context['regime'],
        future_id=f'{pool}-{future:04d}',trial_seed=rng_seed(ns,'history-events',sid,p),
        phase_seed=rng_seed(ns,'history-phase',sid,p),envelope_seed=rng_seed(ns,'history-envelope',sid,p),
        future_seed=rng_seed(ns,'future-events',sid,p,pool,future),
        future_phase_seed=rng_seed(ns,'future-phase',sid,p,pool,future),
        future_burst_seed=rng_seed(ns,'future-envelope',sid,p,pool,future),future_start_ms=c['cutoff_ms'])


def row_path(directory,row):
    root=Path(row.get('source_root',directory)).resolve()
    path=(root/row['path']).resolve()
    if not path.is_relative_to(root):raise ValueError('Native record escapes declared source root')
    return path


def verify_row(directory,row):
    p=row_path(directory,row)
    if not p.is_file() or p.stat().st_size!=row['bytes'] or file_hash(p)!=row['sha256']:
        raise ValueError('Missing or altered native artifact: '+str(p))
    return p


def _record(directory,path,context,spec,trial,seconds,contract_hash):
    return dict(path=str(path.relative_to(directory)),sha256=file_hash(path),bytes=path.stat().st_size,
        trial_id=spec['trial_id'],context_id=context['context_id'],structure_id=context['structure_id'],
        split=context['split'],pool=spec['future_id'].split('-')[0],future_index=int(spec['future_id'].split('-')[1]),
        spec=spec,metadata=trial['metadata'],generation_seconds=seconds,generation_origin='v6_new',
        generation_contract_hash=contract_hash)


def _worker(args):
    c,directory,context,spec,contract_hash=args;directory=Path(directory);tid=spec['trial_id']
    path=directory/'native'/context['split']/context['context_id']/(spec['future_id']+'.h5')
    recordpath=directory/'records'/(tid+'.json')
    if recordpath.exists():
        row=json.loads(recordpath.read_text());verify_row(directory,row)
        if row.get('generation_contract_hash')!=contract_hash or row['spec']!=spec:raise ValueError('Trial resume identity mismatch')
        return row
    started=time.perf_counter()
    try:
        if path.exists():
            # HDF5 was atomically committed but coordinator/worker died before its record.
            import h5py
            with h5py.File(path,'r') as f:attrs=json.loads(f.attrs['metadata_json'])
            if attrs.get('spec')!=spec or attrs.get('generation_contract_hash')!=contract_hash:
                raise ValueError('Orphan native file belongs to another generation contract')
            trial=_read_fork(path);seconds=float(trial['metadata'].get('runtime_seconds',0))
        else:
            if c['backend']=='neuron':from .simulator import generate_trial
            elif c['backend']=='analytical' and c['claim_status']=='engineering':from ...analytical import generate_trial
            else:raise ValueError('No analytical fallback for a NEURON run')
            trial=generate_trial(adapter(c),spec)
            seconds=time.perf_counter()-started
            write_trial(path,trial,dict(schema_version='eegrepr-evaluation-v6.0',spec=spec,
                metadata=trial['metadata'],generation_contract_hash=contract_hash))
        row=_record(directory,path,context,spec,trial,seconds,contract_hash)
        row['worker_peak_rss_bytes']=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        atomic_json(recordpath,row)
        return row
    except BaseException as exc:
        atomic_json(directory/'failures'/(tid+'.json'),dict(status='failed',trial_id=tid,spec=spec,
            error=repr(exc),generation_contract_hash=contract_hash,created_unix=time.time()))
        raise


def import_reused(c,source,requested):
    if not requested:return [],None
    if source is None:raise ValueError('The 320 reused training/development episodes require an explicit native source directory')
    source=Path(source).resolve()
    if not (source/'native/manifest.json').exists() and (source/'generation/native/manifest.json').exists():source=source/'generation'
    if not (source/'native/manifest.json').is_file():raise ValueError('Reuse source lacks native/manifest.json')
    native=json.loads((source/'native/manifest.json').read_text())
    provenance=json.loads((source/'provenance.json').read_text())
    oldconfig=json.loads((source/'resolved_config.json').read_text())
    if c['simulation']['native_dt_ms']!=.0625:raise ValueError('Old coarse-dt recordings cannot be reused under a finer law')
    for key in ('native_dt_ms','burn_in_s','celsius','population_e','population_i','phase_grid_ms','external_synapses_per_cell'):
        if oldconfig['simulation'].get(key)!=c['simulation'].get(key):raise ValueError('Incompatible reused simulation field '+key)
    # Check physics files against the recorded implementation commit, not the global
    # analysis source hash. Failure to establish this identity fails closed.
    commit=provenance.get('commit');mismatches=[]
    for path in physics_source_files():
        relative=str(path.relative_to(ROOT))
        try:old=subprocess.check_output(['git','show',f'{commit}:{relative}'],cwd=ROOT,stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as exc:raise ValueError('Cannot verify historical physical source at '+str(commit)) from exc
        if old!=path.read_bytes():mismatches.append(relative)
    if mismatches:raise ValueError('Physical simulator implementation differs from imported generation: '+str(mismatches))
    wanted={r['context_id']:r for r in requested};rows=[]
    for record in native['trials']:
        if record['context_id'] not in wanted:continue
        ctx=wanted[record['context_id']]
        if ctx['split'] not in ('train','development') or record['split']!=ctx['split'] or record['pool']!='score' or record['future_index']!=0:
            raise ValueError('Reuse must retain the original one-future training/development assignment')
        meta=record['metadata']
        if (meta.get('native_dt_ms')!=.0625 or meta.get('total_duration_ms')!=c['stop_ms'] or
            meta.get('fork',{}).get('cutoff_ms')!=c['cutoff_ms'] or meta.get('sample_boundary_convention')!='(start,stop]'):
            raise ValueError('Reused native support/time convention differs')
        defaults=dict(diffusion=0.,kinetics_scale=1.,shared_fraction=1.,burst=False)
        if {**defaults,**record['spec']['regime']}!={**defaults,**ctx['regime']}:raise ValueError('Imported regime law differs')
        copy=deepcopy(record);copy.update(source_root=str(source),generation_origin='v4_reused',
            original_generation_provenance=provenance,original_generation_config_hash=native.get('config_hash'))
        verify_row(source,copy);rows.append(copy)
    if len(rows)!=len(wanted) or {r['context_id'] for r in rows}!=set(wanted):raise ValueError('Incomplete or duplicate registered reuse')
    return rows,dict(source_root=str(source),manifest_sha256=file_hash(source/'native/manifest.json'),
        original_generation_provenance=provenance,reused_episodes=len(rows),
        compatibility='physical code at recorded commit, simulation law, float64 native support and original IDs verified; analysis/export hashes may differ')


def execute_panel(c,directory,panel_contexts,resume=False,source=None):
    """Internal implementation; public entry points must check their execution gate."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    contract=generation_contract(c);contract_hash=canonical_hash(contract)
    identity=dict(generation_contract=contract,generation_contract_hash=contract_hash,
                  context_allocation=panel_contexts,allocation_hash=canonical_hash(panel_contexts))
    guard=directory/'generation_contract.json'
    with output_lock(directory):
        if guard.exists():
            if not resume:raise FileExistsError('Generation exists; use resume')
            if json.loads(guard.read_text())!=identity:raise ValueError('Generation physics/seed allocation changed; use a new root')
        else:atomic_json(guard,identity)
        imported,reuse=import_reused(c,source,[x for x in panel_contexts if x.get('reuse')])
        known={r['trial_id']:r for r in imported};expected={};tasks=[]
        for ctx in panel_contexts:
            if ctx.get('reuse'):continue
            for pool,count in [('score',ctx['score_futures']),('reference',ctx['reference_futures'])]:
                for future in range(count):
                    spec=trial_spec(c,ctx,pool,future);expected[spec['trial_id']]=spec
                    record=directory/'records'/(spec['trial_id']+'.json')
                    if record.exists():
                        row=json.loads(record.read_text());verify_row(directory,row)
                        if row.get('generation_contract_hash')!=contract_hash or row['spec']!=spec:raise ValueError('Recorded trial generation mismatch')
                        known[spec['trial_id']]=row
                    else:tasks.append((c,str(directory),ctx,spec,contract_hash))
        if reuse:atomic_json(directory/'reuse_manifest.json',reuse)
        started=time.perf_counter();resumed=len(known);audits={};prefix_paths={}
        def accept(row):
            path=verify_row(directory,row);cid=row['context_id']
            if cid in prefix_paths:
                audit=validate_fork_prefixes([_read_fork(prefix_paths[cid]),_read_fork(path)],c['cutoff_ms'])
                audits[cid]=dict(audit,verified_futures=audits.get(cid,{}).get('verified_futures',1)+1)
            else:prefix_paths[cid]=path;audits[cid]=dict(status='single_future',verified_futures=1)
            known[row['trial_id']]=row
        for row in list(known.values()):accept(row)
        planned=sum(x['score_futures']+x['reference_futures'] for x in panel_contexts)
        def save(status):
            rows=sorted(known.values(),key=lambda r:r['trial_id'])
            atomic_json(directory/'native/manifest.json',dict(schema_version='eegrepr-evaluation-v6.0',status=status,
                generation_contract_hash=contract_hash,allocation_hash=identity['allocation_hash'],trials=rows,
                interpretation='Composite import preserves each reused row original source root/provenance; new rows local to this directory'))
            atomic_json(directory/'native/prefix_validation.json',dict(contexts=audits,cutoff_ms=c['cutoff_ms']))
            atomic_json(directory/'progress.json',dict(status=status,completed=len(rows),planned=planned,
                generated_this_call=len(rows)-resumed,resumed_or_imported=resumed,workers=c['data']['workers'],
                updated_unix=time.time(),generation_contract_hash=contract_hash))
        save('running')
        workers=int(c['data']['workers'])
        if workers==1:
            for task in tasks:accept(_worker(task));save('running');print(f'V6 native {len(known)}/{planned}',flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
                pending={};task_iter=iter(tasks)
                for _ in range(min(len(tasks),workers*2)):
                    task=next(task_iter);pending[pool.submit(_worker,task)]=task
                while pending:
                    ready,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                    for done in ready:
                        pending.pop(done);accept(done.result())
                        task=next(task_iter,None)
                        if task is not None:pending[pool.submit(_worker,task)]=task
                    if ready:save('running');print(f'V6 native {len(known)}/{planned}',flush=True)
        if len(known)!=planned:raise ValueError('Generation did not produce the complete registered allocation')
        save('complete')
        rows=list(known.values());new=[r for r in rows if r['generation_origin']=='v6_new']
        result=dict(status='complete',complete=True,continuations=len(rows),contexts=len(panel_contexts),
            new_episodes=len(new),reused_episodes=len(imported),generated_this_call=len(rows)-resumed,
            resumed_this_call=resumed,workers=workers,command_wall_seconds=time.perf_counter()-started,
            summed_generation_seconds=sum(r['generation_seconds'] for r in new),
            mean_continuation_seconds=float(np.mean([r['generation_seconds'] for r in new])) if new else None,
            native_bytes=sum(r['bytes'] for r in rows),new_native_bytes=sum(r['bytes'] for r in new),
            peak_process_rss_bytes=max([r.get('worker_peak_rss_bytes',0) for r in new]+[0]),
            generation_contract_hash=contract_hash,allocation_hash=identity['allocation_hash'],host=platform.node(),
            native_manifest_sha256=file_hash(directory/'native/manifest.json'),
            cutoff_ms=c['cutoff_ms'],episode_stop_ms=c['stop_ms'],integration_dt_ms=c['simulation']['native_dt_ms'],
            cpu_topology=cpu_topology(),output_dt_ms=.0625,full_prefix_replay=True,state_caching=False,
            cpu_note='Independent serial NEURON processes; GPU does not accelerate generation')
        atomic_json(directory/'timing_audit.json',result);atomic_json(directory/'generation_result.json',result)
        return result


def generate(c,root,resume=False,*,execute=False,timing=False,source=None,split=None):
    from .planning import require_execution
    require_execution(c,root,execute,timing=timing,split=split)
    if c['claim_status']=='prospective_confirmation':
        planned_source=json.loads((Path(root)/'plan.json').read_text()).get('native_source')
        if source is None:source=planned_source
        elif planned_source is None or Path(source).resolve()!=Path(planned_source).resolve():raise ValueError('Native reuse source differs from measured plan')
    selected=contexts(c)
    if split is not None:
        splits={split} if isinstance(split,str) else set(split)
        selected=[r for r in selected if r['split'] in splits]
        if not selected:raise ValueError('No registered contexts for requested split')
    # Stable per-split roots allow train/dev fitting before evaluation unblinding.
    directory=Path(root)/'generation';results=[]
    for split_name in dict.fromkeys(r['split'] for r in selected):
        subset=[r for r in selected if r['split']==split_name]
        results.append(execute_panel(c,directory/split_name,subset,resume,source))
        merged=[]
        for subdir in sorted(directory.iterdir()):
            manifest=subdir/'native/manifest.json'
            if subdir.is_dir() and manifest.is_file():
                for row in json.loads(manifest.read_text())['trials']:
                    merged.append({**row,'source_root':row.get('source_root',str(subdir.resolve()))})
        all_planned=sum(r['score_futures']+r['reference_futures'] for r in contexts(c))
        atomic_json(directory/'native/manifest.json',dict(schema_version='eegrepr-evaluation-v6.0',
            status='complete' if len(merged)==all_planned else 'partial',
            trials=sorted(merged,key=lambda r:r['trial_id']),planned=all_planned,completed=len(merged),
            source_policy='Each row resolves against its immutable original source_root; no implicit provenance relabeling'))
    result=dict(status='complete',complete=True,cpu_topology=cpu_topology(),requested_splits=list(dict.fromkeys(r['split'] for r in selected)),
        continuations=sum(v['continuations'] for v in results),results=results,
        summed_generation_seconds=sum(v['summed_generation_seconds'] for v in results),
        native_bytes=sum(v['native_bytes'] for v in results),workers=c['data']['workers'],
        mean_continuation_seconds=sum(v['summed_generation_seconds'] for v in results)/max(1,sum(v['new_episodes'] for v in results)),
        peak_process_rss_bytes=max(v['peak_process_rss_bytes'] for v in results),
        new_episodes=sum(v['new_episodes'] for v in results),reused_episodes=sum(v['reused_episodes'] for v in results))
    atomic_json(Path(root)/('generation_result.json' if split is None else 'generation_result_'+'-'.join(sorted(splits))+'.json'),result)
    atomic_json(directory/'timing_audit.json',result)
    return result
