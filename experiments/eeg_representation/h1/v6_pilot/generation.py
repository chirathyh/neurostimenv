"""Eight-process native pilot generation with strict provenance and a hard cap.

Every simulation enters through _worker and is charged by budget.begin. Imported
recordings retain their original provenance. The compatibility test may reject a
legacy continuation, but never silently changes its physical history or blends it
with the separately registered fresh fallback population.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import json
import importlib.metadata
import multiprocessing
import os
import platform
import resource
import shutil
import time
import traceback
import numpy as np
from ...forks import _read_fork
from ...storage import atomic_json, file_hash, write_trial
from ..dataset import adapter_config
from ..targets import canonical_hash
from ..artifacts import rng_seed, ROOT
from ..v6.generation import cpu_topology
from . import budget, design

_WORKER_CPU=None
_WORKER_BARRIER=None
_THREAD_LIMITER=None


def _read(path):return json.loads(Path(path).read_text())


def adapter_contract():
    """Current capture and simulator identity, independent of imported provenance."""
    from . import state
    from ..v6.generation import physics_source_files
    files=physics_source_files()+[Path(__file__).with_name(x) for x in ('state.py','generation.py','design.py','budget.py')]
    files.append(ROOT/'experiments/eeg_representation/h1/v6/simulator.py')
    return dict(state_schema=state.SCHEMA_VERSION,source={str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(files))},
        versions={name:importlib.metadata.version(name) for name in ('numpy','scipy','NEURON','LFPy')},
        policy='Changed capture schema/code or physical runtime fails before new native calls. Historical native reuse retains original provenance and is independently prefix-qualified.')

def _frozen(root):
    record=_read(Path(root)/'design.json');check=dict(record);digest=check.pop('design_hash')
    if canonical_hash(check)!=digest:raise ValueError('Pilot design checksum changed')
    return record


def preflight(config,root):
    """Read-only hardware checks; never silently downshift requested workers."""
    if int(config.get('workers',8))!=8 or int(config.get('threads_per_worker',1))!=1:
        raise ValueError('This pilot requires eight independent NEURON processes with one CPU thread each')
    affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else list(range(os.cpu_count() or 0))
    mem={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,value=line.split(':',1);mem[key]=int(value.strip().split()[0])*1024
    except OSError:pass
    cgroup={}
    try:
        limit=(Path('/sys/fs/cgroup')/'memory.max').read_text().strip()
        used=int((Path('/sys/fs/cgroup')/'memory.current').read_text().strip())
        if limit!='max':
            cgroup['memory_limit_bytes']=int(limit);cgroup['memory_current_bytes']=used
            mem['MemAvailable']=min(mem.get('MemAvailable',0),max(0,int(limit)-used))
        quota,period=(Path('/sys/fs/cgroup')/'cpu.max').read_text().split()
        if quota!='max':cgroup['cpu_quota_equivalents']=int(quota)/int(period)
    except (OSError,ValueError):pass
    current=Path(root).resolve()
    while not current.exists():current=current.parent
    free=shutil.disk_usage(current).free
    minimum_ram=int(config.get('minimum_available_ram_bytes',7*1024**3))
    minimum_free=int(config.get('minimum_free_bytes',24*1024**3))
    blockers=[]
    if len(affinity)<8:blockers.append(f'CPU affinity exposes {len(affinity)} CPUs; eight required')
    if cgroup.get('cpu_quota_equivalents',8)<8:blockers.append(f'Cgroup CPU quota permits {cgroup["cpu_quota_equivalents"]} CPU equivalents; eight requested')
    for variable in ('OMPI_COMM_WORLD_SIZE','PMI_SIZE','MV2_COMM_WORLD_SIZE'):
        if int(os.environ.get(variable,'1'))>1:blockers.append('Run one coordinator, not a multi-rank MPI launch: '+variable+'='+os.environ[variable])
    if mem.get('MemAvailable',0)<minimum_ram:blockers.append(f'Available RAM {mem.get("MemAvailable")} bytes is below {minimum_ram}')
    if free<minimum_free:blockers.append(f'Free storage {free} bytes is below {minimum_free}')
    return dict(status='blocked' if blockers else 'complete',passed=not blockers,blockers=blockers,
        requested_workers=8,threads_per_worker=1,cpu_affinity=affinity,available_ram_bytes=mem.get('MemAvailable'),
        free_bytes=free,minimum_ram_bytes=minimum_ram,minimum_free_bytes=minimum_free,
        host=platform.node(),cpu_topology=cpu_topology(),cgroup_limits=cgroup,checked_unix=time.time(),
        interpretation='Eight native processes share this host; CUDA fitting is separate; no automatic worker reduction')


def prepare(config,root,source):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    budget.initialize(root,config.get('max_new_invocations',160))
    result=design.register(config,root,source)
    atomic_json(root/'preflight_generation.json',preflight(config,root))
    return result


def verify_prefix(reference,candidate,cutoff_ms):
    """Exact shared prefix audit allowing a cutoff-only replay's shorter tail."""
    cutoff=float(cutoff_ms);a=reference['metadata'];b=candidate['metadata']
    if a.get('structure_hash')!=b.get('structure_hash'):raise ValueError('Physical structure differs from original')
    original=a.get('fork',{});other=b.get('fork',{})
    for key in ('cutoff_ms','physical_state_hash','external_history_hash','pending_recurrent_hash','pending_external_hash','prefix_spike_hash'):
        if key not in original or original[key]!=other.get(key):raise ValueError('Legacy prefix/state incompatibility: '+key)
    am=np.asarray(reference['time_ms'])<=cutoff;bm=np.asarray(candidate['time_ms'])<=cutoff
    for key in ('time_ms','eeg_v','dipole_na_um'):
        if not np.array_equal(np.asarray(reference[key])[am],np.asarray(candidate[key])[bm]):
            raise ValueError('Exact original prefix differs: '+key)
    for key in ('cutoff_physical_state','pending_recurrent_events','pending_external_events'):
        if not np.array_equal(reference['privileged'][key],candidate['privileged'][key]):
            raise ValueError('Physical cutoff array differs: '+key)
    for key in ('phase_rad','burst_depth'):
        timekey='phase_time_ms' if key=='phase_rad' else 'burst_times_ms'
        if key in reference['privileged']:
            at=np.asarray(reference['privileged'][timekey]);bt=np.asarray(candidate['privileged'][timekey])
            if not np.array_equal(at[at<=cutoff],bt[bt<=cutoff]) or not np.array_equal(
                np.asarray(reference['privileged'][key])[at<=cutoff],np.asarray(candidate['privileged'][key])[bt<=cutoff]):
                raise ValueError('Drive history differs: '+key)
    return dict(passed=True,cutoff_ms=cutoff,prefix_samples=int(am.sum()),
        checks=['structure','all original dynamic state','caused pending events','external history','spike prefix','exact EEG/dipole/time prefix','phase/envelope prefix'])


def _native_path(row):
    root=Path(row['source_root']).resolve();path=(root/row['path']).resolve()
    if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']:
        raise ValueError('Original native hash/path mismatch: '+str(path))
    return path


def _initialize_worker(cpus,barrier):
    global _WORKER_CPU,_WORKER_BARRIER,_THREAD_LIMITER
    _WORKER_CPU=cpus.get(timeout=30);_WORKER_BARRIER=barrier
    if hasattr(os,'sched_setaffinity'):os.sched_setaffinity(0,{_WORKER_CPU})
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
        os.environ[name]='1'
    try:
        from threadpoolctl import threadpool_limits
        _THREAD_LIMITER=threadpool_limits(limits=1)
    except ImportError:pass


def _record_path(root,stage,key):return Path(root)/'generation_records'/stage/(key+'.json')


def _task_result(task,trial,native_path,state_path,attempt,runtime,wall_start,validation,*,recovered=False):
    compatibility=None
    if task.get('reference'):
        try:compatibility=verify_prefix(_read_fork(_native_path(task['reference'])),trial,task['spec']['future_start_ms'])
        except ValueError as exc:compatibility=dict(passed=False,error=str(exc))
    return dict(status='complete',task_hash=canonical_hash(task),key=task['key'],stage=task['stage'],spec=task['spec'],
        context_id=task['spec']['history_id'],structure_id=int(task['spec']['structure_seed']),
        native_file=str(native_path),native_sha256=file_hash(native_path),native_bytes=native_path.stat().st_size,
        state_directory=str(state_path),state_validation=validation,prefix_compatibility=compatibility,
        attempt_id=attempt,pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        thread_environment={n:os.environ.get(n) for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        generation_seconds=runtime,started_unix=wall_start,finished_unix=time.time(),
        peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        recovered_complete_orphan=recovered,generation_method='full-prefix NEURON rerun; no checkpoint restoration')


def _close_attempt_if_needed(root,attempt,seconds,recordpath):
    if not (Path(root)/'invocation_ledger'/f'result-{attempt}.json').exists():
        budget.finish(root,attempt,status='complete',seconds=seconds,details=dict(record=str(recordpath),verified_completed_artifact=True))


def _completed_task(task):
    """Verify or recover durable output without allocating another invocation."""
    root=Path(task['root']);stage=task['stage'];spec=task['spec'];key=task['key']
    recordpath=_record_path(root,stage,key)
    from .state import load_state,validate
    if task.get('adapter_contract') is not None and task['adapter_contract']!=adapter_contract():
        raise ValueError('Native task capture/simulator identity changed')
    if recordpath.exists():
        result=_read(recordpath)
        if result['task_hash']!=canonical_hash(task):raise ValueError('Completed native task identity changed')
        if file_hash(result['native_file'])!=result['native_sha256'] or Path(result['native_file']).stat().st_size!=result['native_bytes']:
            raise ValueError('Completed native task changed')
        if validate(load_state(result['state_directory']))!=result['state_validation']:
            raise ValueError('Completed native state binding changed')
        if task.get('reference'):_native_path(task['reference'])
        _close_attempt_if_needed(root,result['attempt_id'],result['generation_seconds'],recordpath)
        return result
    destination=root/'new_native'/stage/key
    # Recover a durably written trial AND complete state without running NEURON.
    # Incomplete attempt folders remain preserved and charged on a new attempt.
    for native_path in sorted(destination.glob('attempt-*/trial.h5')):
        import h5py
        with h5py.File(native_path,'r') as handle:attrs=json.loads(handle.attrs['metadata_json'])
        if attrs.get('pilot_task_hash')!=canonical_hash(task):raise ValueError('Orphan native task contract differs')
        state_path=native_path.parent/'complete_state'
        if not (state_path/'schema.json').exists():continue
        payload=load_state(state_path);trial=_read_fork(native_path);attempt=attrs['invocation_attempt']
        record=_task_result(task,trial,native_path,state_path,attempt,attrs['pilot_runtime_seconds'],attrs['pilot_started_unix'],validate(payload),recovered=True)
        atomic_json(recordpath,record);_close_attempt_if_needed(root,attempt,record['generation_seconds'],recordpath)
        return record
    return None


def _worker(task):
    """Only native invocation site. Any new/retry call consumes one ledger entry."""
    from .state import generate_with_state,export_state,validate
    existing=_completed_task(task)
    if existing is not None:return existing
    root=Path(task['root']);stage=task['stage'];spec=task['spec'];key=task['key']
    recordpath=_record_path(root,stage,key);destination=root/'new_native'/stage/key
    if task.get('concurrency_barrier'):_WORKER_BARRIER.wait(timeout=180)
    started=time.perf_counter();wall_start=time.time()
    attempt=budget.begin(root,stage,dict(key=key,spec=spec,task_hash=canonical_hash(task)))
    try:
        trial,payload=generate_with_state(task['legacy_config'],spec)
        runtime=time.perf_counter()-started;validation=validate(payload)
        attempt_directory=destination/('attempt-'+attempt);native_path=attempt_directory/'trial.h5'
        write_trial(native_path,trial,dict(schema_version='eegrepr-v6-reporting-pilot-native-1',spec=spec,metadata=trial['metadata'],
            original_physics_config_hash=task['original_config_hash'],pilot_design_hash=task['design_hash'],invocation_attempt=attempt,
            pilot_task_hash=canonical_hash(task),pilot_runtime_seconds=runtime,pilot_started_unix=wall_start))
        state_path=attempt_directory/'complete_state';export_state(state_path,payload)
        result=_task_result(task,trial,native_path,state_path,attempt,runtime,wall_start,validation)
        atomic_json(recordpath,result)
        _close_attempt_if_needed(root,attempt,time.perf_counter()-started,recordpath)
        return result
    except BaseException as exc:
        if not (root/'invocation_ledger'/f'result-{attempt}.json').exists():
            budget.finish(root,attempt,status='failed',seconds=time.perf_counter()-started,details=dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


def _legacy_config(plan,cutoff_only=False):
    config=adapter_config(plan['source']['original_configuration'])
    if cutoff_only:
        # Legacy generator requires an interior split; capture is exactly at cutoff.
        # One additional 1-ms tail is excluded from every state-only input.
        cutoff=plan['selection']['state_cohort'][0]['cutoff_ms']
        config['simulation']['duration_s']=(cutoff+1.)/1000-config['simulation']['burn_in_s']
    return config


def _task(root,plan,stage,context,spec=None,cutoff_only=False,key=None,reference=True):
    spec=deepcopy(context['representative']['spec'] if spec is None else spec)
    spec['context_id']=spec['history_id']
    return dict(root=str(Path(root).resolve()),stage=stage,key=key or spec['trial_id'],spec=spec,
        legacy_config=_legacy_config(plan,cutoff_only),original_config_hash=canonical_hash(plan['source']['original_configuration']),
        design_hash=plan['design_hash'],reference=deepcopy(context['representative']) if reference else None,adapter_contract=adapter_contract())


def _run(config,root,stage,tasks,barrier=False):
    root=Path(root)
    pending=[];rows=[]
    for task in tasks:
        complete=_completed_task(task)
        if complete is not None:rows.append(complete)
        else:pending.append(task)
    budget.require_capacity(root,len(pending),stage)
    if barrier and pending and len(pending)!=8:
        # A partly failed gate requires a new eight-way measured check, while old
        # attempts remain charged. It never claims partial resume proved concurrency.
        raise RuntimeError('Concurrency check has partial artifacts. Use concurrency_check again, which registers a full eight-call retry, or inspect the failed gate.')
    started=time.perf_counter()
    if pending:
        hardware=preflight(config,root)
        if not hardware['passed']:raise RuntimeError('Eight-worker preflight blocked: '+'; '.join(hardware['blockers']))
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[name]='1'
        ctx=multiprocessing.get_context('spawn');cpus=ctx.Queue()
        for cpu in hardware['cpu_affinity'][:8]:cpus.put(cpu)
        sync=ctx.Barrier(8) if barrier else None
        with ProcessPoolExecutor(max_workers=8,mp_context=ctx,initializer=_initialize_worker,initargs=(cpus,sync)) as pool:
            futures={pool.submit(_worker,task):task for task in pending}
            for done in as_completed(futures):
                rows.append(done.result())
                atomic_json(root/'progress'/f'{stage}.json',dict(status='running',completed=len(rows),planned=len(tasks),ledger=budget.summary(root),updated_unix=time.time()))
    rows.sort(key=lambda r:r['key'])
    result=dict(status='complete',stage=stage,workers=8,threads_per_worker=1,results=rows,
        command_wall_seconds=time.perf_counter()-started,worker_seconds=sum(r['generation_seconds'] for r in rows),
        generated_this_call=len(pending),reused_completed_calls=len(rows)-len(pending),ledger=budget.summary(root))
    atomic_json(root/'progress'/f'{stage}.json',result)
    return result


def _validate_cached_run(result,tasks):
    """A complete marker cannot hide missing, changed or incompatible artifacts."""
    if result.get('status')!='complete' or len(result.get('results',[]))!=len(tasks):
        raise ValueError('Cached native allocation is incomplete')
    current=[]
    for task in tasks:
        row=_completed_task(task)
        if row is None:raise ValueError('Cached complete allocation has missing native task: '+task['key'])
        current.append(row)
    if canonical_hash(sorted(current,key=lambda r:r['key']))!=canonical_hash(sorted(result['results'],key=lambda r:r['key'])):
        raise ValueError('Cached native allocation differs from verified task records')
    return result


def concurrency_check(config,root,resume=False):
    root=Path(root);plan=_frozen(root);marker=root/'concurrency_check.json'
    if marker.exists():
        result=_read(marker)
        if result['design_hash']!=plan['design_hash'] or result['host']!=platform.node():raise ValueError('Concurrency gate belongs to another design/host')
        if not resume:raise FileExistsError('Concurrency check exists; use resume')
        tasks=[_task(root,plan,result['stage'],x) for x in plan['selection']['concurrency_contexts']]
        for task in tasks:task['concurrency_barrier']=True
        _validate_cached_run(result,tasks)
        return result
    # Distinct attempts after a failed gate keep every previously charged call.
    count=len(list((root/'generation_records').glob('concurrency-*')))
    stage=f'concurrency-{count:02d}'
    tasks=[_task(root,plan,stage,x) for x in plan['selection']['concurrency_contexts']]
    for task in tasks:task['concurrency_barrier']=True
    result=_run(config,root,stage,tasks,barrier=True)
    rows=result['results'];overlap=min(r['finished_unix'] for r in rows)-max(r['started_unix'] for r in rows)
    unique_pids=len({r['pid'] for r in rows});unique_cpus=len({tuple(r['cpu_affinity'] or []) for r in rows})
    if unique_pids!=8 or unique_cpus!=8 or overlap<=0:raise RuntimeError('Actual eight-process concurrent native overlap was not established')
    if any(any(v!='1' for v in r['thread_environment'].values()) for r in rows):raise RuntimeError('Native worker exceeded declared thread count')
    rss=max(r['peak_process_rss_bytes'] for r in rows);hardware=preflight(config,root)
    required_ram=int(8*rss*1.5+1024**3)
    if hardware['available_ram_bytes']<required_ram:raise RuntimeError(f'Measured worker RSS requires {required_ram} available bytes with reserve; host has {hardware["available_ram_bytes"]}')
    result.update(design_hash=plan['design_hash'],host=platform.node(),actual_worker_processes=unique_pids,
        actual_pinned_cpus=unique_cpus,simultaneous_native_overlap_seconds=overlap,measured_peak_worker_rss_bytes=rss,
        required_available_ram_bytes=required_ram,hardware=hardware,
        mean_full_episode_seconds=float(np.mean([r['generation_seconds'] for r in rows])),
        native_bytes_per_full_episode=float(np.mean([r['native_bytes'] for r in rows])))
    result['h2_legacy_compatible']=all(r['prefix_compatibility']['passed'] for r in rows if r['context_id'] in {x['context_id'] for x in plan['selection']['h2_contexts']})
    atomic_json(marker,result)
    return result


def _require_concurrency(config,root):
    result=concurrency_check(config,root,resume=True)
    if result.get('status')!='complete' or result.get('actual_worker_processes')!=8 or result['host']!=platform.node():
        raise RuntimeError('A successful measured eight-process check on this host is required')
    return result


def replay_controls(config,root,resume=False):
    root=Path(root);plan=_frozen(root);gate=_require_concurrency(config,root);tasks=[]
    from .state import load_state,validate_future_invariance
    marker=root/'replay_controls.json'
    cached=_read(marker) if marker.exists() else None
    if cached is not None and not resume:raise FileExistsError('Replay controls exist; use resume')
    # Replay an already recorded distinct future. This tests exact forcing and
    # future-invariant complete inputs using two actual native simulator calls.
    for context in plan['selection']['h2_contexts']:
        match=[r for r in plan['historical_rows'] if r['context_id']==context['context_id'] and r['pool']=='score' and r['future_index']==1]
        if len(match)!=1:raise ValueError('Registered replay future is absent')
        copy=deepcopy(context);copy['representative']=match[0];tasks.append(_task(root,plan,'exact-forcing-replay',copy))
    result=_validate_cached_run(cached,tasks) if cached is not None else _run(config,root,'exact-forcing-replay',tasks)
    audits=[]
    for row,task in zip(result['results'],sorted(tasks,key=lambda x:x['key'])):
        original=_read_fork(_native_path(task['reference']));new=_read_fork(row['native_file'])
        exact=all(np.array_equal(original[k],new[k]) for k in ('time_ms','eeg_v','dipole_na_um'))
        sibling=next(r for r in gate['results'] if r['context_id']==row['context_id'])
        invariant=validate_future_invariance(load_state(sibling['state_directory']),load_state(row['state_directory']))
        audits.append(dict(context_id=row['context_id'],original_trial_id=task['spec']['trial_id'],replica_path=row['native_file'],full_trace_identical=exact,
            eeg_mse_v2=float(np.mean((np.asarray(original['eeg_v'])-np.asarray(new['eeg_v']))**2)),complete_state_future_invariance=invariant,
            original_native_sha256=task['reference']['sha256'],replay_native_sha256=row['native_sha256']))
    if cached is not None:
        if cached.get('audits')!=audits or cached.get('all_exact')!=all(r['full_trace_identical'] for r in audits):
            raise ValueError('Cached replay audit no longer matches native/state artifacts')
        return cached
    result['audits']=audits;result['all_exact']=all(r['full_trace_identical'] for r in audits)
    result['interpretation']='Measured privileged full-prefix replay on two registered original futures; not checkpoint restoration or an EEG predictor. Theory zero is separate.'
    atomic_json(marker,result)
    return result


def recover_states(config,root,resume=False):
    root=Path(root);plan=_frozen(root);gate=_require_concurrency(config,root)
    replay=replay_controls(config,root,resume=True)
    from .state import load_state,validate,from_native
    matched=[];tasks=[];gate_by={r['context_id']:r for r in gate['results']}
    for context in plan['selection']['state_cohort']:
        cid=context['context_id'];record=gate_by.get(cid)
        if record and record['prefix_compatibility']['passed']:
            matched.append(dict(context_id=cid,structure_id=context['structure_id'],split=context['split'],historical_split=context['historical_split'],
                state_directory=record['state_directory'],source='captured concurrency replay',source_trial=context['representative'],prefix_compatibility=record['prefix_compatibility']))
            continue
        existing=from_native(_native_path(context['representative']))
        if isinstance(existing,dict) and existing.get('status')=='unavailable':existing=None
        if existing is not None:
            # Existing native export is reused only through a complete-schema validator.
            from .state import export_state
            validate(existing);dest=root/'state'/'imported'/cid;export_state(dest,existing)
            matched.append(dict(context_id=cid,structure_id=context['structure_id'],split=context['split'],historical_split=context['historical_split'],state_directory=str(dest),source='verified existing complete export',source_trial=context['representative']))
        else:tasks.append(_task(root,plan,'cutoff-state-recovery',context,cutoff_only=True,key=cid))
    recovered=_run(config,root,'cutoff-state-recovery',tasks)
    contexts={x['context_id']:x for x in plan['selection']['state_cohort']}
    for row in recovered['results']:
        ctx=contexts[row['context_id']]
        if not row['prefix_compatibility']['passed']:
            raise RuntimeError('Complete-state recovery cannot be bound to original targets: '+row['context_id']+'; '+row['prefix_compatibility']['error'])
        matched.append(dict(context_id=row['context_id'],structure_id=ctx['structure_id'],split=ctx['split'],historical_split=ctx['historical_split'],
            state_directory=row['state_directory'],source='cutoff recovery plus excluded 1-ms engineering tail',source_trial=ctx['representative'],prefix_compatibility=row['prefix_compatibility']))
    if len(matched)!=56 or len({r['context_id'] for r in matched})!=56:raise ValueError('Incomplete matched state cohort')
    for row in matched:validate(load_state(row['state_directory']))
    result=dict(status='complete',contexts=sorted(matched,key=lambda r:(r['split'],r['context_id'])),
        context_counts=plan['selection']['state_counts'],circuit_ids=plan['selection']['state_circuits'],
        design_hash=plan['design_hash'],ledger=budget.summary(root),future_invariance=replay['audits'],
        target_binding='Exact original cutoff, history, structure, dynamical states, pending events and EEG prefix; calibration IDs separate from development/evaluation',
        input_policy='No unknown future external event times, seeds, RNG state or future path enters predictor inputs')
    atomic_json(root/'state/manifest.json',result)
    return result


def _new_row(record,context,pool,index):
    trial=_read_fork(record['native_file']);path=Path(record['native_file'])
    return dict(path=str(path.name),source_root=str(path.parent),sha256=record['native_sha256'],bytes=record['native_bytes'],
        trial_id=record['spec']['trial_id'],context_id=context['context_id'],structure_id=context['structure_id'],split='evaluation',
        pool=pool,future_index=index,spec=record['spec'],metadata=trial['metadata'],generation_seconds=record['generation_seconds'],
        generation_origin='pilot_new_full_prefix',invocation_attempt=record['attempt_id'])


def _fallback_allocation(root,plan):
    registration=plan['fallback_registration'];sid=registration['circuit_id'];regime=registration['regime'];cid=f's{sid}-p0000-{regime["name"]}'
    context=dict(context_id=cid,structure_id=sid,regime=regime,cutoff_ms=6000.,split='evaluation',score_futures=32,reference_futures=32)
    spec=dict(structure_seed=sid,structure_id=str(sid),history_id=cid,split='evaluation',regime=deepcopy(regime),future_start_ms=6000.,
        trial_seed=rng_seed(registration['namespace'],'history-events',sid,0),phase_seed=rng_seed(registration['namespace'],'history-phase',sid,0),envelope_seed=rng_seed(registration['namespace'],'history-envelope',sid,0))
    context['representative']={'spec':spec};contexts=[context];tasks=[]
    for pool in ('score','reference'):
        for future in range(32):
            s={**deepcopy(spec),'future_id':f'{pool}-{future:04d}','trial_id':cid+f'-{pool}-{future:04d}'}
            for field,kind in [('future_seed','future-events'),('future_phase_seed','future-phase'),('future_burst_seed','future-envelope')]:
                s[field]=rng_seed(registration['namespace'],kind,sid,0,pool,future)
            tasks.append(_task(root,plan,'h2-fresh-fallback',context,s,reference=False))
    return contexts,tasks


def extend_h2(config,root,resume=False,*,state_model_gate=None):
    root=Path(root);plan=_frozen(root);gate=_require_concurrency(config,root)
    # The caller supplies an actual fitted state-model completion record. No H2
    # futures are spent merely because state export succeeded.
    if state_model_gate is None:raise RuntimeError('A measured complete-state training/development/evaluation gate must precede H2 generation')
    state_gate=_read(state_model_gate) if isinstance(state_model_gate,(str,Path)) else state_model_gate
    if state_gate.get('status')!='complete' or not state_gate.get('state_dependent_candidate_fitted') or not state_gate.get('held_out_evaluated'):
        raise RuntimeError('Complete-state predictor engineering completion gate failed')
    marker=root/'h2/generation.json'
    cached=_read(marker) if marker.exists() else None
    if cached is not None and not resume:raise FileExistsError('H2 generation already exists; use resume')
    compatible=gate['h2_legacy_compatible']
    if compatible:
        contexts=plan['selection']['h2_contexts'];byid={x['context_id']:x for x in contexts}
        specs=plan['h2_additional_specs'];tasks=[_task(root,plan,'h2-extension',byid[s['history_id']],s) for s in specs]
        run=_validate_cached_run(cached['generation'],tasks) if cached is not None else _run(config,root,'h2-extension',tasks)
        if any(not r['prefix_compatibility']['passed'] for r in run['results']):
            raise RuntimeError('Unexpected post-gate prefix mismatch; incompatible H2 futures retained separately and NOT merged. Investigate before any fallback capacity decision.')
        selected={x['context_id'] for x in contexts}
        rows=[deepcopy(r) for r in plan['historical_rows'] if r['context_id'] in selected]
        for record in run['results']:
            idx=int(record['spec']['future_id'].split('-')[-1]);rows.append(_new_row(record,byid[record['context_id']],'score',idx))
        population='two registered original contexts extended from M8 to M32, with original disjoint L32'
    else:
        if not config.get('allow_registered_fallback',True):raise RuntimeError('Legacy H2 prefix compatibility failed and registered fresh fallback is disabled')
        contexts,tasks=_fallback_allocation(root,plan);context=contexts[0]
        run=_validate_cached_run(cached['generation'],tasks) if cached is not None else _run(config,root,'h2-fresh-fallback',tasks)
        rows=[];first=None
        from .state import load_state,validate_future_invariance
        for record in run['results']:
            trial=_read_fork(record['native_file'])
            if first is not None:
                verify_prefix(_read_fork(first['native_file']),trial,6000.)
                validate_future_invariance(load_state(first['state_directory']),load_state(record['state_directory']))
            else:first=record
            pool,index=record['spec']['future_id'].split('-');rows.append(_new_row(record,context,pool,int(index)))
        population='one separately registered fresh context M32/L32; narrower population; no legacy futures combined'
    design.verify_pool_seeds(rows,[])
    for context in contexts:
        selected=[r for r in rows if r['context_id']==context['context_id']]
        if {p:sum(r['pool']==p for r in selected) for p in ('score','reference')}!={'score':32,'reference':32}:raise ValueError('H2 pools must be exactly M32/L32')
    for row in rows:_native_path(row)
    if cached is not None:
        native_manifest=_read(root/'h2/native_manifest.json')
        if cached.get('legacy_continuation_compatible')!=compatible or cached.get('trials')!=rows or native_manifest.get('trials')!=rows:
            raise ValueError('Cached H2 population or native provenance changed')
        return cached
    result=dict(status='complete',contexts=[{k:v for k,v in x.items() if k!='representative'} for x in contexts],trials=rows,
        population=population,legacy_continuation_compatible=compatible,scoring_futures=32,reference_futures=32,
        m_values=[1,2,4,8,16,32],ledger=budget.summary(root),generation=run,
        inference='Repeated nested/subsampled scoring estimates share outcomes and L32 references; conditional Monte Carlo uncertainty is distinct from across-circuit variability')
    atomic_json(root/'h2/native_manifest.json',dict(status='complete',trials=rows,population=population))
    atomic_json(marker,result)
    return result


def measured_plan(config,root):
    root=Path(root);plan=_frozen(root);gate=_require_concurrency(config,root);ledger=budget.summary(root)
    # Verify records and recover fully durable orphan outputs before projecting
    # new calls. A charged, completed orphan must resume at the hard cap for free.
    captured={r['context_id'] for r in gate['results'] if r['prefix_compatibility']['passed']}
    state_tasks=[_task(root,plan,'cutoff-state-recovery',x,cutoff_only=True,key=x['context_id'])
        for x in plan['selection']['state_cohort'] if x['context_id'] not in captured]
    state_missing=sum(_completed_task(task) is None for task in state_tasks)
    if gate['h2_legacy_compatible']:
        byid={x['context_id']:x for x in plan['selection']['h2_contexts']}
        h2_tasks=[_task(root,plan,'h2-extension',byid[s['history_id']],s) for s in plan['h2_additional_specs']]
    else:
        if not config.get('allow_registered_fallback',True):raise RuntimeError('Legacy continuation is incompatible and registered fallback is disabled')
        _,h2_tasks=_fallback_allocation(root,plan)
    h2=sum(_completed_task(task) is None for task in h2_tasks)
    replay_tasks=[]
    for context in plan['selection']['h2_contexts']:
        match=[r for r in plan['historical_rows'] if r['context_id']==context['context_id'] and r['pool']=='score' and r['future_index']==1]
        if len(match)!=1:raise ValueError('Registered replay future is absent')
        copy=deepcopy(context);copy['representative']=match[0]
        replay_tasks.append(_task(root,plan,'exact-forcing-replay',copy))
    replays=sum(_completed_task(task) is None for task in replay_tasks)
    remaining=state_missing+h2+replays
    seconds=gate['mean_full_episode_seconds'];ratio=6001./7010.
    worker=seconds*(h2+replays+state_missing*ratio)
    # Measured eight-process throughput includes scheduling/import/export contention.
    elapsed=gate['command_wall_seconds']*(h2+replays+state_missing*ratio)/8
    result=dict(status='measured_planned',workers=8,threads_per_worker=1,host=platform.node(),design_hash=plan['design_hash'],
        already_attempted_calls=ledger['attempted_native_calls'],prospective_remaining_calls=remaining,
        projected_total_calls=ledger['attempted_native_calls']+remaining,hard_cap=160,
        projected_remaining_worker_hours=worker/3600,projected_remaining_elapsed_hours=elapsed/3600,
        projected_new_native_bytes=gate['native_bytes_per_full_episode']*(h2+replays+state_missing*ratio),
        fitting={'status':'unmeasured','policy':'Small fixed ridge grid, bounded one-seed <=500-update neural fits; measured separately on chosen CPU/CUDA device'},
        generation_cost_basis='Actual same-host eight-process measured full episodes; cutoff recovery duration ratio is an approximation; fitting/export excluded',
        future_extension_calls=h2,cutoff_recovery_calls=state_missing,replay_calls=replays,ledger=ledger)
    if result['projected_total_calls']>160:raise RuntimeError('Measured pilot plan exceeds invocation cap before main allocation')
    atomic_json(root/'measured_plan.json',result)
    return result


def h2_trials(config,root):
    """Yield verified numeric native trials per registered H2 context and pool."""
    manifest=_read(Path(root)/'h2/native_manifest.json')
    if manifest.get('status')!='complete':raise ValueError('H2 native allocation is not complete')
    for cid in sorted({r['context_id'] for r in manifest['trials']}):
        row=dict(context_id=cid,score=[],reference=[])
        for item in sorted((r for r in manifest['trials'] if r['context_id']==cid),key=lambda r:(r['pool'],r['future_index'])):
            row[item['pool']].append((item['trial_id'],_read_fork(_native_path(item))))
        if len(row['score'])!=32 or len(row['reference'])!=32:raise ValueError('H2 pool counts differ from M32/L32 registration')
        yield row
