"""Resume completed-native pilot fitting/reporting with native execution disabled.

This additive entry point leaves the frozen generator, state adapter and learning
mathematics unchanged. The narrowly audited TCN metadata fix has explicit source
compatibility records. A fresh Python child runs each unfinished TCN with the
original seed, device, precision and fitting routine. No automatic retries occur.
"""
from contextlib import contextmanager, ExitStack
from pathlib import Path
import argparse
import hashlib
import importlib.abc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from unittest.mock import patch

BLOCKED_IMPORTS=('neuron','LFPy','mpi4py','env.models.neuron','setup.circuits.ballnstick')
UPSTREAM_STAGES=('preflight','engineering','registration','derive_existing','matched_targets',
    'concurrency','measured_plan','replay','state_recovery','state_fit','state_model_gate','h2_generation')


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _denied(*args,**kwargs):
    raise RuntimeError('Recovery prohibits every native simulator invocation; missing evidence must be investigated, never regenerated')


class NativeImportBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if any(fullname==p or fullname.startswith(p+'.') for p in BLOCKED_IMPORTS):
            raise ImportError('Native simulator import prohibited during recovery: '+fullname)
        return None


@contextmanager
def native_disabled():
    """Install before application imports; keep guards active in parent/children."""
    loaded=[n for n in sys.modules if any(n==p or n.startswith(p+'.') for p in BLOCKED_IMPORTS)]
    if loaded:raise RuntimeError('Start recovery in a fresh Python process; native modules already loaded: '+str(loaded))
    blocker=NativeImportBlocker();sys.meta_path.insert(0,blocker)
    try:
        from . import budget,generation,state
        from ... import simulator as legacy
        from ..v6 import simulator as numerical
        with ExitStack() as stack:
            for module,name in ((budget,'begin'),(generation,'_run'),(generation,'_worker'),
                                (state,'generate_with_state'),(legacy,'generate_trial'),(numerical,'generate_trial')):
                stack.enter_context(patch.object(module,name,_denied))
            yield
    finally:
        sys.meta_path.remove(blocker)


class Verifier:
    """Hash each unchanged file once per check; never load model pickle files."""
    def __init__(self):self.cache={};self.files=[]
    def file(self,path,expected=None,size=None):
        p=Path(path).resolve()
        if not p.is_file():raise ValueError('Required recovery artifact missing: '+str(p))
        before=p.stat();key=(str(p),before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
        if size is not None and before.st_size!=int(size):raise ValueError('Artifact size mismatch: '+str(p))
        digest=self.cache.get(key)
        if digest is None:
            digest=sha(p);after=p.stat()
            if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):raise ValueError('Artifact changed during check: '+str(p))
            self.cache[key]=digest;self.files.append(dict(path=str(p),sha256=digest,bytes=before.st_size))
            if len(self.files)%100==0:print(f'Recovery check: verified {len(self.files)} artifact files',flush=True)
        if expected is not None and digest!=expected:raise ValueError('Artifact checksum mismatch: '+str(p))
        return digest
    def record(self,root,row):
        base=Path(root).resolve();p=(base/row['path']).resolve()
        if not p.is_relative_to(base):raise ValueError('Recovery record escapes its declared root')
        self.file(p,row['sha256'],row.get('bytes'));return p


def protected_snapshot(root):
    """Generation accounting and original identity must remain byte-for-byte."""
    root=Path(root);paths=list((root/'invocation_ledger').glob('*.json'))
    paths += [root/n for n in ('pilot_config.json','pilot_provenance.json','design.json','concurrency_check.json',
        'replay_controls.json','state/manifest.json','state/fit.json','state_model_gate.json','h2/generation.json','h2/native_manifest.json')]
    paths += list((root/'generation_records').rglob('*.json'))
    return {p.relative_to(root).as_posix():sha(p) for p in sorted(paths) if p.is_file()}


def assert_protected(root,before):
    if protected_snapshot(root)!=before:raise RuntimeError('Protected native accounting or upstream evidence changed during recovery')


def _state_export(verifier,row):
    directory=Path(row['state_directory']);meta=read(directory/'schema.json')
    verifier.file(directory/'schema.json');verifier.file(directory/'fields.npz',meta['fields_sha256'])
    if meta.get('audit')!=row['state_validation'] or meta.get('input_hash')!=row['state_validation']['input_hash'] or meta.get('schema_hash')!=row['state_validation']['schema_hash']:
        raise ValueError('Saved complete-state export differs from native completion record')


def _verify_state_bindings(v,root,state_manifest,cohorts,statefit):
    from ..targets import canonical_hash
    directory=Path(root)/'state';contract=read(directory/'adapter_input_contract.json');freeze=read(directory/'adapter_freeze.json')
    content={k:val for k,val in freeze.items() if k!='manifest_content_sha256'}
    if freeze.get('status')!='complete' or freeze.get('manifest_content_sha256')!=canonical_hash(content):
        raise ValueError('Frozen state adapter content checksum changed')
    if contract.get('signature')!=canonical_hash(contract['details']) or freeze['signature']!=contract['signature']:
        raise ValueError('State adapter input signature changed')
    if contract['details']['cohorts']!=cohorts or statefit['coverage']['input_signature']!=contract['signature']:
        raise ValueError('Fitted state inputs differ from frozen cohort/adapter')
    population=canonical_hash({k:state_manifest.get(k) for k in
        ('design_hash','context_counts','circuit_ids','target_binding','input_policy','future_invariance')})
    if contract['details']['recovery_population_signature']!=population:raise ValueError('Frozen state recovery population changed')
    for name,digest in freeze['files'].items():v.file(directory/name,digest)
    sources=contract['details']['sources'];entries={r['context_id']:r for r in state_manifest['contexts']}
    expected={(split,cid) for split,ids in cohorts.items() for cid in ids}
    if len(sources)!=len(expected) or {(r['split'],r['context_id']) for r in sources}!=expected:raise ValueError('State input source coverage differs')
    for row in sources:
        path=Path(entries[row['context_id']]['state_directory'])
        if not path.is_absolute():path=Path(root)/path
        v.file(path/'schema.json',row['schema_sha256']);v.file(path/'fields.npz',row['fields_sha256'])
    return dict(contexts=len(sources),adapter_fingerprint=freeze['adapter_fingerprint'],input_signature=contract['signature'])


def verify_upstream(root,source_archive,device=None):
    """Read-only prerequisite check. Missing native/fit data fails closed."""
    from . import config,budget,generation,cli
    from ..targets import canonical_hash
    print('Recovery check: frozen config/device/runtime and source archive',flush=True)
    root=Path(root).resolve();v=Verifier();c=config.resolve(read(root/'pilot_config.json'))
    provenance=read(root/'pilot_provenance.json');actual_device=provenance['identity']['device']
    if device is not None and device!=actual_device:raise ValueError('Recovery device differs from frozen pilot identity')
    expected=dict(config_hash=canonical_hash(c),device=actual_device,dtype=c['dtype'],workers=8,threads_per_worker=1,generation_adapter=generation.adapter_contract())
    if provenance['identity']!=expected:raise ValueError('Frozen pilot config/device/adapter/runtime identity changed')
    v.file(source_archive,c['source_archive_sha256'])
    design=generation._frozen(root);cohorts=cli._cohorts(design);jobs=config.jobs()
    for name in UPSTREAM_STAGES:
        marker=root/'progress'/(name+'.json')
        if read(marker).get('status')!='complete':raise ValueError('Recovery requires completed upstream stage: '+name)
        v.file(marker)
    print('Recovery check: completed upstream stages and immutable invocation ledger',flush=True)
    ledger=budget.summary(root)
    if ledger['unresolved_calls'] or ledger['attempted_native_calls']>160 or ledger['implementation_engineering_calls']!=21:
        raise ValueError('Unexpected unresolved or incompatible invocation ledger; inspect before recovery')
    debit=read(root/'invocation_ledger/implementation_debit.json')
    v.file(generation.ROOT/debit['evidence']['path'],debit['evidence']['sha256'])
    v.file(root/'invocation_ledger/contract.json');v.file(root/'invocation_ledger/implementation_debit.json')
    attempts={}
    for p in (root/'invocation_ledger').glob('attempt-*.json'):
        a=read(p);attempts[a['attempt_id']]=a;v.file(p)
        result=read(root/'invocation_ledger'/('result-'+a['attempt_id']+'.json'))
        if result['attempt_id']!=a['attempt_id']:raise ValueError('Invocation result identity differs')
        v.file(root/'invocation_ledger'/('result-'+a['attempt_id']+'.json'))
    concurrent=read(root/'concurrency_check.json');replay=read(root/'replay_controls.json');h2=read(root/'h2/generation.json')
    if concurrent.get('status')!='complete' or concurrent.get('actual_worker_processes')!=8 or concurrent.get('simultaneous_native_overlap_seconds',0)<=0:
        raise ValueError('Saved measured eight-process gate is incomplete')
    if replay.get('status')!='complete' or replay.get('all_exact') is not True or len(replay.get('audits',[]))!=2 or any(r.get('complete_state_future_invariance',{}).get('passed') is not True for r in replay['audits']):
        raise ValueError('Saved exact-forcing/future-invariance gate is incomplete')
    if h2.get('status')!='complete':raise ValueError('H2 generation must already be complete')
    tasks=[]
    for context in design['selection']['concurrency_contexts']:
        task=generation._task(root,design,concurrent['stage'],context);task['concurrency_barrier']=True;tasks.append(task)
    captured={r['context_id'] for r in concurrent['results'] if r['prefix_compatibility']['passed']}
    tasks += [generation._task(root,design,'cutoff-state-recovery',x,cutoff_only=True,key=x['context_id'])
        for x in design['selection']['state_cohort'] if x['context_id'] not in captured]
    for context in design['selection']['h2_contexts']:
        matching=[r for r in design['historical_rows'] if r['context_id']==context['context_id'] and r['pool']=='score' and r['future_index']==1]
        if len(matching)!=1:raise ValueError('Registered replay source missing')
        tasks.append(generation._task(root,design,'exact-forcing-replay',{**context,'representative':matching[0]}))
    if h2['legacy_continuation_compatible']:
        byid={r['context_id']:r for r in design['selection']['h2_contexts']}
        tasks += [generation._task(root,design,'h2-extension',byid[s['history_id']],s) for s in design['h2_additional_specs']]
    else:tasks += generation._fallback_allocation(root,design)[1]
    print(f'Recovery check: {len(tasks)} completed native tasks, saved state exports and H2 pools',flush=True)
    verified={}
    for task in tasks:
        path=generation._record_path(root,task['stage'],task['key']);row=read(path);v.file(path)
        if row.get('status')!='complete' or row.get('task_hash')!=canonical_hash(task):raise ValueError('Completed native task identity changed')
        attempt=attempts.get(row['attempt_id']);result=read(root/'invocation_ledger'/('result-'+row['attempt_id']+'.json'))
        if not attempt or attempt['stage']!=task['stage'] or attempt['identity']['task_hash']!=row['task_hash'] or result.get('status')!='complete':
            raise ValueError('Native completion is not backed by a completed charged invocation')
        v.file(row['native_file'],row['native_sha256'],row['native_bytes']);_state_export(v,row)
        if task.get('reference'):v.record(task['reference']['source_root'],task['reference'])
        if task.get('reference') and not row.get('prefix_compatibility',{}).get('passed'):raise ValueError('Native prefix compatibility failed')
        verified[(row['stage'],row['key'])]=row
    for marker in (concurrent,replay,h2['generation']):
        if any(verified.get((r['stage'],r['key']))!=r for r in marker['results']):raise ValueError('Aggregate native receipt differs from per-task records')
    native=read(root/'h2/native_manifest.json')
    if native.get('status')!='complete' or native['trials']!=h2['trials']:raise ValueError('H2 native manifest changed')
    for cid in {r['context_id'] for r in h2['trials']}:
        rows=[r for r in h2['trials'] if r['context_id']==cid]
        if {pool:sum(r['pool']==pool for r in rows) for pool in ('score','reference')}!={'score':32,'reference':32}:raise ValueError('H2 M32/L32 pools incomplete')
    for row in h2['trials']:v.record(row['source_root'],row)
    state_manifest=read(root/'state/manifest.json')
    if state_manifest.get('status')!='complete' or state_manifest.get('design_hash')!=design['design_hash'] or len(state_manifest.get('contexts',[]))!=56:
        raise ValueError('Matched complete-state cohort incomplete')
    for split,ids in cohorts.items():
        if {r['context_id'] for r in state_manifest['contexts'] if r['split']==split}!=set(ids):raise ValueError('State cohort differs from registered split')
    statefit=read(root/'state/fit.json');gate=read(root/'state_model_gate.json')
    if statefit.get('status')!='complete' or {r['job'] for r in statefit.get('jobs',[])}!=set(jobs) or statefit.get('cohorts')!=cohorts:
        raise ValueError('All 60 frozen state fits are required')
    if gate.get('status')!='complete' or gate.get('jobs')!=60 or gate.get('held_out_evaluated') is not True or gate.get('state_dependent_candidate_fitted') is not True:
        raise ValueError('Complete-state fitted/held-out gate missing')
    if len(gate.get('artifacts',[]))!=300:raise ValueError('State-model gate artifact inventory incomplete')
    print('Recovery check: complete-state adapter bindings and 60 fitted-state artifact sets',flush=True)
    state_bindings=_verify_state_bindings(v,root,state_manifest,cohorts,statefit)
    for row in gate['artifacts']+[gate['fit_artifact']]:v.record(root,row)
    print('Recovery check: 120 saved derived settings and approved learning-source compatibility',flush=True)
    for cohort in ('matched','broad'):
        stage=root/cohort;manifest=read(stage/'derived/manifest.json')
        if manifest.get('status')!='complete' or {r['job_id'] for r in manifest['jobs']}!=set(jobs):raise ValueError('All 60 derived settings required: '+cohort)
        for row in manifest['jobs']:
            job=read(v.record(stage,row))
            if job.get('status')!='complete':raise ValueError('Derived target is incomplete')
            for data in job['files'].values():v.record(stage,data)
    for name in ('pilot_config.json','pilot_provenance.json','design.json','concurrency_check.json','replay_controls.json','h2/generation.json','h2/native_manifest.json','state/manifest.json','state_model_gate.json'):
        v.file(root/name)
    from .learning_compatibility import compatibility_record
    compatibility=compatibility_record()
    return dict(status='complete',native_calls_allowed=0,device=actual_device,config_hash=canonical_hash(c),
        original_commit=provenance['commit'],ledger=ledger,native_tasks_verified=len(tasks),state_jobs_verified=60,
        derived_settings_verified=120,state_adapter_verified=state_bindings,learning_compatibility=compatibility,files_verified=len(v.files),verified_files=v.files,
        implementation_sha256=sha(__file__),purpose='Finish ordinary fitting, scoring, PDF and export; no native generation or complete-state refit')


def run_tcn_child(root,run_dir,job,history,device,original,panel,manifest,destination,c):
    """Validate completed fits; isolate each unfinished fit without changing it."""
    destination=Path(destination);expected=Path(root)/'matched/learners'/job/('tcn_'+str(history))
    if destination.resolve()!=expected.resolve() or job not in c['neural_jobs'] or history not in c['histories']:
        raise ValueError('Unregistered TCN recovery request')
    if (destination/'selection.json').exists():return original(panel,manifest,destination,c,history,device,True)
    from ..artifacts import atomic_json
    folder=Path(run_dir)/'tcn'/job/('h'+str(history));folder.mkdir(parents=True,exist_ok=False)
    if destination.exists():
        shutil.copytree(destination,folder/'initial_partial_output')
    log=folder/'child.log';command=[sys.executable,'-u','-X','faulthandler','-m',__package__+'.recovery',
        '--root',str(root),'--device',device,'--child-tcn',job,'--history',str(history),'--child-receipt',str(folder/'receipt.json')]
    atomic_json(folder/'before.json',dict(status='starting',job=job,history=history,device=device,command=command,
        parent_pid=os.getpid(),started_unix=time.time(),native_calls_allowed=0,automatic_retries=0))
    env=dict(os.environ,PYTHONUNBUFFERED='1',PYTHONFAULTHANDLER='1')
    print(f'Recovery TCN: starting {job}, history={history}, device={device}; log={log}',flush=True)
    with log.open('wb') as output:
        proc=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,env=env)
        atomic_json(folder/'process.json',dict(pid=proc.pid,started_unix=time.time(),command=command))
        code=proc.wait()
    atomic_json(folder/'exit.json',dict(returncode=code,finished_unix=time.time(),log=str(log),log_sha256=sha(log),native_calls_allowed=0))
    if code:raise RuntimeError(f'TCN recovery child failed with exit {code}; no automatic retry. Inspect {log}')
    receipt=read(folder/'receipt.json')
    if receipt.get('status')!='complete' or receipt.get('job')!=job or receipt.get('history')!=history or receipt.get('native_calls')!=0:
        raise ValueError('TCN child completion receipt invalid')
    if not (destination/'selection.json').is_file():raise ValueError('Child exited without a complete fitted selection')
    if sha(destination/'selection.json')!=receipt['selection_sha256']:raise ValueError('Child fitted selection checksum changed')
    return original(panel,manifest,destination,c,history,device,True)


def child_tcn(args):
    from . import config,learning,generation
    from ..v6.derivation import load_panel
    from ..artifacts import atomic_json
    from ..targets import canonical_hash
    root=Path(args.root).resolve();c=config.resolve(read(root/'pilot_config.json'));identity=read(root/'pilot_provenance.json')['identity']
    if identity['device']!=args.device or identity['config_hash']!=canonical_hash(c) or identity['generation_adapter']!=generation.adapter_contract():
        raise ValueError('Child pilot identity changed')
    if args.child_tcn not in c['neural_jobs'] or args.history not in c['histories']:raise ValueError('Unregistered child fit')
    before=protected_snapshot(root);panel,manifest=load_panel(root/'matched',args.child_tcn)
    dest=root/'matched/learners'/args.child_tcn/('tcn_'+str(args.history))
    try:learning.tcn(panel,manifest,dest,c,args.history,args.device,True)
    finally:assert_protected(root,before)
    atomic_json(args.child_receipt,dict(status='complete',job=args.child_tcn,history=args.history,device=args.device,
        pid=os.getpid(),selection_sha256=sha(dest/'selection.json'),native_calls=0,completed_unix=time.time()))


def execute(root,source_archive,check,destination=None):
    from . import config,cli,analysis,learning,reporting,exporting,budget
    from .learning_compatibility import compatible_ridge,compatibility_record
    from .precision import run as precision
    from ..artifacts import atomic_json,ROOT
    from ..targets import canonical_hash
    root=Path(root).resolve();c=config.resolve(read(root/'pilot_config.json'));device=check['device'];before=protected_snapshot(root)
    if (root/'pilot_completed.json').exists():
        raise FileExistsError('Pilot already has a completion receipt; verify it or use the report-only workflow')
    proposed=Path(destination) if destination else root.parent/(root.name+'-recovered-evidence.zip')
    if proposed.exists():raise FileExistsError('Recovery export already exists; preserve it and choose --destination')
    run_dir=root/'recovery'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]);run_dir.mkdir(parents=True)
    for folder in ('progress','failures'):
        if (root/folder).exists():shutil.copytree(root/folder,run_dir/('initial_'+folder))
    wrapper=ROOT/'scripts/eegrepr/recover_v6_reporting_pilot.sh'
    shutil.copy2(wrapper,run_dir/wrapper.name)
    atomic_json(run_dir/'preflight.json',check)
    atomic_json(run_dir/'execution.json',dict(status='running',pid=os.getpid(),host=platform.node(),started_unix=time.time(),
        source_archive=str(Path(source_archive).resolve()),device=device,config_hash=canonical_hash(c),implementation_sha256=sha(__file__),
        original_provenance_sha256=sha(root/'pilot_provenance.json'),native_calls_allowed=0,protected_snapshot=before,learning_compatibility=compatibility_record(),
        recovery_wrapper=dict(path=wrapper.name,sha256=sha(wrapper)),
        tcn_policy='Unchanged scientific routine in a fresh process per unfinished TCN; original device/seed/precision/budget; no automatic retries'))
    original=learning.tcn;original_ridge=learning.ridge
    def isolated(panel,manifest,destination,options,history,device='cpu',resume=False):
        return run_tcn_child(root,run_dir,Path(destination).parent.name,history,device,original,panel,manifest,destination,options)
    try:
        with patch.object(learning,'tcn',isolated),patch.object(learning,'ridge',lambda *a,**kw:compatible_ridge(original_ridge,*a,**kw)):
            cli.stage(root,'ordinary_fit',lambda:analysis.fit_ordinary(c,root,source_archive,config.jobs(),True,device))
        tables=cli.stage(root,'scoring',lambda:analysis.score_models(c,root,config.jobs()))
        extra=cli.stage(root,'h2_replay_scoring',lambda:precision(c,root))
        results=cli._assemble(c,root,tables,extra)
        report=cli.stage(root,'report',lambda:reporting.report(root,c,results));cli.verify_report(root,c,report)
        assert_protected(root,before)
        dest=Path(destination) if destination else root.parent/(root.name+'-recovered-evidence.zip')
        exported=cli.stage(root,'export',lambda:exporting.export(root,dest))
        assert_protected(root,before)
        receipt=dict(status='complete',purpose=c['purpose'],completed_unix=time.time(),simulation_ledger=budget.summary(root),
            report_receipt_sha256=sha(root/'report/receipt.json'),pdf_sha256=sha(root/'report/pilot_findings.pdf'),
            pilot_results_sha256=sha(root/'pilot_results.json'),config_hash=canonical_hash(c),
            pdf_all_pages_rendered=True,pdf_checks=report['pdf_verification']['automated_checks'],export=exported,
            recovery=dict(path=str(run_dir),native_calls=0,protected_upstream_unchanged=True),
            confirmation='Not launched; separate user decision after pilot review')
        atomic_json(root/'pilot_completed.json',receipt)
        atomic_json(run_dir/'receipt.json',dict(status='complete',native_calls=0,protected_upstream_unchanged=True,
            pilot_completed_sha256=sha(root/'pilot_completed.json'),finished_unix=time.time()))
        return receipt
    except BaseException as exc:
        atomic_json(run_dir/'failure.json',dict(status='failed',error=repr(exc),finished_unix=time.time(),native_calls=0,
            protected_upstream_unchanged=protected_snapshot(root)==before))
        raise
    finally:assert_protected(root,before)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True);p.add_argument('--source-archive');p.add_argument('--device',choices=['cpu','cuda'])
    modes=p.add_mutually_exclusive_group();modes.add_argument('--check-only',action='store_true');modes.add_argument('--execute',action='store_true')
    p.add_argument('--destination');p.add_argument('--child-tcn',help=argparse.SUPPRESS);p.add_argument('--history',type=int,help=argparse.SUPPRESS);p.add_argument('--child-receipt',help=argparse.SUPPRESS)
    args=p.parse_args(argv);root=Path(args.root).expanduser().resolve()
    if not root.is_dir():p.error('An existing completed-native pilot root is required')
    with native_disabled():
        if args.child_tcn:
            if not args.child_receipt or args.history is None or args.device is None:p.error('Incomplete internal child request')
            child_tcn(args);return 0
        if not args.source_archive:p.error('--source-archive is required for verified representation reuse')
        if not args.check_only and not args.execute:p.error('Select --check-only or --execute')
        from ..artifacts import output_lock
        with output_lock(root):
            check=verify_upstream(root,args.source_archive,args.device)
            print(json.dumps({k:v for k,v in check.items() if k!='verified_files'},indent=2),flush=True)
            if args.execute:print(json.dumps(execute(root,args.source_archive,check,args.destination),indent=2),flush=True)
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print('PILOT RECOVERY ERROR: '+str(exc),file=sys.stderr);raise SystemExit(2)
