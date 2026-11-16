"""Explicit v6 stages and a fail-closed bounded campaign runner."""
from pathlib import Path
import argparse,json,os,sys,shutil,time,platform
from .config import load_config,resolve,counts,registry_check
from .artifacts import ROOT,initialize,atomic_json,read_json,canonical_hash,environment,output_lock,progress,failure,module_hash,file_hash

MODES=('doctor','controls','import','audit','audit-existing','explore-existing','geometry-existing','numerical-check','plan','timing-gate','generate','derive','fit','predict','calibrate','score','retention','replication','confirm-statistics','report','export','information-fit','information','replay','run')


def doctor(c,root,device):
    import torch
    parent=Path(root).resolve()
    while not parent.exists():parent=parent.parent
    if device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    if not os.access(parent,os.W_OK):raise PermissionError('Output parent is not writable '+str(parent))
    if c['backend']=='neuron':import neuron,LFPy
    memory={}
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,val=line.split(':',1)
            if key in ('MemTotal','MemAvailable'):memory[key+'_bytes']=int(val.strip().split()[0])*1024
    except OSError:pass
    cpu='unknown'
    try:cpu=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
    except (OSError,StopIteration):pass
    return {'status':'complete','root':str(Path(root).resolve()),'repository':str(ROOT),'python_executable':sys.executable,'environment':environment(),
        'free_bytes':shutil.disk_usage(parent).free,'memory':memory,'cpu_model':cpu,'device':device,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'requested_generation_workers':c['data']['workers'],
        'registry':registry_check(c),'config_hash':canonical_hash(c),'counts':counts(c),
        'scientific_scope':'40-cell canonical6.3C toy, ideal EEG; analytical fallback disabled',
        'generation_accelerator':'CPU NEURON; CUDA accelerates fits only','paths_verified_on_host':platform.node()}


def _source(args,root):
    if args.source:return str(Path(args.source).expanduser().resolve())
    p=Path(root)/'imports/current.json'
    if p.exists():return read_json(p)['source_path']
    raise ValueError('An explicit verified --source archive/native directory is required')


def _stage(args,c,root):
    mode=args.mode
    if mode=='doctor':
        result=doctor(c,root,args.device);atomic_json(root/'doctor.json',result);return result
    if mode=='controls':
        from .controls import run
        return run(c,root,args.device)
    if mode=='import':
        from .evidence import EvidenceSource,import_evidence
        with EvidenceSource(_source(args,root)) as src:return import_evidence(src,root,verify_all=True)
    if mode in ('audit','audit-existing'):
        from .audit import audit_existing
        return audit_existing(c,root,_source(args,root),resume=args.resume)
    if mode=='explore-existing':
        if c['claim_status']!='exploratory_existing':raise ValueError('Exploration must be explicitly labelled inspected existing evidence')
        from .audit import audit_existing
        from .derivation import derive
        from .models import fit
        from .scoring import score
        from .reporting import report
        source=_source(args,root);audit_existing(c,root,source,resume=args.resume)
        derive(c,root,source,resume=args.resume,native_source=args.native_source)
        from ..v4.models import job_ids
        jobs=args.jobs or job_ids(root)
        fit(c,root,jobs=jobs,device=args.device,resume=args.resume);score(c,root,jobs=jobs,resume=args.resume)
        if c.get('extension',{}).get('enabled'):
            from . import information_models,information
            from .replication import replication
            information_models.fit(c,root,device=args.device,resume=args.resume,source=args.native_source or source)
            information.run(c,root,jobs=jobs,resume=args.resume)
            replication(c,root,jobs=jobs,resume=args.resume)
        return report(c,root)
    if mode=='geometry-existing':
        from .geometry import run
        return run(c,root,_source(args,root),resume=args.resume)
    if mode=='timing-gate':
        from .generation import generate
        from .controls import throughput
        result=generate(c,root,execute=args.execute,timing=True,resume=args.resume)
        result['fit_throughput']=throughput(c,root,args.device,resume=args.resume);return result
    if mode=='numerical-check':
        from .numerical import run
        return run(c,root,execute=args.execute,resume=args.resume,qualification_root=args.phase0_root)
    if mode=='plan':
        from .planning import plan
        r=plan(c,root,timing_root=args.timing_root,phase0_root=args.phase0_root,numerical_root=args.numerical_root or root,
            source_archive=args.source_archive or args.source,native_source=args.native_source)
        if (root/'precision_certificate.json').exists():
            from .precision import propose_design
            proposal=propose_design(c,read_json(root/'precision_certificate.json'));atomic_json(root/'pretest_design_proposal.json',proposal)
        return r
    if mode=='generate':
        from .generation import generate
        return generate(c,root,execute=args.execute,resume=args.resume,source=args.native_source,split=args.split)
    if mode=='derive':
        from .derivation import derive
        return derive(c,root,source=args.source,resume=args.resume,native_source=args.native_source if args.source else None)
    if mode in ('fit','predict'):
        from . import models
        return getattr(models,mode)(c,root,jobs=args.jobs,device=args.device,resume=args.resume)
    if mode=='calibrate':
        from .calibration import calibrate
        return calibrate(c,root,jobs=args.jobs,resume=args.resume)
    if mode=='score':
        from .scoring import score
        return score(c,root,jobs=args.jobs,resume=args.resume)
    if mode=='retention':
        from .retention import retention
        return retention(c,root,device=args.device,resume=args.resume)
    if mode=='replication':
        from .replication import replication
        return replication(c,root,jobs=args.jobs or (c['selected_job_ids'] if c.get('extension',{}).get('enabled') else None),resume=args.resume)
    if mode=='information-fit':
        from .information_models import fit
        return fit(c,root,device=args.device,resume=args.resume,source=args.source)
    if mode=='information':
        from .information import run
        return run(c,root,jobs=args.jobs,resume=args.resume)
    if mode=='replay':
        from .replay import run
        if not args.timing_root:raise ValueError('Replay requires the verified --timing-root')
        return run(c,root,args.timing_root,execute=args.execute,resume=args.resume,qualification_root=args.phase0_root)
    if mode=='confirm-statistics':
        from .scoring import confirm_statistics
        return confirm_statistics(c,root)
    if mode=='report':
        from .reporting import report
        return report(c,root)
    if mode=='export':
        from .exporting import export
        if not args.destination:raise ValueError('--destination must name a new ZIP outside the result root')
        return export(c,root,args.destination,full_native=args.full_native,max_bytes=args.max_bytes)
    raise ValueError('Unknown stage '+mode)


def stage(args):
    c=load_config(args.config);root=Path(args.root).expanduser().resolve()
    if args.workers is not None:c=resolve({**c,'data':{**c['data'],'workers':args.workers}})
    if c['claim_status']=='prospective_confirmation' and args.jobs and set(args.jobs)!=set(c['selected_job_ids']):raise ValueError('Fresh campaign jobs cannot silently truncate the frozen grid')
    print(json.dumps({'mode':args.mode,'root':str(root),'config_hash':canonical_hash(c),'claim_status':c['claim_status'],'backend':c['backend']},sort_keys=True),flush=True)
    with output_lock(root):
        initialize(root,c,resume=args.resume)
        progress(root,args.mode,'running',command=[sys.executable,'-m','experiments.eeg_representation.h1.v6.cli',*sys.argv[1:]])
        try:
            value=_stage(args,c,root)
            progress(root,args.mode,value.get('status','complete'),config_hash=canonical_hash(c));return value
        except BaseException as exc:
            failure(root,args.mode,exc);progress(root,args.mode,'failed',error=str(exc));raise



def _campaign_implementation():
    """Content identity, independent of a later commit of identical source bytes."""
    from ..artifacts import source_hash
    paths=list((ROOT/'configs/eeg_representation').rglob('*.yaml'))
    paths+=list((ROOT/'configs/eeg_representation').rglob('*.json'))
    paths+=list((ROOT/'env/eeg').glob('*.py'))
    paths+=[ROOT/'scripts/eegrepr/run_h1_h4_v6.sh',ROOT/'scripts/eegrepr/run_h1_h4_v6_extended.sh']
    return {'scientific_source_hash':source_hash(),
            'additional_dependencies':{str(p.relative_to(ROOT)):file_hash(p) for p in sorted(set(paths)) if p.is_file()}}


def _campaign_request(args,original,base,workers):
    """Capture the original invocation before its pretest selected config changes."""
    from .generation import cpu_topology
    from .planning import _fit_hardware
    source=Path(args.source_archive).expanduser().resolve()
    if not source.is_file():raise FileNotFoundError('Campaign source archive missing: '+str(source))
    return {'resolved_original_config':original,'workers':int(workers),'device':args.device,
            'source_archive':str(source),'source_archive_sha256':file_hash(source),
            'native_source':str(Path(args.native_source).expanduser().resolve()),
            'full_native':bool(args.full_native),'max_bytes':args.max_bytes,
            'destination':str(Path(args.destination or base/'confirmation-standard.zip').expanduser().resolve()),
            'implementation':_campaign_implementation(),
            'environment':{**environment(),'python_executable':str(Path(sys.executable).resolve()),
                'cpu_topology':cpu_topology(),'fitting_hardware':_fit_hardware(args.device),
                'thread_environment':{name:os.environ.get(name) for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}}


def _verify_campaign_export(request,result):
    """Verify the finished ZIP and every payload without extracting or loading models."""
    from ..v4.restore_export import verify_archive
    path=Path(request['destination'])
    if result.get('status')!='complete' or result.get('path')!=str(path) or result.get('full_native')!=request['full_native']:
        raise ValueError('Completed campaign export metadata differs from execution request')
    if not path.is_file() or path.stat().st_size!=result.get('bytes') or file_hash(path)!=result.get('sha256'):
        raise ValueError('Completed campaign export missing or checksum changed')
    if request['max_bytes'] is not None and path.stat().st_size>request['max_bytes']:
        raise ValueError('Completed campaign export exceeds requested size limit')
    checked=verify_archive(path);manifest=checked.pop('manifest')
    if (manifest.get('schema_version')!='eegrepr-v6-export-1' or manifest.get('status')!='complete'
            or manifest.get('full_native')!=request['full_native']):
        raise ValueError('Completed archive is not the requested complete v6 export')
    return checked


def _completed_campaign(args,base,request):
    """Only a checksum-bound completed receipt permits an idempotent return."""
    path=base/'campaign_completed.json'
    if not path.exists():
        if Path(request['destination']).exists():
            raise FileExistsError('Export exists without a verified completed-campaign receipt; choose a new explicit --destination')
        return None
    if not args.resume:raise FileExistsError('Completed campaign exists; use --resume to verify its export')
    receipt=read_json(path)
    if (receipt.get('schema_version')!='eegrepr-v6-completed-campaign-1' or receipt.get('status')!='complete'
            or receipt.get('receipt_hash')!=canonical_hash({k:v for k,v in receipt.items() if k!='receipt_hash'})):
        raise ValueError('Completed campaign receipt invalid or changed')
    if receipt.get('execution_request')!=request or receipt.get('execution_request_hash')!=canonical_hash(request):
        raise ValueError('Completed campaign request/config/source/implementation/environment changed; use a new campaign root')
    _verify_campaign_export(request,receipt['export'])
    return receipt['export']


def campaign(args):
    """No unbounded all mode: fixed sequence, measured gates, explicit execution."""
    if not args.execute:raise ValueError('run requires --execute; use doctor/controls/plan for non-generation review')
    if not args.source_archive or not args.native_source:raise ValueError('run requires explicit --source-archive and --native-source (original v4 generation root)')
    import copy
    args=copy.copy(args)
    base=Path(args.root).expanduser().resolve();base.mkdir(parents=True,exist_ok=True)
    original=load_config(args.config)
    phase0=base/'phase0';timing=base/'timing-gate';core=base/'confirmation';workers=args.workers or original['data']['workers']
    original_args=copy.copy(args);request=_campaign_request(args,original,base,workers)
    completed=_completed_campaign(args,base,request)
    if completed is not None:return completed
    def call(mode,root,config=None,**changes):
        a=copy.copy(args);a.mode=mode;a.root=str(root);a.config=str(config or args.config);a.resume=True;a.jobs=None
        for k,v in changes.items():setattr(a,k,v)
        atomic_json(base/'campaign_progress.json',{'stage':mode,'stage_root':str(root),'updated_unix':time.time(),'status':'running'})
        result=stage(a)
        atomic_json(base/'campaign_progress.json',{'stage':mode,'stage_root':str(root),'updated_unix':time.time(),'status':result.get('status','complete')})
        return result
    existing=ROOT/'configs/eeg_representation/v6/existing.yaml';gate=ROOT/'configs/eeg_representation/v6/timing_gate.yaml'
    # Match prospective optimizer settings in qualification, even if dtype was explicitly configured.
    ec=load_config(existing);ec.update(models=original['models'],retention=original['retention']);ec['data']['workers']=workers
    extended=bool(original.get('extension',{}).get('enabled'))
    if extended:
        ec.update(extension=original['extension'],selected_job_ids=original['selected_job_ids'])
        if 'information_models' in original:ec['information_models']=original['information_models']
    ecpath=base/'phase0-config.json';atomic_json(ecpath,resolve(ec))
    call('doctor',phase0,ecpath);call('controls',phase0,ecpath)
    call('explore-existing',phase0,ecpath,source=args.source_archive)
    try:call('geometry-existing',phase0,ecpath,source=args.source_archive)
    except Exception as exc:
        # Secondary descriptive geometry cannot gate the inferential campaign.
        atomic_json(base/'geometry-secondary-status.json',{'status':'unavailable','error':str(exc),'core_blocking':False})
    from .precision import certificate,propose_design
    reviewed=certificate(original,base/'design-review',args.source_archive,phase0)
    proposal=propose_design(original,reviewed);atomic_json(base/'design-review/pretest_design_proposal.json',proposal)
    if proposal['status']=='unavailable' or not proposal.get('phase0_complete'):
        return {'status':'blocked','stage':'precision-design','proposal':str(base/'design-review/pretest_design_proposal.json')}
    # Authorized pretest revision only: registered IDs, unchanged endpoints, caps
    # still enforced by the measured plan. Never extend after fresh outcomes.
    chosen=proposal['resolved_config'];chosen['data']['workers']=workers
    frozen_path=base/'selected-config.json'
    if frozen_path.exists() and read_json(frozen_path)!=chosen:
        raise ValueError('Pretest chosen config already fixed; changed qualification requires a new campaign root')
    atomic_json(frozen_path,chosen);args.config=str(frozen_path)
    core=base/('confirmation-n'+str(chosen['protocol']['test_circuits']))
    gc=load_config(gate);gc.update(models=chosen['models'],retention=chosen['retention'],simulation=chosen['simulation'])
    gc['data']['workers']=workers
    if extended:
        gc.update(extension=chosen['extension'],selected_job_ids=chosen['selected_job_ids'])
        if 'information_models' in chosen:gc['information_models']=chosen['information_models']
    gatepath=base/'timing-config.json';atomic_json(gatepath,resolve(gc))
    call('timing-gate',timing,gatepath)
    # Precision is decided from inspected evidence before any fresh test generation.
    common=dict(source=None,source_archive=args.source_archive,native_source=args.native_source,phase0_root=str(phase0),timing_root=str(timing),numerical_root=str(core))
    pre=call('plan',core,**common)
    if not pre['eligible_numerical_execution']:
        call('report',core);return {'status':'blocked','stage':'pre-numerical','root':str(core),'reasons':pre['blocked_reasons']}
    if extended:call('replay',core,**common)
    call('numerical-check',core,**common)
    final=call('plan',core,**common)
    if not final['eligible_for_generation']:
        call('report',core);return {'status':'blocked','stage':'pre-core','root':str(core),'reasons':final['blocked_reasons'],'proposal':str(core/'pretest_design_proposal.json')}
    for split in ('train','development'):call('generate',core,split=split,**common)
    call('derive',core,source=None);call('fit',core)
    if extended:call('information-fit',core,source=None)
    call('generate',core,split='calibration',**common);call('derive',core,source=None);call('predict',core);call('calibrate',core)
    if extended:call('information-fit',core,source=None)
    call('generate',core,split='evaluation',**common);call('derive',core,source=None);call('predict',core)
    if extended:call('information-fit',core,source=None)
    call('retention',core);call('score',core);call('replication',core);call('confirm-statistics',core)
    if extended:call('information',core)
    call('report',core)
    if _campaign_request(original_args,original,base,workers)!=request:
        raise ValueError('Campaign execution request/source/implementation/environment changed during execution')
    result=call('export',core,destination=request['destination'])
    verification=_verify_campaign_export(request,result)
    receipt={'schema_version':'eegrepr-v6-completed-campaign-1','status':'complete',
             'execution_request':request,'execution_request_hash':canonical_hash(request),
             'export':result,'export_verification':verification}
    receipt['receipt_hash']=canonical_hash(receipt);atomic_json(base/'campaign_completed.json',receipt)
    atomic_json(base/'campaign_progress.json',{'status':'complete','export':result,'updated_unix':time.time()});return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=MODES)
    p.add_argument('--root',required=True);p.add_argument('--config',default=str(ROOT/'configs/eeg_representation/v6/confirmation.yaml'))
    p.add_argument('--source');p.add_argument('--source-archive');p.add_argument('--native-source');p.add_argument('--phase0-root');p.add_argument('--timing-root');p.add_argument('--numerical-root')
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--workers',type=int);p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--jobs',nargs='+');p.add_argument('--split',choices=('train','development','calibration','evaluation'))
    p.add_argument('--destination');p.add_argument('--full-native',action='store_true');p.add_argument('--max-bytes',type=int)
    return p


def main(argv=None):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    args=parser().parse_args(argv)
    try:
        value=campaign(args) if args.mode=='run' else stage(args)
        if args.mode=='run':atomic_json(Path(args.root)/'campaign_result.json',value)
        print(json.dumps(value,sort_keys=True,indent=2,allow_nan=False),flush=True)
        return 2 if value.get('status')=='blocked' else 0
    except (Exception,KeyboardInterrupt) as exc:
        print('V6 ERROR: '+str(exc),file=sys.stderr,flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
