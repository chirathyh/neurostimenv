"""Explicit manuscript-v4 stages; no generic all mode and no implicit simulation."""
from pathlib import Path
import argparse,json,os,sys,shutil
from .config import load_config
from .artifacts import initialize,environment,source_hash,atomic_json,read_json,ROOT,output_lock
from ..targets import canonical_hash
from .evidence import EvidenceSource,import_evidence

MODES=('doctor','controls','audit-existing','qualify-existing','geometry-existing','plan','timing-gate','generate','derive','fit','score','retention','replication','report','export')


def doctor(config,root,device):
    import torch
    ancestors=Path(root).expanduser().resolve()
    while not ancestors.exists():ancestors=ancestors.parent
    usage=shutil.disk_usage(ancestors)
    result={'status':'complete','repository':str(ROOT),'environment':environment(),'python_executable':sys.executable,
        'root':str(Path(root).resolve()),'nearest_existing_root':str(ancestors),'free_bytes':usage.free,
        'write_access':os.access(ancestors,os.W_OK),'backend':config['backend'],'device':device,'cuda_available':torch.cuda.is_available(),
        'config_hash':canonical_hash(config),'analysis_source_hash':source_hash(),
        'scientific_scope':'canonical6.3C40-cell toy; idealEEG; no human calibration',
        'native_fallback':'disabled; NEURON failure is an error, never analytical substitution',
        'workstation_path_status':'These are actual current-host paths; historical remote paths are not assumed to exist'}
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA unavailable on this host')
    if device=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8',':16:8'):raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG before Python imports CUDA')
    if not result['write_access']:raise PermissionError('Output parent is not writable: '+str(ancestors))
    if config['backend']=='neuron':
        import neuron,LFPy
        result['neuron_import']='available'
    from .planning import registry_check
    result['registry']=registry_check(config)
    return result


def controls(config,root,device):
    from .measurement import qualify
    from .models import known_learnable_control
    from .geometry import geometry_controls
    from .driver import controls as driver_controls
    parts={'measurement':qualify(root),'same_neural_path':known_learnable_control(device=device),
           'geometry':geometry_controls(),'common_events':driver_controls()}
    result={'status':'complete','passed':all(v['passed'] for v in parts.values()),'controls':parts,
            'scope':'engineering and measurement checks; not new biological evidence'}
    atomic_json(Path(root)/'controls.json',result)
    if not result['passed']:raise RuntimeError('One or more v4 controls failed; inspect controls.json')
    return result


def _source(args,root):
    if args.source:return EvidenceSource(args.source)
    current=root/'imports/current.json'
    if current.exists():return EvidenceSource(read_json(current)['source_path'])
    if (root/'generation/native/manifest.json').exists():return EvidenceSource(root/'generation')
    raise ValueError('Supply explicit --source for existing evidence, or complete native generation first')


def run(args):
    config=load_config(args.config);root=Path(args.root).expanduser().resolve()
    if config['claim_status']=='prospective_bounded' and args.jobs is not None and args.jobs!=config['selected_job_ids']:
        raise ValueError('Prospective jobs must match frozen configured selection; exploratory changes require a new protocol/root')
    print(json.dumps({'mode':args.mode,'root':str(root),'config_hash':canonical_hash(config),
        'analysis_source_hash':source_hash(),'backend':config['backend'],'claim_status':config['claim_status']},sort_keys=True),flush=True)
    if args.mode=='doctor':return doctor(config,root,args.device)
    with output_lock(root):
        initialize(root,config,args.resume)
        if args.mode=='controls':return controls(config,root,args.device)
        if args.mode=='plan':
            from .planning import plan
            return plan(config,root,args.timing_root,args.qualification_root)
        if args.mode in ('timing-gate','generate'):
            from .generation import generate
            return generate(config,root,args.resume,execute=args.execute,timing=args.mode=='timing-gate')
        if args.mode in ('derive','audit-existing','qualify-existing','geometry-existing','retention'):
            with _source(args,root) as source:
                import_evidence(source,root)
                if args.mode in ('derive','audit-existing'):
                    from .derivation import derive,audit_existing
                    return (derive if args.mode=='derive' else audit_existing)(config,root,source,args.resume)
                if args.mode=='qualify-existing':
                    from .derivation import derive
                    from .models import fit
                    from .scoring import score
                    derive(config,root,source,args.resume)
                    fitted=fit(config,root,args.jobs,args.device,args.resume)
                    return {'status':'complete','models':fitted,'scores':score(config,root,args.jobs,args.resume)}
                if args.mode=='geometry-existing':
                    from .geometry import geometry
                    return geometry(config,root,source,args.resume)
                from .retention import retention
                return retention(config,root,source,args.device,args.resume)
        if args.mode=='fit':
            from .models import fit
            return fit(config,root,args.jobs,args.device,args.resume)
        if args.mode=='score':
            from .scoring import score
            return score(config,root,args.jobs,args.resume)
        if args.mode=='replication':
            from .replication import replication
            return replication(config,root,args.jobs,args.resume)
        if args.mode=='report':
            from .reporting import report
            return report(config,root)
        if args.mode=='export':
            from .reporting import export
            if not args.destination:raise ValueError('Export requires --destination outside result root')
            source=None
            if args.source or (root/'imports/current.json').exists() or (root/'generation/native/manifest.json').exists():source=_source(args,root)
            try:return export(config,root,args.destination,source,args.full_native,args.max_bytes,include_imported=args.include_imported,deduplicate_npz=not args.plain_npz)
            finally:
                if source is not None:source.close()
    raise AssertionError('Unknown mode')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=MODES);p.add_argument('--config',required=True);p.add_argument('--root',required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu');p.add_argument('--source')
    p.add_argument('--resume',action='store_true');p.add_argument('--execute',action='store_true',help='Explicit native/analytical generation after a saved eligible plan')
    p.add_argument('--timing-root');p.add_argument('--qualification-root');p.add_argument('--jobs',nargs='+')
    p.add_argument('--destination');p.add_argument('--full-native',action='store_true');p.add_argument('--max-bytes',type=int)
    p.add_argument('--plain-npz',action='store_true',help='Keep original NPZ containers instead of compact lossless numeric-array deduplication')
    p.add_argument('--include-imported',action='store_true',help='Also include original imported nonraw artifacts; default compact export hashes their omissions')
    args=p.parse_args(argv)
    try:
        result=run(args)
        # Full numeric tables remain on disk; terminal status is bounded and readable.
        summary={k:v for k,v in result.items() if k in ('status','passed','path','bytes','sha256','completed_jobs','planned_jobs','total_episodes','eligible_for_generation','summed_generation_hours','counts','cost_status','protocol_hash','complete','continuations','mean_continuation_seconds','circuit_counts')}
        if args.mode=='doctor':summary=result
        print(json.dumps(summary,indent=2,sort_keys=True));return 0
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print('V4 ERROR: '+str(exc),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
