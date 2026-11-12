"""Explicit, cost-gated H1 information campaign; legacy CLI remains unchanged."""
from pathlib import Path
import argparse,json,subprocess,sys
from .artifacts import (atomic_json,read_json,file_hash,jsonable,initialize_output,require_compatible,
                        output_lock,resume_stage,stage_complete,ROOT,source_hash)
from .config import load_config
from .planning import plan


def audit(pilot_root=None,verify_raw=False):
    from ..storage import source_hash as legacy_hash
    def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()
    result={'schema_version':'eegrepr-h1-audit-v2','current_commit':git('rev-parse','HEAD'),
      'branch':git('branch','--show-current'),'git_status':git('status','--porcelain'),
      'base_pilot_commit':'4532e19f29cfcf318f8afbe7c25850f29981803e',
      'legacy_scientific_source_hash':legacy_hash(),'h1_source_hash':source_hash(),
      'legacy_hash_matches_reviewed_pilot':legacy_hash()=='9a1fa6f48f04125143569c062b839859c820c61df17ba90df3487ee3389ae3d0',
      'already_existing':['physical checkpoint and pending caused events','exact prefix/phase audits','sample covariance and independent-pair estimator','finite-M corrected leave-one-out oracle'],
      'new':['shared physical-time TargetSpec','explicit active job execution','matched context/future scoring','finite-replication/finite-rate/scale diagnostics'],
      'legacy_protocol':'Existing experiments.eeg_representation.cli/configs remain the separately named legacy_filtered_stream_v1 benchmark; no backward timestamp shift',
      'pilot':{'root':str(pilot_root) if pilot_root else None,'status':'not_inspected'}}
    if pilot_root:
        root=Path(pilot_root)
        if not (root/'manifest.json').exists():result['pilot']['status']='not_accessible_or_missing'
        else:
            original=read_json(root/'provenance.json');records=read_json(root/'manifest.json')['trials'];missing=[];verified=0
            for r in records:
                p=root/r['path']
                if not p.is_file():missing.append(r['path'])
                elif verify_raw:
                    if file_hash(p)!=r['sha256']:raise ValueError('Pilot native checksum mismatch '+str(p))
                    verified+=1
            derived=[]
            for manifest in (root/'derived').glob('*/manifest.json'):
                for r in read_json(manifest)['trials']:
                    p=root/r['path']
                    if not p.is_file():derived.append(r['path'])
                    elif verify_raw and file_hash(p)!=r['sha256']:raise ValueError('Derived checksum mismatch '+str(p))
            result['pilot'].update(status='metadata_accessible',recorded_provenance=original,native_records=len(records),missing_native=len(missing),missing_native_paths=missing,
              missing_derived=len(derived),verified_native_hashes=verified,derived_bytes_warning='Legacy derived byte fields were inherited from native rows; hashes, not those sizes, are checked',
              raw_rescoring='not performed by audit; original raw/probe predictions cannot be recreated by copying')
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('h1-audit');a.add_argument('--pilot-root',type=Path);a.add_argument('--verify-raw',action='store_true');a.add_argument('--output-root',type=Path)
    sub.add_parser('doctor')
    for name in ['h1-plan','h1-controls','h1-generate','h1-forks','h1-derive','h1-benchmark','h1-replication','h1-compression','h1-scale','h1-summarize','h1-export']:
        q=sub.add_parser(name);q.add_argument('--config',type=Path,required=True);q.add_argument('--output-root',type=Path,required=True)
        q.add_argument('--resume',action='store_true');q.add_argument('--timing-root',type=Path)
        if name in ('h1-generate','h1-forks'):q.add_argument('--execute-large',action='store_true',help='Explicit opt-in for expanded/confirmatory or fixed-budget campaigns after measured plan')
        if name in ('h1-benchmark','h1-compression'):q.add_argument('--device',choices=['cpu','cuda'],default='cpu')
        if name=='h1-export':q.add_argument('--archive',type=Path,required=True);q.add_argument('--include-native',action='store_true')
    return p


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor':
            from ..cli import doctor
            result=doctor()
        elif args.command=='h1-audit':
            result=audit(args.pilot_root,args.verify_raw)
            if args.output_root:
                dest=args.output_root/'audit.json'
                if dest.exists():raise FileExistsError('Audit output exists; choose a new audit directory')
                atomic_json(dest,result)
        else:
            c=load_config(args.config);root=args.output_root
            planned=None
            if args.command in ('h1-plan','h1-generate','h1-forks'):
                planned=plan(c,args.timing_root)
                large=c['name'] in ('expanded_reference','fixed_budget_extension','confirmation_plan')
                if args.command!='h1-plan' and large and not args.execute_large:raise ValueError('Read the measured cost plan first; expanded/fixed-budget/confirmation generation requires explicit --execute-large')
            with output_lock(root):
                initialize_output(root,c,args.resume)
                if args.command=='h1-plan':
                    atomic_json(root/'plan.json',planned);result=planned
                elif args.command=='h1-controls':
                    done=resume_stage(root,'controls',args.resume)
                    if done:result=read_json(root/'controls/controls.json')
                    else:
                        from .controls import run_controls
                        from .compression import exact_controls
                        from .scale import fft_controls
                        result=run_controls(root/'controls');exact_controls(root/'controls/information');fft_controls(root/'controls/fft')
                        if not result['passed']:raise RuntimeError('Exact-control qualification failed')
                        stage_complete(root,'controls',[p for p in (root/'controls').rglob('*') if p.is_file()],{'passed':True})
                elif args.command in ('h1-generate','h1-forks'):
                    from .dataset import generate
                    atomic_json(root/'plan.json',planned)
                    result=generate(c,root,args.resume,splits=['evaluation'] if args.command=='h1-forks' else ['train','development'])
                elif args.command=='h1-derive':
                    from .dataset import derive
                    result=derive(c,root,args.resume)
                elif args.command=='h1-benchmark':
                    from .benchmark import run_benchmark
                    result=run_benchmark(c,root,args.device,args.resume)
                elif args.command=='h1-replication':
                    from .replication import run_replication
                    result=run_replication(c,root,args.resume)
                elif args.command=='h1-compression':
                    from .compression import run_compression
                    result=run_compression(c,root,args.device,args.resume)
                elif args.command=='h1-scale':
                    from .scale import run_scale
                    result=run_scale(c,root,args.resume)
                elif args.command=='h1-summarize':
                    from .reporting import summarize
                    result=summarize(c,root)
                elif args.command=='h1-export':
                    from .reporting import export_evidence
                    result=export_evidence(c,root,args.archive,args.include_native)
        print(json.dumps(jsonable(result),indent=2,allow_nan=False));return 0
    except (ValueError,FileExistsError,FileNotFoundError,RuntimeError) as exc:
        print(f'H1 ERROR: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
