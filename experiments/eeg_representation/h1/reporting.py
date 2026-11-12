"""Reaggregate numeric H1 scores and export a self-contained evidence package."""
from pathlib import Path
import csv,json,zipfile,hashlib,subprocess,collections,sys,argparse
import numpy as np
from .artifacts import atomic_json,read_json,artifact_record,file_hash,source_hash,ROOT
from .scoring import circuit_summary
from .targets import canonical_hash


def _safe_path(root, relative):
    relative=Path(relative)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe evidence artifact path')
    return Path(root)/relative


def reaggregate_numeric(root):
    """Standalone benchmark rescoring: no original checkout, Git or pickle needed.

    Recompute MSE from saved truth/prediction arrays, verify all available context
    score aggregates, and retain circuit weighting. Other proper-score components
    are reaggregated as saved; this is not a new distribution fit.
    """
    root=Path(root);rows=[];checks=[];index=root/'benchmarks/manifest.json'
    if not index.exists():return dict(status='not_run',rows=rows,checks=checks)
    inventory=read_json(index)
    if inventory.get('schema_version')!='eegrepr-h1-benchmark-index-v2':raise ValueError('Stale benchmark index')
    for entry in inventory['jobs']:
        report_path=_safe_path(root,entry['path']);report=read_json(report_path);base=report_path.parent
        if report.get('schema_version')!='eegrepr-h1-benchmark-v2':raise ValueError('Stale benchmark summary')
        for item in report['models']:
            path=_safe_path(base,item['prediction_file'])
            if report['artifacts'].get(str(path.relative_to(base)))!=file_hash(path):raise ValueError('Stale/corrupt raw score arrays')
            with np.load(path,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
            if str(raw['target_fingerprint'])!=report['target_fingerprint']:raise ValueError('Raw target fingerprint mismatch')
            truth=np.asarray(raw['truth'],float);mean=np.asarray(raw['mean'],float);mask=np.asarray(raw['future_mask'],bool)
            if truth.ndim!=3 or mean.shape!=(len(truth),truth.shape[-1]) or mask.shape!=truth.shape[:2] or np.any(mask.sum(1)<1):raise ValueError('Invalid raw score shapes')
            if not np.isfinite(truth[mask]).all() or not np.isfinite(mean).all():raise ValueError('Nonfinite raw predictions/targets')
            mse=((truth-mean[:,None])**2).mean(-1)
            actual=np.where(mask,mse,0).sum(1)/mask.sum(1)
            if not np.allclose(actual,raw['context__mean_risk'],rtol=1e-9,atol=1e-12):raise ValueError('Saved MSE does not match numeric predictions/targets')
            values={k.removeprefix('context__'):v for k,v in raw.items() if k.startswith('context__')}
            fresh=circuit_summary(values,raw['structure_id'],bootstrap=0)
            for metric,value in fresh['summary'].items():
                old=item['summary'][metric]['mean'];new=value['mean']
                if (old is None)!=(new is None) or (old is not None and not np.isclose(old,new,rtol=1e-10,atol=1e-14)):raise ValueError('Raw score reaggregation mismatch '+metric)
            checks.append({'job':entry['job_id'],'model':item['model'],'raw_scores_reaggregate':True,'mse_recomputed_from_truth_and_mean':True})
            for circuit in fresh['per_circuit']:
                rows.append({'job':entry['job_id'],'model':item['model'],**{k:v for k,v in circuit.items() if not k.endswith('_available_contexts')}})
    return dict(status='complete',rows=rows,checks=checks,
      scope='MSE recomputed from raw truth/mean; all saved context score components reaggregated with equal circuit weighting; no model refit')


def _qualifications(value, path='$'):
    """Collect explicit status records, not infer scientific success from presence."""
    records=[]
    if isinstance(value,dict):
        for key,item in value.items():
            location=path+'.'+str(key)
            if (key=='status' or key.endswith('_status')) and isinstance(item,str):
                text=item.lower()
                category=next((word for word in ('failed','pending','unsupported','unavailable','disabled','not_run') if word in text),None)
                if category is None and ('not run' in text or 'not computed' in text):category='not_run'
                if category:records.append(dict(category=category,path=location,detail=item))
            elif key in ('passed','within_tolerance') and item is False:
                records.append(dict(category='failed_diagnostic',path=location,detail=f'{key}=false; execution success does not establish qualification'))
            records.extend(_qualifications(item,location))
    elif isinstance(value,list):
        for i,item in enumerate(value):records.extend(_qualifications(item,f'{path}[{i}]'))
    return records


def phase_summaries(config,root):
    root=Path(root);statuses={};expected=len(config.get('jobs',[]))
    locations={'P0_targets':'panels/manifest.json','P1_exact_controls':'controls/controls.json','P1_matched_benchmark':'benchmarks/manifest.json',
      'P2_replication':'replication/summary.json','P3_compression':'compression/summary.json','P4_scale':'scale/summary.json'}
    for phase,rel in locations.items():
        path=root/rel
        if not path.exists():
            statuses[phase]={'status':'not_run','execution_status':'not_run','path':rel};continue
        payload=read_json(path);qualifications=_qualifications(payload)
        if phase=='P1_matched_benchmark':
            for job in payload.get('jobs',[]):
                child=_safe_path(root,job['path'])
                if not child.exists():qualifications.append(dict(category='failed',path=job['path'],detail='Recorded benchmark job artifact missing'))
                else:qualifications.extend(_qualifications(read_json(child),job['path']))
        jobs=payload.get('executed_jobs',payload.get('jobs',[]))
        completed=payload.get('completed_jobs',len(jobs) if isinstance(jobs,list) else None)
        if phase!='P1_exact_controls' and completed is not None and completed<expected:
            qualifications.append(dict(category='pending',path=rel,detail=f'{completed}/{expected} configured jobs represented'))
        if phase=='P1_exact_controls' and payload.get('passed') is not True:
            qualifications.append(dict(category='failed',path=rel,detail='Exact controls have not explicitly passed'))
        counts=dict(collections.Counter(r['category'] for r in qualifications))
        execution='failed' if counts.get('failed') else ('partial' if completed is not None and completed<expected and phase!='P1_exact_controls' else 'complete')
        statuses[phase]=dict(status='complete_with_qualifications' if execution=='complete' and qualifications else execution,
            execution_status=execution,path=rel,recorded_status=payload.get('status'),completed_jobs=completed,
            qualification_status='inspect_pending_unsupported_or_failed_items' if qualifications else 'no_explicit_qualification_failures_recorded',
            status_record_counts=counts,qualification_examples=qualifications[:100],qualification_examples_omitted=max(0,len(qualifications)-100),
            counts_scope='explicit nested status/diagnostic records; these counts are not independent contexts/circuits or a scientific pass')
    return statuses


def summarize(config,root):
    root=Path(root);directory=root/'summary';directory.mkdir(exist_ok=True)
    rescored=reaggregate_numeric(root);rows=rescored['rows'];checks=rescored['checks']
    path=directory/'per_circuit_scores.csv'
    if rows:
        fields=sorted({k for r in rows for k in r})
        with path.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    statuses=phase_summaries(config,root)
    report={'schema_version':'eegrepr-h1-summary-v2','source_hash':source_hash(),'config_hash':canonical_hash(config),
      'name':config['name'],'claim_status':config['claim_status'],'backend':config['backend'],'phases':statuses,
      'raw_reaggregation_checks':checks,'n_per_circuit_model_rows':len(rows),
      'identifiability_statement_is_theoretical_not_execution_evidence':True,
      'identifiable':['state-conditioned innovation dispersion for declared target','frozen predictor risk on identical contexts/futures','combined observation/compression/fitting excess with finite-M correction'],
      'not_identified':['separate nonlinear observation/compression/fitting gaps','true EEG-history Bayes floor without a validated posterior-state model','human biological prediction thresholds'],
      'H2':'task-supervised/progressive comparison remains work in progress; no new full H2 leaderboard',
      'statistical_unit':'circuit structure; futures, views, overlapping histories and optimizer seeds are repeats'}
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        jobs=sorted({r['job'] for r in rows})
        for job in jobs:
            subset=[r for r in rows if r['job']==job];names=sorted({r['model'] for r in subset})
            fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
            for i,name in enumerate(names):
                group=[r for r in subset if r['model']==name]
                axes[0].scatter([i]*len(group),[r['risk_over_r0'] for r in group],s=28)
                axes[1].scatter([i]*len(group),[r['corrected_excess'] for r in group],s=28)
            axes[0].axhline(1,color='black',ls='--',lw=1);axes[1].axhline(0,color='black',ls='--',lw=1)
            for ax in axes:ax.set_xticks(range(len(names)),names,rotation=45,ha='right',fontsize=7)
            axes[0].set_ylabel('Model risk / training-mean test risk');axes[1].set_ylabel('Bias-corrected combined excess (standardized)')
            fig.suptitle(f'{job}: individual circuit means; {config["claim_status"]}, {config["backend"]}')
            for ext in ['png','pdf']:fig.savefig(directory/f'{job}_matched_risk.{ext}',dpi=180)
            plt.close(fig)
    atomic_json(directory/'summary.json',report)
    (directory/'README.txt').write_text('H1 EVIDENCE STATUS\n\n'+json.dumps(report,indent=2)+'\n\nPaper mapping: H1a exact controls + matched_risk figures; H1b replication curves; H1c compression/rate_distortion_task_surface; H1d scale/executed_scale_table. All JSON/NPZ/CSV records retain actual executed job IDs. Original pilot evidence and new engineering outputs are separate. Inspect per-phase pending/unsupported entries; file presence is not scientific qualification.\n')
    return report


def _source_snapshot_files():
    """Tracked source/config/docs/dependencies plus current untracked H1 work.

    Include the simulator/source-hash dependency closure, not only the H1 folder.
    External installed packages and generated data/binaries are not vendored.
    """
    allowed_roots={'experiments','env','setup','configs','agent','utils','visualiser','tests','scripts','requirements','docs'}
    suffixes={'.py','.hoc','.mod','.yaml','.yml','.toml','.cfg','.ini','.sh','.md','.rst','.txt','.json'}
    try:
        tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0')
    except (OSError,subprocess.CalledProcessError):
        tracked=[p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()]
    paths=set()
    for value in tracked:
        if not value:continue
        p=Path(value)
        root_file=len(p.parts)==1 and (p.suffix.lower() in {'.md','.txt','.toml','.cfg','.ini','.yml','.yaml'} or p.name.lower().startswith(('license','copying')) or p.name in {'.gitignore','.gitattributes'})
        source_file=p.parts[0] in allowed_roots and p.suffix.lower() in suffixes
        if p.suffix.lower()=='.json' and p.parts[0] not in {'configs','docs'}:source_file=False
        if root_file or source_file:paths.add(ROOT/p)
    additions=list((ROOT/'experiments/eeg_representation/h1').rglob('*.py'))+list((ROOT/'configs/eeg_representation/h1').rglob('*.yaml'))+list((ROOT/'configs/eeg_representation/h1').glob('*.json'))
    additions+=list((ROOT/'tests').glob('test_eegrepr_h1*.py'))+list((ROOT/'docs').glob('eegrepr_h1*'))+[ROOT/'scripts/eegrepr/run_h1.sh']
    paths.update(p for p in additions if p.is_file())
    # This exact closure must agree with artifacts.source_hash, including untracked
    # scientific files. Other tracked source above supplies imports/runtime setup.
    paths.update((ROOT/'experiments/eeg_representation').rglob('*.py'))
    paths.update((ROOT/'setup/circuits/ballnstick').glob('*.py'))
    paths.update((ROOT/'setup/circuits/ballnstick').glob('*.hoc'))
    paths.update((ROOT/'env/models/neuron').glob('*.py'))
    paths.add(ROOT/'configs/env/ballnstick.yaml')
    result=[]
    for p in sorted(paths):
        if p.is_symlink():raise ValueError('Symlink in source snapshot')
        if p.is_file():result.append(p)
    return result


def _stream_sha256(stream):
    digest=hashlib.sha256()
    for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def export_evidence(config,root,output,include_native=False):
    root=Path(root).resolve();output=Path(output).resolve();sidecar=Path(str(output)+'.sha256')
    if output.exists() or sidecar.exists():raise FileExistsError('Refusing to replace evidence archive/checksum')
    output.parent.mkdir(parents=True,exist_ok=True)
    if root==output or root in output.parents:raise ValueError('Evidence archive must be outside its input root')
    records=[];files=[]
    for p in sorted(root.rglob('*')):
        if p.is_symlink():raise ValueError('Symlink in evidence tree')
        if not p.is_file() or p.name=='.eegrepr.lock':continue
        rel=p.relative_to(root)
        if not include_native and rel.parts[0]=='native' and p.suffix=='.h5':continue
        files.append(p);records.append(artifact_record(root,p))
    scientific_hash=source_hash();source_files=_source_snapshot_files()
    source_records=[{'path':'source/'+p.relative_to(ROOT).as_posix(),'sha256':file_hash(p),'bytes':p.stat().st_size} for p in source_files]
    paths=[r['path'] for r in records+source_records]+['EVIDENCE_MANIFEST.json','RESCORE_README.txt']
    if len(set(paths))!=len(paths):raise ValueError('Evidence/source archive path collision')
    manifest={'schema_version':'eegrepr-h1-export-v2','source_hash':scientific_hash,'config_hash':canonical_hash(config),
      'includes_native_hdf5':include_native,'includes_all_existing_result_files_except_declared_omissions':True,
      'phase_status':phase_summaries(config,root),
      'source_scope':'tracked code/config/docs/license/requirements plus current H1 files and complete scientific source-hash closure; installed external packages are not bundled',
      'omission':'native raw simulation HDF5 intentionally excluded from compact export; their manifests/hashes are included' if not include_native else None,
      'files':records,'source_files':source_records}
    instructions=(
      'H1 NUMERIC RESCORING\n\n'
      'This package records actual phase status; file presence is not scientific qualification.\n'
      'Extract it into a new directory. Use a Python environment with numpy and scipy installed\n'
      '(see source/requirements* and saved provenance versions for the wider study environment).\n'
      'From the extracted directory run:\n'
      '  PYTHONPATH=source python -m experiments.eeg_representation.h1.reporting --rescore-root . --output independent_rescore.json\n\n'
      'This recomputes benchmark MSE from numeric truth/means and verifies saved per-circuit score\n'
      'aggregates. It needs no Git repository, NEURON, model pickle, or new model fitting. Other\n'
      'proper-score components are reaggregated as saved. Missing benchmark evidence is reported\n'
      'as not_run. Existing output files are never replaced. Native HDF5 is omitted by default;\n'
      'full simulator reruns require native dependencies and the study workstation setup.\n')
    generated_records=[{'path':'RESCORE_README.txt','bytes':len(instructions.encode()),'sha256':hashlib.sha256(instructions.encode()).hexdigest()}]
    manifest['generated_files']=generated_records
    partial=output.with_suffix(output.suffix+'.partial')
    if partial.exists():raise FileExistsError(partial)
    try:
        with zipfile.ZipFile(partial,'w',zipfile.ZIP_DEFLATED,compresslevel=4,allowZip64=True) as z:
            for p in files:z.write(p,arcname=p.relative_to(root).as_posix())
            for p in source_files:z.write(p,arcname='source/'+p.relative_to(ROOT).as_posix())
            z.writestr('EVIDENCE_MANIFEST.json',json.dumps(manifest,indent=2))
            z.writestr('RESCORE_README.txt',instructions)
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:raise ValueError('ZIP integrity failure')
            for r in records+source_records+generated_records:
                with z.open(r['path']) as stream:
                    if _stream_sha256(stream)!=r['sha256']:raise ValueError('ZIP checksum mismatch')
        if source_hash()!=scientific_hash:raise ValueError('Scientific source changed during export')
        import os
        os.link(partial,output);partial.unlink()
    finally:
        if partial.exists():partial.unlink()
    digest=file_hash(output)
    with sidecar.open('x') as stream:stream.write(digest+'  '+output.name+'\n')
    return {'path':str(output),'bytes':output.stat().st_size,'sha256':digest,'files':len(files),'source_files':len(source_files),'includes_native':include_native}


def main(argv=None):
    parser=argparse.ArgumentParser(description='Standalone numeric H1 benchmark rescoring; no original checkout/Git/model pickle required')
    parser.add_argument('--rescore-root',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(argv)
    if args.output.exists():parser.error('Output exists; choose a new independent rescoring file')
    try:
        result=reaggregate_numeric(args.rescore_root);atomic_json(args.output,result)
    except (ValueError,OSError,KeyError) as exc:
        print('H1 rescore error: '+str(exc),file=sys.stderr);return 2
    print(json.dumps({'status':result['status'],'models_checked':len(result['checks']),'output':str(args.output)},indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
