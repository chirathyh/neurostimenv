"""Quota-checked S1 qualification/discovery submission; dry run by default."""
import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.s1_analysis import load_qualification,code_hashes,latest_suite
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.nci.submit_l23net_r1 import PROJECTS,parse_balance,allocate_projects


def jobs_for_stage(stage):
    if stage=='qualification':
        return [{'name':name,'run':{'seed':8451,'condition':condition,'arm':arm}}
                for name,condition,arm in [('q_reference_sham','reference','sham'),
                                           ('q_mdd_sham','mdd','sham'),('q_mdd_alpha','mdd','fixed_alpha')]]
    jobs=[{'name':f's{seed}_{arm}','run':{'seed':seed,'condition':'mdd','arm':arm}}
          for seed in s0.PROTOCOL['discovery_seeds']
          for arm in ['sham','fixed_theta','fixed_alpha','fixed_low_beta']]
    jobs.append({'name':'s8401_transverse_alpha','run':{'seed':8401,'condition':'mdd','arm':'transverse_alpha'}})
    return jobs


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['qualification','discovery'],required=True)
    parser.add_argument('--s0',type=Path,default=ROOT/'experiments/l23net_analysis/frozen/s0_full_spectrum_v1.json')
    parser.add_argument('--qualification',type=Path)
    parser.add_argument('--qualification-latest',action='store_true')
    parser.add_argument('--max-concurrent',type=int,default=4)
    parser.add_argument('--available-ksu',nargs=3,metavar='PROJECT=KSU')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.max_concurrent<=8:raise ValueError('Concurrency must be 1--8')
    gate=s0.load_gate(args.s0)
    if args.qualification_latest:
        if args.qualification is not None:raise ValueError('Use a qualification path OR --qualification-latest')
        args.qualification=latest_suite('qualification')/'s1_summary.json'
        print(f'QUALIFICATION_REPORT={args.qualification}',flush=True)
    if args.stage=='discovery':
        if args.qualification is None:raise ValueError('Discovery requires --qualification PATH/s1_summary.json')
        load_qualification(args.qualification,gate['sha256'])
    accounts={}
    if args.available_ksu:
        balances={k:float(v) for k,v in (x.split('=',1) for x in args.available_ksu)}
        if set(balances)!=set(PROJECTS) or any(not math.isfinite(v) or v<0 for v in balances.values()):
            raise ValueError('Supply current nonnegative sj53/fa32/ny83 balances')
    else:
        accounts={p:subprocess.check_output(['nci_account','-P',p],text=True) for p in PROJECTS}
        balances={p:parse_balance(v) for p,v in accounts.items()}
    jobs=allocate_projects(jobs_for_stage(args.stage),balances)
    print(json.dumps({'stage':args.stage,'trajectories':len(jobs),'duration_s':60,
        'resources_per_worker':'624 CPUs / 256GB / normal / 02:30:00; one trajectory',
        'maximum_worker_reservation_ksu':len(jobs)*3.12,
        'projects':{p:sum(j['project']==p for j in jobs) for p in PROJECTS},
        'note':'One KSU retained per project. Quota expiry/queue start not guaranteed. No phase-matched arms.'},indent=2))
    if not args.submit:
        print('DRY RUN. Add --submit only after inspecting this plan.');return 0
    for command in [['git','diff','--quiet'],['git','diff','--cached','--quiet']]:
        subprocess.run(command,cwd=ROOT,check=True)
    if not shutil.which('qsub'):raise ValueError('Submit only from NCI')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (ROOT/'results').mkdir(exist_ok=True)
    suite=Path(tempfile.mkdtemp(prefix=f'l23net_s1_{args.stage}_{datetime.now():%Y%m%d_%H%M%S}_',dir=ROOT/'results'))
    shutil.copy2(args.s0,suite/'s0_gate.json')
    if args.qualification:shutil.copy2(args.qualification,suite/'qualification.json')
    manifest={'stage':args.stage,'commit':commit,'code_sha256':code_hashes(),'s0_sha256':gate['sha256'],
              'jobs':jobs,'balances':balances,'account_snapshots':accounts,'max_concurrent':args.max_concurrent,
              'submission_status':'submitting'}
    path=suite/'submission.json';save_json(path,manifest)
    print(f'SUITE_DIRECTORY={suite}',flush=True)
    try:
        for index,job in enumerate(jobs):
            folder=suite/job['name'];folder.mkdir()
            env=f'EXPECTED_COMMIT={commit},SUITE_DIRECTORY={suite},JOB_INDEX={index}'
            cmd=['qsub','-P',job['project'],'-N',f's1_{args.stage[0]}{index:02d}','-v',env,
                 '-o',str(folder/'pbs.out'),'-e',str(folder/'pbs.err')]
            if index>=args.max_concurrent:
                cmd+=['-W','depend=afterok:'+jobs[index-args.max_concurrent]['job_id']]
            job['submission_attempted']=True;save_json(path,manifest)
            cmd.append(str(ROOT/'experiments/l23net_analysis/nci/run_l23net_s1_worker.sh'))
            job['job_id']=subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
            if not re.fullmatch(r'\d+(?:\.[\w.-]+)?',job['job_id']):raise ValueError('Unexpected qsub response; inspect manifest')
            save_json(path,manifest)
            print(f"{job['name']} -> {job['project']}: {job['job_id']}",flush=True)
        manifest['submission_status']='submitted'
    except Exception as exc:
        manifest['submission_status']='partial_failure';manifest['submission_error']=repr(exc)
        save_json(path,manifest)
        print('Submission stopped; inspect this manifest and existing job IDs before retrying.',file=sys.stderr)
        raise
    save_json(path,manifest)
    print(f'After ALL jobs finish: python3 experiments/l23net_analysis/analyze_l23net_s1.py "{suite}"')
    print('Do not update this checkout while jobs are queued/running. Failed dependencies must be inspected/cancelled, not bypassed.')
    return 0


if __name__=='__main__':raise SystemExit(main())
