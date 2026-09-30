"""Dry-run-first submission of ten frozen sham/14-Hz pairs (20 workers)."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from experiments.l23net_analysis import s2_pilot_protocol as p
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.nci.submit_l23net_r1 import PROJECTS, parse_balance


def allocate(balances):
    if set(balances)!=set(PROJECTS) or any(not math.isfinite(v) or v<0 for v in balances.values()):
        raise ValueError('Require finite nonnegative sj53/fa32/ny83 balances')
    remaining={k:max(0.,v-1.) for k,v in balances.items()}; jobs=p.jobs()
    for job in jobs:
        project=next((k for k in PROJECTS if remaining[k]+1e-10>=p.RESERVATION_KSU),None)
        if project is None:
            raise ValueError('Insufficient full-walltime quota for ALL 20 jobs plus 1 KSU/project; no submission')
        job['project']=project; remaining[project]-=p.RESERVATION_KSU
    return jobs


def submit_jobs(suite,manifest,max_concurrent):
    path=suite/'submission.json'; save_json(path,manifest)
    print('SUITE_DIRECTORY='+str(suite),flush=True)
    try:
        for index,job in enumerate(manifest['jobs']):
            folder=suite/job['name'];folder.mkdir()
            env=f"EXPECTED_COMMIT={manifest['commit']},SUITE_DIRECTORY={suite},JOB_INDEX={index}"
            cmd=['qsub','-P',job['project'],'-N',f's2p_{index:02d}','-v',env,
                 '-o',str(folder/'pbs.out'),'-e',str(folder/'pbs.err')]
            if index>=max_concurrent:
                cmd+=['-W','depend=afterok:'+manifest['jobs'][index-max_concurrent]['job_id']]
            cmd.append(str(ROOT/'experiments/l23net_analysis/nci/run_l23net_s2_pilot_worker.sh'))
            job['submission_attempted']=True; save_json(path,manifest)
            job['job_id']=subprocess.check_output(cmd,cwd=ROOT,text=True).strip()
            if not re.fullmatch(r'\d+(?:\.[\w.-]+)?',job['job_id']):
                raise ValueError('Unexpected qsub response; inspect partial manifest')
            save_json(path,manifest)
            print(f"{job['name']} -> {job['project']}: {job['job_id']}",flush=True)
        manifest['submission_status']='submitted'
    except Exception as exc:
        manifest.update(submission_status='partial_failure',submission_error=repr(exc));save_json(path,manifest)
        print('STOP: inspect recorded job IDs; do not blindly resubmit the suite.',file=sys.stderr)
        raise
    save_json(path,manifest)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,help='Completed Reference60 directory; defaults to latest local NCI suite')
    parser.add_argument('--available-ksu',nargs=3,metavar='PROJECT=KSU',help='Manual fallback using freshly checked balances')
    parser.add_argument('--max-concurrent',type=int,default=20)
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args(argv)
    if not 1<=args.max_concurrent<=20: raise ValueError('Concurrency must be 1--20')
    source=(args.reference or p.latest_reference()).resolve();gate,target=p.load_prerequisites(source)
    if args.available_ksu:
        pairs=[v.split('=',1) for v in args.available_ksu]
        if len({k for k,v in pairs})!=3: raise ValueError('Duplicate quota project')
        balances={k:float(v) for k,v in pairs}; accounts={}
    else:
        accounts={k:subprocess.check_output(['nci_account','-P',k],text=True) for k in PROJECTS}
        balances={k:parse_balance(v) for k,v in accounts.items()}
    jobs=allocate(balances)
    plan={'stage':'s2_pilot','candidate_pairs':10,'trajectories':20,'duration_s':60,
          'max_concurrent':args.max_concurrent,'resources_per_worker':'624 CPUs / 256GB / normal / 01:40:00',
          'maximum_worker_reservation_ksu':20*p.RESERVATION_KSU,'estimated_actual_ksu':32.3,
          'projects':{k:{'jobs':sum(j['project']==k for j in jobs),'available_ksu':balances[k],
                         'maximum_reservation_ksu':round(sum(j['project']==k for j in jobs)*p.RESERVATION_KSU,6)} for k in PROJECTS},
          'note':'1 KSU/project retained. Prior active maximum ~82 min; request 100 min. Queue start/expiry and runtime NOT guaranteed. Fixed pilot, not powered S2.'}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.submit:
        print('DRY RUN: no jobs or directories created. Add --submit after checking this plan.');return 0
    expected=gate['reference_contract']['package_versions']
    versions={name:importlib.metadata.version(name) for name in expected}
    if versions!=expected:
        raise ValueError('Activate the qualified NCI environment before spending quota: '+json.dumps({'expected':expected,'actual':versions}))
    for cmd in (['git','diff','--quiet'],['git','diff','--cached','--quiet']):
        subprocess.run(cmd,cwd=ROOT,check=True)
    if not shutil.which('qsub'): raise ValueError('Submit on NCI only')
    previous=list((ROOT/'results').glob('l23net_s2_pilot_*/submission.json'))
    if previous:
        raise ValueError('An S2-P submission already exists; inspect it rather than duplicating frozen seeds: '+str(previous[0]))
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    (ROOT/'results').mkdir(exist_ok=True)
    suite=Path(tempfile.mkdtemp(prefix=f'l23net_s2_pilot_{datetime.now():%Y%m%d_%H%M%S}_',dir=ROOT/'results'))
    for name in p.PREREQUISITES: shutil.copy2(source/name,suite/name)
    manifest={'stage':'s2_pilot','protocol':p.PROTOCOL,'commit':commit,'code_sha256':p.code_hashes(),
              's0_sha256':gate['sha256'],'target_sha256':target['sha256'], 'reference_source':str(source),
              'jobs':jobs,'balances':balances,'account_snapshots':accounts,'max_concurrent':args.max_concurrent,
              'submission_status':'submitting'}
    submit_jobs(suite,manifest,args.max_concurrent)
    print('After ALL jobs finish: python3 experiments/l23net_analysis/analyze_l23net_s2_pilot.py --latest')
    print('Keep checkout/environment unchanged until jobs finish. Copy the COMPLETE suite, including all PBS logs.')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
