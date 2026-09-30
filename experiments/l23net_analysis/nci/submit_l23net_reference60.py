"""Submit exactly 16 Healthy reference runs; quota checked, dry run by default."""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiments.l23net_analysis import reference60_protocol as p
from experiments.l23net_analysis import s0_protocol as s0
from experiments.l23net_analysis.s1_analysis import latest_suite, load_qualification
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.nci.submit_l23net_r1 import PROJECTS, parse_balance


def allocate(balances):
    if set(balances) != set(PROJECTS) or any(not math.isfinite(v) or v < 0 for v in balances.values()):
        raise ValueError('Require finite nonnegative sj53/fa32/ny83 balances')
    remaining = {k: max(0., v-1.) for k,v in balances.items()}
    jobs = p.jobs()
    for job in jobs:
        project = next((k for k in PROJECTS if remaining[k]+1e-12 >= p.RESERVATION_KSU), None)
        if project is None:
            raise ValueError('Insufficient quota for all 16 jobs plus 1 KSU per-project buffer; no jobs submitted')
        job['project'] = project
        remaining[project] -= p.RESERVATION_KSU
    return jobs


def submit_jobs(suite, manifest, max_concurrent):
    """Persist every attempted/submitted job; never blindly retry a partial suite."""
    path = suite/'submission.json'
    save_json(path, manifest)
    print(f'SUITE_DIRECTORY={suite}', flush=True)
    try:
        for index, job in enumerate(manifest['jobs']):
            folder = suite/job['name']; folder.mkdir()
            environment = f"EXPECTED_COMMIT={manifest['commit']},SUITE_DIRECTORY={suite},JOB_INDEX={index}"
            cmd = ['qsub', '-P', job['project'], '-N', f'r60_{job["run"]["seed"]}', '-v', environment,
                   '-o', str(folder/'pbs.out'), '-e', str(folder/'pbs.err')]
            if index >= max_concurrent:
                cmd += ['-W', 'depend=afterok:'+manifest['jobs'][index-max_concurrent]['job_id']]
            cmd.append(str(ROOT/'experiments/l23net_analysis/nci/run_l23net_reference60_worker.sh'))
            job['submission_attempted'] = True; save_json(path, manifest)
            job['job_id'] = subprocess.check_output(cmd, cwd=ROOT, text=True).strip()
            if not re.fullmatch(r'\d+(?:\.[\w.-]+)?', job['job_id']):
                raise ValueError('Unexpected qsub response; inspect manifest before retrying')
            save_json(path, manifest)
            print(f"{job['name']} -> {job['project']}: {job['job_id']}", flush=True)
        manifest['submission_status'] = 'submitted'
    except Exception as exc:
        manifest.update(submission_status='partial_failure', submission_error=repr(exc))
        save_json(path, manifest)
        print('STOP: inspect the manifest and listed job IDs. Do not resubmit the entire suite.', file=sys.stderr)
        raise
    save_json(path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--s0', type=Path, default=ROOT/'experiments/l23net_analysis/frozen/s0_full_spectrum_v1.json')
    parser.add_argument('--qualification', type=Path, help='Default: latest S1 qualification in results')
    parser.add_argument('--core', type=Path, help='Default: locate the exact S0-frozen R1 core')
    parser.add_argument('--available-ksu', nargs=3, metavar='PROJECT=KSU', help='Otherwise query current nci_account balances')
    parser.add_argument('--max-concurrent', type=int, default=16, help='1--16; default one eligible wave, not a start-time guarantee')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.max_concurrent <= 16:
        raise ValueError('Concurrency must be 1--16')
    gate = s0.load_gate(args.s0)
    qualification = args.qualification or latest_suite('qualification')/'s1_summary.json'
    load_qualification(qualification, gate['sha256'])
    core = args.core or p.find_core(gate)
    sources = p.prepare_sources(core, gate)
    if args.available_ksu:
        pairs = [x.split('=',1) for x in args.available_ksu]
        if len({k for k,v in pairs}) != 3:
            raise ValueError('Duplicate quota project')
        balances = {k: float(v) for k,v in pairs}; accounts = {}
    else:
        accounts = {k: subprocess.check_output(['nci_account','-P',k], text=True) for k in PROJECTS}
        balances = {k: parse_balance(v) for k,v in accounts.items()}
    jobs = allocate(balances)
    print(json.dumps({'stage': 'reference60', 'trajectories': 16, 'duration_s': 60, 'max_concurrent': args.max_concurrent,
                      'resources_per_worker': '624 CPUs / 256GB / normal / 02:00:00',
                      'maximum_worker_reservation_ksu': 16*p.RESERVATION_KSU,
                      'estimated_actual_ksu': 25.6,
                      'projects': {k: {'jobs': sum(j['project']==k for j in jobs), 'available_ksu': balances[k],
                                        'maximum_reservation_ksu': sum(j['project']==k for j in jobs)*p.RESERVATION_KSU} for k in PROJECTS},
                      'note': '1 KSU/project buffer. Quota model uses normal 2 SU/core-hour. Queue/deadline not guaranteed. Healthy sham only.'}, indent=2))
    if not args.submit:
        print('DRY RUN: no jobs submitted. Repeat with --submit only after inspecting the plan.'); return 0
    for cmd in (['git','diff','--quiet'], ['git','diff','--cached','--quiet']):
        subprocess.run(cmd, cwd=ROOT, check=True)
    if not shutil.which('qsub'):
        raise ValueError('Submit from NCI only')
    commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    (ROOT/'results').mkdir(exist_ok=True)
    suite = Path(tempfile.mkdtemp(prefix=f'l23net_reference60_{datetime.now():%Y%m%d_%H%M%S}_', dir=ROOT/'results'))
    shutil.copy2(args.s0, suite/'s0_gate.json')
    shutil.copy2(qualification, suite/'qualification.json')
    save_json(suite/'reference_sources.json', sources)
    manifest = {'stage': 'reference60', 'protocol': p.PROTOCOL, 'commit': commit, 'code_sha256': p.code_hashes(),
                's0_sha256': gate['sha256'], 'reference_sources_sha256': sources['sha256'],
                'jobs': jobs, 'balances': balances, 'account_snapshots': accounts, 'core_source': str(core),
                'max_concurrent': args.max_concurrent, 'submission_status': 'submitting'}
    submit_jobs(suite, manifest, args.max_concurrent)
    print('After ALL jobs finish: python3 experiments/l23net_analysis/analyze_l23net_reference60.py --latest')
    print('Keep this checkout/environment unchanged until all jobs finish. Copy the COMPLETE suite afterwards.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
