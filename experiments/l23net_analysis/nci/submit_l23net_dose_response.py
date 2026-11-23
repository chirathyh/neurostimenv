"""Plan or submit a gated L23Net dose-response stage on NCI project fa32."""

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from experiments.l23net_analysis import dose_response_protocol as p
from experiments.l23net_analysis.nci.submit_l23net_r1 import parse_balance
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.s2_pilot_protocol import PREREQUISITES

WORKER = Path(__file__).with_name('run_l23net_dose_response_worker.sh')
PROJECT_RESERVE_KSU = 1.
HISTORICAL_KSU_PER_TRAJECTORY = 1.645


def check_budget(stage, available_ksu):
    count = len(p.jobs(stage))
    reservation = round(count * p.RESERVATION_KSU, 3)
    if not math.isfinite(available_ksu) or available_ksu < reservation + PROJECT_RESERVE_KSU:
        raise ValueError(f'{p.PROJECT} needs {reservation + PROJECT_RESERVE_KSU:.3f} KSU available for {stage}, including 1 KSU reserve')
    return reservation


def find_reference():
    results = ROOT / 'results'
    paths = [results / p.REFERENCE_SUITE_NAME, results / 'reference60_results' / p.REFERENCE_SUITE_NAME]
    paths.extend(sorted(results.glob('l23net_s2_pilot_*')))
    for path in paths:
        if all((path / name).is_file() for name in PREREQUISITES):
            return path
    raise FileNotFoundError('Completed Reference60 artifacts not found. Supply --reference with the original suite directory.')


def worker_command(suite, manifest, index):
    job = manifest['jobs'][index]
    folder = suite / job['name']
    variables = {'SUITE_DIRECTORY': suite, 'DOSE_STAGE': manifest['stage'], 'DOSE_JOB_NAME': job['name'],
                 'DOSE_SEED': job['seed'], 'DOSE_ARM': job['arm'], 'EXPECTED_COMMIT': manifest['commit']}
    variable_list = ','.join(f'{key}={value}' for key, value in variables.items())
    resources = p.RESOURCES
    resource_list = (f"ncpus={resources['ncpus']},mem={resources['memory_gb']}GB,walltime={resources['walltime']},"
                     'jobfs=10GB,storage=gdata/ny83')
    command = ['qsub', '-P', p.PROJECT, '-q', resources['queue'], '-l', resource_list,
               '-N', f"l23d_{manifest['stage'][0]}{index:02d}", '-v', variable_list,
               '-o', str(folder / 'pbs.out'), '-e', str(folder / 'pbs.err')]
    previous = index - manifest['max_concurrent']
    if previous >= 0:
        command.extend(['-W', f"depend=afterok:{manifest['jobs'][previous]['job_id']}"])
    return [*command, str(WORKER)]


def submit_jobs(suite, manifest):
    path = suite / 'submission.json'
    save_json(path, manifest)
    print(f'SUITE_DIRECTORY={suite}', flush=True)
    try:
        for index, job in enumerate(manifest['jobs']):
            (suite / job['name']).mkdir()
            job['submission_attempted'] = True
            save_json(path, manifest)
            job['job_id'] = subprocess.check_output(worker_command(suite, manifest, index), cwd=ROOT, text=True).strip()
            if not re.fullmatch(r'\d+(?:\.[\w.-]+)?', job['job_id']):
                raise ValueError('Unexpected qsub response')
            save_json(path, manifest)
            print(f"{job['name']} -> {job['job_id']}", flush=True)
    except (Exception, KeyboardInterrupt) as exc:
        manifest.update(submission_status='partial_failure', submission_error=repr(exc))
        save_json(path, manifest)
        print(f'Inspect {path} and qstat before retrying a partial submission.', file=sys.stderr)
        raise
    manifest['submission_status'] = 'submitted'
    save_json(path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=p.STAGES, default='qualification')
    parser.add_argument('--reference', type=Path, help='Completed Reference60 directory, located automatically by default')
    parser.add_argument('--available-ksu', type=float, help='Balance for an offline dry run. Submission checks the live fa32 balance.')
    parser.add_argument('--max-concurrent', type=int, default=8, help='Maximum active workers per stage, default 8')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    if not 1 <= args.max_concurrent <= len(p.jobs('discovery')):
        parser.error('--max-concurrent must be 1–24')
    if args.submit and args.available_ksu is not None:
        parser.error('--available-ksu is for dry runs. Submission queries live fa32 accounting.')

    stage = args.stage
    suite = p.suite_path(stage)
    if any(char in str(suite) for char in ',\n\r'):
        raise ValueError('The suite path contains a delimiter unsupported by qsub -v')
    qualification = None
    if stage == 'discovery':
        qualification_dir = p.suite_path('qualification')
        report = qualification_dir / p.UPPER_GATE
        if not report.is_file():
            raise FileNotFoundError(f'Run the qualification analyzer after both jobs exit. It must produce {report}.')
        qualification = p.load_upper_gate(report)
        reference = (args.reference or qualification_dir).resolve()
    else:
        reference = (args.reference or find_reference()).resolve()
    gate, _, _ = p.load_reference(reference)
    account = None
    if args.available_ksu is None:
        account = subprocess.check_output(['nci_account', '-P', p.PROJECT], text=True)
        try:
            available_ksu = parse_balance(account)
        except ValueError as exc:
            raise ValueError('Could not parse the live fa32 balance. Save the output of nci_account -P fa32 for diagnosis.') from exc
    else:
        available_ksu = args.available_ksu
    reservation = check_budget(stage, available_ksu)
    jobs = p.jobs(stage)
    concurrency = min(args.max_concurrent, len(jobs))
    print(json.dumps({'stage': stage, 'project': p.PROJECT, 'available_ksu': available_ksu,
                      'worker_count': len(jobs), 'max_concurrent': concurrency, 'resources': p.RESOURCES,
                      'max_stage_reservation_ksu': reservation, 'kept_unreserved_ksu': PROJECT_RESERVE_KSU,
                      'expected_at_historical_runtime_ksu': round(len(jobs) * HISTORICAL_KSU_PER_TRAJECTORY, 2),
                      'qualification_plus_discovery_max_ksu': round(sum(len(p.jobs(s)) for s in p.STAGES) * p.RESERVATION_KSU, 3),
                      'source_reference': str(reference), 'suite': str(suite), 'suite_exists': suite.exists()}, indent=2), flush=True)
    if not args.submit:
        print('Dry run. Add --submit when ready to submit this stage.')
        return 0

    if not shutil.which('qsub'):
        raise ValueError('Run submission on the Gadi login node')
    expected = gate['reference_contract']['package_versions']
    installed = {name: importlib.metadata.version(name) for name in expected}
    if installed != expected:
        raise ValueError(f'Simulation environment differs from Reference60: expected {expected}, installed {installed}')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if qualification and qualification['commit'] != commit:
        raise ValueError('Discovery must use the checkout that passed upper-dose qualification')
    manifest = {'stage': stage, 'protocol': p.PROTOCOL, 'commit': commit, 'code_sha256': p.code_hashes(),
                'mechanism_sha256': p.mechanism_hashes(), 's0_sha256': gate['sha256'], 'jobs': jobs,
                'max_concurrent': concurrency, 'available_ksu': available_ksu,
                'reserved_ksu': reservation, 'account_snapshot': account, 'submission_status': 'submitting',
                'reference_source': str(reference)}
    suite.parent.mkdir(exist_ok=True)
    # An exclusive directory creation prevents duplicate launches, even from two login shells.
    try:
        suite.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'This stage already has a suite at {suite}. Inspect submission.json and existing job IDs.') from exc
    for name in PREREQUISITES:
        shutil.copy2(reference / name, suite / name)
    if qualification:
        manifest['upper_qualification_sha256'] = qualification['sha256']
        shutil.copy2(qualification_dir / p.UPPER_GATE, suite / p.UPPER_GATE)
    submit_jobs(suite, manifest)
    print(f'After all jobs exit: python3 experiments/l23net_analysis/analyze_l23net_dose_response.py --stage {stage}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
