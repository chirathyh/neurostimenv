"""Plan or submit one preplanned L23Net theta frequency wave on Gadi."""

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

from experiments.l23net_analysis import dose_response_protocol as dose
from experiments.l23net_analysis import theta_frequency_protocol as p
from experiments.l23net_analysis.nci.submit_l23net_r1 import parse_balance
from experiments.l23net_analysis.nci.submit_l23net_repeatability import reference_directory
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.s2_pilot_protocol import PREREQUISITES

WORKER = Path(__file__).with_name('run_l23net_theta_frequency_worker.sh')
RESERVE_KSU = 1.
HISTORICAL_KSU_PER_RUN = 28.81356 / 18


def reservation_for_history(history, balance):
    reservation = round(len(p.jobs(history)) * dose.RESERVATION_KSU, 3)
    required = reservation + RESERVE_KSU
    if not math.isfinite(balance) or balance < required:
        raise ValueError(f'{p.PROJECT} needs {required:.3f} KSU available for history {history}, including 1 KSU reserve')
    return reservation


def worker_command(suite, manifest, index):
    job = manifest['jobs'][index]
    folder = suite / job['name']
    variables = {'SUITE_DIRECTORY': suite, 'THETA_JOB_NAME': job['name'], 'STRUCTURE_SEED': job['seed'],
                 'HISTORY_SEED': job['history_seed'], 'THETA_ARM': job['arm'], 'EXPECTED_COMMIT': manifest['commit']}
    options = ','.join(f'{key}={value}' for key, value in variables.items())
    resource = dose.RESOURCES
    request = (f"ncpus={resource['ncpus']},mem={resource['memory_gb']}GB,walltime={resource['walltime']},"
               'jobfs=10GB,storage=gdata/ny83')
    command = ['qsub', '-P', p.PROJECT, '-q', resource['queue'], '-l', request,
               '-N', f"l23th{manifest['history']}_{index:02d}", '-v', options,
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
        print(f'Inspect {path} and qstat before recovering a partial submission.', file=sys.stderr)
        raise
    manifest['submission_status'] = 'submitted'
    save_json(path, manifest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', choices=p.HISTORIES, type=int, default=1)
    parser.add_argument('--reference', type=Path, help='Directory containing the frozen S0/Reference60 artifacts')
    parser.add_argument('--available-ksu', type=float, help='Offline balance for a dry run')
    parser.add_argument('--max-concurrent', type=int, default=8)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    jobs = p.jobs(args.history)
    if not 1 <= args.max_concurrent <= len(jobs):
        parser.error(f'--max-concurrent must be 1–{len(jobs)}')
    if args.submit and args.available_ksu is not None:
        parser.error('Submission uses the live fa32 balance. --available-ksu is for dry runs.')

    first_path = p.suite_path(1) / p.SUMMARY
    first = p.load_first_history(first_path) if args.history == 2 else None
    suite = p.suite_path(args.history)
    if any(char in str(suite) for char in ',\n\r'):
        raise ValueError('Suite path contains a delimiter unsupported by qsub -v')
    source = (args.reference or (p.suite_path(1) if first is not None else reference_directory(1))).resolve()
    gate, _, _ = dose.load_reference(source)
    if set(p.ARMS) - set(gate['allowed_arms']):
        raise ValueError('One of the fixed frequencies is missing from S0 qualification')
    account = None
    if args.available_ksu is None:
        account = subprocess.check_output(['nci_account', '-P', p.PROJECT], text=True)
        try:
            balance = parse_balance(account)
        except ValueError as exc:
            raise ValueError('Could not parse the live fa32 balance. Save nci_account -P fa32 for diagnosis.') from exc
    else:
        balance = args.available_ksu
    reservation = reservation_for_history(args.history, balance)
    print(json.dumps({'history': args.history, 'project': p.PROJECT, 'available_ksu': balance,
                      'workers': len(jobs), 'max_concurrent': args.max_concurrent, 'resources': dose.RESOURCES,
                      'expected_ksu_this_wave': round(len(jobs) * HISTORICAL_KSU_PER_RUN, 2),
                      'max_ksu_this_wave': reservation, 'reserve_ksu': RESERVE_KSU,
                      'source_reference': str(source), 'suite': str(suite), 'suite_exists': suite.exists()},
                     indent=2), flush=True)
    if not args.submit:
        print('Dry run. Add --submit when ready to submit this wave.')
        return 0

    if not shutil.which('qsub'):
        raise ValueError('Run submission on a Gadi login node')
    expected = gate['reference_contract']['package_versions']
    installed = {name: importlib.metadata.version(name) for name in expected}
    if installed != expected:
        raise ValueError(f'Environment differs from Reference60: expected {expected}, installed {installed}')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--quiet'], cwd=ROOT, check=True)
    subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT, check=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if first is not None and commit != first['commit']:
        raise ValueError('Use the same checkout for both history waves')
    manifest = {'history': args.history, 'protocol': p.PROTOCOL, 'commit': commit, 'code_sha256': p.code_hashes(),
                'mechanism_sha256': dose.mechanism_hashes(), 's0_sha256': gate['sha256'], 'jobs': jobs,
                'max_concurrent': args.max_concurrent, 'available_ksu': balance, 'reserved_ksu': reservation,
                'account_snapshot': account, 'submission_status': 'submitting', 'reference_source': str(source),
                'history1_sha256': first['sha256'] if first else None}
    suite.parent.mkdir(exist_ok=True)
    try:
        suite.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'History {args.history} already has a suite at {suite}. Inspect its recorded job IDs.') from exc
    for name in PREREQUISITES:
        shutil.copy2(source / name, suite / name)
    if first is not None:
        shutil.copy2(first_path, suite / p.FIRST_HISTORY)
    submit_jobs(suite, manifest)
    print(f'After all workers exit: python3 experiments/l23net_analysis/analyze_l23net_theta_frequency.py --history {args.history}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
