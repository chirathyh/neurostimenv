"""Plan or submit one L23Net OU-history wave to Gadi project fa32."""

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
from experiments.l23net_analysis import repeatability_protocol as p
from experiments.l23net_analysis.nci.submit_l23net_dose_response import find_reference
from experiments.l23net_analysis.nci.submit_l23net_r1 import parse_balance
from experiments.l23net_analysis.r1_protocol import save_json
from experiments.l23net_analysis.s2_pilot_protocol import PREREQUISITES

WORKER = Path(__file__).with_name('run_l23net_repeatability_worker.sh')
RESERVE_KSU = 1.
HISTORICAL_KSU_PER_TRAJECTORY = 38.80415 / 24


def reservation_for_wave(wave, available_ksu):
    reservation = round(len(p.jobs(wave)) * dose.RESERVATION_KSU, 3)
    required = reservation + RESERVE_KSU
    if not math.isfinite(available_ksu) or available_ksu < required:
        raise ValueError(f'{p.PROJECT} needs {required:.3f} KSU available for wave {wave}, including 1 KSU reserve')
    return reservation


def reference_directory(wave):
    if wave == 2:
        return p.suite_path(1)
    discovery = dose.suite_path('discovery')
    if all((discovery / name).is_file() for name in PREREQUISITES):
        return discovery
    return find_reference()


def worker_command(suite, manifest, index):
    job = manifest['jobs'][index]
    folder = suite / job['name']
    variables = {'SUITE_DIRECTORY': suite, 'WAVE_JOB_NAME': job['name'], 'STRUCTURE_SEED': job['seed'],
                 'HISTORY_SEED': job['history_seed'], 'WAVE_ARM': job['arm'], 'EXPECTED_COMMIT': manifest['commit']}
    variable_list = ','.join(f'{key}={value}' for key, value in variables.items())
    resources = dose.RESOURCES
    resource_list = (f"ncpus={resources['ncpus']},mem={resources['memory_gb']}GB,walltime={resources['walltime']},"
                     'jobfs=10GB,storage=gdata/ny83')
    command = ['qsub', '-P', p.PROJECT, '-q', resources['queue'], '-l', resource_list,
               '-N', f"l23rpt{manifest['wave']}_{index:02d}", '-v', variable_list,
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
    parser.add_argument('--wave', choices=p.WAVES, type=int, default=1)
    parser.add_argument('--reference', type=Path, help='Directory containing the frozen S0/Reference60 artifacts')
    parser.add_argument('--available-ksu', type=float, help='Offline balance for a dry run')
    parser.add_argument('--max-concurrent', type=int, default=8)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args(argv)
    jobs = p.jobs(args.wave)
    if not 1 <= args.max_concurrent <= len(jobs):
        parser.error(f'--max-concurrent must be 1–{len(jobs)}')
    if args.submit and args.available_ksu is not None:
        parser.error('--available-ksu is for dry runs. Submission queries live fa32 accounting.')

    p.load_archive()
    first_path = p.suite_path(1) / p.SUMMARY
    first = p.load_first_wave(first_path) if args.wave == 2 else None
    suite = p.suite_path(args.wave)
    if any(char in str(suite) for char in ',\n\r'):
        raise ValueError('The suite path contains a delimiter unsupported by qsub -v')
    reference = (args.reference or reference_directory(args.wave)).resolve()
    gate, _, _ = dose.load_reference(reference)
    account = None
    if args.available_ksu is None:
        account = subprocess.check_output(['nci_account', '-P', p.PROJECT], text=True)
        available_ksu = parse_balance(account)
    else:
        available_ksu = args.available_ksu
    reservation = reservation_for_wave(args.wave, available_ksu)
    print(json.dumps({'wave': args.wave, 'project': p.PROJECT, 'available_ksu': available_ksu,
                      'workers': len(jobs), 'max_concurrent': args.max_concurrent, 'resources': dose.RESOURCES,
                      'expected_at_historical_runtime_ksu': round(len(jobs) * HISTORICAL_KSU_PER_TRAJECTORY, 2),
                      'maximum_wave_reservation_ksu': reservation, 'kept_unreserved_ksu': RESERVE_KSU,
                      'source_reference': str(reference), 'suite': str(suite), 'suite_exists': suite.exists()},
                     indent=2), flush=True)
    if not args.submit:
        print('Dry run. Add --submit when ready to submit this wave.')
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
    if first is not None and first['commit'] != commit:
        raise ValueError('Wave 2 must use the checkout that completed wave 1')
    manifest = {'wave': args.wave, 'protocol': p.PROTOCOL, 'commit': commit, 'code_sha256': p.code_hashes(),
                'mechanism_sha256': dose.mechanism_hashes(), 's0_sha256': gate['sha256'], 'jobs': jobs,
                'max_concurrent': args.max_concurrent, 'available_ksu': available_ksu,
                'reserved_ksu': reservation, 'account_snapshot': account, 'submission_status': 'submitting',
                'reference_source': str(reference), 'history1_sha256': first['sha256'] if first else None}
    suite.parent.mkdir(exist_ok=True)
    try:
        suite.mkdir()
    except FileExistsError as exc:
        raise ValueError(f'Wave {args.wave} already has a suite at {suite}. Inspect existing job IDs.') from exc
    for name in PREREQUISITES:
        shutil.copy2(reference / name, suite / name)
    if first is not None:
        shutil.copy2(first_path, suite / p.PREVIOUS_WAVE)
    submit_jobs(suite, manifest)
    print(f'After all workers exit: python3 experiments/l23net_analysis/analyze_l23net_repeatability.py --wave {args.wave}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
