"""Bounded status output; no scientific libraries or NCI access needed."""
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
manifest = json.loads((root/'submission.json').read_text())
print('SUITE:', root, '\nStage:', manifest['stage'], '\nSubmission:', manifest['submission_status'])
print('job\tproject\tPBS_ID\texit\tphase')
failures = []
for job in manifest['jobs']:
    folder = root/job['name']
    def read(name):
        p = folder/name
        return p.read_text().strip() if p.exists() else 'pending'
    code = read('worker_exit_code.txt')
    print('\t'.join([job['name'], job['project'], job.get('job_id','not_submitted'), code, read('current_phase.txt')]))
    if code not in ('0', 'pending'):
        failures.append(folder)
for folder in failures[:3]:
    for name in ('pbs.err', 'job_live.log', 'heartbeat.tsv'):
        path = folder/name
        if path.exists():
            print('\nFILE:', path, flush=True)
            subprocess.run(['tail', '-n', '25', str(path)], check=False)
    for path in sorted(folder.glob('seed_*/*/failure*.json'))[:2]:
        print(path, path.read_text())
for name in ('r1_summary.md', 'summary_pbs.err'):
    path = root/name
    if path.exists():
        print('\nFILE:', path, flush=True)
        subprocess.run(['tail', '-n', '90', str(path)], check=False)
print('\nCopy the entire suite, including traces and final PBS epilogues, after all jobs finish.')
