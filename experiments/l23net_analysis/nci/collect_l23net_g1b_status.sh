#!/bin/bash
set -euo pipefail
: "${1:?Supply the suite directory printed by submit_l23net_g1b.sh}"
SUITE_DIRECTORY=$1
python3 - "${SUITE_DIRECTORY}" <<'PY'
import json
from pathlib import Path
import subprocess
import sys
root = Path(sys.argv[1])
print('SUITE:', root)
print((root/'submission.json').read_text())
for seed in (7101, 7102):
    folder = root/f'seed_{seed}'
    print('\nSEED', seed)
    for name in ('current_phase.txt', 'worker_exit_code.txt', 'heartbeat.tsv', 'pbs.err', 'pbs.out', 'job_live.log', 'qstat_at_exit.txt', 'g1b_pair_summary.md'):
        path = folder/name
        if path.exists():
            print('\nFILE:', path)
            subprocess.run(['tail', '-n', '30', str(path)], check=False)
    for condition in ('reference', 'mdd'):
        p = folder/condition/'l23net_g1b_run.json'
        if p.exists():
            d = json.loads(p.read_text())
            print(condition, json.dumps({k:d.get(k) for k in ('status','completed_simulated_ms','errors','failure','performance')}, indent=2))
        failures = sorted((folder/condition).glob('failure*.json'))
        if failures:
            print('Failure reports:', len(failures), '(showing at most three)')
        for failure in failures[:3]:
            print('FAILURE:', failure, failure.read_text())
for name in ('summary_live.log', 'summary_pbs.err', 'g1b_summary.md'):
    p = root/name
    if p.exists():
        print('\nFILE:', p)
        subprocess.run(['tail', '-n', '60', str(p)], check=False)
PY
