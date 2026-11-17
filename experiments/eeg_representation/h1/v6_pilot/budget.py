"""Process-safe, append-only accounting of every attempted pilot NEURON call.

The start record is fsynced immediately before invoking the simulator. Failed,
interrupted and orphaned attempts remain charged; resuming never refunds them.
"""
from contextlib import contextmanager
from pathlib import Path
import fcntl
import json
import os
import time
import uuid
from ..targets import canonical_hash
from ...storage import atomic_json

SCHEMA = 'eegrepr-pilot-invocations-1'
HARD_CAP = 160


@contextmanager
def _lock(root):
    path = Path(root) / 'invocation_ledger'; path.mkdir(parents=True, exist_ok=True)
    with (path / '.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try: yield path
        finally: fcntl.flock(handle, fcntl.LOCK_UN)


def initialize(root, limit=HARD_CAP):
    limit = int(limit)
    if not 1 <= limit <= HARD_CAP:
        raise ValueError('Pilot invocation limit must be between 1 and 160')
    with _lock(root) as path:
        identity = dict(schema_version=SCHEMA, hard_limit=limit,
            accounting='Every attempted native call is charged before simulator entry; failed/interrupted calls and retries count; no refunds')
        marker = path / 'contract.json'
        if marker.exists() and json.loads(marker.read_text()) != identity:
            raise ValueError('Immutable simulation budget changed')
        if not marker.exists(): atomic_json(marker, identity)
    return summary(root)


def summary(root):
    path = Path(root) / 'invocation_ledger'
    contract = json.loads((path/'contract.json').read_text()) if (path/'contract.json').exists() else dict(hard_limit=HARD_CAP)
    starts = [json.loads(p.read_text()) for p in sorted(path.glob('attempt-*.json'))]
    external=json.loads((path/'implementation_debit.json').read_text()) if (path/'implementation_debit.json').exists() else {'count':0,'evidence':None}
    total=len(starts)+int(external['count'])
    if len({r['attempt_id'] for r in starts}) != len(starts): raise ValueError('Duplicate invocation ledger IDs')
    done = {p.stem.removeprefix('result-'): json.loads(p.read_text()) for p in path.glob('result-*.json')}
    stages = {}
    for r in starts: stages[r['stage']] = stages.get(r['stage'], 0) + 1
    finished = [done[r['attempt_id']] for r in starts if r['attempt_id'] in done]
    return dict(schema_version=SCHEMA, hard_limit=contract['hard_limit'], attempted_native_calls=total,local_attempted_native_calls=len(starts),
        implementation_engineering_calls=int(external['count']),implementation_debit_evidence=external['evidence'],
        remaining_native_calls=contract['hard_limit']-total, calls_by_stage=stages,
        successful_calls=sum(r['status']=='complete' for r in finished),
        failed_calls=sum(r['status']=='failed' for r in finished),
        unresolved_calls=len(starts)-len(finished),
        worker_seconds=sum(float(r.get('seconds', 0)) for r in finished),
        interpretation='Invocation count is computational accounting; workers/futures are not independent circuit samples')


def require_capacity(root, additional, purpose):
    with _lock(root):
        current = summary(root)
        if int(additional) > current['remaining_native_calls']:
            raise RuntimeError(f"Pilot hard cap blocks {purpose}: needs {additional} additional calls; "
                f"{current['remaining_native_calls']} remain, including all failed/interrupted attempts. "
                "Use already exported states, reduce the explicitly registered optional requirement, or stop; no automatic expansion.")
    return current


def begin(root, stage, identity):
    """Call directly before a native simulator invocation, including each retry."""
    with _lock(root) as path:
        current = summary(root)
        if current['remaining_native_calls'] <= 0: raise RuntimeError('Pilot hard cap of 160 attempted NEURON calls reached')
        attempt = f"{current['attempted_native_calls']+1:04d}-{uuid.uuid4().hex}"
        record = dict(schema_version=SCHEMA, attempt_id=attempt, stage=str(stage), identity=identity,
            identity_hash=canonical_hash(identity), pid=os.getpid(), started_unix=time.time(),
            charged=True, status='started')
        atomic_json(path/f'attempt-{attempt}.json', record)
        # Durability is part of honest recovery accounting, not a performance hint.
        with (path/f'attempt-{attempt}.json').open('rb') as handle: os.fsync(handle.fileno())
        directory_fd=os.open(path,os.O_RDONLY)
        try:os.fsync(directory_fd)
        finally:os.close(directory_fd)
        return attempt


def finish(root, attempt, *, status, seconds, details=None):
    if status not in ('complete','failed'): raise ValueError('Unknown attempt result')
    with _lock(root) as path:
        if not (path/f'attempt-{attempt}.json').exists(): raise ValueError('Unregistered simulation attempt')
        result = dict(attempt_id=attempt, status=status, seconds=float(seconds),
            finished_unix=time.time(), details=details or {})
        target=path/f'result-{attempt}.json'
        if target.exists(): raise ValueError('Invocation result is immutable')
        atomic_json(target,result)
    return result


def record_implementation_debit(root,count,evidence):
    """Charge independently documented implementation native calls exactly once.

    These are an aggregate historical debit, not invented per-call records. The
    caller supplies the actual retained engineering accounting artifact/hash.
    """
    count=int(count)
    if count<0 or count>HARD_CAP:raise ValueError('Invalid external engineering invocation count')
    record=dict(schema_version=SCHEMA,count=count,evidence=evidence,
        interpretation='Actual earlier implementation engineering NEURON invocations, including failures; aggregate evidence, not fabricated local attempts')
    with _lock(root) as path:
        marker=path/'implementation_debit.json'
        if marker.exists():
            if json.loads(marker.read_text())!=record:raise ValueError('Implementation debit is immutable')
        else:
            current=summary(root)
            if count>current['remaining_native_calls']:raise RuntimeError('Implementation debit plus prior attempts exceeds pilot hard cap')
            atomic_json(marker,record)
    return summary(root)
