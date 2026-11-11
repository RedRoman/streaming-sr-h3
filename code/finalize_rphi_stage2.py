"""Wait once for both finite stage-2 queues, audit, and archive their evidence.

This CPU dependency step neither starts training nor acquires an accelerator.
It exits on upstream failure, timeout, or audit failure. Existing run evidence
is never overwritten. A completed audit still requires a scientific review.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def inspect_states(paths):
    records = {str(p): json.loads(p.read_text()) if p.exists() else {'state': 'not_started'}
               for p in paths}
    states = [r.get('state') for r in records.values()]
    if any(s == 'failed' for s in states):
        return 'failed', records
    if all(s == 'completed' for s in states):
        return 'completed', records
    if any(s not in ('not_started', 'waiting', 'running', 'completed') for s in states):
        raise ValueError('unexpected upstream state')
    return 'waiting', records


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=10800)
    p.add_argument('--poll-seconds', type=int, default=30)
    a = p.parse_args(argv)
    if min(a.wait_seconds, a.poll_seconds) < 1:
        p.error('wait/poll budgets must be positive')
    root = a.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    status = root / 'finalize_status.json'
    with status.open('x') as stream:
        json.dump({'state': 'starting'}, stream)
    queues = [root / (name + '_queue_status.json') for name in ('native', 'controls')]

    def save(state, **fields):
        record = {'state': state, 'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields}
        temp = status.with_suffix('.partial')
        temp.write_text(json.dumps(record, indent=2) + '\n')
        temp.replace(status)
        print(json.dumps(record), flush=True)

    try:
        deadline, last = time.monotonic() + a.wait_seconds, None
        while True:
            state, upstream = inspect_states(queues)
            if state == 'failed':
                save('failed', reason='upstream queue failed', upstream=upstream)
                return 2
            if state == 'completed':
                break
            marker = [(r.get('state'), r.get('step')) for r in upstream.values()]
            if marker != last:
                save('waiting', upstream=upstream)
                last = marker
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                save('failed', reason='bounded dependency wait expired', upstream=upstream)
                return 2
            time.sleep(min(a.poll_seconds, remaining))
        audit = root / 'audit.json'
        command = [sys.executable, str(Path(__file__).with_name('audit_rphi_stage2.py')),
                   '--run-dir', str(root), '--out', str(audit)]
        save('auditing', argv=command)
        process = subprocess.run(command, check=False)
        if process.returncode:
            save('failed', reason='stage2 evidence audit failed', returncode=process.returncode)
            return process.returncode
        archive = root / 'evidence.tar.gz'
        # Small records only: no checkpoints, source video, or generated images.
        records = sorted(p for p in root.rglob('*') if p.is_file()
                         and p.suffix in ('.json', '.jsonl', '.log')
                         and p != status and p.stat().st_size <= 32 * 1024 * 1024)
        with tarfile.open(archive, 'x:gz') as bundle:
            for record in records:
                bundle.add(record, arcname=str(record.relative_to(root)))
        save('completed', quality_verdict='requires_review', audit=str(audit),
             archive=str(archive), records=len(records))
        return 0
    except Exception as exc:
        save('failed', reason=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
