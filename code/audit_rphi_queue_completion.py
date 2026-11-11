"""Finish one running R_phi pilot with a CPU evidence audit, then exit.

This is a bounded dependency step, not a recurring monitor. It never acquires an
NPU or starts training. An upstream failure or timeout is recorded separately;
the original queue status and experiment artifacts are never modified.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--wait-seconds', type=int, default=5400)
    p.add_argument('--poll-seconds', type=int, default=30)
    a = p.parse_args()
    if min(a.wait_seconds, a.poll_seconds) < 1:
        p.error('wait and poll intervals must be positive')
    root = a.run_dir.resolve()
    upstream = root / 'queue_status.json'
    status = root / 'audit_status.json'
    if not upstream.is_file() or (root / 'audit.json').exists():
        p.error('a started queue and a new audit output are required')
    with status.open('x') as f:
        json.dump({'state': 'starting'}, f)

    def save(state, **fields):
        record = {'state': state, 'updated_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'upstream': str(upstream), **fields}
        tmp = status.with_suffix('.partial')
        tmp.write_text(json.dumps(record, indent=2) + '\n')
        tmp.replace(status)
        print(json.dumps(record), flush=True)

    deadline, last = time.monotonic() + a.wait_seconds, None
    try:
        while True:
            queue = json.loads(upstream.read_text())
            state = queue['state']
            if state == 'failed':
                save('failed', reason='upstream queue failed', upstream_snapshot=queue)
                return 2
            if state == 'completed':
                break
            if state not in ('waiting', 'running'):
                raise ValueError('unexpected upstream state: ' + state)
            marker = (state, queue.get('step'))
            if marker != last:
                save('waiting', upstream_state=state, upstream_step=queue.get('step'))
                last = marker
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                save('failed', reason='bounded wait expired')
                return 2
            time.sleep(min(a.poll_seconds, remaining))
        argv = [sys.executable, str(Path(__file__).with_name('audit_rphi_results.py')),
                '--known', str(root / 'known/eval/metrics.json'),
                '--blind', str(root / 'blind/eval/metrics.json'),
                '--confidence', str(root / 'blind_confidence/eval/metrics.json'),
                '--out', str(root / 'audit.json')]
        save('running', argv=argv)
        result = subprocess.run(argv, check=False)
        save('completed' if result.returncode == 0 else 'failed', returncode=result.returncode,
             quality_verdict='requires_review', output=str(root / 'audit.json'))
        return result.returncode
    except Exception as exc:
        save('failed', reason=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
