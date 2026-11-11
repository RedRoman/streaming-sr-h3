"""Run a finite, reviewed experiment plan after an Ascend card becomes idle.

Never kills other jobs. Locks coordinate cooperating jobs only: they are not a
cluster scheduler reservation. Unknown npu-smi formats fail closed.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def parse_npu(text):
    cards, current = {}, None
    for line in text.splitlines():
        match = re.match(r'\|\s*(\d+)\s+(910\S*)\s*\|\s*(\S+)\s*\|', line)
        if match:
            current = int(match[1])
            if current in cards:
                raise ValueError('duplicate card in npu-smi table')
            cards[current] = {'health': match[3], 'pids': []}
            continue
        if current is not None and ':' in line and '|' in line:
            cells = line.split('|')
            nums = re.findall(r'(\d+)\s*/\s*(\d+)', line)
            if len(nums) >= 2:
                cards[current].update(used_mb=int(nums[-1][0]), total_mb=int(nums[-1][1]),
                                      utilization=int(cells[3].split()[0]))
                current = None
        proc = re.match(r'\|\s*(\d+)\s+\d+\s*\|\s*(\d+)\s*\|', line)
        if proc:
            idx = int(proc[1])
            if idx not in cards:
                raise ValueError('process table references unknown device')
            cards[idx]['pids'].append(int(proc[2]))
    if not cards or 'Process id' not in text:
        raise ValueError('missing device or process table')
    if any('total_mb' not in c for c in cards.values()):
        raise ValueError('incomplete device memory table')
    return cards


def available(card, min_free):
    return (card['health'] == 'OK' and not card['pids'] and card['utilization'] <= 5
            and card['total_mb'] - card['used_mb'] >= min_free)


def inspect():
    result = subprocess.run(['npu-smi', 'info'], text=True, capture_output=True, timeout=30, check=True)
    return parse_npu(result.stdout)


def save(path, **record):
    record['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(record, indent=2) + '\n')
    temp.replace(path)


def run_plan(plan, status, card):
    root = Path(plan['root']).resolve()
    for relative, expected in plan.get('implementation_sha256', {}).items():
        path = root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            save(status, state='failed', reason='reviewed implementation changed', path=str(path))
            return 2
    env = {**os.environ, **plan.get('environment', {}), 'ASCEND_RT_VISIBLE_DEVICES': str(card),
           'SSR0': str(root), 'OMP_NUM_THREADS': '8'}
    done = []
    for step in plan['steps']:
        for guard in plan.get('required_json_values', []):
            if json.loads(Path(guard['path']).read_text()).get(guard['key']) != guard['value']:
                save(status, state='failed', card=card, reason='external admission condition changed',
                     completed=done, condition=guard)
                return 2
        # Another non-cooperating queue may acquire the card between commands.
        # Recheck each process boundary and stop rather than sharing its memory.
        now = inspect()
        if card not in now or not available(now[card], plan['min_free_mb']):
            save(status, state='failed', card=card, reason='card acquired between steps',
                 completed=done, devices=now)
            return 2
        argv = step['argv']
        if not isinstance(argv, list) or not argv or any(not isinstance(x, str) for x in argv):
            raise ValueError('every command must be a nonempty argv list')
        log = root / step['log']
        log.parent.mkdir(parents=True, exist_ok=True)
        # A new run directory is required; rerunning a partial experiment must be deliberate.
        with log.open('x') as output:
            save(status, state='running', card=card, step=step['name'], completed=done, argv=argv)
            result = subprocess.run(['bash', str(root / 'code/run_takeover_command.sh'), *argv],
                                    cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT)
        if result.returncode:
            save(status, state='failed', card=card, step=step['name'], completed=done,
                 returncode=result.returncode, log=str(log))
            return result.returncode
        done.append(step['name'])
    save(status, state='completed', card=card, completed=done)
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('plan', type=Path)
    p.add_argument('--poll-seconds', type=int, default=120)
    p.add_argument('--stable-seconds', type=int, default=15)
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    root = Path(plan['root']).resolve()
    status = root / plan['status']
    status.parent.mkdir(parents=True, exist_ok=True)
    lock_dir = Path(plan['lock_dir'])
    lock_dir.mkdir(parents=True, exist_ok=True)
    queue_lock = status.with_suffix('.lock').open('a')
    fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if status.exists() and json.loads(status.read_text()).get('state') in ('completed', 'running', 'failed'):
        p.error('plan has run already; inspect outputs and create a new run path')
    while True:
        try:
            cards = inspect()
        except (ValueError, subprocess.SubprocessError, OSError) as exc:
            save(status, state='waiting', reason='npu-smi inspection failed', detail=str(exc))
            if a.inspect_only:
                raise
            time.sleep(max(30, a.poll_seconds))
            continue
        eligible = [i for i in plan['cards'] if i in cards and available(cards[i], plan['min_free_mb'])]
        save(status, state='waiting', reason='waiting for an idle card', cards=cards, eligible=eligible)
        if a.inspect_only:
            print(status.read_text())
            return 0
        for card in eligible:
            with (lock_dir / ('card_%d.lock' % card)).open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                time.sleep(max(1, a.stable_seconds))
                try:
                    final = inspect()
                except (ValueError, subprocess.SubprocessError, OSError):
                    continue
                if card not in final or not available(final[card], plan['min_free_mb']):
                    continue
                try:
                    return run_plan(plan, status, card)
                except Exception as exc:
                    save(status, state='failed', card=card, reason='experiment runner failed', detail=str(exc))
                    raise
        time.sleep(max(30, a.poll_seconds))


if __name__ == '__main__':
    raise SystemExit(main())
