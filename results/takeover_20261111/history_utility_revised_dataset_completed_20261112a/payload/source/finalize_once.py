"""Explicitly admitted, bounded terminal wait and single CPU dataset audit."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RESOURCE = {'wait_seconds': 4200, 'poll_seconds': 30, 'audit_total_seconds': 900,
            'worker_seconds': 880, 'cleanup_seconds': 10,
            'owned_group_RSS_bytes': 4 << 30, 'CPU_cores': 2, 'threads': 1}
OWN = Path(__file__).resolve().parent


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'); temp.replace(path)


def wait_terminal(path, limit=4200, poll=30, *, clock=time.monotonic, sleep=time.sleep):
    started = clock()
    while True:
        if Path(path).is_file():
            row = read(path)
            require(row.get('state') in ('starting', 'running', 'completed', 'failed'), 'unknown producer state')
            if row['state'] in ('completed', 'failed'):
                return row, sha(path), clock() - started
        remaining = limit - (clock() - started)
        if remaining <= 0: raise TimeoutError('4200s terminal wait expired; no audit/retry')
        sleep(min(poll, remaining))


def members(pgid):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        try:
            fields = (path / 'stat').read_text().rsplit(')', 1)[1].split()
            if fields[0] == 'Z' or int(fields[2]) != pgid: continue
            rss = next((int(s.split()[1]) * 1024 for s in (path / 'status').read_text().splitlines()
                        if s.startswith('VmRSS:')), 0)
            result.append({'pid': int(path.name), 'rss_bytes': rss})
        except (FileNotFoundError, PermissionError, ProcessLookupError): pass
    return result


def stop_owned(process, grace=10):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        process.poll()
        if not members(process.pid): return
        time.sleep(.05)
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=1)


@contextmanager
def spawned_owned(command, **kwargs):
    """Close TERM/INT fork race; child and descendants have one new owned PG."""
    blocked = {signal.SIGTERM, signal.SIGINT}
    oldmask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    old = {s: signal.getsignal(s) for s in blocked}; process = None
    def unmask(): signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
    def interrupted(signum, frame): raise InterruptedError('external signal ' + str(signum))
    try:
        process = subprocess.Popen(command, start_new_session=True, preexec_fn=unmask, **kwargs)
        for s in blocked: signal.signal(s, interrupted)
        signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)
        yield process
    finally:
        for s in blocked: signal.signal(s, signal.SIG_IGN)
        try:
            if process is not None and (process.poll() is None or members(process.pid)): stop_owned(process)
        finally:
            for s, handler in old.items(): signal.signal(s, handler)
            signal.pthread_sigmask(signal.SIG_SETMASK, oldmask)


def monitor(process, status, *, worker_seconds=880, memory_bytes=4 << 30,
            clock=time.monotonic, sleep=time.sleep, group_members=members):
    started = clock()
    while process.poll() is None:
        rss = sum(x['rss_bytes'] for x in group_members(process.pid))
        status['peak_owned_group_rss_bytes'] = max(status.get('peak_owned_group_rss_bytes', 0), rss)
        if rss > memory_bytes: raise MemoryError('owned audit PG RSS exceeded4GiB')
        if clock() - started >= worker_seconds: raise TimeoutError('audit worker exceeded880s')
        sleep(.1)
    require(not group_members(process.pid), 'audit worker left live owned descendants')
    require(process.returncode == 0, 'dataset auditor failed; no retry or readiness promotion')


def archive_remaining(started, clock=time.monotonic):
    remaining = RESOURCE['audit_total_seconds'] - (clock() - started)
    if remaining <= 0: raise TimeoutError('900s total exhausted; archive not attempted; no retry')
    return remaining


def validate_freeze(path, expected):
    require(Path(path).resolve() == OWN / 'freeze.json' and sha(path) == expected, 'audit freeze SHA/path differs')
    freeze = read(path)
    require(freeze['schema'] == 'openh3.history_utility_revised_terminal_audit_freeze.v1'
            and freeze['resource'] == RESOURCE, 'freeze schema/resources changed')
    for name, digest in freeze['files'].items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe source pin path')
        require(sha(OWN / relative) == digest, 'source pin changed: ' + name)
    require({'finalize_once.py', 'audit_terminal.py', 'small_archive.py', 'execution_plan.json',
             'code/history_utility_data.py', 'code/history_utility_revised_data.py'} <= set(freeze['files']),
            'freeze lacks required executable pins')
    config = read(OWN / 'execution_plan.json')
    require(config['schema'] == 'openh3.history_utility_revised_terminal_audit_plan.v1'
            and config['resource'] == RESOURCE and config['host'] == '31984', 'execution plan changed')
    root = Path(config['root'])
    require(Path(config['out']).parent == root / 'evidence'
            and Path(config['producer_run']).parent == root / 'runs'
            and config['producer_status'] == config['producer_run'] + '_supervisor/status.json', 'owned paths differ')
    # These two small immutable documents suffice here. Do NOT iterate producer
    # freeze.pins: that includes large original MP4/target files not used here.
    for key in ('producer_contract', 'producer_freeze', 'producer_admission'):
        ref = config[key]; require(sha(ref['path']) == ref['sha256'], 'producer metadata pin changed: ' + key)
    return freeze, config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--freeze', type=Path, required=True); p.add_argument('--freeze-sha256', required=True)
    p.add_argument('--execute', action='store_true'); p.add_argument('--admission', type=Path)
    p.add_argument('--admission-sha256'); args = p.parse_args()
    freeze, config = validate_freeze(args.freeze, args.freeze_sha256)
    if not args.execute:
        print(json.dumps({'status': 'prepared_not_dispatched', 'producer_not_polled': True,
                          'actual_NPZ_reads': 0, 'resource': RESOURCE})); return 0
    require(sys.platform == 'linux' and Path('/proc').is_dir(), 'guarded execution requires Linux procfs')
    require(args.admission and args.admission_sha256 and sha(args.admission) == args.admission_sha256,
            'explicit SHA-bound root admission required')
    admission = read(args.admission)
    require(admission.get('approved') is True and admission.get('scope') == 'revised_dataset_terminal_audit_only'
            and admission.get('freeze_sha256') == args.freeze_sha256
            and admission.get('host') == '31984', 'not admitted for these exact frozen bytes')
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=False)
    status = {'schema': 'openh3.history_utility_revised_audit_supervision.v1', 'state': 'waiting',
              'complete': False, 'revised_pair_ready': False, 'training_admitted': False,
              'freeze_sha256': args.freeze_sha256, 'admission_sha256': args.admission_sha256,
              'resource': RESOURCE, 'audit_invocations': 0, 'peak_owned_group_rss_bytes': 0}
    write(out / 'status.json', status)
    write(out / 'admission.json', admission)
    process = None; active_process = None; alarm_started = None
    def deadline(signum, frame):
        # Never address a reaped worker PID during later archive work.
        if active_process is not None and (active_process.poll() is None or members(active_process.pid)):
            try: os.killpg(active_process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        raise TimeoutError('audit total900s including cleanup/archive expired')
    original_alarm = signal.getsignal(signal.SIGALRM)
    try:
        terminal, terminal_sha, waited = wait_terminal(config['producer_status'])
        status.update(wait_elapsed_seconds=waited, producer_status_sha256=terminal_sha)
        write(out / 'producer_status.json', terminal)
        signal.signal(signal.SIGALRM, deadline); signal.alarm(900); alarm_started = time.monotonic()
        require(terminal['state'] == 'completed' and terminal['complete'] is True,
                'producer failed; no dataset scan or promotion permitted')
        affinity = sorted(os.sched_getaffinity(0))[:2]
        require(len(affinity) == 2, 'two allowed CPU cores required')
        os.sched_setaffinity(0, set(affinity))
        status['cpu_affinity'] = affinity
        env = os.environ.copy(); env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
        command = [sys.executable, str(OWN / 'audit_terminal.py'), '--config', str(OWN / 'execution_plan.json'),
                   '--config-sha256', freeze['files']['execution_plan.json'], '--status-sha256', terminal_sha,
                   '--out', str(out / 'dataset_audit')]
        status.update(state='auditing', audit_invocations=1, command=command)
        write(out / 'status.json', status)
        with (out / 'worker.log').open('xb') as log:
            try:
                with spawned_owned(command, cwd=config['root'], env=env, stdout=log, stderr=subprocess.STDOUT) as process:
                    active_process = process
                    status['owned_pgid'] = process.pid; write(out / 'status.json', status)
                    monitor(process, status)
            finally:
                active_process = None
        audit = read(out / 'dataset_audit/audit.json')
        require(audit['complete'] is True and audit['revised_pair_ready'] is True
                and audit['training_admitted'] is False and audit['legacy_research_ready'] is False,
                'dataset final audit not accepted')
        status.update(state='completed', complete=True, revised_pair_ready=True,
                      audit_sha256=sha(out / 'dataset_audit/audit.json'))
    except BaseException as exc:
        status.update(state='failed', complete=False, revised_pair_ready=False,
                      error={'type': type(exc).__name__, 'message': str(exc)})
    finally:
        if alarm_started is None:
            # Even wait timeout/producer failure gets one bounded small archive.
            signal.signal(signal.SIGALRM, deadline); signal.alarm(900); alarm_started = time.monotonic()
        status['returncode'] = process.poll() if process is not None else None
        if alarm_started is not None: status['audit_elapsed_before_archive_seconds'] = time.monotonic() - alarm_started
        write(out / 'status.json', status)
    try:
        remaining = archive_remaining(alarm_started)
        # A caught one-shot alarm must not grant unbounded packaging afterwards.
        signal.setitimer(signal.ITIMER_REAL, remaining)
        from small_archive import pack
        archive = pack(out, config, OWN)
        print(json.dumps({'state': status['state'], 'archive': archive}))
    except BaseException as exc:
        status.update(state='failed', complete=False, revised_pair_ready=False,
                      archive_error={'type': type(exc).__name__, 'message': str(exc)},
                      retry_permitted=False)
        write(out / 'status.json', status)
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM, original_alarm)
    return 0 if status['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
