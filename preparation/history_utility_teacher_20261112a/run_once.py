"""Single bounded P1 job, with a frozen supervisor and no retry/continuation.

The reused supervisor signals only the process group it creates. This runner
adds job identity, resource and fresh card checks; locks are cooperative only.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import time

SCHEMA = 'rphi.history_utility_bounded_once.v1'
LIMITS = {'prepare_schedules': (900, 2, 32 << 30),
          'backend_smoke': (900, 4, 32 << 30), 'teacher1000': (7200, 4, 32 << 30)}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate(plan_path):
    p = read(plan_path); root = Path(p['root'])
    require(p['schema'] == SCHEMA and root.is_absolute() and root.is_dir(), 'invalid once plan/root')
    kind = p['kind']; require(kind in LIMITS, 'unsupported bounded job')
    maximum, threads, rss = LIMITS[kind]; r = p['resource']
    require(0 < r['wall_seconds'] <= maximum and r['CPU_threads'] == threads
            and r['rss_bytes'] == rss and r['grace_seconds'] == 30, 'resource scope changed')
    card = p['physical_card']
    require((card is None) if kind == 'prepare_schedules' else (type(card) is int and 0 <= card < 8), 'wrong device scope')
    pins = p['pins']; require(pins and all(Path(q).is_absolute() and sha(q) == v for q, v in pins.items()), 'source/input SHA differs')
    required = [str(Path(__file__).resolve()), p['supervision_source'], p['execution_contract'],
                str(root/'code/takeover_queue.py'), str(root/'code/run_takeover_command.sh'), str(root/'env.sh')]
    require(all(q in pins for q in required), 'missing runner/runtime pin')
    argv = p['argv']
    require(len(argv) == 7 and all(isinstance(q, str) for q in argv)
            and Path(argv[0]).is_absolute() and argv[1] in pins
            and argv[2] == '--execution-contract' and argv[3] == p['execution_contract']
            and argv[4] == '--out' and argv[5] == p['out'] and argv[6] == '--execute', 'unexpected command')
    run, out = Path(p['run']), Path(p['out'])
    require(run.parent == root/'runs' and out == run/'output', 'fresh isolated run/output required')
    require(p['report_file'] in ('report.json', 'complete.json', 'manifest.json') and isinstance(p['report_requires'], dict)
            and p['report_requires'].get('complete') is True, 'terminal report requirements missing')
    a = read(p['admission'])
    require(a['schema'] == 'rphi.history_utility_once_admission.v1' and a['approved'] is True
            and a['plan_sha256'] == sha(plan_path) and a['runner_sha256'] == sha(__file__)
            and a['kind'] == kind and a['physical_card'] == card, 'root launch admission differs')
    if kind == 'teacher1000':
        smoke = p['accepted_backend_smoke']
        require(sha(smoke['path']) == smoke['sha256'], 'backend acceptance changed')
        accepted = read(smoke['path'])
        require(accepted['accepted'] is True and accepted['scope'] == 'P1_revised_real_NPU_two_step_backend'
                and accepted['teacher_entry_sha256'] == pins[argv[1]], 'actual backend acceptance required')
    return p


def resources(root):
    available = next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
    free = shutil.disk_usage(root).free
    require(available >= 48 << 30 and free >= 10 << 30, 'insufficient host memory/disk')
    return {'available_RAM_bytes': available, 'disk_free_bytes': free}


def run(plan_path):
    p = validate(plan_path); original_sha = sha(plan_path)
    root, runpath, card = Path(p['root']), Path(p['run']), p['physical_card']
    s = load_module(p['supervision_source'], 'p1_frozen_supervision')
    q = load_module(root/'code/takeover_queue.py', 'p1_device_inspection')
    runpath.mkdir(parents=False, exist_ok=False)
    status_path = runpath/'status.json'
    status = {'schema': 'rphi.history_utility_once_status.v1', 'state': 'checking', 'complete': False,
              'kind': p['kind'], 'plan_sha256': original_sha, 'producer_started': False,
              'physical_card': card, 'resource': p['resource'], 'automatic_retry': False,
              'cooperative_lock_only': True}
    old_affinity = os.sched_getaffinity(0)
    try:
        with ExitStack() as stack, s.signal_guard():
            lock = stack.enter_context((runpath/'status.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status['first_resources'] = resources(root)
            if card is not None:
                first = q.inspect(); status['first_devices'] = first
                require(card in first and q.available(first[card], 60000), 'card unavailable; no waiting')
                locks = Path('/cache/zihaohe/.npu-reservations'); locks.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((locks/f'card_{card}.lock').open('a'))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                status['state'] = 'stability_check'; s.write(status_path, status)
                time.sleep(15)
                require(sha(plan_path) == original_sha, 'plan changed during stability check')
                validate(plan_path)
                second = q.inspect(); status['second_devices'] = second
                require(card in second and q.available(second[card], 60000), 'second idle check failed')
                status['second_resources'] = resources(root)
            affinity = sorted(old_affinity)[:p['resource']['CPU_threads']]
            require(len(affinity) == p['resource']['CPU_threads'], 'CPU allocation unavailable')
            os.sched_setaffinity(0, set(affinity)); status['cpu_affinity'] = affinity
            env = {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': '' if card is None else str(card),
                   'OMP_NUM_THREADS': str(len(affinity)), 'MKL_NUM_THREADS': str(len(affinity)),
                   'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}
            # NumPy-only preparation needs no CANN environment; env.sh requires
            # a nonempty visible-card selection and must not be used to fake one.
            command = p['argv'] if card is None else ['bash', str(root/'code/run_takeover_command.sh'), *p['argv']]
            s.supervise(command, root, env,
                runpath/'worker.log', status, status_path, wall=p['resource']['wall_seconds'],
                rss_limit=p['resource']['rss_bytes'], grace=p['resource']['grace_seconds'])
            if status['state'] == 'worker_completed':
                report_path = Path(p['out'])/p['report_file']; report = read(report_path)
                require(all(report.get(k) == v for k, v in p['report_requires'].items()), 'terminal report did not meet declared scope')
                status.update(state='completed', complete=True, report_sha256=sha(report_path),
                              acceptance='execution complete; independent result acceptance pending')
    except BaseException as error:
        status.update(state='failed' if status['producer_started'] else 'not_launched', complete=False,
                      error={'type': type(error).__name__, 'message': str(error)})
    finally:
        os.sched_setaffinity(0, old_affinity)
        try:
            require(sha(plan_path) == original_sha, 'plan changed during execution')
            validate(plan_path); status['pins_unchanged'] = True
        except BaseException as error:
            status.update(state='failed', complete=False, pins_unchanged=False,
                          error={'type': type(error).__name__, 'message': str(error)})
        s.write(status_path, status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    args = parser.parse_args(); result = run(args.plan)
    print(json.dumps({k: result.get(k) for k in ('state', 'complete', 'error')}))
    raise SystemExit(0 if result['complete'] else 2)
