"""One CPU default-inspection child; no --execute or model runtime import."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

p = Path(__file__).resolve().parent
root = p.parent.parent
code = root / 'preparation/h3_full22_joint_cpu_validation_20261112b/code'
sources = json.loads((p / 'source_manifest.json').read_text())
assert all(hashlib.sha256((p / f).read_bytes()).hexdigest() == h for f, h in sources.items())
assert not (p / 'inspection_receipt.json').exists() and not (p / 'inspect.log').exists()
assert not (p / 'actual_input_inspection').exists()
os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(code))
argv = [sys.executable, str(code / 'check_h3_full22_joint_rgb_grad.py'),
        '--pretrained-manifest', str(root / 'configs/openh3_20261111/refiner_turbo_30794.json'),
        '--execution-contract', str(p / 'execution_contract.json'),
        '--out', str(p / 'actual_input_inspection')]
started = time.monotonic()
peak, reason, proc = 0, None, None


def cleanup():
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()


def interrupted(signum, frame):
    global reason
    reason = 'external_signal_' + str(signum)
    raise KeyboardInterrupt(reason)


for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, interrupted)
try:
    with (p / 'inspect.log').open('xb') as log:
        proc = subprocess.Popen(argv, cwd=root, env=env, stdout=log,
                                stderr=subprocess.STDOUT, start_new_session=True)
    while proc.poll() is None:
        rss = 0
        for d in Path('/proc').iterdir():
            if not d.name.isdecimal():
                continue
            try:
                stat = (d / 'stat').read_text().rsplit(')', 1)[1].split()
                if int(stat[2]) == proc.pid:
                    rss += int(stat[21]) * os.sysconf('SC_PAGE_SIZE')
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        peak = max(peak, rss)
        if rss > 4 * (1 << 30) or time.monotonic() - started > 120:
            reason = 'RSS' if rss > 4 * (1 << 30) else 'wall'
            break
        time.sleep(.1)
finally:
    cleanup()
    inspection = p / 'actual_input_inspection/inspection.json'
    passed = bool(proc is not None and proc.returncode == 0 and reason is None and inspection.is_file())
    receipt = {'schema': 'openh3.full22_joint_actual_input_CPU_inspection.v1',
        'complete': passed, 'returncode': None if proc is None else proc.returncode,
        'limit_failure': reason, 'elapsed_seconds': time.monotonic() - started,
        'peak_owned_group_RSS_bytes': peak, 'wall_cap_seconds': 120, 'kill_grace_seconds': 10,
        'RSS_cap_bytes': 4 * (1 << 30), 'CPU_affinity': sorted(os.sched_getaffinity(0)),
        'threads': 1, 'argv': argv, 'source_manifest_sha256': hashlib.sha256((p / 'source_manifest.json').read_bytes()).hexdigest(),
        'execution_contract_sha256': hashlib.sha256((p / 'execution_contract.json').read_bytes()).hexdigest(),
        'inspection_sha256': hashlib.sha256(inspection.read_bytes()).hexdigest() if inspection.exists() else None,
        'log_sha256': hashlib.sha256((p / 'inspect.log').read_bytes()).hexdigest(),
        'scope': 'Actual cache bytes and source/target identity inspection only; zero model calls. No NPU admission.'}
    with (p / 'inspection_receipt.json').open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt), flush=True)
sys.exit(0 if passed else 2)
