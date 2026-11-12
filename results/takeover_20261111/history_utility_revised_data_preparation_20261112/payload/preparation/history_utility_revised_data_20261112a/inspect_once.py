"""One bounded CPU evidence check. Never construct a dataset or import models."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
C = json.loads((P / 'execution_contract.json').read_text())
ROOT = Path(C['root'])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker():
    for relative, expected in {**C['source_sha256'], **C['reference_sha256']}.items():
        assert sha(ROOT / relative) == expected, relative
    sys.path.insert(0, str(P / 'code'))
    import history_utility_revised_data as R
    result = R.inspect_cohort_evidence(cohort=C['cohort'], plan=C['plan'],
        train_manifest=C['train_manifest'], dev_manifest=C['dev_manifest'], evidence_root=ROOT)
    result['execution_contract_sha256'] = sha(P / 'execution_contract.json')
    result['actual_source_sha256'] = C['source_sha256']
    result['actual_registry_input_sha256'] = {k: sha(C[k]) for k in ('plan', 'train_manifest', 'dev_manifest')}
    result['forbidden_modules_loaded'] = sorted(k for k in sys.modules
        if k.split('.')[0] in ('torch', 'torch_npu', 'comfy', 'comfy_kitchen', 'h3_comfy', 'rphi_model'))
    assert not result['forbidden_modules_loaded']
    assert result['actual_selected_counts'] == {'train': 19, 'dev': 6}
    with (P / 'inspection.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({'complete': True, 'primary_sources': len(result['primary_evidence']),
                      'revised_dataset_ready': False, 'model_calls': 0}), flush=True)


def supervise():
    assert not (P / 'inspection.json').exists() and not (P / 'inspect.log').exists()
    assert not (P / 'inspection_supervision.json').exists()
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    start = time.monotonic()
    peak, reason, proc = 0, None, None
    def interrupted(signum, frame):
        nonlocal reason
        reason = 'signal_' + str(signum)
        raise KeyboardInterrupt(reason)
    for sig in (signal.SIGTERM, signal.SIGINT): signal.signal(sig, interrupted)
    try:
        with (P / 'inspect.log').open('xb') as log:
            proc = subprocess.Popen([sys.executable, __file__, '--worker'], cwd=ROOT,
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            rss = 0
            for d in Path('/proc').iterdir():
                if not d.name.isdecimal(): continue
                try:
                    stat = (d / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(stat[2]) == proc.pid: rss += int(stat[21]) * os.sysconf('SC_PAGE_SIZE')
                except (FileNotFoundError, ProcessLookupError, PermissionError): pass
            peak = max(peak, rss)
            if rss > C['budget']['owned_group_RSS_bytes'] or time.monotonic() - start > C['budget']['wall_seconds']:
                reason = 'RSS' if rss > C['budget']['owned_group_RSS_bytes'] else 'wall'
                break
            time.sleep(.1)
    finally:
        if proc is not None:
            try: os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: proc.wait(timeout=C['budget']['kill_grace_seconds'])
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        success = proc is not None and proc.returncode == 0 and reason is None and (P / 'inspection.json').is_file()
        receipt = {'schema': 'openh3.history_utility_revised_evidence_CPU_supervision.v1',
            'complete': success, 'returncode': None if proc is None else proc.returncode,
            'limit_failure': reason, 'peak_owned_group_RSS_bytes': peak,
            'elapsed_seconds': time.monotonic() - start, 'budget': C['budget'],
            'CPU_affinity': sorted(os.sched_getaffinity(0)),
            'inspection_sha256': sha(P / 'inspection.json') if (P / 'inspection.json').is_file() else None,
            'log_sha256': sha(P / 'inspect.log'), 'entry_sha256': sha(__file__),
            'execution_contract_sha256': sha(P / 'execution_contract.json')}
        with (P / 'inspection_supervision.json').open('x') as f:
            json.dump(receipt, f, indent=2)
            f.write('\n')
        print(json.dumps(receipt), flush=True)
    return 0 if success else 2


if __name__ == '__main__':
    if sys.argv[1:] == ['--worker']: worker()
    else:
        assert len(sys.argv) == 1
        sys.exit(supervise())
