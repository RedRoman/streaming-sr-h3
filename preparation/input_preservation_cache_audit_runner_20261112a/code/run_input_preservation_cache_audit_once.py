"""One admitted CPU audit of one completed input-preservation cache.

Reuse the exact previously tested owned-process supervisor, without modifying
the running followup runner. No device inspector, card lock, CANN wrapper,
model, retry, waiting loop or training launch is available through this entry.
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

SCHEMA = 'rphi.input_preservation_cache_audit_once.v1'
SUPERVISOR_SHA = 'efabd330e5b49653fc01275bba18c9c408de43fd56e194685102bb313e097388'
REFERENCE_RUNNER_SHA = '8df017e4f1aa48bff3bd80a1f3361061f28f84b01c65991bc119eddeb023f30e'
ENTRY_NAME = 'audit_input_preservation_cache.py'
ENTRY_SHA = 'ee9cbcd1dcb1a1de8cee7b09f19f1437e2d665e33added8bf42f606581d55aa1'
AUDIT_SCHEMA = 'rphi.input_preservation_cache_actual_audit.v1'
AUDIT_STATUS = 'actual_cache_data_features_and_labels_verified_no_training_admission'
LIMITS = {'train': {'wall_seconds': 900, 'CPU_threads': 2, 'rss_bytes': 32 << 30, 'grace_seconds': 30},
          'dev': {'wall_seconds': 900, 'CPU_threads': 2, 'rss_bytes': 4 << 30, 'grace_seconds': 30}}


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def absolute(path, root=None):
    p = Path(path)
    if not p.is_absolute():
        require(root is not None, 'absolute path required')
        p = Path(root) / p
    return p.resolve()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def report_requirements(role):
    rows, files, arrays = (16000, 1000, 7000) if role == 'train' else (2880, 180, 1440)
    result = {'schema': AUDIT_SCHEMA, 'complete': True, 'status': AUDIT_STATUS, 'role': role,
        'actual_rows': rows, 'actual_cache_files': files, 'actual_cache_arrays': arrays,
        'raw_feature_rows_recomputed': rows, 'restorer_calls': 0, 'restorer_constructed': False,
        'checkpoint_deserialized': False, 'NPU_calls': 0, 'head_training_admitted': False,
        'original_history_headroom_remains_failed': True}
    if role == 'dev': result.update(original_cache_files_verified=180, original_cache_arrays_verified=1080)
    return result


def typed_values(document, expected, message):
    require(all(type(document.get(k)) is type(v) and document[k] == v for k, v in expected.items()), message)


def validate(plan_path):
    p = read(plan_path); root = Path(p['root']); role = p['role']
    require(p['schema'] == SCHEMA and root.is_absolute() and root.is_dir()
        and root == root.resolve() and role in LIMITS, 'invalid CPU audit plan/root/role')
    require(p['resource'] == LIMITS[role] and p['physical_card'] is None, 'fixed CPU budget only; no device admission')
    pins = p['pins']
    require(pins and all(Path(name).is_absolute() and sha(name) == digest for name, digest in pins.items()),
        'source/input pins differ')
    required = (str(Path(__file__).resolve()), p['supervision_source'], p['entry'], p['execution_contract'])
    require(all(name in pins for name in required), 'runner/entry/supervisor/contract pin missing')
    require(sha(p['supervision_source']) == SUPERVISOR_SHA, 'owned-process supervisor is not the frozen tested version')
    require(Path(p['entry']).name == ENTRY_NAME and p['entry_sha256'] == pins[p['entry']] == ENTRY_SHA, 'wrong cache auditor entry/SHA')
    require(Path(p['python']).is_absolute() and Path(p['python']).is_file() and os.access(p['python'], os.X_OK), 'existing absolute executable Python required')
    expected = [p['python'], p['entry'], '--execution-contract', p['execution_contract'], '--out', p['out'], '--execute']
    require(p['argv'] == expected, 'audit argv differs; extra options/roles/commands forbidden')
    run, out = Path(p['run']), Path(p['out'])
    require(run.parent == root / 'runs' and run == run.absolute() and out == run / 'output'
        and not run.is_symlink(), 'new isolated runs/<name>/output required')
    require(p['report_file'] == 'audit.json' and p['report_requires'] == report_requirements(role),
        'actual-cache acceptance scope must not be weakened')
    c = read(p['execution_contract'])
    require(c['schema'] == 'rphi.input_preservation_cache_audit_execution.v1' and c['root'] == str(root)
        and c['role'] == role and c['resources'] == {'wall_seconds': 900, 'CPU_cores': 2,
            'RSS_limit_bytes': LIMITS[role]['rss_bytes'], 'max_small_output_bytes': 32 << 20},
        'numerical audit contract role/resource differs')
    require(c['implementation_sha256'].get(str(Path(p['entry']).resolve())) == p['entry_sha256'],
        'execution contract does not bind actual auditor entry')
    require(all(pins.get(str(absolute(name))) == digest for name, digest in c['implementation_sha256'].items()),
        'auditor dependency pin missing from runner')
    for key in ('producer_manifest', 'producer_execution_contract', 'admission'):
        path = absolute(c[key]['path'], root)
        require(pins.get(str(path)) == c[key]['sha256'] == sha(path), 'producer/audit admission binding differs: ' + key)
    producer = read(absolute(c['producer_manifest']['path'], root))
    typed_values(producer, {'schema': 'rphi.input_preservation_cache.v1', 'role': role, 'complete': True,
        'status': 'cache_complete_pending_independent_audit_no_training_admission',
        'completed_rows': 16000 if role == 'train' else 2880, 'head_training_admitted': False,
        'original_history_headroom_remains_failed': True, 'usable_for_training': role == 'train'},
        'producer not complete, wrong role or scope changed')
    execution = read(absolute(c['producer_execution_contract']['path'], root))
    require(execution == producer['execution_contract'] and execution['role'] == role
        and producer['execution_contract_file_sha256'] == c['producer_execution_contract']['sha256'],
        'producer execution identity differs')
    permission = read(absolute(c['admission']['path'], root))
    typed_values(permission, {'schema': 'rphi.input_preservation_cache_audit_admission.v1', 'approved': True,
        'role': role, 'execution_sha256': canonical_sha({k: v for k, v in c.items() if k != 'admission'}),
        'entry_sha256': p['entry_sha256'], 'out': str(out)}, 'exact numerical audit permission differs')
    launch = read(p['admission'])
    require(absolute(p['admission']) != absolute(c['admission']['path'], root), 'launch and numerical admission must be distinct')
    typed_values(launch, {'schema': 'rphi.input_preservation_cache_audit_launch_admission.v1', 'approved': True,
        'scope': 'one_completed_cache_CPU_audit_no_training', 'role': role,
        'plan_sha256': sha(plan_path), 'runner_sha256': sha(__file__),
        'execution_contract_sha256': sha(p['execution_contract']), 'entry_sha256': p['entry_sha256'],
        'physical_card': None}, 'root CPU audit launch permission differs')
    return p


def resources(root, role):
    memory = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
        if line.startswith('MemAvailable:'))
    disk = shutil.disk_usage(root).free
    require(memory >= (48 if role == 'train' else 8) << 30 and disk >= 1 << 30,
        'insufficient host memory/disk for CPU audit')
    return {'MemAvailable_bytes': memory, 'disk_free_bytes': disk}


def verify_terminal(p):
    out = Path(p['out']); report_path = out / 'audit.json'; report = read(report_path)
    typed_values(report, report_requirements(p['role']), 'terminal actual cache audit failed or weakened')
    c = read(p['execution_contract'])
    require(report['producer_manifest'] == c['producer_manifest'], 'terminal audited another producer manifest')
    typed_values(report['runtime'], {'device': 'cpu', 'threads': 2, 'torch_npu_imported_at_end': False}, 'audit runtime not fixed CPU')
    require(report['runtime']['versions']['torch_npu'] is None, 'audit imported NPU backend')
    receipt = read(out / 'archive_receipt.json'); archive = out / 'evidence.tar.gz'
    require(absolute(receipt['path']) == archive.resolve() and not archive.is_symlink()
        and archive.stat().st_size == receipt['bytes'] and 0 < receipt['bytes'] <= 32 << 20
        and sha(archive) == receipt['sha256'], 'small archive missing/truncated/changed')
    require(receipt['NPZ_or_checkpoint_arrays_included'] is False and type(receipt['payload_files']) is int
        and receipt['payload_files'] > 0, 'small archive scope differs')
    return {'report_sha256': sha(report_path), 'archive_receipt_sha256': sha(out / 'archive_receipt.json'),
        'archive_sha256': receipt['sha256'], 'archive_bytes': receipt['bytes']}


def run(plan_path):
    p = validate(plan_path); plan_sha = sha(plan_path)
    root, runpath = Path(p['root']), Path(p['run'])
    # mkdir alone provides a one-attempt claim, including a prior failed attempt.
    runpath.mkdir(parents=False, exist_ok=False)
    s = load_module(p['supervision_source'], 'frozen_input_cache_audit_supervisor')
    status_path = runpath / 'status.json'
    status = {'schema': 'rphi.input_preservation_cache_audit_once_status.v1', 'state': 'checking',
        'complete': False, 'producer_started': False, 'role': p['role'], 'plan_sha256': plan_sha,
        'entry_sha256': p['entry_sha256'], 'resource': p['resource'], 'physical_card': None,
        'automatic_retry': False, 'head_training_admitted': False, 'model_calls': 0,
        'RSS_scope': 'simultaneously sampled sum of owned process group; not a kernel quota'}
    old_affinity = os.sched_getaffinity(0)
    try:
        with ExitStack() as stack, s.signal_guard():
            lock = stack.enter_context((runpath / 'status.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            status['resources'] = resources(root, p['role'])
            affinity = sorted(old_affinity)[:2]; require(len(affinity) == 2, 'two CPU cores unavailable')
            os.sched_setaffinity(0, set(affinity)); status['cpu_affinity'] = affinity
            env = {**os.environ, 'ASCEND_RT_VISIBLE_DEVICES': '', 'CUDA_VISIBLE_DEVICES': '',
                'TORCH_DEVICE_BACKEND_AUTOLOAD': '0', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
                'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONHASHSEED': '0'}
            os.nice(10)
            s.supervise(p['argv'], root, env, runpath / 'worker.log', status, status_path,
                wall=900, rss_limit=LIMITS[p['role']]['rss_bytes'], grace=30)
            if status['state'] == 'worker_completed':
                terminal = verify_terminal(p)
                status.update(terminal, state='completed', complete=True,
                    acceptance='actual cache audit complete only; head training remains separately unadmitted')
    except BaseException as error:
        status.update(state='failed' if status['producer_started'] else 'not_launched', complete=False,
            error={'type': type(error).__name__, 'message': str(error)})
    finally:
        os.sched_setaffinity(0, old_affinity)
        try:
            require(sha(plan_path) == plan_sha, 'plan changed during audit')
            validate(plan_path); status['pins_unchanged'] = True
        except BaseException as error:
            status.update(state='failed', complete=False, pins_unchanged=False,
                error={'type': type(error).__name__, 'message': str(error)})
        s.write(status_path, status)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('plan', type=Path)
    parser.add_argument('--inspect-only', action='store_true'); args = parser.parse_args()
    if args.inspect_only:
        validate(args.plan); print(json.dumps({'valid': True, 'worker_started': False, 'NPU_calls': 0}))
        raise SystemExit(0)
    result = run(args.plan)
    print(json.dumps({k: result.get(k) for k in ('state', 'complete', 'role', 'error')}))
    raise SystemExit(0 if result['complete'] else 2)
