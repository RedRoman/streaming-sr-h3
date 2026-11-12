"""One bounded extraction from an already verified target NPZ; no video decode."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
import hashlib, io, json, signal, subprocess, sys, tarfile, time
from pathlib import Path

ROOT = Path('/cache/zihaohe/omni_openh3_20261111')
OUT = ROOT / 'evidence/history_utility_target_followup_20261112b'
RECEIPT = ROOT / 'runs/history_utility_target_screen_20261112b/sources/SJ-_0uGIL1c/complete.json'
RECEIPT_SHA = '92524019ef353f29ea293a0ee3a476ee28541aaa0b37274794a7485c78866828'
INDICES = [23, 24, 25, 78, 79, 80]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def put(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)

def worker():
    import numpy as np
    from PIL import Image
    assert sha(RECEIPT) == RECEIPT_SHA
    receipt = json.loads(RECEIPT.read_text())
    target = receipt['target']
    assert sha(target['path']) == target['file_sha256']
    assert Path(target['path']).stat().st_size == target['bytes_on_disk']
    with np.load(target['path'], allow_pickle=False) as data:
        frames = data['frames']
    assert frames.dtype == np.uint8 and frames.shape == (96, 480, 832, 3)
    assert hashlib.sha256(memoryview(np.ascontiguousarray(frames)).cast('B')).hexdigest() == target['pixel_sha256']
    rows = []
    for index in INDICES:
        path = OUT / ('SJ-_0uGIL1c_frame%03d.png' % index)
        assert not path.exists()
        Image.fromarray(frames[index], mode='RGB').save(path)
        with Image.open(path) as image:
            assert np.array_equal(np.asarray(image), frames[index])
        rows.append({'index': index, 'target_timestamp': receipt['decode']['timestamps']['target'][index],
                     'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size,
                     'pixel_sha256': hashlib.sha256(frames[index].tobytes()).hexdigest(),
                     'shape': [480, 832, 3], 'dtype': 'uint8'})
    assert not any(x.split('.')[0] in {'torch', 'torch_npu', 'comfy', 'h3_comfy'} for x in sys.modules)
    report = {'schema': 'openh3.history_utility_original_target_frame_followup.v1', 'complete': True,
              'source_id': receipt['source']['source_id'], 'split': receipt['source']['split'],
              'source_sha256': receipt['source']['source_sha256'], 'window_key': receipt['window_key'],
              'target_specification': receipt['target_specification'], 'target': target,
              'target_receipt': {'path': str(RECEIPT), 'sha256': RECEIPT_SHA},
              'frames': rows, 'new_video_decodes': 0, 'model_calls': 0, 'NPU_calls': 0,
              'transform': 'Exact target RGB pixels, no resizing, no annotation, lossless PNG.',
              'continuity_approved': False, 'script_sha256': sha(__file__)}
    put(OUT / 'extraction.json', report)

def group_rss(pgid):
    total = 0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            fields = (p / 'stat').read_text().rsplit(')', 1)[1].split()
            if int(fields[2]) == pgid:
                for line in (p / 'status').read_text().splitlines():
                    if line.startswith('VmRSS:'):
                        total += int(line.split()[1]) * 1024
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total

def stop(proc):
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()

def supervise():
    OUT.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    started = time.monotonic(); peak = 0; error = None; proc = None
    old = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise RuntimeError('supervisor interrupted: %s' % signum)
    try:
        for sig in old:
            signal.signal(sig, interrupted)
        with (OUT / 'worker.log').open('wb') as log:
            proc = subprocess.Popen([sys.executable, __file__, '--worker'], stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            while proc.poll() is None:
                peak = max(peak, group_rss(proc.pid))
                if peak > 768 << 20:
                    raise RuntimeError('owned PG RSS exceeds768MiB')
                if time.monotonic() - started > 120:
                    raise RuntimeError('owned worker exceeds120seconds')
                time.sleep(.05)
            if proc.returncode != 0:
                raise RuntimeError('worker failed: %s' % proc.returncode)
    except BaseException as exc:
        error = repr(exc)
    finally:
        if proc is not None:
            stop(proc)
        for sig, handler in old.items():
            signal.signal(sig, handler)
        put(OUT / 'supervision.json', {'complete': error is None, 'error': error,
            'worker_pid': None if proc is None else proc.pid, 'seconds': time.monotonic()-started,
            'peak_owned_PG_RSS_bytes': peak, 'wall_limit_seconds': 120, 'RSS_limit_bytes': 768 << 20,
            'CPU_affinity': sorted(os.sched_getaffinity(0)), 'new_video_decodes': 0, 'model_calls': 0})
    if error:
        raise RuntimeError(error)
    files = sorted(p for p in OUT.iterdir() if p.suffix in {'.png', '.json', '.log'})
    manifest = {'payloads': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]}
    archive = OUT / 'evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as tar:
        for p in files:
            tar.add(p, arcname=p.name, recursive=False)
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('MANIFEST.json'); info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    assert archive.stat().st_size <= 8 << 20
    result = {'path': str(archive), 'sha256': sha(archive), 'bytes': archive.stat().st_size,
              'payloads': len(files), 'target_NPZ_included': False}
    put(OUT / 'archive_receipt.json', result)
    print(json.dumps(result))

if __name__ == '__main__':
    worker() if '--worker' in sys.argv else supervise()
