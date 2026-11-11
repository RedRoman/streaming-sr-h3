"""One finite CPU-only RGB damage audit of existing TRAIN control outputs.

No inference, LPIPS, flow, fitting, coefficient selection or independent test claim.
Uses the pinned evaluator's exact RGB metrics via its two pure NumPy functions.
"""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
import numpy as np

ROOT = Path('/cache/zihaohe/omni_openh3_20261111')
BASE = Path(__file__).resolve().parent
GUIDES = ROOT / 'runs/rphi_train_controls_legacy_smoke_20261111a/guides'
MANIFEST_SHA = '3f40c6a49119f091c1bc8745e7a1fdd04e9f7671f2730b9211a7157dbc5a6943'
METRIC_SHA = '6f2526a0f503813b1e7364710433e40a87fd8cc442fbf29f56dad34914c64809'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 << 20), b''): h.update(part)
    return h.hexdigest()

def array_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()

def main():
    if (BASE/'report.json').exists() or (BASE/'started.json').exists():
        raise FileExistsError('use a fresh audit directory')
    os.sched_setaffinity(0, {11})
    os.nice(10)
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    started = time.monotonic()
    (BASE/'started.json').write_text(json.dumps({'pid': os.getpid(), 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}))
    assert sha(GUIDES/'manifest.json') == MANIFEST_SHA
    manifest = json.loads((GUIDES/'manifest.json').read_text())
    assert manifest['complete'] and manifest['role'] == 'train' and manifest['samples'] == 8
    source = ROOT/'preparation/train_control_export_20261111_v1/code/eval_rphi.py'
    assert sha(source) == METRIC_SHA
    nodes = [n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in ('_matching_rgb', 'rgb_metrics')]
    assert len(nodes) == 2
    ns = {'np': np}; exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    metric = ns['rgb_metrics']
    rows = []
    for row in manifest['per_sample']:
        assert row['split'] == row['role'] == 'train'
        target_path = Path(row['target_npz']); cache = GUIDES/row['guide_npz']
        assert sha(target_path) == row['target_file_sha256']
        assert sha(cache) == row['guide_file_sha256']
        with np.load(target_path, allow_pickle=False) as z: target = z['frames']
        assert target.dtype == np.uint8 and array_sha(target) == row['target_pixel_sha256']
        target = target.astype(np.float32)/255
        with np.load(cache, allow_pickle=False) as z: arrays = {k: z[k] for k in row['arrays']}
        for key, value in arrays.items():
            spec = row['arrays'][key]
            assert value.dtype == np.float32 and list(value.shape) == spec['shape'] and array_sha(value) == spec['array_sha256']
        assert np.array_equal(arrays['rphi_rgb'], np.clip(arrays['rphi_raw_rgb'], 0, 1))
        scores = {key: metric(value, target) for key, value in arrays.items()}
        rows.append({'sample_id': row['sample_id'], 'source_id': row['source_id'], 'task': row['task'], 'control': row['control'],
            'target_pixel_sha256': row['target_pixel_sha256'], 'cache_sha256': row['guide_file_sha256'], 'scores_vs_original_RGB': scores,
            'rphi_edit_vs_input': metric(arrays['rphi_rgb'], arrays['raw_draft']), 'clipping': row['clipping'],
            'clipping_MSE_reduction': scores['rphi_raw_rgb']['mse']-scores['rphi_rgb']['mse'],
            'rphi_MSE_damage_vs_input': scores['rphi_rgb']['mse']-scores['raw_draft']['mse']})
        print(json.dumps({'sample': row['sample_id'], 'scores': scores}), flush=True)
        del arrays, target
    assert sha(GUIDES/'manifest.json') == MANIFEST_SHA
    assert 'torch' not in sys.modules and 'torch_npu' not in sys.modules
    report = {'schema': 'rphi.train_control_cached_RGB_damage.v1', 'complete': True,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'seconds': time.monotonic()-started,
        'manifest_sha256': MANIFEST_SHA, 'metric_source_sha256': METRIC_SHA, 'script_sha256': sha(__file__),
        'sources': sorted({r['source_id'] for r in rows}), 'samples': len(rows), 'per_sample': rows,
        'model_forwards': 0, 'NPU_calls': 0, 'training_steps': 0, 'lpips_measured': False, 'temporal_measured': False,
        'limits': ['One existing training source; no generalization, independent test or quality acceptance.',
            'Four identity task labels repeat identical pixels; no four-source interpretation.',
            'Mean per-frame RGB PSNR; perfect frames capped at 120dB by existing evaluator.',
            'No coefficient selection or training objective change from this audit.']}
    with (BASE/'report.json').open('x') as f: json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({'complete': True, 'report_sha256': sha(BASE/'report.json'), 'seconds': report['seconds']}), flush=True)

if __name__ == '__main__': main()
