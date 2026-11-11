"""CPU-only fixed residual grid on hash-bound saved RGB; never imports torch.

The analytic oracle minimizes whole-clip pooled MSE BEFORE the blend's own clamp.
Saved G is AS-RETURNED H3 RGB: Comfy VAE already clamps inside _finalize_pixels.
It is not pre-VAE-clamp RGB and cannot diagnose that internal clipping.
It uses GT and is diagnostic only, not a deployable or selected method.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

GRID = (0., .05, .10, .25, .50, .75, 1.)
SCHEMA = 'openh3.saved_residual_mix.v1'


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_sha(a):
    return hashlib.sha256(memoryview(np.ascontiguousarray(a)).cast('B')).hexdigest()


def canonical_sha(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def safe_child(parent, name):
    p = (Path(parent) / name).resolve()
    if not p.is_relative_to(Path(parent).resolve()):
        raise ValueError('cache path escapes its manifest directory')
    return p


def require_hash(path, expected):
    actual = sha_file(path)
    if actual != expected:
        raise ValueError('file SHA mismatch: ' + str(path))
    return actual


def metric_summary(mses, maes):
    mses = np.asarray(mses, dtype=np.float64)
    pooled = float(mses.mean())
    return {'mse': pooled, 'mae': float(np.mean(maes)),
            'psnr': float(np.mean(-10 * np.log10(np.maximum(mses, 1e-12)))),
            'psnr_from_mean_mse': float(-10 * np.log10(max(pooled, 1e-12))),
            'perfect_frames': int(np.sum(mses == 0)), 'frames': len(mses)}


def scan_mix(restored, generated, target, grid=GRID):
    """Float64 arithmetic, protocol float32 clipped outputs, frame-bounded RAM."""
    if (restored.dtype != np.float32 or generated.dtype != np.float32 or
            target.dtype != np.uint8 or restored.shape != generated.shape or
            target.shape != restored.shape or restored.ndim != 4 or restored.shape[-1] != 3):
        raise ValueError('expected matching float32 R/G and uint8 target THWC')
    if (not np.isfinite(restored).all() or not np.isfinite(generated).all() or
            restored.min() < 0 or restored.max() > 1):
        raise ValueError('R must be clipped and R/G finite')
    numer = energy = base_sq = 0.
    n = int(target.size)
    for rf, gf, yf in zip(restored, generated, target):
        r, g = rf.astype(np.float64), gf.astype(np.float64)
        y = (yf.astype(np.float32) / np.float32(255)).astype(np.float64)
        residual, desired = g-r, y-r
        numer += float(np.sum(desired * residual))
        energy += float(np.sum(residual * residual))
        base_sq += float(np.sum(desired * desired))
    unconstrained = numer / energy if energy > 0 else 0.
    oracle_t = float(np.clip(unconstrained, 0., 1.))
    coefficients = list(dict.fromkeys([float(t) for t in grid] + [oracle_t]))
    if any(not np.isfinite(t) or not 0 <= t <= 1 for t in coefficients):
        raise ValueError('mix coefficient outside [0,1]')
    stats = {t: {'pre_mse': [], 'pre_mae': [], 'post_mse': [], 'post_mae': [],
                 'clipped_components': 0, 'cast_change_sq': 0., 'pre_min': float('inf'),
                 'pre_max': -float('inf')} for t in coefficients}
    for rf, gf, yf in zip(restored, generated, target):
        r, g = rf.astype(np.float64), gf.astype(np.float64)
        y = (yf.astype(np.float32) / np.float32(255)).astype(np.float64)
        delta = g-r
        for t in coefficients:
            z = r+t*delta
            post = np.clip(z, 0, 1).astype(np.float32).astype(np.float64)
            a = stats[t]
            a['pre_mse'].append(float(np.mean((z-y)**2)))
            a['pre_mae'].append(float(np.mean(np.abs(z-y))))
            a['post_mse'].append(float(np.mean((post-y)**2)))
            a['post_mae'].append(float(np.mean(np.abs(post-y))))
            a['clipped_components'] += int(np.count_nonzero((z < 0) | (z > 1)))
            a['cast_change_sq'] += float(np.sum((post-z)**2))
            a['pre_min'] = min(a['pre_min'], float(z.min()))
            a['pre_max'] = max(a['pre_max'], float(z.max()))
    def finish(t):
        a = stats[t]
        return {'t': t, 'pre_pixel_clip_float64': metric_summary(a['pre_mse'], a['pre_mae']),
                'post_pixel_clip_float32': metric_summary(a['post_mse'], a['post_mae']),
                'clipped_component_fraction': a['clipped_components']/n,
                'clip_and_float32_cast_change_mse': a['cast_change_sq']/n,
                'pre_pixel_clip_range': [a['pre_min'], a['pre_max']],
                'analytic_preclip_pooled_mse': (base_sq-2*t*numer+t*t*energy)/n}
    return {'grid': [finish(float(t)) for t in grid],
            'oracle': {'purpose': 'GT diagnostic only; not a learned, deployable or selected method',
                       'objective': 'whole-clip pooled MSE before the blend additional clamp; cached G already has upstream VAE clamp; not mean-frame PSNR',
                       'formula': 'clip(sum((Y-R)*(G-R))/sum((G-R)^2),0,1)',
                       'unconstrained_t': unconstrained, 't': oracle_t,
                       'zero_residual': energy == 0, 'dot_desired_residual': numer,
                       'residual_squared_sum': energy, 'component_count': n,
                       'scored': finish(oracle_t)}}


def assert_quality_match(actual, expected, label):
    tolerances = {'psnr': 1e-4, 'psnr_from_mean_mse': 1e-4, 'mse': 1e-8, 'mae': 1e-7}
    diffs = {k: abs(actual[k]-expected[k]) for k in tolerances}
    if any(diffs[k] > tol for k, tol in tolerances.items()):
        raise ValueError(label + ' cached RGB reproduction failed: ' + str(diffs))
    return {'passed': True, 'max_absolute_differences': diffs, 'absolute_tolerances': tolerances}


def analyze_metrics(metrics_path, guide_path, dev_rows):
    metrics_path, guide_path = Path(metrics_path), Path(guide_path)
    d, manifest = json.loads(metrics_path.read_text()), json.loads(guide_path.read_text())
    if not d['complete'] or not manifest['complete']:
        raise ValueError('requires complete input evidence')
    require_hash(guide_path, d['contract']['guide_manifest_sha256'])
    exported = {r['sample_id']: r for r in manifest['per_sample']}
    result = []
    for row in d['per_sample']:
        begin = time.monotonic()
        e = exported[row['sample_id']]
        key = (row['source_id'], e['target_key'], row['task'])
        record = dev_rows[key]
        if canonical_sha(record) != row['record_sha256']:
            raise ValueError('development record binding mismatch')
        for field in ['source_sha256', 'target_pixel_sha256']:
            if row[field] != record[field] or e[field] != record[field]:
                raise ValueError('source/target binding mismatch')
        target_path = Path(record['target_npz'])
        target_file_sha = require_hash(target_path, e['target_file_sha256'])
        with np.load(target_path, allow_pickle=False) as archive:
            target = archive['frames']
        if array_sha(target) != row['target_pixel_sha256']:
            raise ValueError('target pixel hash mismatch')
        guide_file = safe_child(guide_path.parent, e['guide_npz'])
        guide_file_sha = require_hash(guide_file, e['guide_file_sha256'])
        with np.load(guide_file, allow_pickle=False) as archive:
            restored = archive['rgb']
        if array_sha(restored) != e['arrays']['rgb']['array_sha256']:
            raise ValueError('R_phi clipped RGB hash mismatch')
        decoded_file = safe_child(metrics_path.parent, row['decoded_file']['path'])
        decoded_sha = require_hash(decoded_file, row['decoded_file']['sha256'])
        with np.load(decoded_file, allow_pickle=False) as archive:
            generated = archive['rphi']
        generated_hash = array_sha(generated)
        if generated_hash != row['decoded_file']['arrays']['rphi']:
            raise ValueError('as-returned decoded RGB hash mismatch')
        if row['frame_indices'] != list(range(len(target))):
            raise ValueError('this diagnostic requires complete adjacent-frame cached score')
        scored = scan_mix(restored, generated, target)
        checks = {'t0_rphi_only': assert_quality_match(scored['grid'][0]['post_pixel_clip_float32'], row['arms']['rphi_only']['quality'], 't0'),
                  't1_h3_rphi': assert_quality_match(scored['grid'][-1]['post_pixel_clip_float32'], row['arms']['h3_rphi']['quality'], 't1')}
        if row['task'] != 'sr':
            checks['t075_native_alpha025_anchor'] = assert_quality_match(
                next(v for v in scored['grid'] if v['t']==.75)['post_pixel_clip_float32'],
                row['arms']['h3_rphi_anchor']['quality'], 'native t.75')
        else:
            checks['t075_native_alpha025_anchor'] = {'applicable': False, 'reason': 'SR uses canonical Q4 projection; full-RGB mixture is a different operator.'}
        entry = {'sample_id': row['sample_id'], 'source_id': row['source_id'], 'task': row['task'],
                 'control': row.get('control', 'ordinary'), 'frame_indices': row['frame_indices'],
                 'evidence': {'metrics_path': str(metrics_path), 'metrics_sha256': sha_file(metrics_path),
                              'guide_manifest_path': str(guide_path), 'guide_manifest_sha256': sha_file(guide_path),
                              'record_sha256': row['record_sha256'], 'source_sha256': row['source_sha256'],
                              'target_path': str(target_path), 'target_file_sha256': target_file_sha,
                              'target_pixel_sha256': row['target_pixel_sha256'],
                              'guide_cache_path': str(guide_file), 'guide_cache_sha256': guide_file_sha,
                              'clipped_rphi_array_sha256': e['arrays']['rgb']['array_sha256'],
                              'decoded_cache_path': str(decoded_file), 'decoded_cache_sha256': decoded_sha,
                              'as_returned_generated_array_sha256': generated_hash,
                              'as_returned_generated_range': [float(generated.min()),float(generated.max())]},
                 'baselines': {k: row['arms'][k]['quality'] for k in ['input','rphi_only','h3_rphi','h3_rphi_anchor']},
                 'grid_and_oracle': scored, 'reproduction_checks': checks,
                 'elapsed_seconds': time.monotonic()-begin}
        result.append(entry)
        print(json.dumps({'done': len(result), 'metrics': metrics_path.name, 'sample': row['sample_id'], 'control': entry['control'], 'task': row['task'], 'seconds': entry['elapsed_seconds']}), flush=True)
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ['ordinary-metrics','ordinary-guides','controls-metrics','controls-guides','dev-manifest','out']:
        ap.add_argument('--'+name, required=True, type=Path)
    args = ap.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(args.out)
    dev = [json.loads(line) for line in args.dev_manifest.read_text().splitlines() if line.strip()]
    index = {(r['source_id'],r['target_key'],r['task']): r for r in dev}
    if len(index) != len(dev):
        raise ValueError('duplicate development records')
    for p in [args.ordinary_metrics,args.controls_metrics]:
        if json.loads(p.read_text())['contract']['dev_manifest_sha256'] != sha_file(args.dev_manifest):
            raise ValueError('development manifest SHA mismatch')
    args.out.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    result = {'schema': SCHEMA, 'complete': False, 'quality_verdict': 'requires_review',
              'formula': 'Z_t=clip(R+t*(G_R-R),0,1), output float32; R=stored clipped R_phi, G_R=stored as-returned Rphi-guided H3 RGB',
              'fixed_grid': list(GRID), 'no_new_h3_calls': True, 'no_training': True,
              'scope': 'single-source exploratory CPU distortion diagnostic; no model or coefficient selection',
              'psnr_semantics': 'psnr=mean frame PSNR; psnr_from_mean_mse uses pooled whole-clip MSE; analytic oracle minimizes only preclip pooled MSE',
              'lpips_evaluated': False, 'temporal_evaluated': False,
              'clipping_scope': 'Comfy VAE _finalize_pixels applies upstream pixel_std/mean and clamp[0,1] before H3.decode returns. Cached G is not true pre-VAE-clamp RGB. All pre/postclip fields here describe only the additional blend clamp; this diagnostic cannot exclude or quantify upstream VAE clipping harm.',
              'limitations': ['Does not replace LPIPS/temporal acceptance.', 'GT oracle is not deployable and is not a learned method result.', 'Additional pixel clipping changes the objective in general; analytic t is not guaranteed postclip optimal.', 'Identity copies and near-clean task-keyed draws share one source.', 'A zero final blend/anchor clipping fraction does not imply zero internal VAE clipping.'],
              'implementation_sha256': sha_file(__file__), 'numpy_version': np.__version__,
              'dev_manifest_sha256': sha_file(args.dev_manifest), 'per_sample': []}
    try:
        for mp,gp in [(args.ordinary_metrics,args.ordinary_guides),(args.controls_metrics,args.controls_guides)]:
            result['per_sample'].extend(analyze_metrics(mp,gp,index))
        if len(result['per_sample']) != 12:
            raise ValueError('expected ordinary4 plus controls8')
        result.update(complete=True, elapsed_seconds=time.monotonic()-begin)
        with (args.out/'metrics.json').open('x') as f:
            json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    except Exception as exc:
        with (args.out/'failure.json').open('x') as f:
            json.dump({'complete':False,'error':str(exc)},f,indent=2)
        raise
    print(json.dumps({'complete': True, 'records':len(result['per_sample']),'seconds':result['elapsed_seconds']}))

if __name__ == '__main__':
    main()
