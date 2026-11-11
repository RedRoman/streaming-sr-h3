"""Complementary bicubic-band error diagnosis (metric schema v2).

Lowband L=UD and detail I-L sum to identity, but are not generally orthogonal.
MSE equals lowband MSE + detail MSE + 2*mean(low_error*detail_error).
The GT-lowband substitution is an oracle diagnostic, not achievable recovery.
Historical decomp.json is preserved; v2 reports direct and clipped intervention MSE.
"""
import json, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from restoration_metrics import (METRIC_SCHEMA_VERSION, assert_compatible_output, band_error_terms,
                                 metric_metadata, psnr_from_mse)

ROOT = Path(os.environ.get('SSR0', os.environ.get('STREAMING_SR_ROOT', '/cache/zihaohe/streaming_sr0')))
D_ = ROOT / 'data/stage2_eval'
OUT = ROOT / 'results/mech5_error_decomp'
ARMS = {'indep': ROOT / 'results/stage2_blocks/indep',
        'cache': ROOT / 'results/stage5_blocks/cache/indep'}
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    """Return complementary low/detail bands, not an orthogonal range/null split."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def describe_terms(terms):
    """PSNR from actual MSE, with explicitly named oracle intervention gains."""
    out = dict(terms)
    out['psnr'] = psnr_from_mse(out['mse_actual'])
    out['psnr_gt_lowband_substitution'] = psnr_from_mse(out['mse_gt_lowband_substitution'])
    out['psnr_gt_lowband_substitution_clipped'] = psnr_from_mse(out['mse_gt_lowband_substitution_clipped'])
    out['oracle_lowband_substitution_gain_dB'] = out['psnr_gt_lowband_substitution'] - out['psnr']
    out['oracle_lowband_substitution_clipped_gain_dB'] = out['psnr_gt_lowband_substitution_clipped'] - out['psnr']
    # Compatibility aliases, not independent contributions to total error.
    out['mse_range'], out['mse_null'] = out['mse_lowband'], out['mse_detail']
    out['range_share'] = out['mse_lowband'] / max(out['mse_lowband'] + out['mse_detail'], 1e-12)
    out['psnr_if_range_exact'] = out['psnr_gt_lowband_substitution']
    out['headroom_dB'] = out['oracle_lowband_substitution_gain_dB']
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / 'decomp_v2.json'
    assert_compatible_output(output)
    man = json.loads((D_ / 'manifest.json').read_text())
    acc = {a: [] for a in ARMS}
    rows = []
    for name in man['clips']:
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        r = {'clip': name}
        for arm, base in ARMS.items():
            sd = base / name / 'stream'
            if not sd.exists():
                continue
            frames = []
            for t in range(0, T, STRIDE):
                f = sd / f'{t:06d}.png'
                if not f.exists():
                    continue
                g, y = load(D_ / name / 'hr' / f'{t:06d}.png'), load(f)
                gr, gn = split(g[None])
                yr, yn = split(y[None])
                frames.append(band_error_terms(g.numpy(), y.numpy(), gr[0].numpy(), yr[0].numpy()))
            if not frames:
                continue
            terms = {key: float(np.mean([frame[key] for frame in frames])) for key in frames[0]}
            r[arm] = {**describe_terms(terms), 'frames': len(frames)}
            acc[arm].append(r[arm])
        rows.append(r)
        print(json.dumps({'clip': name, **{a: {k: round(v, 5) if isinstance(v, float) else v
                                               for k, v in r[a].items()} for a in ARMS if a in r}}), flush=True)

    agg = {}
    for a, samples in acc.items():
        if not samples:
            continue
        summary = {key: {'mean': float(np.mean([s[key] for s in samples])),
                         'median': float(np.median([s[key] for s in samples])),
                         'ci95': ci([s[key] for s in samples]), 'n': len(samples)}
                   for key in samples[0] if key != 'frames'}
        agg[a] = {key: value['mean'] for key, value in summary.items()}
        agg[a].update({'range_share': summary['range_share'], 'headroom_dB': summary['headroom_dB'],
                      'statistics': summary})
    schema = metric_metadata()
    schema['decomposition_identity'] = 'mse_actual = mse_lowband + mse_detail + mse_cross_term'
    schema['legacy_decomposition_aliases'] = {
        'mse_range': 'mse_lowband', 'mse_null': 'mse_detail',
        'range_share': 'lowband/(lowband+detail) energy ratio only; excludes cross term; NOT fraction of total error',
        'psnr_if_range_exact': 'psnr_gt_lowband_substitution (oracle, unclipped)',
        'headroom_dB': 'oracle_lowband_substitution_gain_dB; may be negative, not achievable headroom',
    }
    schema['aggregation'] = 'Per clip: mean frame MSE then PSNR. Aggregate PSNR: mean of per-clip PSNR. statistics.ci95 bootstraps clips.'
    verdict = 'Diagnostic only: nonorthogonal band energy shares cannot establish recoverable error or architectural necessity.'
    out = {'metric_schema_version': METRIC_SCHEMA_VERSION, 'metric_schema': schema,
           'n': len(rows), 'eval_set': str(D_), 'per_clip': rows, 'aggregate': agg, 'verdict': verdict}
    output.write_text(json.dumps(out, indent=1, allow_nan=False))
    print(json.dumps({'written': str(output), 'aggregate': agg, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
