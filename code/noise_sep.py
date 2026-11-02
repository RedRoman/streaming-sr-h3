"""Can the observation tell us how degraded it is, before any network sees it?

The range branch's whole failure mode is routing. Trained on the mixture it applies one average correction
and loses 14.17 dB on the 25% of clips whose LR is a clean Pillow-BICUBIC downsample (already 58 dB), while
gaining 1.98 dB on the realistic ones. Both the relative and the hinge losses that were tried to fix this
collapsed it to the identity, because a scale-invariant loss hands the near-perfect samples 10-20x the
gradient weight. The way out is to stop asking one unconditioned network to infer the regime and give it
the regime instead -- but only if the regime is actually estimable from y.

descriptor_select measured spatial statistics at AUC 0.43-0.55 and concluded they cannot, on the argument
that blur lowers high-frequency content while noise raises it. That argument applies to band energies; it
does not apply to an estimator built to be blind to image structure. Immerkaer's sigma convolves with
[[1,-2,1],[-2,4,-2],[1,-2,1]], which annihilates any locally-linear patch, so a smooth blurred image and a
smooth clean image both read near zero while noise and blocking read high.

Reported per val clip: the estimator, the base PSNR it is supposed to predict, the rank correlation
between them, and the AUC separating the two degradation kinds. An AUC near 1 makes conditioning trivial.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DATA = Path('/temp/zihaohe_range_pairs')


def immerkaer(x):
    """x float [H,W] in 0..1 -> sigma estimate. Annihilates locally-linear structure."""
    n = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
    h, w = x.shape
    a = np.lib.stride_tricks.sliding_window_view(x, (3, 3))
    r = np.abs((a * n).sum(axis=(-1, -2)))
    return float(np.sqrt(np.pi / 2) * r.sum() / (6 * (w - 2) * (h - 2)))


def mad_hp(x):
    """Median absolute deviation of the same high pass -- the robust twin, less sensitive to edges."""
    n = np.array([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]])
    a = np.lib.stride_tricks.sliding_window_view(x, (3, 3))
    r = np.abs((a * n).sum(axis=(-1, -2)))
    return float(1.4826 * np.median(r) / 6.0 * np.sqrt(np.pi / 2))


def one(p):
    d = np.load(p)
    y = d['y'].astype(np.float32) / 255
    t = d['target'].astype(np.float32) / 255
    meta = json.loads(str(d['meta']))
    g = y.mean(-1)
    f = min(4, len(y))
    return {'file': Path(p).name, 'kind': meta['degradation'].get('kind', '?'),
            'immerkaer': float(np.mean([immerkaer(g[i]) for i in range(f)])),
            'mad_hp': float(np.mean([mad_hp(g[i]) for i in range(f)])),
            'temporal': float(np.abs(np.diff(g[:f], axis=0)).mean()) if f > 1 else 0.0,
            'base_psnr': float(10 * np.log10(1 / max(float(((y - t) ** 2).mean()), 1e-12)))}


def auc(pos, neg):
    """P(a random positive scores above a random negative), ties counted as half."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else 'val'
    fs = sorted((DATA / split).glob('*.npz'))
    with Pool(24) as pool:
        rows = pool.map(one, [str(p) for p in fs])
    real = [r for r in rows if r['kind'] == 'realistic']
    pill = [r for r in rows if r['kind'] != 'realistic']
    print('%s: %d realistic, %d pillow_bicubic' % (split, len(real), len(pill)))
    print()
    print('%-10s %10s %10s %10s %8s' % ('stat', 'realistic', 'pillow', 'AUC(r>p)', 'spearman'))
    for k in ('immerkaer', 'mad_hp', 'temporal'):
        a = auc([r[k] for r in real], [r[k] for r in pill])
        x = np.array([r[k] for r in rows]); b = np.array([r['base_psnr'] for r in rows])
        rk = lambda v: np.argsort(np.argsort(v))
        sp = float(np.corrcoef(rk(x), rk(b))[0, 1])
        print('%-10s %10.5f %10.5f %10.3f %8.3f'
              % (k, np.mean([r[k] for r in real]), np.mean([r[k] for r in pill]), a, sp))
    print()
    print('base_psnr  realistic %.2f dB   pillow %.2f dB'
          % (np.mean([r['base_psnr'] for r in real]), np.mean([r['base_psnr'] for r in pill])))
    lo = min(r['immerkaer'] for r in pill); hi = max(r['immerkaer'] for r in real)
    ov = [r for r in real if r['immerkaer'] < max(r2['immerkaer'] for r2 in pill)]
    print('immerkaer: pillow range [%.5f, %.5f], realistic range [%.5f, %.5f], %d realistic below the '
          'pillow maximum' % (lo, max(r['immerkaer'] for r in pill),
                              min(r['immerkaer'] for r in real), hi, len(ov)))
    if ov:
        print('  those overlapping realistic clips sit at base PSNR %s'
              % ' '.join('%.1f' % r['base_psnr'] for r in ov[:10]))
    Path('/cache/zihaohe/streaming_sr0/results/range_branch/noise_sep_%s.json' % split).write_text(
        json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
