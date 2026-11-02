"""mech8: how badly is the range space corrupted, as a function of degradation complexity?

The per-kind read-out of the range branch produced a number worth following up: the observation's range
component sits at 58.1 dB under the field's standard Pillow-BICUBIC x4 and at 29.0 dB under the realistic
chain. If that gap is systematic, it explains why nobody targets the range space -- on a bicubic benchmark
it is already solved, so all of the error is in the null space and all of the effort goes to the prior and
to temporal modules. Under all-degradation it is not solved, and mech5 measured that it then holds 43.5%
of a trained restorer's error.

This uses the 89 validation pairs, each of which carries its own recorded degradation parameters, and needs
no model: PSNR(y, D x) is exactly "how much range-space work is left to do".
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')
VAL = ROOT / 'data/range_pairs/val'
OUT = ROOT / 'results/mech8_severity'


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12)))


def severity(d):
    """A 0-1 composite of the drawn degradation parameters, in the spirit of OmniVR's severity scalar."""
    if d.get('kind') == 'bicubic':
        return 0.0, {}
    parts = {}
    b = d.get('blur1')
    if b:
        parts['blur'] = min(1.0, float(max(b[1], b[2])) / 2.5)
    if d.get('noise1'):
        parts['noise'] = 1.0
    if d.get('jpeg_q') is not None:
        parts['jpeg'] = 1.0 - (float(d['jpeg_q']) - 35) / 60.0
    if d.get('crf') is not None:
        parts['h264'] = (float(d['crf']) - 18) / 19.0
    if d.get('second'):
        parts['second'] = 1.0
    v = float(np.mean(list(parts.values()))) if parts else 0.0
    return min(1.0, max(0.0, v)), parts


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in sorted(VAL.glob('*.npz')):
        d = np.load(p)
        meta = json.loads(str(d['meta']))
        deg = meta['degradation']
        y = d['y'].astype(np.float32) / 255
        t = d['target'].astype(np.float32) / 255
        s, parts = severity(deg)
        rows.append({'id': meta['id'], 'kind': deg.get('kind', '?'), 'severity': s,
                     'psnr_range': psnr(y, t), 'parts': parts})
    by = {}
    for r in rows:
        by.setdefault(r['kind'], []).append(r['psnr_range'])
    agg = {k: {'mean': float(np.mean(v)), 'ci95': ci(v), 'n': len(v)} for k, v in by.items()}

    real = [r for r in rows if r['kind'] != 'bicubic' and r['severity'] > 0]
    out = {'n': len(rows), 'by_kind': agg, 'per_clip': rows}
    if len(real) >= 8:
        x = np.array([r['severity'] for r in real])
        yv = np.array([r['psnr_range'] for r in real])
        A = np.vstack([x, np.ones_like(x)]).T
        slope, inter = np.linalg.lstsq(A, yv, rcond=None)[0]
        rho = float(np.corrcoef(x, yv)[0, 1])
        # bootstrap the slope so the trend is not read off a single fit
        g = np.random.default_rng(0)
        sl = []
        for _ in range(10000):
            i = g.choice(len(x), len(x), replace=True)
            sl.append(np.linalg.lstsq(np.vstack([x[i], np.ones(len(i))]).T, yv[i], rcond=None)[0][0])
        out['severity_fit'] = {'slope_dB_per_unit': float(slope), 'intercept': float(inter),
                               'pearson_r': rho, 'slope_ci95': [float(np.percentile(sl, 2.5)),
                                                                float(np.percentile(sl, 97.5))],
                               'n': len(real)}
    (OUT / 'severity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
