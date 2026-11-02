"""Re-check the load-bearing streaming conclusion on real footage.

Every streaming design decision in this project rests on one measurement: that the unobserved band
decorrelates fast enough that cross-block memory carries nothing (0.103 across a block stride, against
0.614 for the range component), and that motion compensation makes it worse rather than better.

That was measured on generated clips. Three separate findings have since shown that generated material
behaves differently from real footage -- its ground truth was soft, its unobserved band carries ~20x less
energy, and the temporal statistic that identified degradation there reverses sign here. A band with 20x
more energy could easily have different temporal structure, so the conclusion has to be re-measured before
it is built on any further.

Same protocol as the original: correlation of each subspace against itself across a lag, with and without
motion compensation estimated from the observation.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/real_eval/const'
OUT = ROOT / 'results/recheck_transport'
DELTAS = (1, 4, 17, 34)
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def warp(img, fl):
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def corr(a, b):
    a, b = np.asarray(a).reshape(-1).astype(np.float64), np.asarray(b).reshape(-1).astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    per = {sp: {d: {'nowarp': [], 'warp': []} for d in DELTAS} for sp in ('range', 'null')}
    band = []
    for name in man['clips']:
        T = man['frames']
        hr = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in range(T)])
        lr = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
        up = F.interpolate(lr.permute(0, 3, 1, 2), size=hr.shape[1:3], mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
        R, N = split(hr)
        band.append(float(N.square().mean() / hr.square().mean()))
        sub = {'range': R.numpy(), 'null': N.numpy()}
        g_obs = (up.mean(-1).numpy() * 255).astype(np.uint8)
        for d in DELTAS:
            ts = list(range(0, T - d, STRIDE))
            fl = [flow(g_obs[t], g_obs[t + d]) for t in ts]
            for sp, arr in sub.items():
                nw = [corr(arr[t], arr[t + d]) for t in ts]
                wp = [corr(warp(arr[t], fl[i]), arr[t + d]) for i, t in enumerate(ts)]
                per[sp][d]['nowarp'].append(float(np.mean(nw)))
                per[sp][d]['warp'].append(float(np.mean(wp)))
        print(json.dumps({'clip': name, 'band_ratio': round(band[-1], 5),
                          'null_d17': round(per['null'][17]['nowarp'][-1], 3),
                          'range_d17': round(per['range'][17]['nowarp'][-1], 3)}), flush=True)

    agg = {sp: {f'd{d}': {k: {'mean': float(np.mean(v)), 'ci95': ci(v)} for k, v in per[sp][d].items()}
                for d in DELTAS} for sp in per}
    out = {'n': len(man['clips']), 'source': 'real_eval/const',
           'band_ratio_mean': float(np.mean(band)), 'aggregate': agg,
           'generated_reference': {'null_d17_nowarp': 0.103, 'range_d17_nowarp': 0.614,
                                   'band_ratio': 0.005}}
    (OUT / 'transport.json').write_text(json.dumps(out, indent=1))
    print()
    print('band_ratio real %.4f vs generated 0.005' % out['band_ratio_mean'])
    print('%-6s %-8s %18s %18s' % ('lag', 'subspace', 'no-warp', 'warped'))
    for d in DELTAS:
        for sp in ('range', 'null'):
            a = agg[sp][f'd{d}']
            print('%-6s %-8s %7.3f %-10s %7.3f %-10s' % (
                f'd{d}', sp, a['nowarp']['mean'], str([round(x, 2) for x in a['nowarp']['ci95']]),
                a['warp']['mean'], str([round(x, 2) for x in a['warp']['ci95']])))


if __name__ == '__main__':
    main()
