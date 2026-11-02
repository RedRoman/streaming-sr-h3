"""Is the unobserved band predictable across a block gap? (PROTOCOL_mech2.md) No model; GT + LR only.

Every memory mechanism in this project assumes there is something worth carrying across blocks. That has
never been measured directly -- we only ever measured whether a particular carry helped. This measures the
information itself: take GT detail N(x_t), transport it with motion estimated from the DEGRADED observation
(model-free, so it cannot drift), and see how much of N(x_{t+d}) it explains.

The comparison that decides the question is against what the trained model already achieves on its own:
Stage 1 reports cosine 0.434 between its generated band and GT. Transported real detail has to beat that,
otherwise regenerating each block from the fresh observation is the better move and cross-block memory is
the wrong direction.

Read-out rules, fixed before running (d = 17 frames, the block stride):
  strong : corr(warped, future) >= 0.60  -> memory is justified, our failures were implementation
  none   : corr(warped, future) <  0.434 -> carrying detail cannot beat regenerating it; drop memory and
           report that streaming restoration needs no cross-block detail memory
  partial: in between
Reported alongside: the no-warp baseline, the GT-flow oracle, and a wrong-clip scramble control.
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D = ROOT / 'data/stage2_eval'
OUT = ROOT / 'results/mech2_premise'
DELTAS = (1, 4, 17)          # adjacent / the KV-cache gap / the block stride
STAGE1_COS = 0.434           # what the trained model's own regenerated band scores against GT
STRIDE = 8                   # subsample start frames


def unobserved(x):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def flow(a, b):
    """Farneback flow from grayscale a -> b, both uint8 [H,W]."""
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def warp(img, fl):
    """Pull img forward along fl so it lands on the later frame's grid. img: [H,W,3] float."""
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def corr(a, b):
    a, b = a.reshape(-1), b.reshape(-1)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12))


def bootstrap(v, n=10000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    clips = man['clips']
    per = {d: {'warp_lr': [], 'warp_gt': [], 'nowarp': [], 'scramble': []} for d in DELTAS}
    rows = []
    cache = {}
    for ci, name in enumerate(clips):
        meta = json.loads((D / name / 'meta.json').read_text())
        T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        hr = torch.stack([load(D / name / 'hr' / f'{t:06d}.png') for t in range(T)])
        lr = torch.stack([load(D / name / 'lr' / f'{t:06d}.png') for t in range(T)])
        up = F.interpolate(lr.permute(0, 3, 1, 2), size=hw, mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
        N = unobserved(hr).numpy()
        g_obs = (up.mean(-1).numpy() * 255).astype(np.uint8)      # grayscale of the degraded observation
        g_gt = (hr.mean(-1).numpy() * 255).astype(np.uint8)
        cache[name] = N
        r = {'clip': name}
        for d in DELTAS:
            w_lr, w_gt, nw, sc = [], [], [], []
            for t in range(0, T - d, STRIDE):
                f_obs = flow(g_obs[t], g_obs[t + d])
                f_gt = flow(g_gt[t], g_gt[t + d])
                tgt = N[t + d]
                w_lr.append(corr(warp(N[t], f_obs), tgt))
                w_gt.append(corr(warp(N[t], f_gt), tgt))
                nw.append(corr(N[t], tgt))
                # control: the same transport applied to a different clip's detail must explain nothing
                other = cache[clips[ci - 1]] if ci else None
                if other is not None and other.shape == N.shape:
                    sc.append(corr(warp(other[t], f_obs), tgt))
            r[f'd{d}'] = {'warp_lr': float(np.mean(w_lr)), 'warp_gt': float(np.mean(w_gt)),
                          'nowarp': float(np.mean(nw)),
                          'scramble': float(np.mean(sc)) if sc else None, 'n_pairs': len(w_lr)}
            for k in ('warp_lr', 'warp_gt', 'nowarp'):
                per[d][k].append(r[f'd{d}'][k])
            if sc:
                per[d]['scramble'].append(float(np.mean(sc)))
        rows.append(r)
        print(json.dumps({'clip': name, **{f'd{d}': {k: round(v, 3) for k, v in r[f'd{d}'].items() if isinstance(v, float)}
                                           for d in DELTAS}}), flush=True)
        if ci:
            cache.pop(clips[ci - 1], None)

    agg = {}
    for d in DELTAS:
        agg[f'd{d}'] = {k: {'mean': float(np.mean(v)), 'ci95': bootstrap(v)} for k, v in per[d].items() if v}
    m17 = agg['d17']['warp_lr']['mean']
    verdict = ('strong: cross-block detail memory is justified' if m17 >= 0.60 else
               'none: carrying detail cannot beat regenerating it from the fresh observation'
               if m17 < STAGE1_COS else 'partial')
    out = {'n': len(rows), 'stage1_model_cos_vs_gt': STAGE1_COS, 'deltas': DELTAS,
           'per_clip': rows, 'aggregate': agg, 'verdict': verdict}
    (OUT / 'premise.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
