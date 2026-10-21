"""mech6: the same transport question, asked of BOTH subspaces (PROTOCOL_mech.md). No model; GT + LR only.

mech2 found the null space cannot be transported: warping attenuates exactly the high-frequency band it is
moving. Read the other way, that argument predicts the opposite for the range space, which is low-frequency
and therefore survives resampling. If that holds, temporal memory is not useless -- it is useful in one
subspace and useless in the other, and the field has been spending it on the wrong one.

Read-out rule, fixed before running (d = 17, the block stride):
  range transport works iff corr(warped range) >= 0.60 AND warping beats no-warp by >= 0.05 with a CI
  excluding 0 -- i.e. motion compensation HELPS here, the opposite of what it did for the null space.


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
OUT = ROOT / 'results/mech6_subspace_transport'
DELTAS = (1, 4, 17)          # adjacent / the KV-cache gap / the block stride
STAGE1_COS = 0.434           # what the trained model's own regenerated band scores against GT
STRIDE = 8                   # subsample start frames


def split(x):
    """(range, null) = (U(D x), x - U(D x)) for the x4 antialiased bicubic observation operator."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


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
    per = {sp: {d: {'warp_lr': [], 'nowarp': []} for d in DELTAS} for sp in ('range', 'null')}
    rows = []
    cache = {}
    for ci, name in enumerate(clips):
        meta = json.loads((D / name / 'meta.json').read_text())
        T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        hr = torch.stack([load(D / name / 'hr' / f'{t:06d}.png') for t in range(T)])
        lr = torch.stack([load(D / name / 'lr' / f'{t:06d}.png') for t in range(T)])
        up = F.interpolate(lr.permute(0, 3, 1, 2), size=hw, mode='bicubic',
                           align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1)
        R_, N_ = split(hr)
        SUB = {'range': R_.numpy(), 'null': N_.numpy()}
        g_obs = (up.mean(-1).numpy() * 255).astype(np.uint8)      # grayscale of the degraded observation
        g_gt = (hr.mean(-1).numpy() * 255).astype(np.uint8)
        r = {'clip': name}
        for d in DELTAS:
            flows = [flow(g_obs[t], g_obs[t + d]) for t in range(0, T - d, STRIDE)]
            for sp, arr in SUB.items():
                w_lr, nw = [], []
                for i, t in enumerate(range(0, T - d, STRIDE)):
                    tgt = arr[t + d]
                    w_lr.append(corr(warp(arr[t], flows[i]), tgt))
                    nw.append(corr(arr[t], tgt))
                r[f'{sp}_d{d}'] = {'warp_lr': float(np.mean(w_lr)), 'nowarp': float(np.mean(nw)),
                                   'warp_gain': float(np.mean(w_lr)) - float(np.mean(nw)),
                                   'n_pairs': len(w_lr)}
                per[sp][d]['warp_lr'].append(r[f'{sp}_d{d}']['warp_lr'])
                per[sp][d]['nowarp'].append(r[f'{sp}_d{d}']['nowarp'])
        rows.append(r)
        print(json.dumps({'clip': name, **{f'{sp}_d{d}': {k: round(v, 3) for k, v in r[f'{sp}_d{d}'].items()
                                                          if isinstance(v, float)}
                                           for sp in SUB for d in DELTAS}}), flush=True)

    agg = {sp: {f'd{d}': {k: {'mean': float(np.mean(v)), 'ci95': bootstrap(v)}
                          for k, v in per[sp][d].items()} for d in DELTAS} for sp in per}
    gain17 = [a - b for a, b in zip(per['range'][17]['warp_lr'], per['range'][17]['nowarp'])]
    agg['range']['d17']['warp_gain'] = {'mean': float(np.mean(gain17)), 'ci95': bootstrap(gain17)}
    gn17 = [a - b for a, b in zip(per['null'][17]['warp_lr'], per['null'][17]['nowarp'])]
    agg['null']['d17']['warp_gain'] = {'mean': float(np.mean(gn17)), 'ci95': bootstrap(gn17)}
    r17 = agg['range']['d17']['warp_lr']['mean']
    ok = r17 >= 0.60 and agg['range']['d17']['warp_gain']['mean'] >= 0.05 and agg['range']['d17']['warp_gain']['ci95'][0] > 0
    verdict = ('range transports and motion compensation helps there -- memory belongs in the range space'
               if ok else 'range transport does not clear the bar')
    out = {'n': len(rows), 'deltas': DELTAS, 'per_clip': rows, 'aggregate': agg, 'verdict': verdict}
    (OUT / 'premise.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
