"""Pick the degradation descriptors by measuring separation, before training anything on them.

The first attempt (fine/coarse spatial residual ratio) separated the two regimes by only ~10%, which is why
it was not trained. This scores several candidates by how well a single scalar separates the Pillow-BICUBIC
clips from the realistic ones on the validation pairs, using AUC. Only a candidate that actually separates
is worth feeding to the model.

The temporal candidates matter most on priors: sensor noise and codec noise are independent across frames
while scene structure is not, and the branch already receives K frames, so the signal is free.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VAL = Path('/cache/zihaohe/streaming_sr0/data/range_pairs/val')
OUT = Path('/cache/zihaohe/streaming_sr0/results/mech7_desc')


def blur(x, sigma):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=x.dtype)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, 1, -1)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2))


def feats(y):
    """y: [T,1,H,W] grey in [0,1]. Returns a dict of scalar candidates."""
    f = {}
    fine = (y - blur(y, 0.7)).abs()
    mid = (y - blur(y, 1.5)).abs()
    coarse = (y - blur(y, 3.0)).abs()
    f['fine'] = float(fine.mean())
    f['fine_over_coarse'] = float(fine.mean() / coarse.mean().clamp_min(1e-6))
    # pixel-level roughness restricted to flat regions: clean bicubic is smooth there, noise and codec are not
    flat = (blur(coarse, 2.0) < coarse.mean() * 0.6).float()
    f['fine_in_flat'] = float((fine * flat).sum() / flat.sum().clamp_min(1.0))
    # temporal: noise is independent across frames, structure is not
    if y.shape[0] > 1:
        td = (y[1:] - y[:-1]).abs()
        f['temporal'] = float(td.mean())
        f['temporal_in_flat'] = float((td * flat[1:]).sum() / flat[1:].sum().clamp_min(1.0))
        f['temporal_over_fine'] = float(td.mean() / fine.mean().clamp_min(1e-6))
        # high-frequency part of the frame difference: pure noise has a flat spectrum, motion does not
        tdf = (td - blur(td, 1.5)).abs()
        f['temporal_hf'] = float(tdf.mean())
        f['temporal_hf_over_mid'] = float(tdf.mean() / mid.mean().clamp_min(1e-6))
    return f


def auc(pos, neg):
    """P(a random positive scores above a random negative); 0.5 = no separation, 1 or 0 = perfect."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return float(w / (len(pos) * len(neg)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = {}
    for p in sorted(VAL.glob('*.npz')):
        d = np.load(p)
        kind = json.loads(str(d['meta']))['degradation'].get('kind', '?')
        y = torch.from_numpy(d['y'][:8]).float().permute(0, 3, 1, 2).mean(1, keepdim=True) / 255
        groups.setdefault(kind, []).append(feats(y))
    kinds = list(groups)
    assert len(kinds) == 2, kinds
    a, b = groups['pillow_bicubic'], groups['realistic']
    rows = []
    for k in a[0]:
        va, vb = [x[k] for x in a], [x[k] for x in b]
        s = auc(vb, va)                       # realistic scoring above bicubic
        rows.append({'feature': k, 'bicubic_mean': float(np.mean(va)), 'realistic_mean': float(np.mean(vb)),
                     'auc_realistic_over_bicubic': s, 'separation': abs(s - 0.5) * 2})
    rows.sort(key=lambda r: -r['separation'])
    for r in rows:
        print(json.dumps({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.items()}), flush=True)
    (OUT / 'descriptor_selection.json').write_text(json.dumps({'n_bicubic': len(a), 'n_realistic': len(b),
                                                               'features': rows}, indent=1))


if __name__ == '__main__':
    main()
