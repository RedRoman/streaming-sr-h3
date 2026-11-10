"""Pad-or-resize, and which rows: register OmniVR's 1088-row output against the 1080-row input properly.

omnibench_align.py scored raw MAE and got an argmin that wandered between 0 and 3 with a spread of only
0.7-1.9 MAE units out of 20-44. That is the wrong instrument: between a degraded input and its restoration
the appearance difference swamps an 8-row shift, so the curve is flat and the argmin is noise.

This uses a registration measure instead -- normalised cross-correlation of the vertical Laplacian, which
responds to edge position and not to brightness or texture change -- and it tests the two hypotheses
separately:
    crop   pred[dy:dy+1080] for dy in 0..8        (they padded 8 rows and did not crop back)
    resize pred resampled 1088 -> 1080            (they resized rather than padded)
Whichever scores higher, and by how much, decides the preprocessing. Getting this wrong would shift their
frames by up to 8 of 480 rows in our crops and make their arm look worse for a reason that is ours.
"""
import json, sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OB = Path('/cache/zihaohe/baselines/omnibench')
NCLIP = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def frames(p, n=4, stride=30):
    out = []
    with av.open(str(p)) as c:
        for i, f in enumerate(c.decode(video=0)):
            if i % stride == 0:
                out.append(f.to_ndarray(format='rgb24'))
            if len(out) >= n:
                break
    return out


def lap(x):
    """vertical Laplacian of a [H,W] float tensor, zero-mean and unit-norm for NCC."""
    t = x[None, None]
    k = torch.tensor([[-1.], [2.], [-1.]]).reshape(1, 1, 3, 1)
    g = F.conv2d(t, k, padding=(1, 0))[0, 0]
    g = g - g.mean()
    return g / (g.norm() + 1e-8)


def ncc(a, b):
    return float((lap(a) * lap(b)).sum())


def main():
    names = sorted(x.name for x in (OB / 'omnibench/with_gt/lq').glob('*.mp4'))[:NCLIP]
    rows, per = [], {}
    for n in names:
        lq = frames(OB / 'omnibench/with_gt/lq' / n)
        pr = frames(OB / 'predictions/withgt_step02400_neg_cfg' / n)
        if not lq or not pr:
            continue
        H = lq[0].shape[0]
        pad = pr[0].shape[0] - H
        L = [torch.from_numpy(x).float().mean(-1) for x in lq]
        P = [torch.from_numpy(x).float().mean(-1) for x in pr]
        crop = [float(np.mean([ncc(p[dy:dy + H], l) for p, l in zip(P, L)])) for dy in range(pad + 1)]
        rs = float(np.mean([ncc(F.interpolate(p[None, None], size=(H, p.shape[1]), mode='bilinear',
                                              align_corners=False, antialias=True)[0, 0], l)
                            for p, l in zip(P, L)]))
        best_dy = int(np.argmax(crop))
        per[n] = {'pad_rows': pad, 'ncc_crop_by_offset': [round(x, 4) for x in crop],
                  'best_crop_offset': best_dy, 'ncc_best_crop': round(max(crop), 4),
                  'ncc_resize': round(rs, 4),
                  'winner': 'crop' if max(crop) >= rs else 'resize',
                  'margin': round(max(crop) - rs, 4)}
        rows.append(per[n])
        print(json.dumps({'clip': n[:26], **{k: v for k, v in per[n].items() if k != 'ncc_crop_by_offset'}}),
              flush=True)
    offs = [r['best_crop_offset'] for r in rows]
    res = {'clips': len(rows), 'best_crop_offsets': offs,
           'consensus_offset': int(np.bincount(offs).argmax()) if offs else None,
           'unanimous': len(set(offs)) == 1 if offs else None,
           'crop_wins': sum(r['winner'] == 'crop' for r in rows),
           'mean_ncc_best_crop': round(float(np.mean([r['ncc_best_crop'] for r in rows])), 4),
           'mean_ncc_resize': round(float(np.mean([r['ncc_resize'] for r in rows])), 4),
           'per_clip': per}
    (ROOT / 'results/deploy/omnibench_align2.json').write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != 'per_clip'}, indent=1))


main()
