"""Is the temporal instability inside blocks or at their seams? The answer picks the fix.

Measured warp error against ground truth is 2.34 for our streaming output and 1.275 for a per-frame CNN
baseline, so our output is the less temporally stable of the two. But the pairs that measurement sampled
(every eighth frame) mostly sit inside a single block -- blocks span 22 frames on a stride of 17 -- so it
does not say where the instability comes from.

  seam     pairs (t, t+1) where t+1 is the first frame a new block emits: 22, 39, 56, 73, 90, 107
  interior every other pair

If the seam pairs are much worse, cross-block mechanisms are the fix: shared noise indexed by absolute
frame, or the KV cache, which real footage now justifies (null-space correlation 0.487 across a stride,
against the 0.103 the original negative result was measured on).

If the two are alike, the instability is produced inside a single sampling run and no cross-block mechanism
will touch it; the fix has to be in the sampler or the conditioning instead.
"""
import json, os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
ARM = ROOT / os.environ.get('ARM_DIR', 'results/stage7_eval/raw')
OUT = ROOT / 'results/flicker_split'
SEAM_FIRST = [17 * k + 5 for k in range(1, 7)]          # 22, 39, 56, 73, 90, 107


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def unobserved(x, s=4):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def warp(img, fl):
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    # one interior pair just before each seam, so the two sets sample the same moments of the clip
    interior_first = [t - 6 for t in SEAM_FIRST]
    agg = {'seam': {'flicker': [], 'warp': []}, 'interior': {'flicker': [], 'warp': []}}
    rows = []
    for name in man['clips']:
        sd = ARM / name / 'stream'
        if not sd.exists():
            continue
        r = {'clip': name}
        skip = False
        for label, firsts in (('seam', SEAM_FIRST), ('interior', interior_first)):
            fn, fd, wn, wd = [], [], [], []
            for t1 in firsts:
                t0 = t1 - 1
                f0, f1 = sd / f'{t0:06d}.png', sd / f'{t1:06d}.png'
                if not (f0.exists() and f1.exists()):
                    continue
                y0, y1 = load(f0)[None], load(f1)[None]
                g0 = load(D_ / name / 'hr' / f'{t0:06d}.png')[None]
                g1 = load(D_ / name / 'hr' / f'{t1:06d}.png')[None]
                fn.append(float((unobserved(y1) - unobserved(y0)).square().mean()))
                fd.append(float((unobserved(g1) - unobserved(g0)).square().mean()))
                fw = flow((g0[0].mean(-1).numpy() * 255).astype(np.uint8),
                          (g1[0].mean(-1).numpy() * 255).astype(np.uint8))
                wn.append(float(np.abs(warp(y0[0].numpy(), fw) - y1[0].numpy()).mean()))
                wd.append(float(np.abs(warp(g0[0].numpy(), fw) - g1[0].numpy()).mean()))
            if not fn or np.mean(fd) < 1e-7 or np.mean(wd) < 1e-5:
                skip = True
                break
            r[f'{label}_flicker'] = float(np.mean(fn) / np.mean(fd))
            r[f'{label}_warp'] = float(np.mean(wn) / np.mean(wd))
        if skip:
            continue
        for label in ('seam', 'interior'):
            agg[label]['flicker'].append(r[f'{label}_flicker'])
            agg[label]['warp'].append(r[f'{label}_warp'])
        rows.append(r)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    out = {'arm': str(ARM), 'n': len(rows), 'per_clip': rows,
           'groups': {k: {m: {'mean': float(np.mean(v[m])), 'ci95': ci(v[m])} for m in ('flicker', 'warp')}
                      for k, v in agg.items() if v['flicker']}}
    d_fl = [r['seam_flicker'] - r['interior_flicker'] for r in rows]
    d_wp = [r['seam_warp'] - r['interior_warp'] for r in rows]
    out['seam_minus_interior'] = {'flicker': {'mean': float(np.mean(d_fl)), 'ci95': ci(d_fl)},
                                  'warp': {'mean': float(np.mean(d_wp)), 'ci95': ci(d_wp)}}
    (OUT / (Path(os.environ.get('ARM_DIR', 'raw')).name + '.json')).write_text(json.dumps(out, indent=1))
    print()
    for k, v in out['groups'].items():
        print('%-9s flicker %7.3f %-14s warp %6.3f %s' % (
            k, v['flicker']['mean'], str([round(x, 2) for x in v['flicker']['ci95']]),
            v['warp']['mean'], [round(x, 2) for x in v['warp']['ci95']]))
    d = out['seam_minus_interior']
    print('seam - interior   flicker %+.3f %s | warp %+.3f %s' % (
        d['flicker']['mean'], [round(x, 2) for x in d['flicker']['ci95']],
        d['warp']['mean'], [round(x, 2) for x in d['warp']['ci95']]))


if __name__ == '__main__':
    main()
