"""Does the unobserved-subspace volume order the tasks, as the omni plan assumes?

The plan rests on a claim that has not been measured: that restoration tasks are not a flat label set but
a one-dimensional family ordered by how much of the signal the operator leaves unobserved, and that this
volume decides how much a generative prior can contribute at all. Denoising (D = identity) has no null
space and nothing for a prior to do; x4 SR leaves 15/16 of the spectrum unobserved. If that ordering is
real, the value of generation should fall off with the null volume in a measurable way -- and if it does
not, the organising principle is wrong and the multi-operator work should not be built on it.

The scale family D_s = bicubic downsample by s is the cheapest way to vary exactly that one quantity: U_s
composed with D_s stays a projection for every s, so the decomposition is structurally identical and only
the null volume changes. Measured on outputs already on disk, at s = 2, 4, 8, 16:

  null_frac   the share of the ground truth's energy that D_s does not observe
  ceiling     LPIPS of U_s(D_s(GT)) alone: a perfect estimator with NO generation. This is what a
              non-generative method can reach, so the gap to 0 is the whole prize generation competes for.
  with_ours   LPIPS of U_s(D_s(GT)) + N_s(ours): the same perfect low band plus our generated detail.
              ceiling - with_ours is what our generation is actually worth at that operator scale.

x4 is the operator this project has been solving; the other scales say whether the framework's value is
specific to it.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / 'data/real_eval/const'
ARM = ROOT / 'results/deploy/t4r2_q'
SCALES = (2, 4, 8, 16)
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u[0].permute(1, 2, 0).float()
    return r, x - r


def frame(d, name, t):
    p = d / name / 'stream' / f'{t:06d}.png'
    return p if p.exists() else d / name / f'{t:06d}.png'


def one(name):
    torch.set_num_threads(2)
    from revisit_metrics import lpips_dist
    m = json.loads((D_ / name / 'meta.json').read_text())
    ts = [t for t in range(0, m['frames'], STRIDE) if frame(ARM, name, t).exists()]
    acc = {s: {'null_frac': [], 'ceiling': [], 'with_ours': []} for s in SCALES}
    for t in ts:
        g = load(D_ / name / 'hr' / f'{t:06d}.png')
        o = load(frame(ARM, name, t))
        gm = g - g.mean()
        for s in SCALES:
            gr, gn = split(g, s)
            _, on = split(o, s)
            acc[s]['null_frac'].append(float(gn.square().sum() / gm.square().sum().clamp_min(1e-12)))
            acc[s]['ceiling'].append(lpips_dist(gr.clamp(0, 1), g))
            acc[s]['with_ours'].append(lpips_dist((gr + on).clamp(0, 1), g))
    out = {'clip': name, 'degradation': m['degradation'].get('kind', '?'), 'n_frames': len(ts)}
    for s, d in acc.items():
        for k, v in d.items():
            out[f's{s}.{k}'] = float(np.mean(v))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=20)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/null_volume.json'))
    a = ap.parse_args()
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, names), key=lambda r: r['clip'])
    Path(a.out).write_text(json.dumps({'arm': str(ARM), 'scales': SCALES, 'per_clip': rows}, indent=1))
    for g, sel in (('realistic', lambda r: r['degradation'] == 'realistic'),
                   ('bicubic', lambda r: r['degradation'] != 'realistic')):
        rs = [r for r in rows if sel(r)]
        print('### %s n=%d' % (g, len(rs)))
        print('  %5s %11s %10s %11s %14s' % ('scale', 'null_frac', 'ceiling', 'with_ours', 'generation buys'))
        for s in SCALES:
            c = np.mean([r[f's{s}.ceiling'] for r in rs])
            w = np.mean([r[f's{s}.with_ours'] for r in rs])
            print('  x%-4d %11.4f %10.4f %11.4f %14.4f'
                  % (s, np.mean([r[f's{s}.null_frac'] for r in rs]), c, w, c - w))
        print()
    print('ceiling is a perfect estimator with no generation; "generation buys" is how much our detail')
    print('improves on it at that operator scale. x4 is the operator the project solves.')


if __name__ == '__main__':
    main()
