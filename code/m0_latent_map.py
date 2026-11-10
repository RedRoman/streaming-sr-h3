"""Is there a latent-space correction that improves the generated detail? (mapper feasibility, step 2)

Step 1 (null_roundtrip.py) established that the frozen H3 decoder can emit a detail-only image: round-tripping
the true null component n = x - U(D x) through encode/decode returns it at cosine 0.9155 and 31.93 dB, only
1.8 dB behind round-tripping a natural image. So the decoder's output manifold is not the obstacle.

The remaining question is whether the deficit in our generated detail (null energy 0.544 of the truth at
cosine 0.4876) has a systematic component that a map in latent space can remove. The scalar version of that
map is the `gain` knob, which trades cosine for energy (gain 1.3: energy 0.544 -> 0.893, cosine 0.4588 ->
0.4531). This fits the smallest non-scalar version instead -- one 24->24 3x3 convolution with bias, 5208
parameters, in closed form by ridge regression -- on a disjoint set of clips, and measures it on the rest.

Three arms are written so the comparison is honest:
  orig     the shipped stream, untouched
  rt       decode(encode(stream)): the round-trip control. The mapper has to beat THIS, not `orig`, because
           at deployment the mapper would consume the sampler's own latent and no round-trip would happen.
  map      decode(W * encode(stream) + b)
A 5208-parameter map fit on 20 clips cannot overfit 76 held-out ones, so if `map` does not beat `rt` there is
no systematic latent-space correction to be had and a trained mapper would be inventing detail, not fixing
calibration.

Phases: `fit` accumulates the normal equations over a shard of the fit clips (encode only);
`solve` inverts them; `apply` writes the three arms for a shard of the test clips.
"""
import argparse, json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))

D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/heldout')
SRC = ROOT / 'results/deploy_runs/t4_plain2'
OUT = ROOT / 'results/deploy'
NFIT = int(os.environ.get('NFIT', 20))
BLOCKS = [(17 * k, 17 * k + 22) for k in range(7)]
C, K = 24, 3


def splits():
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    names = sorted(n for n in names if (SRC / n / 'block000').is_dir())
    return names[:NFIT], names[NFIT:]


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def block_px(clip, k, s, e):
    d = SRC / clip / f'block{k:03d}'
    return torch.stack([load(d / f'{f:06d}.png') for f in range(s, e)])


def gt_px(clip, s, e):
    d = D_ / clip / 'hr'
    return torch.stack([load(d / f'{f:06d}.png') for f in range(s, e)])


def feat(z):
    """z [1,C,T,h,w] -> [N, C*K*K+1] design matrix (3x3 neighbourhood per latent site, plus bias)."""
    t = z[0].permute(1, 0, 2, 3)                              # [T,C,h,w]
    p = F.unfold(t, K, padding=K // 2)                        # [T, C*K*K, h*w]
    p = p.permute(0, 2, 1).reshape(-1, C * K * K)
    return torch.cat([p, torch.ones(p.shape[0], 1, dtype=p.dtype)], 1)


def target(z):
    return z[0].permute(1, 2, 3, 0).reshape(-1, C)            # [T*h*w, C]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['fit', 'solve', 'apply'])
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--lam', type=float, default=1e-3)
    ap.add_argument('--tag', default='m0')
    a = ap.parse_args()
    fitc, testc = splits()
    nq = OUT / f'{a.tag}_normal'
    nq.mkdir(parents=True, exist_ok=True)

    if a.phase == 'solve':
        d = C * K * K + 1
        A = torch.zeros(d, d, dtype=torch.float64)
        B = torch.zeros(d, C, dtype=torch.float64)
        n = 0
        for f in sorted(nq.glob('shard*.pt')):
            s = torch.load(f, map_location='cpu', weights_only=True)
            A += s['A']; B += s['B']; n += int(s['n'])
        scale = A.diagonal().mean()
        W = torch.linalg.solve(A + a.lam * scale * torch.eye(d, dtype=torch.float64), B)
        torch.save({'W': W.float(), 'n': n, 'lam': a.lam, 'fit_clips': fitc},
                   OUT / f'{a.tag}_W.pt')
        print(json.dumps({'solved': True, 'sites': n, 'W_norm': float(W.norm()),
                          'identity_dist': float((W[:, :].norm())), 'fit_clips': len(fitc)}))
        return

    from h3_comfy import H3
    h3 = H3(load_dit=False)

    if a.phase == 'fit':
        d = C * K * K + 1
        A = torch.zeros(d, d, dtype=torch.float64)
        B = torch.zeros(d, C, dtype=torch.float64)
        n = 0
        mine = [c for i, c in enumerate(fitc) if i % a.nshards == a.shard]
        for c in mine:
            for k, (s, e) in enumerate(BLOCKS):
                z = h3.encode(block_px(c, k, s, e)).float().cpu()
                zs = h3.encode(gt_px(c, s, e)).float().cpu()
                X, Y = feat(z).double(), target(zs).double()
                A += X.T @ X; B += X.T @ Y; n += X.shape[0]
            print(json.dumps({'fit': c, 'sites': n}), flush=True)
        torch.save({'A': A, 'B': B, 'n': n}, nq / f'shard{a.shard:02d}.pt')
        return

    W = torch.load(OUT / f'{a.tag}_W.pt', map_location='cpu', weights_only=True)['W']
    mine = [c for i, c in enumerate(testc) if i % a.nshards == a.shard]
    for c in mine:
        dirs = {t: OUT / f'{a.tag}_{t}' / c / 'stream' for t in ('orig', 'rt', 'map')}
        for p in dirs.values():
            p.mkdir(parents=True, exist_ok=True)
        for k, (s, e) in enumerate(BLOCKS):
            px = block_px(c, k, s, e)
            z = h3.encode(px)
            zf = z.float().cpu()
            rt = h3.decode(z).float()[:e - s].clamp(0, 1)
            zm = (feat(zf) @ W).reshape(zf.shape[2], zf.shape[3], zf.shape[4], C)
            zm = zm.permute(3, 0, 1, 2)[None].to(dtype=z.dtype, device=z.device)
            mp = h3.decode(zm).float()[:e - s].clamp(0, 1)
            first = 0 if k == 0 else 5          # the stream keeps frames from b['new'][0] = 17k+5 on
            for i in range(first, e - s):
                f = s + i
                for t, im in (('orig', px[i]), ('rt', rt[i]), ('map', mp[i])):
                    Image.fromarray(np.uint8(im.numpy().clip(0, 1) * 255 + .5)).save(
                        dirs[t] / f'{f:06d}.png')
        print(json.dumps({'applied': c}), flush=True)


if __name__ == '__main__':
    main()
