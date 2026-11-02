"""Whose low-frequency estimate is better: ours or the judge's?

judge13 says that on the 19 realistic clips we lose on LPIPS (0.2309 vs InfVSR's 0.2183) while winning on
the gtrange diagnostic (0.1431 vs 0.1541). Those two facts together can only mean our range component is
worse than InfVSR's, because gtrange replaces exactly that component with the truth. This measures the
range component directly instead of inferring it.

Everything is measured at LR resolution, where the range space is defined: D(x) is the evaluation
operator (torch bicubic, antialias) and the arms are

  y            the observation the stream receives -- the floor any estimator must beat
  rphi         R_phi(y), what the pipeline actually writes into the low band
  D(ours)      the low band our final output implies
  D(infvsr)    the same for the judge
  D(gt) = 0    by construction, the target

A gap here is directly actionable: it is a 1/4-resolution regression problem with no generative component
and a 0.6M-parameter incumbent, so it is the cheapest place in the system to buy LPIPS.
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
ARMS = {'ours': ROOT / 'results/deploy/t4dcr_q', 'infvsr': ROOT / 'results/baseline_infvsr'}
STRIDE = 8
H, W = 480, 832


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def down(x):
    """D: the evaluation operator. [H,W,3] -> [H/4,W/4,3]."""
    t = x.permute(2, 0, 1)[None].double()
    d = F.interpolate(t, size=(H // 4, W // 4), mode='bicubic', align_corners=False, antialias=True)
    return d[0].permute(1, 2, 0).float().clamp(0, 1)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


_RB = {}


def rphi(lr, ckpt):
    """lr [T,H,W,3] -> [T,H,W,3]. One batched call; the k1 checkpoint is K=1 so frames are independent."""
    if ckpt not in _RB:
        from range_branch import RangeBranch
        st = torch.load(ckpt, map_location='cpu')
        assert st['cfg']['k'] == 1, st['cfg']
        m = RangeBranch(**st['cfg']).eval()
        m.load_state_dict(st['state'])
        _RB[ckpt] = m
    with torch.no_grad():
        return _RB[ckpt](lr.permute(0, 3, 1, 2)[:, None]).clamp(0, 1).permute(0, 2, 3, 1)


def one(job):
    name, ckpts = job
    m = json.loads((D_ / name / 'meta.json').read_text())
    T = m['frames']
    ts = list(range(0, T, STRIDE))
    tgt = torch.stack([down(load(D_ / name / 'hr' / f'{t:06d}.png')) for t in ts])
    y = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in ts])
    acc = {'y': [psnr(y[i], tgt[i]) for i in range(len(ts))]}
    for tag, ck in ckpts.items():
        e = rphi(y, ck)
        acc[tag] = [psnr(e[i], tgt[i]) for i in range(len(ts))]
    for tag, d in ARMS.items():
        fs = [d / name / 'stream' / f'{t:06d}.png' for t in ts]
        if not all(f.exists() for f in fs):
            fs = [d / name / f'{t:06d}.png' for t in ts]
        if not all(f.exists() for f in fs):
            continue
        acc[tag] = [psnr(down(load(f)), tgt[i]) for i, f in enumerate(fs)]
    return {'clip': name, 'degradation': m['degradation'].get('kind', '?'),
            **{k: float(np.mean(v)) for k, v in acc.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', action='append', default=[], help='tag=path, repeatable')
    ap.add_argument('--procs', type=int, default=12)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/range_gap.json'))
    a = ap.parse_args()
    ckpts = dict(x.split('=', 1) for x in a.ckpt) or {'rphi': str(ROOT / 'results/range_branch/k1.pt')}
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, [(n, ckpts) for n in names]), key=lambda r: r['clip'])
    tags = ['y'] + list(ckpts) + [t for t in ARMS if t in rows[0]]
    groups = {'all': lambda r: True, 'realistic': lambda r: r['degradation'] == 'realistic',
              'bicubic': lambda r: r['degradation'] != 'realistic'}
    summ = {g: {'n': sum(1 for r in rows if s(r)),
                **{t: float(np.mean([r[t] for r in rows if s(r)])) for t in tags}}
            for g, s in groups.items()}
    Path(a.out).write_text(json.dumps({'tags': tags, 'summary': summ, 'per_clip': rows}, indent=1))
    print('%-10s %4s %s' % ('group', 'n', ' '.join('%9s' % t for t in tags)))
    for g, v in summ.items():
        print('%-10s %4d %s' % (g, v['n'], ' '.join('%9.3f' % v[t] for t in tags)))
    print()
    print('range-space PSNR against D(GT), dB. Higher is better; "y" is the untouched observation.')


if __name__ == '__main__':
    main()
