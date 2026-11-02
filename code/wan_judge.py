"""Score the ported route on the 24-clip protocol, same frames and same code as judge13.

The arms produced by wan_sr_route only cover frames 0..120, which is exactly the stride-8 evaluation grid,
so the comparison against our shipped t4dcr_q and against InfVSR is like for like. Reported by degradation
kind, because the all-24 average mixes the 19 realistic clips with 5 clean bicubic ones and hides which
system wins where.

Also reports the range-space PSNR of each arm against D(GT): the ported generator is conditioned on a
bicubic upsample rather than on R_phi, so its low band is a different object, and that is the quantity our
post-stage would replace.
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
STRIDE, LAST = 8, 120


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def down(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    return d[0].permute(1, 2, 0).float().clamp(0, 1)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def frame(d, name, t):
    p = Path(d) / name / 'stream' / f'{t:06d}.png'
    return p if p.exists() else Path(d) / name / f'{t:06d}.png'


def one(job):
    name, arms = job
    torch.set_num_threads(2)
    from revisit_metrics import lpips_dist
    m = json.loads((D_ / name / 'meta.json').read_text())
    ts = list(range(0, LAST + 1, STRIDE))
    r = {'clip': name, 'degradation': m['degradation'].get('kind', '?')}
    gt = {t: load(D_ / name / 'hr' / f'{t:06d}.png') for t in ts}
    for tag, d in arms.items():
        fs = [frame(d, name, t) for t in ts]
        if not all(f.exists() for f in fs):
            continue
        p, l, rp = [], [], []
        for t, f in zip(ts, fs):
            x = load(f)
            if x.shape != gt[t].shape:
                r[tag + '_shape'] = list(x.shape)
                break
            p.append(psnr(x, gt[t])); l.append(lpips_dist(x, gt[t]))
            rp.append(psnr(down(x), down(gt[t])))
        if p:
            r[tag + '.psnr'] = float(np.mean(p)); r[tag + '.lpips'] = float(np.mean(l))
            r[tag + '.rpsnr'] = float(np.mean(rp))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='tag=path')
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/wan_judge.json'))
    a = ap.parse_args()
    arms = dict(x.split('=', 1) for x in a.arm)
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, [(n, arms) for n in names]), key=lambda r: r['clip'])
    groups = {'all': lambda r: True, 'realistic': lambda r: r['degradation'] == 'realistic',
              'bicubic': lambda r: r['degradation'] != 'realistic'}
    summ = {}
    for g, sel in groups.items():
        rs = [r for r in rows if sel(r)]
        summ[g] = {'n': len(rs)}
        for tag in arms:
            have = [r for r in rs if tag + '.psnr' in r]
            if not have:
                continue
            summ[g][tag] = {'n': len(have),
                            **{k: float(np.mean([r[f'{tag}.{k}'] for r in have]))
                               for k in ('psnr', 'lpips', 'rpsnr')}}
    Path(a.out).write_text(json.dumps({'arms': arms, 'summary': summ, 'per_clip': rows}, indent=1))
    for g, v in summ.items():
        print('### %s n=%d' % (g, v['n']))
        print('   %-14s %4s %8s %8s %10s' % ('arm', 'n', 'PSNR', 'LPIPS', 'rangePSNR'))
        for tag in arms:
            if tag in v:
                x = v[tag]
                print('   %-14s %4d %8.3f %8.4f %10.3f' % (tag, x['n'], x['psnr'], x['lpips'], x['rpsnr']))
        print()


if __name__ == '__main__':
    main()
