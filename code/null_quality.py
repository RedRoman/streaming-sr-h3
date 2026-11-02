"""Whose generated detail is better, once the low band is taken out of the comparison?

Porting the generator moved the bottleneck. Our post-stage lifted the Wan route's PSNR from 23.78 to
25.77 on the 19 realistic clips -- level with our own 20.11B four-step pipeline -- and its range-space
accuracy from 27.07 to 31.25 dB, but LPIPS only moved 0.2607 -> 0.2572. The guide_dose curve says +4.2 dB
of low band is worth about 0.05 LPIPS, so the discrepancy has to live in the component the post-stage
does not touch: the detail the generator invents.

This substitutes the ground truth's range into every arm, U(D x_GT) + N(arm), which is the same
diagnostic judge13 reports as lpips_gtrange. What is left is the detail alone. Reference points already
measured on these clips: ours 0.1431, InfVSR 0.1541.
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


def split(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u[0].permute(1, 2, 0).float()
    return r, x - r


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
    for tag, d in arms.items():
        fs = [frame(d, name, t) for t in ts]
        if not all(f.exists() for f in fs):
            continue
        lp, en = [], []
        for t, f in zip(ts, fs):
            g = load(D_ / name / 'hr' / f'{t:06d}.png')
            gr, gn = split(g)
            _, n = split(load(f))
            lp.append(lpips_dist((gr + n).clamp(0, 1), g))
            en.append(float(n.square().sum() / gn.square().sum().clamp_min(1e-12)))
        r[tag + '.lpips_gtrange'] = float(np.mean(lp))
        r[tag + '.null_energy'] = float(np.mean(en))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True)
    ap.add_argument('--procs', type=int, default=28)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/null_quality.json'))
    a = ap.parse_args()
    arms = dict(x.split('=', 1) for x in a.arm)
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, [(n, arms) for n in names]), key=lambda r: r['clip'])
    Path(a.out).write_text(json.dumps({'arms': arms, 'per_clip': rows}, indent=1))
    for g, sel in (('realistic', lambda r: r['degradation'] == 'realistic'),
                   ('bicubic', lambda r: r['degradation'] != 'realistic'),
                   ('all', lambda r: True)):
        rs = [r for r in rows if sel(r)]
        print('### %s n=%d' % (g, len(rs)))
        print('   %-10s %4s %16s %13s' % ('arm', 'n', 'LPIPS(gt range)', 'null energy'))
        for tag in arms:
            have = [r for r in rs if tag + '.lpips_gtrange' in r]
            if not have:
                continue
            print('   %-10s %4d %16.4f %13.3f'
                  % (tag, len(have), np.mean([r[tag + '.lpips_gtrange'] for r in have]),
                     np.mean([r[tag + '.null_energy'] for r in have])))
        print()
    print('reference: GT detail would score 0; ours 0.1431 and InfVSR 0.1541 on the realistic 19.')


if __name__ == '__main__':
    main()
