"""Band-resolved energy of each arm's output against ground truth, as a pooled ratio.

The sharpness question could not be answered from the existing metric. `null_energy` is a per-frame ratio
sum|N(out)|^2 / sum|N(gt)|^2, and on near-static or smooth clips the denominator approaches zero, so
individual clips reach 6.4e12 and the MEAN over clips is meaningless (reported as 8.3e10 for our arm).
Only its median was usable, and a median of ratios still hides which band the deficit is in.

This pools instead of averaging ratios: the numerator and denominator are summed over every scored frame
of every clip first, then divided once. That is defined even when some clips have almost no detail, and it
is the quantity "how much of the truth's energy in this band did the method produce".

Three bands, from the same x4 operator the whole project is defined by:
  range     U(D x)                      what the observation determines
  null      x - U(D x)                  what it does not -- where sharpness lives
  mid       the sigma=2 high pass of the range band, at HR; cross_band identified the mid-frequency
            deficit as the actual defect, and R_phi's own mid_ratio has plateaued at 0.8831

A ratio of 1.0 means matched energy. Below 1.0 is under-sharp, above is over-sharp: FlashVSR's null
median was 1.06, i.e. it overshoots, while ours is 0.60. Energy alone does not say the detail is correct,
so the cosine against the truth's band is reported next to it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def gauss(sigma):
    r = max(1, int(round(3 * sigma)))
    g = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)[None, None].repeat(3, 1, 1, 1), r


def highpass(x, sigma=2.0):
    k, r = gauss(sigma)
    t = x.permute(0, 3, 1, 2)
    lo = F.conv2d(F.pad(t, (r, r, r, r), mode='reflect'), k, groups=3)
    return (t - lo).permute(0, 2, 3, 1)


def arm_frame(arm, D_, name, t, hw):
    if arm == 'bicubic':
        lr = load(D_ / name / 'lr' / f'{t:06d}.png')
        return F.interpolate(lr.permute(2, 0, 1)[None], size=hw, mode='bicubic',
                             align_corners=False).clamp(0, 1)[0].permute(1, 2, 0)
    for p in (Path(arm) / name / 'stream' / f'{t:06d}.png', Path(arm) / name / f'{t:06d}.png'):
        if p.exists():
            return load(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='tag=path | tag=bicubic')
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clips', type=int, default=0, help='0 = all realistic clips')
    ap.add_argument('--out', default='results/band_energy.json')
    a = ap.parse_args()
    D_ = ROOT / a.eval_set
    arms = dict(x.split('=', 1) for x in a.arm)
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    names = [n for n in names
             if json.loads((D_ / n / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    if a.clips:
        names = names[:a.clips]

    acc = {k: {b: {'num': 0.0, 'den': 0.0, 'dot': 0.0, 'n2': 0.0} for b in ('range', 'null', 'mid')}
           for k in arms}
    frames = 0
    for i, name in enumerate(names):
        m = json.loads((D_ / name / 'meta.json').read_text())
        T, hw = m['frames'], (m['hr_wh'][1], m['hr_wh'][0])
        for t in range(0, T, STRIDE):
            gt = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
            gr, gn = split(gt)
            gm = highpass(gr)
            bands_gt = {'range': gr, 'null': gn, 'mid': gm}
            for tag, path in arms.items():
                y = arm_frame(path, D_, name, t, hw)
                if y is None:
                    continue
                yr, yn = split(y[None])
                ym = highpass(yr)
                for b, o in (('range', yr), ('null', yn), ('mid', ym)):
                    g = bands_gt[b]
                    acc[tag][b]['num'] += float(o.square().sum())
                    acc[tag][b]['den'] += float(g.square().sum())
                    acc[tag][b]['dot'] += float((o * g).sum())
                    acc[tag][b]['n2'] += float(o.square().sum())
            frames += 1
        if (i + 1) % 10 == 0:
            print(json.dumps({'clips_done': i + 1, 'of': len(names)}), flush=True)

    out = {'eval_set': a.eval_set, 'clips': len(names), 'frames_scored': frames // max(len(arms), 1),
           'stride': STRIDE, 'arms': {}}
    for tag in arms:
        out['arms'][tag] = {}
        for b, v in acc[tag].items():
            den = max(v['den'], 1e-12)
            out['arms'][tag][b] = {
                'energy_ratio': v['num'] / den,
                'cos': v['dot'] / (max(v['n2'], 1e-12) ** 0.5 * den ** 0.5),
            }
    (ROOT / a.out).write_text(json.dumps(out, indent=1))

    print()
    print('pooled over %d clips, every %dth frame. 1.0 = matched energy' % (len(names), STRIDE))
    print('%-16s %10s %8s %10s %8s %10s %8s'
          % ('arm', 'range E', 'cos', 'null E', 'cos', 'mid E', 'cos'))
    for tag in arms:
        r = out['arms'][tag]
        print('%-16s %10.4f %8.4f %10.4f %8.4f %10.4f %8.4f'
              % (tag, r['range']['energy_ratio'], r['range']['cos'],
                 r['null']['energy_ratio'], r['null']['cos'],
                 r['mid']['energy_ratio'], r['mid']['cos']))
    print()
    print('wrote', ROOT / a.out)


if __name__ == '__main__':
    main()
