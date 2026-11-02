"""The low band is over-smoothed; does putting the mid frequencies back recover the LPIPS?

cross_band located the loss precisely. On the 19 realistic clips our low band carries mid-frequency RMS
0.01407 against the truth's 0.01959 (-28%), InfVSR's 0.01698 (-13%). It is 2.95 dB closer to D(GT) than
theirs and still perceptually worse: with the truth's detail substituted it scores 0.0849 LPIPS against
their 0.0820. That is what an L1-trained estimator does -- an L1-optimal estimate of D(x) given y is a
conditional median, and conditional medians are smooth.

Two things ride on that low band, which is why the cost compounds:
  1. it is the output's entire low frequency content, U(R_phi(y));
  2. the quota gate scales the generated detail by its own local std, w = clip(std_8x8(guide)/0.02, 0, 1),
     so a smoother guide also admits less of the detail that would have made up for it.

The sampled stream is on disk and the quota step is pure post-processing, so the guide can be replaced
without touching the diffusion model. Variants, all at LR resolution before the bicubic upsample:

  rphi     R_phi(y), as shipped          y        the raw observation, full mid band but noisy
  mix50    half of each                  sharpNN  unsharp mask on R_phi(y), alpha = NN/10: restores the
                                                  band without re-adding the noise R_phi just removed
  gt       D(GT), the ceiling            rphi_gategt  shipped guide, gate driven by D(GT)'s local std,
                                                  to separate the guide's content from the gate it feeds

  oracNN   (1-NN/100)*R_phi(y) + NN/100*D(GT): a dose-response curve between the estimator we have and
           the one we cannot have, so the LPIPS bought per dB of low-band accuracy is measured rather
           than assumed. This is what decides whether training a better estimator is worth it.
  _gg      suffix: same guide, gate driven by D(GT)'s local std instead of the guide's own.
  blendNN  (1-NN/100) * U(sharpened R_phi(y)) + NN/100 * R(sampled): stop overwriting the sampler's own
           low band unconditionally and mix it in. On the 77 held-out realistic clips the margin by which
           R_phi beats the sampler's low band falls from +0.99 dB on the mildest quarter to +0.30 dB on
           the worst, and the LPIPS difference against InfVSR tracks it quartile for quartile (-0.0233 to
           +0.0027). Committing to one estimate is only worth its cost while that margin is large.

Reported with mid_rms of each composed output, and the range-space PSNR of the composed output against
D(GT), so the mechanism and the accuracy are visible next to the score.
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SHARP = 0.3                      # the unsharp alpha the shipped configuration uses
sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / 'data/real_eval/const'
SRC = Path(os.environ.get('GS_SRC', str(ROOT / 'results/deploy_runs/t4_dcr')))
CKPT = ROOT / 'results/range_branch/k1.pt'
STRIDE, S0 = 8, 0.02


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u[0].permute(1, 2, 0).float()
    return r, x - r


def up(lr, hw):
    return F.interpolate(lr.permute(2, 0, 1)[None].double(), size=hw, mode='bicubic',
                         align_corners=False)[0].permute(1, 2, 0).float()


def down(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    return d[0].permute(1, 2, 0).float().clamp(0, 1)


def _local_std(x, k=8):
    y = x.mean(-1)[None, None]
    m = F.avg_pool2d(y, k, k)
    s = (F.avg_pool2d(y * y, k, k) - m * m).clamp_min(0).sqrt()
    return F.interpolate(s, size=y.shape[-2:], mode='bilinear', align_corners=False)[0, 0, ..., None]


def _gk(sigma):
    r = max(1, int(round(3 * sigma)))
    a = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (a / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def blur(x, sigma):
    t = x.permute(2, 0, 1)[None]
    g = _gk(sigma).expand(3, 1, 1, -1)
    r = g.shape[-1] // 2
    b = F.conv2d(F.pad(t, (r, r, 0, 0), mode='reflect'), g, groups=3)
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2), groups=3)
    return b[0].permute(1, 2, 0)


def mid_rms(x, sigma=2.0):
    return float((x - blur(x, sigma)).square().mean().sqrt())


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def one(job):
    name, variants, ckpt, s0, gain = job
    torch.set_num_threads(2)
    from revisit_metrics import lpips_dist
    from range_branch import RangeBranch
    st = torch.load(ckpt, map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    m = json.loads((D_ / name / 'meta.json').read_text())
    T, hw = m['frames'], (m['hr_wh'][1], m['hr_wh'][0])
    ts = [t for t in range(0, T, STRIDE) if (SRC / name / 'stream' / f'{t:06d}.png').exists()]
    y = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in ts])
    with torch.no_grad():
        est = rb(y.permute(0, 3, 1, 2)[:, None]).clamp(0, 1).permute(0, 2, 3, 1)
    acc = {v: {'psnr': [], 'lpips': [], 'mid': [], 'rpsnr': []} for v in variants}
    obs = []
    for i, t in enumerate(ts):
        g = load(D_ / name / 'hr' / f'{t:06d}.png')
        gr, _ = split(g)
        ro, nn = split(load(SRC / name / 'stream' / f'{t:06d}.png'))
        r, yy = est[i], y[i]
        for v in variants:
            gate_src = None
            vv = v
            if vv.endswith('_gg'):
                vv, gate_src = vv[:-3], down(g)
            if vv == 'rphi':
                e = r
            elif vv == 'y':
                e = yy
            elif vv == 'gt':
                e = down(g)
            elif vv == 'rphi_gategt':
                e, gate_src = r, down(g)
            elif vv.startswith('mix'):
                a = int(vv[3:]) / 100
                e = (1 - a) * r + a * yy
            elif vv.startswith('orac'):
                a = int(vv[4:]) / 100
                e = (1 - a) * r + a * down(g)
            elif vv.startswith('sharp'):
                a = int(vv[5:]) / 10
                e = r + a * (r - blur(r, 1.0))
            elif vv.startswith('blend'):
                a = int(vv[5:]) / 100
                e = r + SHARP * (r - blur(r, 1.0))
            else:
                raise ValueError(v)
            guide = up(e.clamp(0, 1), hw).clamp(0, 1)
            if vv.startswith('blend'):
                b = int(vv[5:]) / 100
                guide = ((1 - b) * guide + b * ro).clamp(0, 1)
            gate = guide if gate_src is None else up(gate_src, hw).clamp(0, 1)
            w = (_local_std(gate) / s0).clamp(0, 1) * gain
            o = (guide + nn * w).clamp(0, 1)
            acc[v]['psnr'].append(psnr(o, g))
            acc[v]['lpips'].append(lpips_dist(o, g))
            acc[v]['mid'].append(mid_rms(split(o)[0]))
            acc[v]['rpsnr'].append(psnr(down(o), down(g)))
        acc.setdefault('_gt', {'mid': []})['mid'].append(mid_rms(gr))
        # severity of the observation, GT-free at inference but available here for grouping
        gl = F.interpolate(g.permute(2, 0, 1)[None].double(), size=yy.shape[:2], mode='bicubic',
                           align_corners=False, antialias=True)[0].permute(1, 2, 0).float().clamp(0, 1)
        obs.append(psnr(yy, gl))
    out = {'clip': name, 'degradation': m['degradation'].get('kind', '?'), 'n_frames': len(ts),
           'obs_psnr': float(np.mean(obs)),
           'rphi_shift': float((est - y).abs().mean())}      # GT-free severity proxy: how much R_phi moved y
    for v, d in acc.items():
        for k, vals in d.items():
            out[f'{v}.{k}'] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variants', default='rphi,y,mix25,mix50,sharp03,sharp06,sharp10,rphi_gategt,gt')
    ap.add_argument('--ckpt', default=str(CKPT))
    ap.add_argument('--gain', type=float, default=1.0,
                    help='multiplier on the admitted detail. The gate only ever removes detail, and the '
                         'detail energy of the 4-step output already sits at 0.41 of the truth, so a '
                         'value above 1 asks whether the generator is under-producing rather than '
                         'over-producing once the flat regions are handled. Costs nothing to run.')
    ap.add_argument('--s0', type=float, default=S0,
                    help='gate scale. Tuned to 0.02 on a grid against the incumbent R_phi; a less smooth '
                         'guide raises the same local std, so the optimum moves with the estimator.')
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/guide_sweep.json'))
    a = ap.parse_args()
    vs = a.variants.split(',')
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, [(n, vs, a.ckpt, a.s0, a.gain) for n in names]),
                      key=lambda r: r['clip'])
    groups = {'all': lambda r: True, 'realistic': lambda r: r['degradation'] == 'realistic',
              'bicubic': lambda r: r['degradation'] != 'realistic'}
    summ = {}
    for g, s in groups.items():
        rs = [r for r in rows if s(r)]
        summ[g] = {'n': len(rs), **{k: float(np.mean([r[k] for r in rs]))
                                    for k in rows[0] if k not in ('clip', 'degradation')}}
    Path(a.out).write_text(json.dumps({'ckpt': a.ckpt, 's0': a.s0, 'gain': a.gain, 'summary': summ,
                                       'per_clip': rows}, indent=1))
    for g, v in summ.items():
        print('### %s  n=%d  s0=%.3f  gain=%.2f  ckpt=%s  (GT low-band mid_rms %.5f)'
              % (g, v['n'], a.s0, a.gain, Path(a.ckpt).stem, v['_gt.mid']))
        print('   %-12s %8s %8s %9s %9s' % ('variant', 'PSNR', 'LPIPS', 'mid_rms', 'rangePSNR'))
        for x in vs:
            print('   %-12s %8.3f %8.4f %9.5f %9.3f'
                  % (x, v[f'{x}.psnr'], v[f'{x}.lpips'], v[f'{x}.mid'], v[f'{x}.rpsnr']))
        print()


if __name__ == '__main__':
    main()
