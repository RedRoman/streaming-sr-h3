"""Upper bound on any per-site energy recalibration of the null band.

The proposal is to learn a multiplicative factor g per site (and per band) so the decoded null band's
energy matches a severity-predicted target, on the argument that the direction is already right and only
the amplitude is wrong. That argument rests on a latent-domain number (detail cosine 0.86 against the
guide) which is a different decomposition from the pixel null band, where the cosine is 0.48.

Rather than argue, bound it. A per-site gain can only rescale a fixed direction, never rotate it, so the
best any such scheme can do is the ORACLE gain computed from the ground truth:

    g*(site) = <n_ours, n_gt> / ||n_ours||^2        (least-squares optimal, per site)

Any learned gain must predict g* from the observation alone, so the oracle is a hard ceiling. Three
variants, increasingly generous:
    global      one g for the whole frame
    site        one g per 8x8 window
    site_band   one g per 8x8 window per Laplacian band (3 bands), i.e. the full "per site per band" form
Reported as the null-band PSNR against the truth, the resulting null cosine (which a gain cannot change
within a site but can across sites), and the energy ratio. Compared against the shipped stream untouched.

The 12-point hand-tuned version of this axis (results/deploy/judge_sharp.json, _quota with gain and
sharpness on 96 clips) was monotonically worse than shipping on every metric. This says whether that was
a tuning failure or a ceiling.
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
SRC = ROOT / os.environ.get('SRC', 'results/deploy_runs/t4_plain2')
STRIDE = 8
WIN = 8
EPS = 1e-8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def null_of(x):
    t = x.permute(2, 0, 1)[None].double()
    d = F.interpolate(t, scale_factor=.25, mode='bicubic', align_corners=False, antialias=True)
    r = F.interpolate(d, size=t.shape[-2:], mode='bicubic', align_corners=False)
    return (t - r)[0]                                        # [3,H,W] double


def pool(x, w=WIN):
    """sum over non-overlapping w x w windows, per channel -> [3, H/w, W/w]"""
    return F.avg_pool2d(x[None], w, w)[0] * (w * w)


def bands(x, n=3):
    """Laplacian bands of [3,H,W]: n-1 differences of successive blurs plus the residual."""
    out, cur = [], x[None]
    for i in range(n - 1):
        lo = F.interpolate(F.interpolate(cur, scale_factor=.5, mode='bilinear', align_corners=False,
                                         antialias=True), size=cur.shape[-2:], mode='bilinear',
                           align_corners=False)
        out.append((cur - lo)[0]); cur = lo
    out.append(cur[0])
    return out


def apply_gain(n_o, n_g, w):
    """oracle least-squares gain on w x w windows, applied; returns the corrected null band."""
    num = pool(n_o * n_g, w)
    den = pool(n_o * n_o, w).clamp_min(EPS)
    g = num / den
    gu = F.interpolate(g[None], size=n_o.shape[-2:], mode='nearest')[0]
    return n_o * gu, g


def score(nh, ng):
    a, b = nh.reshape(-1), ng.reshape(-1)
    # The energy ratio divides by the truth's own null energy, which is ~0 on near-static or flat frames
    # and produced inf/nan in the first run. Those frames are dropped from the ratio only; PSNR and cosine
    # stay defined on them, so the sample size differs per metric and is reported.
    bb = float(b @ b)
    r = float((a @ a) / bb) if bb > 1e-6 else None
    return {'null_psnr': float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12)),
            'null_cos': float((a @ b) / (a.norm() * b.norm() + EPS)),
            'null_energy_ratio': r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=24)
    a = ap.parse_args()
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    names = [n for n in names if (SRC / n / 'stream').is_dir()][:a.clips]
    acc = {}
    for c in names:
        m = json.loads((D_ / c / 'meta.json').read_text())
        for t in range(0, m['frames'], STRIDE):
            f = SRC / c / 'stream' / f'{t:06d}.png'
            if not f.exists():
                continue
            n_o, n_g = null_of(load(f)), null_of(load(D_ / c / 'hr' / f'{t:06d}.png'))
            acc.setdefault('shipped', []).append(score(n_o, n_g))
            gl = (n_o.reshape(-1) @ n_g.reshape(-1)) / (n_o.reshape(-1) @ n_o.reshape(-1)).clamp_min(EPS)
            acc.setdefault('oracle_global', []).append(score(n_o * gl, n_g))
            acc['oracle_global'][-1]['g_mean'] = float(gl)
            nh, g = apply_gain(n_o, n_g, WIN)
            r = score(nh, n_g); r['g_mean'] = float(g.mean()); r['g_median'] = float(g.median())
            acc.setdefault('oracle_site', []).append(r)
            bo, bg = bands(n_o), bands(n_g)
            nb = sum(apply_gain(x, y, WIN)[0] for x, y in zip(bo, bg))
            acc.setdefault('oracle_site_band', []).append(score(nb, n_g))
            # how much of the per-site gain's win is reachable from the observation at all: the oracle g
            # is a per-site number, so report its spread -- a learned predictor has to reproduce it
            acc.setdefault('g_spread', []).append({'g_p10': float(g.flatten().quantile(.1)),
                                                   'g_p50': float(g.flatten().quantile(.5)),
                                                   'g_p90': float(g.flatten().quantile(.9))})
        print(json.dumps({'clip': c, 'frames': len(acc['shipped'])}), flush=True)
    out = {}
    for k, v in acc.items():
        out[k] = {}
        for m in v[0]:
            vals = [d[m] for d in v if d.get(m) is not None]
            out[k][m] = round(float(np.mean(vals)), 4) if vals else None
            if len(vals) != len(v):
                out[k][m + '_n'] = len(vals)
    out['_note'] = ('oracle_* use the ground truth to pick the gain, so they are ceilings no learned '
                    'predictor can exceed. If oracle_site_band is close to shipped, the axis has no room.')
    out['_clips'] = len(names)
    out['_frames'] = len(acc['shipped'])
    (ROOT / 'results/deploy/gain_oracle.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


main()
