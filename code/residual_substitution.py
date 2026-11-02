"""mech5b: how much of the +2.84 dB range-space headroom is actually reachable from the DEGRADED observation?

error_decomposition.py showed 43.5% of the restorer's error lies in the range space, using the GT's own
range component U(D x_GT). That is an upper bound: at inference we hold y, which for most clips is not
D(x_GT) but a blurred / noisy / compressed version of it. This substitutes what we really have and reports
the gain that survives, per degradation kind.

Arms:
  as_is        the restorer's output, unchanged (baseline)
  sub_lr       x_hat <- U(y) + N(x_hat)            -- raw degraded observation supplies the range part
  sub_oracle   x_hat <- U(D x_GT) + N(x_hat)       -- the bound from error_decomposition, for reference

Read-out rule, fixed before running:
  sub_lr - as_is >= +0.5 dB with a CI excluding 0 -> a residual parameterisation is worth building, and the
  reachable share of the bound is reported; otherwise the bound is not reachable from y alone and the
  mechanism needs a range-space estimator trained for it, which must be said explicitly.

LPIPS is reported alongside PSNR and is the binding check. Substituting the upsampled observation puts a
bicubic low band back into the output, which can raise PSNR while making the frame look softer -- this
project's first lesson is that a proxy metric has to live in the same domain as the deliverable. The gain
only counts if LPIPS does not get worse.
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, '/cache/zihaohe/streaming_sr0/code')
from revisit_metrics import lpips_dist

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/stage2_eval'
SRC = ROOT / 'results/stage2_blocks/indep'
OUT = ROOT / 'results/mech5_error_decomp'
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def rng_null(x, s=4):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def up(lr, hw):
    return F.interpolate(lr.permute(0, 3, 1, 2).double(), size=hw, mode='bicubic',
                         align_corners=False).permute(0, 2, 3, 1).float()


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    man = json.loads((D_ / 'manifest.json').read_text())
    rows = []
    for name in man['clips']:
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        kind = meta['degradation'].get('kind', '?')
        a = {'as_is': [], 'sub_lr': [], 'sub_oracle': []}
        lp = {'as_is': [], 'sub_lr': [], 'sub_oracle': []}
        for t in range(0, T, STRIDE):
            f = SRC / name / 'stream' / f'{t:06d}.png'
            if not f.exists():
                continue
            g = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
            y = load(f)[None]
            lr = load(D_ / name / 'lr' / f'{t:06d}.png')[None]
            _, yn = rng_null(y)
            gr, _ = rng_null(g)
            ulr = up(lr, hw)
            s_lr = (ulr + yn).clamp(0, 1)
            s_or = (gr + yn).clamp(0, 1)
            a['as_is'].append(psnr(y, g))
            a['sub_lr'].append(psnr(s_lr, g))
            a['sub_oracle'].append(psnr(s_or, g))
            for tag, im in (('as_is', y), ('sub_lr', s_lr), ('sub_oracle', s_or)):
                lp[tag].append(lpips_dist(im[0], g[0]))
        r = {'clip': name, 'degradation': kind, **{k: float(np.mean(v)) for k, v in a.items()},
             **{f'lpips_{k}': float(np.mean(v)) for k, v in lp.items()}}
        r['gain_lr'] = r['sub_lr'] - r['as_is']
        r['gain_oracle'] = r['sub_oracle'] - r['as_is']
        r['dlpips_lr'] = r['lpips_sub_lr'] - r['lpips_as_is']
        r['dlpips_oracle'] = r['lpips_sub_oracle'] - r['lpips_as_is']
        r['reachable_share'] = r['gain_lr'] / r['gain_oracle'] if abs(r['gain_oracle']) > 1e-9 else float('nan')
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    keys = ('as_is', 'sub_lr', 'sub_oracle', 'gain_lr', 'gain_oracle',
            'lpips_as_is', 'lpips_sub_lr', 'lpips_sub_oracle', 'dlpips_lr', 'dlpips_oracle')
    agg = {k: {'mean': float(np.mean([r[k] for r in rows])), 'ci95': ci([r[k] for r in rows])} for k in keys}
    reach = [r['reachable_share'] for r in rows if np.isfinite(r['reachable_share'])]
    agg['reachable_share'] = {'mean': float(np.mean(reach)), 'ci95': ci(reach)}
    psnr_ok = agg['gain_lr']['mean'] >= 0.5 and agg['gain_lr']['ci95'][0] > 0
    lpips_ok = agg['dlpips_lr']['ci95'][1] <= 0.0        # not worse, interval must not reach above 0
    verdict = ('residual parameterisation is worth building from y alone' if psnr_ok and lpips_ok else
               'PSNR improves but LPIPS gets worse: raw substitution trades perception for fidelity, so the '
               'range part must come from a trained estimator rather than from U(y) directly' if psnr_ok else
               'the bound needs a trained range-space estimator; raw y is not enough')
    out = {'n': len(rows), 'per_clip': rows, 'aggregate': agg, 'verdict': verdict}
    (OUT / 'substitution_lpips.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'aggregate': agg, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
