"""mech5 (PROTOCOL_mech.md): how much of the restorer's error lies in the subspace the observation fixes?

Range-null decomposition: x = U(D x) + N(x). The observation determines the first term exactly, so any
error there is error the model did not have to make. Step 2 measured that H3 under the same guide is 2.55 dB
behind a classical projection, which suggests the share is not negligible -- but that was before Stage-1
training and before innerDC. This measures it on the arm we actually ship.

If the range share is large, a residual parameterisation (predict only N, take the range part from the
observation) buys that part of the error for free, and it is a training-time change, not a projection.
If innerDC has already driven it to ~0, the remaining gap is purely null-space, which says the prior is the
only lever left and no observation-side mechanism can help.

Read-out rule, fixed before running:
  range share >= 0.20 -> residual parameterisation is worth building; report the implied PSNR headroom
  range share <  0.05 -> the observation side is already exhausted; say so and stop proposing mechanisms there
"""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/stage2_eval'
OUT = ROOT / 'results/mech5_error_decomp'
ARMS = {'indep': ROOT / 'results/stage2_blocks/indep',
        'cache': ROOT / 'results/stage5_blocks/cache/indep'}
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    """(range, null) = (U(D x), x - U(D x)) for the x4 antialiased bicubic observation operator."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    acc = {a: {'range': [], 'null': [], 'share': [], 'psnr': [], 'psnr_resid': []} for a in ARMS}
    rows = []
    for name in man['clips']:
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        r = {'clip': name}
        for arm, base in ARMS.items():
            sd = base / name / 'stream'
            if not sd.exists():
                continue
            er, en, nf = [], [], 0
            for t in range(0, T, STRIDE):
                f = sd / f'{t:06d}.png'
                if not f.exists():
                    continue
                g, y = load(D_ / name / 'hr' / f'{t:06d}.png'), load(f)
                gr, gn = split(g[None])
                yr, yn = split(y[None])
                er.append(float((yr - gr).square().mean()))
                en.append(float((yn - gn).square().mean()))
                nf += 1
            if not nf:
                continue
            mr, mn = float(np.mean(er)), float(np.mean(en))
            share = mr / max(mr + mn, 1e-12)
            # a residual parameterisation makes the range error exactly zero; the rest is unchanged
            psnr = 10 * np.log10(1.0 / max(mr + mn, 1e-12))
            psnr_r = 10 * np.log10(1.0 / max(mn, 1e-12))
            r[arm] = {'mse_range': mr, 'mse_null': mn, 'range_share': share,
                      'psnr': float(psnr), 'psnr_if_range_exact': float(psnr_r),
                      'headroom_dB': float(psnr_r - psnr), 'frames': nf}
            for k, v in (('range', mr), ('null', mn), ('share', share),
                         ('psnr', psnr), ('psnr_resid', psnr_r)):
                acc[arm][k].append(v)
        rows.append(r)
        print(json.dumps({'clip': name, **{a: {k: round(v, 5) if isinstance(v, float) else v
                                               for k, v in r[a].items()} for a in ARMS if a in r}}), flush=True)

    agg = {}
    for a in ARMS:
        if not acc[a]['share']:
            continue
        head = [p - q for p, q in zip(acc[a]['psnr_resid'], acc[a]['psnr'])]
        agg[a] = {'range_share': {'mean': float(np.mean(acc[a]['share'])), 'ci95': ci(acc[a]['share'])},
                  'mse_range': float(np.mean(acc[a]['range'])), 'mse_null': float(np.mean(acc[a]['null'])),
                  'psnr': float(np.mean(acc[a]['psnr'])),
                  'psnr_if_range_exact': float(np.mean(acc[a]['psnr_resid'])),
                  'headroom_dB': {'mean': float(np.mean(head)), 'ci95': ci(head)}}
    s = agg.get('indep', {}).get('range_share', {}).get('mean', 0.0)
    verdict = ('residual parameterisation is worth building' if s >= 0.20 else
               'the observation side is already exhausted' if s < 0.05 else 'partial')
    out = {'n': len(rows), 'per_clip': rows, 'aggregate': agg, 'verdict': verdict}
    (OUT / 'decomp.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'aggregate': agg, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
