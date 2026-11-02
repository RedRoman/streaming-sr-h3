"""Our streaming pipeline on the real-footage evaluation set, against the published floor.

Everything before this ran on 12 generated clips whose unobserved band carries 20x less energy than real
footage (band_ratio 0.005 vs 0.09-0.10 measured during set construction), so the numbers could move a lot.
The arms are the same four as the generated-set table, computed with the same metric code:

  ours         frozen H3 + Stage-1 LoRA + innerDC, streamed block by block
  ours+range   the same outputs with the range component replaced by U(R_phi(y))
  oracle       the range component replaced by U(D x_GT) -- the bound, not a method
  realesrgan   read from the baseline run on this same set
"""
import json, os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, '/cache/zihaohe/streaming_sr0/code')
from revisit_metrics import lpips_dist
from range_branch import RangeBranch

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
SRC = ROOT / os.environ.get('OUR_OUT', 'results/real_const_indep/indep')
BASE = ROOT / os.environ.get('BASE_JSON', 'results/baseline_realesrgan_real_const/metrics.json')
CKPT = ROOT / ('results/range_branch/%s.pt' % os.environ.get('RB_TAG', 'k1'))
OUTD = ROOT / 'results/real_comparison'
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
    OUTD.mkdir(parents=True, exist_ok=True)
    st = torch.load(CKPT, map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    K = st['cfg']['k']
    base = {r['clip']: r for r in json.loads(BASE.read_text())['per_clip']} if BASE.exists() else {}
    man = json.loads((D_ / 'manifest.json').read_text())
    rows = []
    with torch.no_grad():
        for name in man['clips']:
            sd = SRC / name / 'stream'
            if not sd.exists():
                continue
            meta = json.loads((D_ / name / 'meta.json').read_text())
            T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
            lr_all = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
            acc = {k: {'psnr': [], 'lpips': []} for k in ('ours', 'ours_range', 'oracle')}
            for t in range(0, T, STRIDE):
                f = sd / f'{t:06d}.png'
                if not f.exists():
                    continue
                g = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
                y = load(f)[None]
                _, yn = rng_null(y)
                gr, _ = rng_null(g)
                stack = lr_all[max(0, t - K + 1):t + 1]
                if stack.shape[0] < K:
                    stack = torch.cat([stack[:1].repeat(K - stack.shape[0], 1, 1, 1), stack])
                est = rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0)[None]
                cand = {'ours': y,
                        'ours_range': (up(est, hw) + yn).clamp(0, 1),
                        'oracle': (gr + yn).clamp(0, 1)}
                for k, im in cand.items():
                    acc[k]['psnr'].append(psnr(im, g)); acc[k]['lpips'].append(lpips_dist(im[0], g[0]))
            r = {'clip': name, 'degradation': meta['degradation'].get('kind', '?'),
                 **{f'{k}_{m}': float(np.mean(v[m])) for k, v in acc.items() for m in ('psnr', 'lpips')}}
            if name in base:
                r['realesrgan_psnr'] = base[name]['psnr']; r['realesrgan_lpips'] = base[name]['lpips']
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    arms = [a for a in ('realesrgan', 'ours', 'ours_range', 'oracle') if f'{a}_psnr' in rows[0]]
    out = {'n': len(rows), 'rb_tag': os.environ.get('RB_TAG', 'k1'), 'K': K, 'per_clip': rows, 'arms': {}}
    for a in arms:
        out['arms'][a] = {m: {'mean': float(np.mean([r[f'{a}_{m}'] for r in rows])),
                              'ci95': ci([r[f'{a}_{m}'] for r in rows])} for m in ('psnr', 'lpips')}
    if 'realesrgan' in arms:
        out['vs_realesrgan'] = {}
        for a in ('ours', 'ours_range'):
            dp = [r[f'{a}_psnr'] - r['realesrgan_psnr'] for r in rows]
            dl = [r[f'{a}_lpips'] - r['realesrgan_lpips'] for r in rows]
            out['vs_realesrgan'][a] = {'psnr': {'mean': float(np.mean(dp)), 'ci95': ci(dp)},
                                       'lpips': {'mean': float(np.mean(dl)), 'ci95': ci(dl)}}
    dp = [r['ours_range_psnr'] - r['ours_psnr'] for r in rows]
    dl = [r['ours_range_lpips'] - r['ours_lpips'] for r in rows]
    out['range_branch_gain'] = {'psnr': {'mean': float(np.mean(dp)), 'ci95': ci(dp)},
                                'lpips': {'mean': float(np.mean(dl)), 'ci95': ci(dl)}}
    (OUTD / 'metrics.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
