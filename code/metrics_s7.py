"""Stage-7 decode-domain metrics: the number that decides whether the super-resolution part is finished.

Reference points on this same 24-clip real-footage set, same metric code:
  Real-ESRGAN   PSNR 24.470  LPIPS 0.2359   (the published floor)
  previous H3   PSNR 25.478  LPIPS 0.2774   (Stage-1 LoRA, trained on generated clips)
  + range branch              LPIPS 0.2467
  oracle bound  PSNR 28.148  LPIPS 0.1912   (range component replaced by the truth)

Velocity MSE said the oracle-guided training is 39% better, but that is a proxy: this project's first
lesson was a latent metric that moved the opposite way from decoded quality, so only this table counts.
"""
import json, os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import sys
sys.path.insert(0, '/cache/zihaohe/streaming_sr0/code')
from revisit_metrics import lpips_dist

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
EV = ROOT / 'results/stage7_eval'
BASE = ROOT / 'results/baseline_realesrgan_real_const/metrics.json'
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    man = json.loads((D_ / 'manifest.json').read_text())
    base = {r['clip']: r for r in json.loads(BASE.read_text())['per_clip']} if BASE.exists() else {}
    arms = [a for a in ('raw', 'oracle', 'perc', 'perc_hi') if (EV / a).exists()]
    rows = []
    for name in man['clips']:
        r = {'clip': name, 'degradation': json.loads((D_ / name / 'meta.json').read_text())['degradation'].get('kind', '?')}
        ok = True
        for a in arms:
            sd = EV / a / name / 'stream'
            if not sd.exists():
                ok = False
                continue
            ps, lp = [], []
            for t in range(0, man['frames'], STRIDE):
                f = sd / f'{t:06d}.png'
                if not f.exists():
                    continue
                g = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
                y = load(f)[None]
                ps.append(psnr(y, g)); lp.append(lpips_dist(y[0], g[0]))
            if not ps:
                ok = False
                continue
            r[f'{a}_psnr'], r[f'{a}_lpips'] = float(np.mean(ps)), float(np.mean(lp))
        if name in base:
            r['realesrgan_psnr'], r['realesrgan_lpips'] = base[name]['psnr'], base[name]['lpips']
        if ok:
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    present = [a for a in ('realesrgan', *arms) if f'{a}_psnr' in rows[0]]
    out = {'n': len(rows), 'per_clip': rows, 'arms': {}}
    for a in present:
        out['arms'][a] = {m: {'mean': float(np.mean([r[f'{a}_{m}'] for r in rows])),
                              'ci95': ci([r[f'{a}_{m}'] for r in rows])} for m in ('psnr', 'lpips')}
    out['paired'] = {}
    for a in arms:
        for ref in ('realesrgan',):
            if f'{ref}_psnr' not in rows[0]:
                continue
            dp = [r[f'{a}_psnr'] - r[f'{ref}_psnr'] for r in rows]
            dl = [r[f'{a}_lpips'] - r[f'{ref}_lpips'] for r in rows]
            out['paired'][f'{a} - {ref}'] = {'psnr': {'mean': float(np.mean(dp)), 'ci95': ci(dp)},
                                             'lpips': {'mean': float(np.mean(dl)), 'ci95': ci(dl)}}
    if 'oracle' in arms and 'raw' in arms:
        dp = [r['oracle_psnr'] - r['raw_psnr'] for r in rows]
        dl = [r['oracle_lpips'] - r['raw_lpips'] for r in rows]
        out['paired']['oracle - raw'] = {'psnr': {'mean': float(np.mean(dp)), 'ci95': ci(dp)},
                                         'lpips': {'mean': float(np.mean(dl)), 'ci95': ci(dl)}}
    (EV / 'metrics.json').write_text(json.dumps(out, indent=1))
    print()
    print('%-12s %22s %22s' % ('arm', 'PSNR', 'LPIPS'))
    for a, v in out['arms'].items():
        print('%-12s %8.3f %-13s %7.4f %-13s' % (
            a, v['psnr']['mean'], str([round(x, 2) for x in v['psnr']['ci95']]),
            v['lpips']['mean'], str([round(x, 3) for x in v['lpips']['ci95']])))
    print()
    for k, v in out['paired'].items():
        print('%-22s PSNR %+.3f %s | LPIPS %+.4f %s' % (
            k, v['psnr']['mean'], [round(x, 2) for x in v['psnr']['ci95']],
            v['lpips']['mean'], [round(x, 4) for x in v['lpips']['ci95']]))


if __name__ == '__main__':
    main()
