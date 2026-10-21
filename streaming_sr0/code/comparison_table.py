"""Assemble the one comparison that tells us where we actually stand.

Every arm here is evaluated on the same 12 clips, the same LR frames and the same metric code:
  realesrgan   a published real-world SR baseline (RRDBNet x4, 16.7M params, per-frame)
  ours         frozen H3 + Stage-1 LoRA + innerDC, streamed block by block
  ours+range   the same outputs with the range component replaced by U(R_phi(y))
  oracle       the same outputs with the range component replaced by U(D x_GT) -- the bound, not a method

Split by degradation kind, because mech8 measured that the range space sits at 58 dB under Pillow-BICUBIC
and 30 dB under the realistic chain, and the two regimes behave differently throughout this project.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    base = json.loads((ROOT / 'results/baseline_realesrgan/metrics.json').read_text())
    sub = json.loads((ROOT / 'results/mech5_error_decomp/substitution_lpips.json').read_text())
    hr = json.loads((ROOT / 'results/mech7_hr/hr_k1.json').read_text())
    bl = {r['clip']: r for r in base['per_clip']}
    su = {r['clip']: r for r in sub['per_clip']}
    rp = {r['clip']: r for r in hr['per_clip']}
    rows = []
    for c, s in su.items():
        if c not in bl or c not in rp:
            continue
        rows.append({'clip': c, 'degradation': s['degradation'],
                     'realesrgan_psnr': bl[c]['psnr'], 'realesrgan_lpips': bl[c]['lpips'],
                     'ours_psnr': s['as_is'], 'ours_lpips': s['lpips_as_is'],
                     'ours_range_psnr': rp[c]['sub_rphi'], 'ours_range_lpips': rp[c]['lpips_sub_rphi'],
                     'oracle_psnr': s['sub_oracle'], 'oracle_lpips': s['lpips_sub_oracle']})
    arms = ['realesrgan', 'ours', 'ours_range', 'oracle']
    out = {'n': len(rows), 'per_clip': rows, 'groups': {}}
    for label, sel in (('all', lambda r: True),
                       ('realistic', lambda r: r['degradation'] == 'realistic'),
                       ('bicubic', lambda r: r['degradation'] != 'realistic')):
        g = [r for r in rows if sel(r)]
        if not g:
            continue
        out['groups'][label] = {'n': len(g), **{a: {
            'psnr': {'mean': float(np.mean([r[f'{a}_psnr'] for r in g])), 'ci95': ci([r[f'{a}_psnr'] for r in g])},
            'lpips': {'mean': float(np.mean([r[f'{a}_lpips'] for r in g])), 'ci95': ci([r[f'{a}_lpips'] for r in g])}}
            for a in arms}}
        # paired differences against the published baseline
        for a in ('ours', 'ours_range'):
            dp = [r[f'{a}_psnr'] - r['realesrgan_psnr'] for r in g]
            dl = [r[f'{a}_lpips'] - r['realesrgan_lpips'] for r in g]
            out['groups'][label][f'{a}_minus_realesrgan'] = {
                'psnr': {'mean': float(np.mean(dp)), 'ci95': ci(dp)},
                'lpips': {'mean': float(np.mean(dl)), 'ci95': ci(dl)}}
    (ROOT / 'results/comparison.json').write_text(json.dumps(out, indent=1))

    for label, g in out['groups'].items():
        print(f"\n=== {label} (n={g['n']}) ===")
        print('%-12s %18s %18s' % ('arm', 'PSNR', 'LPIPS'))
        for a in arms:
            print('%-12s %8.3f %-9s %7.4f %-9s' % (
                a, g[a]['psnr']['mean'], str([round(x, 2) for x in g[a]['psnr']['ci95']]),
                g[a]['lpips']['mean'], str([round(x, 3) for x in g[a]['lpips']['ci95']])))
        for a in ('ours', 'ours_range'):
            d = g[f'{a}_minus_realesrgan']
            print('  %-20s PSNR %+.3f %s | LPIPS %+.4f %s' % (
                f'{a} - realesrgan', d['psnr']['mean'], [round(x, 2) for x in d['psnr']['ci95']],
                d['lpips']['mean'], [round(x, 4) for x in d['lpips']['ci95']]))


if __name__ == '__main__':
    main()
