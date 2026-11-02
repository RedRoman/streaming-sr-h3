"""Spatially adaptive detail quota: suppress generation where the observation says there is nothing there.

flat_hallucination.py measured that on realistic footage our output carries 2.95x the ground truth's
unobserved-band energy in the flattest half of the pixels, where InfVSR carries 1.03x, and that our squared
error there is three times InfVSR's and twice plain bicubic's. In the textured fifth of the pixels we are
within 4% of InfVSR. So the perceptual gap is not a detail deficit -- it is detail in the wrong places.

Stage 3 already tried an energy quota, but a single scalar per block: it cannot separate a flat sky from a
brick wall inside the same frame. This one is per pixel and reads only the observation, so it is deployable:

    w(x) = clip((s(x) / s0) ** p, 0, 1),  out' = R(out) + N(out) * w

where s is the local standard deviation of U(R_phi(y)) in an 8x8 window -- the same range estimate the
substitution arm already uses, which is denoised relative to raw y and so a steadier activity signal.

The one hyper-parameter pair is chosen on the 16 clips that never entered Stage-7 training and reported on
all 24, so the choice cannot be read off the clips it is scored on.
"""
import itertools, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
import deploy_eval as DE                                     # noqa: E402

OUT = ROOT / 'results/deploy'
STRIDE = 8
GRID = list(itertools.product([0.01, 0.02, 0.04], [0.5, 1.0]))


def local_std(x, k=8):
    y = x.mean(-1)[None, None]
    m = F.avg_pool2d(y, k, k)
    s = (F.avg_pool2d(y * y, k, k) - m * m).clamp_min(0).sqrt()
    return F.interpolate(s, size=y.shape[-2:], mode='bilinear', align_corners=False)[0, 0, ..., None]


def work(job):
    name, src, rphi = job
    torch.set_num_threads(3)
    from revisit_metrics import lpips_dist
    from range_branch import RangeBranch
    st = torch.load(rphi, map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    K = st['cfg']['k']
    m = DE.meta(name)
    T, hw = m['frames'], (m['hr_wh'][1], m['hr_wh'][0])
    lr = torch.stack([DE.load(DE.D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
    acc = {cfg: {'psnr': [], 'lpips': []} for cfg in GRID + ['base']}
    with torch.no_grad():
        for t in range(0, T, STRIDE):
            y = DE.arm_frame(src, name, t)
            if y is None:
                continue
            g = DE.load(DE.D_ / name / 'hr' / f'{t:06d}.png')
            stack = lr[max(0, t - K + 1):t + 1]
            if stack.shape[0] < K:
                stack = torch.cat([stack[:1].repeat(K - stack.shape[0], 1, 1, 1), stack])
            est = rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0)
            guide = DE.up(est, hw).clamp(0, 1)
            s = local_std(guide)
            yr, yn = DE.split(y)
            base = (guide + yn).clamp(0, 1)                   # the range-substituted arm, unchanged
            acc['base']['psnr'].append(DE.psnr(base, g)); acc['base']['lpips'].append(lpips_dist(base, g))
            for (s0, p) in GRID:
                w = (s / s0).clamp(0, 1) ** p if p != 1.0 else (s / s0).clamp(0, 1)
                o = (guide + yn * w).clamp(0, 1)
                acc[(s0, p)]['psnr'].append(DE.psnr(o, g)); acc[(s0, p)]['lpips'].append(lpips_dist(o, g))
    return name, {str(k): {m_: float(np.mean(v[m_])) for m_ in ('psnr', 'lpips')}
                  for k, v in acc.items() if v['psnr']}


def main():
    src = sys.argv[1]
    rphi = ROOT / 'results/range_branch/k1.pt'
    names = DE.clips()
    rows = {}
    with Pool(int(os.environ.get('PROCS', 24))) as pool:
        for name, a in pool.imap_unordered(work, [(n, src, rphi) for n in names]):
            rows[name] = a
            print(json.dumps({'done': name}), flush=True)
    kinds = {n: DE.meta(n)['degradation'].get('kind', '?') for n in names}
    keys = ['base'] + [str(c) for c in GRID]
    out = {'src': src, 'grid': [list(c) for c in GRID], 'per_clip': rows, 'groups': {}}
    for gname, sel in (('all', lambda n: True), ('realistic', lambda n: kinds[n] == 'realistic'),
                       ('clean16', lambda n: n not in DE.LEAKED)):
        ns = [n for n in names if sel(n) and n in rows]
        out['groups'][gname] = {'n': len(ns)}
        for k in keys:
            v = [rows[n][k] for n in ns if k in rows[n]]
            if not v:
                continue
            d = [rows[n][k]['lpips'] - rows[n]['base']['lpips'] for n in ns if k in rows[n]]
            out['groups'][gname][k] = {
                'psnr': float(np.mean([x['psnr'] for x in v])),
                'lpips': float(np.mean([x['lpips'] for x in v])),
                'lpips_vs_base': float(np.mean(d)), 'ci95': DE.ci(d) if any(d) else [0.0, 0.0],
                'better': int(sum(x < 0 for x in d)), 'n': len(v)}
    (OUT / 'spatial_quota.json').write_text(json.dumps(out, indent=1))
    for gname, g in out['groups'].items():
        print(f"\n=== {gname} (n={g['n']})")
        print('%-14s %8s %8s %16s %10s' % ('cfg(s0,p)', 'PSNR', 'LPIPS', 'dLPIPS vs base', 'better'))
        for k in keys:
            if k not in g:
                continue
            v = g[k]
            print('%-14s %8.3f %8.4f %+10.4f %s %4d/%d' % (
                k, v['psnr'], v['lpips'], v['lpips_vs_base'],
                str([round(x, 4) for x in v['ci95']]), v['better'], v['n']))


if __name__ == '__main__':
    main()
