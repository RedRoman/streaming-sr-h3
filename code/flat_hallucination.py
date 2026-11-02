"""Where does our generated detail go wrong: is it misplaced, or just wrong?

The panels show our output covering flat, dark regions with texture that is not in the ground truth, while
InfVSR leaves them flat. If that is the whole story, the error should concentrate where the ground truth
has little local structure, and our unobserved-band energy there should exceed GT's by a large factor while
InfVSR's does not.

Pixels are bucketed by GT local activity -- the std of the GT inside an 8x8 window, measured on the
ground truth so the bucketing never depends on the method being scored. For each bucket:

  e_ratio   mean squared N(output) over mean squared N(GT): 1.0 = as much detail as the truth has
  err       mean squared (N(output) - N(GT)): how wrong the detail is, in absolute terms

A method that hallucinates in flat areas shows e_ratio >> 1 in the lowest buckets. A method that is merely
blurry shows e_ratio < 1 everywhere.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
import deploy_eval as DE                                     # noqa: E402

OUT = ROOT / 'results/deploy'
EDGES = [0.0, 0.01, 0.02, 0.04, 0.08, 1.0]                   # GT local std, in [0,1] intensity units
STRIDE = 8


def local_std(g, k=8):
    y = g.mean(-1)[None, None]
    m = F.avg_pool2d(y, k, k)
    m2 = F.avg_pool2d(y * y, k, k)
    s = (m2 - m * m).clamp_min(0).sqrt()
    return F.interpolate(s, size=y.shape[-2:], mode='nearest')[0, 0]


def work(job):
    name, arms = job
    torch.set_num_threads(3)
    m = DE.meta(name)
    T = m['frames']
    nb = len(EDGES) - 1
    acc = {t: {'num': np.zeros(nb), 'den': np.zeros(nb), 'err': np.zeros(nb), 'n': np.zeros(nb)} for t in arms}
    for t in range(0, T, STRIDE):
        g = DE.load(DE.D_ / name / 'hr' / f'{t:06d}.png')
        _, gn = DE.split(g)
        s = local_std(g)
        bucket = torch.bucketize(s, torch.tensor(EDGES[1:-1])).numpy()
        gpow = gn.square().mean(-1).numpy()
        for tag, p in arms.items():
            y = DE.arm_frame(p, name, t)
            if y is None:
                continue
            _, yn = DE.split(y)
            ypow = yn.square().mean(-1).numpy()
            epow = (yn - gn).square().mean(-1).numpy()
            for b in range(nb):
                sel = bucket == b
                if not sel.any():
                    continue
                a = acc[tag]
                a['num'][b] += ypow[sel].sum(); a['den'][b] += gpow[sel].sum()
                a['err'][b] += epow[sel].sum(); a['n'][b] += sel.sum()
    return name, {t: {k: v.tolist() for k, v in a.items()} for t, a in acc.items()}


def main():
    arms = dict(x.split('=', 1) for x in sys.argv[1:])
    names = DE.clips()
    rows = {}
    with Pool(int(os.environ.get('PROCS', 24))) as pool:
        for name, a in pool.imap_unordered(work, [(n, arms) for n in names]):
            rows[name] = a
            print(json.dumps({'done': name}), flush=True)

    kinds = {n: DE.meta(n)['degradation'].get('kind', '?') for n in names}
    out = {'edges': EDGES, 'arms': arms, 'per_clip': rows, 'groups': {}}
    nb = len(EDGES) - 1
    for gname, sel in (('all', lambda n: True), ('realistic', lambda n: kinds[n] == 'realistic')):
        ns = [n for n in names if sel(n)]
        out['groups'][gname] = {'n': len(ns)}
        for tag in arms:
            num = np.zeros(nb); den = np.zeros(nb); err = np.zeros(nb); cnt = np.zeros(nb)
            for n in ns:
                a = rows[n].get(tag)
                if not a:
                    continue
                num += np.array(a['num']); den += np.array(a['den'])
                err += np.array(a['err']); cnt += np.array(a['n'])
            out['groups'][gname][tag] = {
                'e_ratio': (num / np.maximum(den, 1e-12)).tolist(),
                'err_x1e4': (1e4 * err / np.maximum(cnt, 1)).tolist(),
                'pixel_share': (cnt / max(cnt.sum(), 1)).tolist()}
    (OUT / 'flat_hallucination.json').write_text(json.dumps(out, indent=1))

    for gname, g in out['groups'].items():
        print(f"\n=== {gname} (n={g['n']})  GT local std buckets {EDGES}")
        print('%-12s %s' % ('arm', ' '.join('%9s' % f'[{EDGES[i]:.2f},{EDGES[i+1]:.2f})' for i in range(nb))))
        print('%-12s %s' % ('pixel share', ' '.join('%9.3f' % x for x in g[list(arms)[0]]['pixel_share'])))
        for tag in arms:
            print('%-12s %s   <- energy / GT' % (tag, ' '.join('%9.2f' % x for x in g[tag]['e_ratio'])))
        for tag in arms:
            print('%-12s %s   <- squared error x1e4' % (tag, ' '.join('%9.2f' % x for x in g[tag]['err_x1e4'])))


if __name__ == '__main__':
    main()
