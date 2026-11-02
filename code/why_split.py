"""Two claims I made without checking, checked.

(1) "the operator code barely learns": theta MAE sat at 0.277 after 4200 steps. That number means nothing
    without the trivial predictors. The code is 8-dimensional in [0,1] and a quarter of the clips are the
    clean regime whose code is all-zero, so predicting the dataset mean, or predicting zeros, already
    scores something. Those baselines are computed here.

(2) "the LPIPS lead shrank because the knobs overfit the 24 clips": on the 77 held-out realistic clips we
    win 43/77 with a sign-test p of 0.36, i.e. a coin flip with heavy tails (-0.162 to +0.161). That is
    not the signature of a small systematic bias; it is clip-dependent. So the question is what separates
    the clips we win from the ones we lose. Severity is the obvious candidate: our whole pipeline is a
    denoise-then-regenerate, which should pay off more the more degraded the observation is. Measured as
    the range-space PSNR of the raw observation against D(GT) -- the same quantity range_gap reports --
    and correlated against the per-clip LPIPS difference.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
HO = ROOT / 'data/real_eval/heldout'


def theta_baselines():
    sys.path.insert(0, str(ROOT / 'code'))
    from operator_estimator import params_vector
    import glob
    th = []
    for p in sorted(glob.glob(str(Path('/temp/zihaohe_range_pairs/val') / '*.npz'))):
        d = np.load(p)
        th.append(params_vector(json.loads(str(d['meta']))['degradation']))
    th = np.stack(th)
    mean = th.mean(0)
    out = {'n': len(th), 'dim': th.shape[1],
           'mae_predict_zeros': float(np.abs(th).mean()),
           'mae_predict_mean': float(np.abs(th - mean).mean()),
           'mae_predict_median': float(np.abs(th - np.median(th, 0)).mean()),
           'per_dim_std': [round(float(x), 3) for x in th.std(0)],
           'mean_code': [round(float(x), 3) for x in mean]}
    return out


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def down(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    return d[0].permute(1, 2, 0).float().clamp(0, 1)


def severity(name):
    torch.set_num_threads(2)
    m = json.loads((HO / name / 'meta.json').read_text())
    ts = list(range(0, m['frames'], 16))
    v = []
    for t in ts:
        g = down(load(HO / name / 'hr' / f'{t:06d}.png'))
        y = load(HO / name / 'lr' / f'{t:06d}.png')
        v.append(float(10 * np.log10(1.0 / max(float((y - g).square().mean()), 1e-12))))
    return {'clip': name, 'obs_psnr': float(np.mean(v)),
            'degradation': m['degradation'].get('kind', '?'),
            'params': {k: val for k, val in m['degradation'].items() if k != 'kind'}}


def main():
    print(json.dumps({'theta_baselines': theta_baselines()}, indent=1))
    j = json.load(open(ROOT / 'results/deploy/judge_ho.json'))
    rows = [r for r in j['per_clip'] if r['degradation'] == 'realistic']
    names = [r['clip'] for r in rows]
    with Pool(24) as pool:
        sev = {s['clip']: s for s in pool.map(severity, names)}
    dl = np.array([r['arms']['t4r2_q']['lpips'] - r['arms']['infvsr']['lpips'] for r in rows])
    ob = np.array([sev[r['clip']]['obs_psnr'] for r in rows])
    rk = lambda v: np.argsort(np.argsort(v))
    print()
    print('n=%d  spearman(dLPIPS, observation PSNR) = %+.3f  pearson = %+.3f'
          % (len(rows), np.corrcoef(rk(dl), rk(ob))[0, 1], np.corrcoef(dl, ob)[0, 1]))
    q = np.quantile(ob, [0, .25, .5, .75, 1])
    print('%-22s %6s %9s %9s' % ('observation PSNR band', 'n', 'dLPIPS', 'ours better'))
    for i in range(4):
        sel = (ob >= q[i]) & (ob <= q[i + 1]) if i == 3 else (ob >= q[i]) & (ob < q[i + 1])
        print('%6.1f - %-14.1f %6d %+9.4f %6d/%d'
              % (q[i], q[i + 1], sel.sum(), dl[sel].mean(), int((dl[sel] < 0).sum()), sel.sum()))
    for flag in ('h264_crf', 'jpeg_q', 'noise1', 'second_order'):
        has = np.array([flag in sev[r['clip']]['params'] for r in rows])
        if has.sum() and (~has).sum():
            print('%-14s present n=%2d dLPIPS %+.4f | absent n=%2d dLPIPS %+.4f'
                  % (flag, has.sum(), dl[has].mean(), (~has).sum(), dl[~has].mean()))
    (ROOT / 'results/deploy/why_split.json').write_text(json.dumps(
        {'per_clip': [{'clip': r['clip'], 'dlpips': float(x), 'obs_psnr': float(o)}
                      for r, x, o in zip(rows, dl, ob)]}, indent=1))


if __name__ == '__main__':
    main()
