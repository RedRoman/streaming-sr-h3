"""Why do we lose on the most degraded clips and win on the mildest? The opposite of what I assumed.

On the 77 held-out realistic clips the per-clip LPIPS difference against InfVSR correlates with how
degraded the observation is, spearman -0.228, and the quartiles run

  observation 19.8-25.6 dB   dLPIPS +0.0027   we win  8/19
  observation 30.5-53.0 dB   dLPIPS -0.0233   we win 14/20

A denoise-then-regenerate pipeline should pay off MORE as the observation gets worse, so the sign is
backwards and something specific is happening. The candidate mechanism is that our output's entire low
band is U(R_phi(y)): we commit to one explicit estimate of the observed subspace, whereas InfVSR never
commits and can hedge. If R_phi degrades faster with severity than the sampler's own low band does, then
on the worst clips we are actively replacing a better estimate with a worse one -- and the gate makes it
worse again, because it reads the guide's local std and an over-smoothed guide closes it.

Measured per clip at LR resolution against D(GT): the observation, R_phi's estimate, and the low band the
sampler produced on its own, grouped by severity. If the R_phi curve crosses the sampler curve somewhere,
that crossing is the operating point where the guide should stop being trusted.
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
RAW = ROOT / 'results/deploy_runs/t4_r2_ho'
CKPT = ROOT / 'results/range_branch/r2_hs.pt'
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def down(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    return d[0].permute(1, 2, 0).float().clamp(0, 1)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def one(name):
    torch.set_num_threads(2)
    from range_branch import RangeBranch
    st = torch.load(CKPT, map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    m = json.loads((HO / name / 'meta.json').read_text())
    ts = [t for t in range(0, m['frames'], STRIDE) if (RAW / name / 'stream' / f'{t:06d}.png').exists()]
    if not ts:
        return None
    tgt = torch.stack([down(load(HO / name / 'hr' / f'{t:06d}.png')) for t in ts])
    y = torch.stack([load(HO / name / 'lr' / f'{t:06d}.png') for t in ts])
    with torch.no_grad():
        est = rb(y.permute(0, 3, 1, 2)[:, None]).clamp(0, 1).permute(0, 2, 3, 1)
    raw = [down(load(RAW / name / 'stream' / f'{t:06d}.png')) for t in ts]
    return {'clip': name, 'degradation': m['degradation'].get('kind', '?'),
            'obs': float(np.mean([psnr(y[i], tgt[i]) for i in range(len(ts))])),
            'rphi': float(np.mean([psnr(est[i], tgt[i]) for i in range(len(ts))])),
            'sampler': float(np.mean([psnr(raw[i], tgt[i]) for i in range(len(ts))]))}


def main():
    names = [r['clip'] for r in json.load(open(ROOT / 'results/deploy/judge_ho.json'))['per_clip']
             if r['degradation'] == 'realistic']
    with Pool(24) as pool:
        rows = [r for r in pool.map(one, names) if r]
    j = {r['clip']: r for r in json.load(open(ROOT / 'results/deploy/judge_ho.json'))['per_clip']}
    for r in rows:
        a = j[r['clip']]['arms']
        r['dlpips'] = a['t4r2_q']['lpips'] - a['infvsr']['lpips']
    (ROOT / 'results/deploy/sev_split.json').write_text(json.dumps(rows, indent=1))
    ob = np.array([r['obs'] for r in rows])
    q = np.quantile(ob, [0, .25, .5, .75, 1])
    print('%-20s %4s %8s %8s %9s %10s %9s' % ('observation PSNR', 'n', 'obs', 'R_phi', 'sampler',
                                              'R_phi-samp', 'dLPIPS'))
    for i in range(4):
        sel = (ob >= q[i]) & (ob <= q[i + 1]) if i == 3 else (ob >= q[i]) & (ob < q[i + 1])
        g = [r for r, s in zip(rows, sel) if s]
        f = lambda k: np.mean([r[k] for r in g])
        print('%6.1f - %-11.1f %4d %8.2f %8.2f %9.2f %10.2f %+9.4f'
              % (q[i], q[i + 1], len(g), f('obs'), f('rphi'), f('sampler'),
                 f('rphi') - f('sampler'), f('dlpips')))
    print()
    print('R_phi-samp > 0 means the guide is the better low band and overwriting is right.')


if __name__ == '__main__':
    main()
