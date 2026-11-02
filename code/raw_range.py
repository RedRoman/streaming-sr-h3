"""Is the diffusion model's own low band better or worse than the 0.6M CNN we overwrite it with?

The pipeline spends its consistency projection and its whole post-hoc guide on R_phi(y), on the premise
that the sampler cannot be trusted with the observed subspace. That premise was never measured against the
sampler's own output. If a 20.1B model already produces a more accurate D(x) than a 0.6M CNN does, then
every projection step is throwing information away, and the fix is not a better CNN.

Measured at LR resolution against D(GT), on the streams already on disk:
  t8        8 steps, projection onto the raw observation y
  t8_dcr    8 steps, projection onto R_phi(y)      -- the change worth -0.075 LPIPS
  t4_dcr    4 steps, projection onto R_phi(y)      -- what the shipped t4dcr_q is built from
and R_phi(y) and y themselves as the two references.
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
D_ = ROOT / 'data/real_eval/const'
RUNS = ROOT / 'results/deploy_runs'
ARMS = ['t8', 't8_dcr', 't4_dcr']
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
    st = torch.load(ROOT / 'results/range_branch/k1.pt', map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    m = json.loads((D_ / name / 'meta.json').read_text())
    ts = list(range(0, m['frames'], STRIDE))
    tgt = torch.stack([down(load(D_ / name / 'hr' / f'{t:06d}.png')) for t in ts])
    y = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in ts])
    with torch.no_grad():
        est = rb(y.permute(0, 3, 1, 2)[:, None]).clamp(0, 1).permute(0, 2, 3, 1)
    out = {'clip': name, 'degradation': m['degradation'].get('kind', '?'),
           'y': float(np.mean([psnr(y[i], tgt[i]) for i in range(len(ts))])),
           'rphi': float(np.mean([psnr(est[i], tgt[i]) for i in range(len(ts))]))}
    for a in ARMS:
        fs = [RUNS / a / name / 'stream' / f'{t:06d}.png' for t in ts]
        ok = [i for i, f in enumerate(fs) if f.exists()]
        if len(ok) < len(ts) // 2:
            continue
        out[a] = float(np.mean([psnr(down(load(fs[i])), tgt[i]) for i in ok]))
        out[a + '_n'] = len(ok)
    return out


def main():
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(24) as pool:
        rows = sorted(pool.imap_unordered(one, names), key=lambda r: r['clip'])
    tags = ['y', 'rphi'] + [a for a in ARMS if a in rows[0]]
    (ROOT / 'results/deploy/raw_range.json').write_text(json.dumps(rows, indent=1))
    print('%-10s %4s %s' % ('group', 'n', ' '.join('%9s' % t for t in tags)))
    for g, sel in (('all', lambda r: True), ('realistic', lambda r: r['degradation'] == 'realistic'),
                   ('bicubic', lambda r: r['degradation'] != 'realistic')):
        rs = [r for r in rows if sel(r)]
        print('%-10s %4d %s' % (g, len(rs), ' '.join(
            '%9.3f' % np.mean([r[t] for r in rs if t in r]) for t in tags)))
    print()
    print('LR-resolution PSNR against D(GT). Arms are the sampled streams before any post-processing.')


if __name__ == '__main__':
    main()
