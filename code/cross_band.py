"""Which band actually costs us the LPIPS loss on realistic clips?

range_gap settled one thing: our low band is not the weak one. Measured against D(GT) at LR resolution on
the 19 realistic clips, R_phi(y) scores 30.57 dB and InfVSR's implied low band 27.73 dB -- we are 2.95 dB
ahead. Yet judge13 has us losing LPIPS there, 0.2309 against 0.2183, while winning the gtrange diagnostic,
0.1431 against 0.1541. Both components look better and the total looks worse, so LPIPS is not reading the
two bands the way PSNR does.

The decomposition is exact and the frames are already on disk, so the cross terms can simply be built:

  r_o + n_o   ours, as shipped                     r_i + n_i   InfVSR, as shipped
  r_i + n_o   our detail on their low band         r_o + n_i   their detail on our low band
  g_r + n_o   gtrange (reproduces judge13)         g_r + n_i
  r_o + g_n   our low band with perfect detail     r_i + g_n

Reading: if (r_o + g_n) is worse than (r_i + g_n), our low band is perceptually worse despite being closer
in L2, which would point at regression blur -- R_phi is trained with L1 against D(x) and an L1-optimal
estimate is a conditional mean. mid_rms reports the mid-frequency energy of each low band against the
truth's, which is the signature that would confirm it.
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / 'data/real_eval/const'
ARMS = {'ours': ROOT / 'results/deploy/t4dcr_q', 'infvsr': ROOT / 'results/baseline_infvsr'}
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u[0].permute(1, 2, 0).float()
    return r, x - r


def _gauss(sigma):
    r = max(1, int(round(3 * sigma)))
    a = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (a / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def mid_rms(x, sigma=2.0):
    """RMS of what a sigma=2 blur removes: the mid band the low component is supposed to carry."""
    t = x.permute(2, 0, 1)[None]
    g = _gauss(sigma).expand(3, 1, 1, -1)
    r = g.shape[-1] // 2
    b = F.conv2d(F.pad(t, (r, r, 0, 0), mode='reflect'), g, groups=3)
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2), groups=3)
    return float((t - b).square().mean().sqrt())


def frame_path(d, name, t):
    p = d / name / 'stream' / f'{t:06d}.png'
    return p if p.exists() else d / name / f'{t:06d}.png'


def one(name):
    from revisit_metrics import lpips_dist
    m = json.loads((D_ / name / 'meta.json').read_text())
    ts = list(range(0, m['frames'], STRIDE))
    acc, mids = {}, {}
    for t in ts:
        g = load(D_ / name / 'hr' / f'{t:06d}.png')
        gr, gn = split(g)
        o = load(frame_path(ARMS['ours'], name, t))
        i = load(frame_path(ARMS['infvsr'], name, t))
        ro, no = split(o)
        ri, ni = split(i)
        combos = {'ours': ro + no, 'infvsr': ri + ni,
                  'rinf_nours': ri + no, 'rours_ninf': ro + ni,
                  'gtr_nours': gr + no, 'gtr_ninf': gr + ni,
                  'rours_gtn': ro + gn, 'rinf_gtn': ri + gn}
        for k, v in combos.items():
            acc.setdefault(k, []).append(lpips_dist(v.clamp(0, 1), g))
        for k, v in (('gt', gr), ('ours', ro), ('infvsr', ri)):
            mids.setdefault(k, []).append(mid_rms(v))
    return {'clip': name, 'degradation': m['degradation'].get('kind', '?'),
            **{k: float(np.mean(v)) for k, v in acc.items()},
            **{'mid_' + k: float(np.mean(v)) for k, v in mids.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--procs', type=int, default=24)
    ap.add_argument('--out', default=str(ROOT / 'results/deploy/cross_band.json'))
    a = ap.parse_args()
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(a.procs) as pool:
        rows = sorted(pool.imap_unordered(one, names), key=lambda r: r['clip'])
    keys = [k for k in rows[0] if k not in ('clip', 'degradation')]
    groups = {'all': lambda r: True, 'realistic': lambda r: r['degradation'] == 'realistic',
              'bicubic': lambda r: r['degradation'] != 'realistic'}
    summ = {g: {'n': sum(1 for r in rows if s(r)),
                **{k: float(np.mean([r[k] for r in rows if s(r)])) for k in keys}}
            for g, s in groups.items()}
    Path(a.out).write_text(json.dumps({'summary': summ, 'per_clip': rows}, indent=1))
    lp = [k for k in keys if not k.startswith('mid_')]
    print('%-10s %4s %s' % ('group', 'n', ' '.join('%11s' % k for k in lp)))
    for g, v in summ.items():
        print('%-10s %4d %s' % (g, v['n'], ' '.join('%11.4f' % v[k] for k in lp)))
    print()
    print('%-10s %4s %s' % ('group', 'n', ' '.join('%11s' % k for k in keys if k.startswith('mid_'))))
    for g, v in summ.items():
        print('%-10s %4d %s' % (g, v['n'], ' '.join('%11.5f' % v[k] for k in keys if k.startswith('mid_'))))


if __name__ == '__main__':
    main()
