"""mech7 judge 2: does the range branch's estimate, substituted into the HR output, beat the raw observation?

mech5b established the two reference points on the same 12 clips and the same restorer outputs:
    substituting U(y)            -> +0.711 dB, LPIPS -0.0016 (neutral)
    substituting U(D x_GT)       -> +3.426 dB, LPIPS -0.0901   (the bound)
This substitutes U(R_phi(y)) and reports where it lands between them, split by degradation kind because the
two regimes behave completely differently (mech8: the range space sits at 58 dB under Pillow-BICUBIC and
30 dB under the realistic chain).

The branch used here is the plain-L1 model: it is the only one that learned a non-trivial mapping (+1.95 dB
in the LR domain on realistic clips). The normalised and hinge objectives both collapsed to the identity,
so they have nothing to substitute.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, '/cache/zihaohe/streaming_sr0/code')
from range_branch import RangeBranch, OUT as RB_OUT
from revisit_metrics import lpips_dist

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/stage2_eval'
SRC = ROOT / 'results/stage2_blocks/indep'
OUTD = ROOT / 'results/mech7_hr'
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
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='k5')
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    st = torch.load(RB_OUT / f'{a.tag}.pt', map_location='cpu')
    model = RangeBranch(**st['cfg']).eval()
    model.load_state_dict(st['state'])
    K = st['cfg']['k']
    man = json.loads((D_ / 'manifest.json').read_text())
    rows = []
    with torch.no_grad():
        for name in man['clips']:
            meta = json.loads((D_ / name / 'meta.json').read_text())
            T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
            kind = meta['degradation'].get('kind', '?')
            lr_all = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
            acc = {k: [] for k in ('as_is', 'sub_lr', 'sub_rphi', 'sub_oracle')}
            lp = {k: [] for k in acc}
            for t in range(0, T, STRIDE):
                f = SRC / name / 'stream' / f'{t:06d}.png'
                if not f.exists():
                    continue
                g = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
                y = load(f)[None]
                _, yn = rng_null(y)
                gr, _ = rng_null(g)
                i0 = max(0, t - K + 1)
                stack = lr_all[i0:t + 1]
                if stack.shape[0] < K:                       # causal pad at the clip start
                    stack = torch.cat([stack[:1].repeat(K - stack.shape[0], 1, 1, 1), stack])
                est = model(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0)[None]
                cand = {'as_is': y,
                        'sub_lr': (up(lr_all[t][None], hw) + yn).clamp(0, 1),
                        'sub_rphi': (up(est, hw) + yn).clamp(0, 1),
                        'sub_oracle': (gr + yn).clamp(0, 1)}
                for k, im in cand.items():
                    acc[k].append(psnr(im, g)); lp[k].append(lpips_dist(im[0], g[0]))
            r = {'clip': name, 'degradation': kind,
                 **{k: float(np.mean(v)) for k, v in acc.items()},
                 **{f'lpips_{k}': float(np.mean(v)) for k, v in lp.items()}}
            r['gain_rphi'] = r['sub_rphi'] - r['as_is']
            r['gain_lr'] = r['sub_lr'] - r['as_is']
            r['gain_oracle'] = r['sub_oracle'] - r['as_is']
            r['dlpips_rphi'] = r['lpips_sub_rphi'] - r['lpips_as_is']
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    def agg(sel, label):
        s = [r for r in rows if sel(r)]
        if not s:
            return None
        return {'n': len(s), **{k: {'mean': float(np.mean([r[k] for r in s])), 'ci95': ci([r[k] for r in s])}
                                for k in ('gain_rphi', 'gain_lr', 'gain_oracle', 'dlpips_rphi')}}
    out = {'tag': a.tag, 'K': K, 'per_clip': rows,
           'all': agg(lambda r: True, 'all'),
           'realistic': agg(lambda r: r['degradation'] == 'realistic', 'realistic'),
           'bicubic': agg(lambda r: r['degradation'] != 'realistic', 'bicubic')}
    (OUTD / f'hr_{a.tag}.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
