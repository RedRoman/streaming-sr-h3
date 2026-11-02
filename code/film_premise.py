"""Does the range/null mechanism have anything to grab under film-type degradation?

The method rests on a known linear observation operator D (antialiased bicubic x4). mech5 measured that
43.5% of the restorer's squared error lies in the range of that operator -- the subspace the observation
determines exactly -- and that is the share R_phi and the innerDC projection target act on. Old film
degradation is not one operator: grain is roughly noise, soft focus is roughly blur, but scratches are
sparse outliers that lie in no downsampling operator's range, flicker is a per-frame gain, and gate weave
is a geometric warp. If those components push the range share down, there is nothing for the mechanism to
recover and the direction is dead before any training happens.

So this measures the share per component rather than for one combined chain, which also says which film
artifacts the existing machinery can host and which need separate mechanisms.

Read-out rule. The 0.20 threshold in error_decomposition.py does NOT transfer here and must not be used:
mech5's 43.5% is the share of a RESTORER's error in the range space, while this measures the share of the
error a DEGRADATION introduces. Different quantities. The reference point measured the same way is the
chain this project already trains on and demonstrably works on -- the `current_chain` row, the
bicubic-upsampled observation against ground truth. A component above that row is more addressable by the
range branch than what we already handle, below it is less.

The absolute squared error is reported next to every share, because a component can be almost entirely in
the range space and still be negligible (flicker), or dominate the total error and be mostly outside it
(weave). Both of those happen. Note also that weave and flicker are a global warp and a global gain: both
have standard classical solutions (registration, gain normalisation), so they belong in preprocessing and
their error should not be charged to the restoration model at all.

No real footage is involved and none is needed for this question: the share is a property of the
degradation, not of any particular film, and it is measured against ground truth we have. Fitting the
chain to an actual print is the next step and needs source material.

Components, each applied alone at a strength taken from the film-restoration literature's usual range:
  grain      signal-dependent noise, stronger in the mid-tones, correlated across a small kernel
  softfocus  mild isotropic blur, no extra downsampling
  scratch    a few bright/dark vertical lines per frame, 1-3 px wide, temporally unstable
  dust       sparse blobs, a new random set every frame
  flicker    per-frame multiplicative gain, random walk
  weave      per-frame sub-pixel translation, random walk
  telecine   4:3 pillarbox resample then back, i.e. a resolution loss with a different kernel
  compress   heavy JPEG, standing in for the transfers that actually circulate
  film_all   the composition, which is what a real print looks like
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    """(range, null) = (U(D x), x - U(D x)) for the x4 antialiased bicubic observation operator.

    Copied from error_decomposition.py rather than imported, because that module runs a whole study on
    import; the arithmetic is identical so the two numbers stay comparable.
    """
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def _gauss_kernel(sigma, ch=3):
    r = max(1, int(round(3 * sigma)))
    g = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())
    k = torch.outer(g, g)[None, None].repeat(ch, 1, 1, 1)
    return k, r


def blur(x, sigma):
    k, r = _gauss_kernel(sigma)
    t = x.permute(0, 3, 1, 2)
    t = F.conv2d(F.pad(t, (r, r, r, r), mode='reflect'), k, groups=3)
    return t.permute(0, 2, 3, 1)


# --------------------------------------------------------------------------------- film components
def grain(x, rng, strength=0.055):
    """Signal-dependent, spatially correlated. Peaks in the mid-tones the way film emulsion does."""
    n = torch.from_numpy(rng.standard_normal(tuple(x.shape[:3]) + (1,)).astype(np.float32))
    n = blur(n.repeat(1, 1, 1, 3), 0.6)[..., :1]
    w = (1 - (2 * x.mean(-1, keepdim=True) - 1).abs()).clamp(0.15, 1.0)
    return (x + strength * w * n).clamp(0, 1)


def softfocus(x, rng, sigma=1.1):
    return blur(x, sigma).clamp(0, 1)


def scratch(x, rng, n_lines=(1, 4)):
    y = x.clone()
    T, H, W, _ = y.shape
    for t in range(T):
        for _ in range(rng.integers(*n_lines)):
            c = int(rng.integers(0, W))
            w = int(rng.integers(1, 4))
            val = float(rng.choice([0.0, 1.0]))
            a = float(rng.uniform(0.5, 1.0))
            lo, hi = max(0, c), min(W, c + w)
            y[t, :, lo:hi] = (1 - a) * y[t, :, lo:hi] + a * val
    return y


def dust(x, rng, density=4e-4):
    y = x.clone()
    T, H, W, _ = y.shape
    for t in range(T):
        m = torch.from_numpy((rng.random((H, W)) < density).astype(np.float32))
        m = blur(m[None, ..., None].repeat(1, 1, 1, 3), 0.8)[0]
        m = (m / m.max().clamp_min(1e-6)).clamp(0, 1)
        val = float(rng.choice([0.0, 1.0]))
        y[t] = (1 - m) * y[t] + m * val
    return y


def flicker(x, rng, sd=0.06):
    g = 1.0 + np.cumsum(rng.standard_normal(x.shape[0]) * sd)
    g = np.clip(g / g.mean(), 0.75, 1.25).astype(np.float32)
    return (x * torch.from_numpy(g)[:, None, None, None]).clamp(0, 1)


def weave(x, rng, sd=0.6):
    """Sub-pixel per-frame translation: the print moving in the gate, not scene motion."""
    T, H, W, _ = x.shape
    dx = np.cumsum(rng.standard_normal(T) * sd)
    dy = np.cumsum(rng.standard_normal(T) * sd)
    out = []
    t_ = x.permute(0, 3, 1, 2)
    for t in range(T):
        theta = torch.tensor([[1.0, 0.0, 2 * float(dx[t]) / W],
                              [0.0, 1.0, 2 * float(dy[t]) / H]])[None]
        grid = F.affine_grid(theta, (1, 3, H, W), align_corners=False)
        out.append(F.grid_sample(t_[t:t + 1], grid, mode='bilinear',
                                 padding_mode='reflection', align_corners=False))
    return torch.cat(out).permute(0, 2, 3, 1).clamp(0, 1)


def telecine(x, rng, frac=0.72):
    """Resolution lost to a 4:3 transfer and back, with a different kernel than the eval operator."""
    t = x.permute(0, 3, 1, 2)
    h, w = t.shape[-2:]
    small = F.interpolate(t, size=(int(h * frac), int(w * frac)), mode='area')
    back = F.interpolate(small, size=(h, w), mode='bilinear', align_corners=False)
    return back.permute(0, 2, 3, 1).clamp(0, 1)


def compress(x, rng, q=28):
    import io
    out = []
    for t in range(x.shape[0]):
        b = io.BytesIO()
        Image.fromarray((x[t].numpy() * 255).round().astype(np.uint8)).save(b, 'JPEG', quality=q)
        b.seek(0)
        out.append(load_pil(Image.open(b)))
    return torch.stack(out)


def load_pil(im):
    return torch.from_numpy(np.asarray(im.convert('RGB')).copy()).float() / 255


COMPONENTS = {'grain': grain, 'softfocus': softfocus, 'scratch': scratch, 'dust': dust,
              'flicker': flicker, 'weave': weave, 'telecine': telecine, 'compress': compress}
# ordered the way a print actually degrades: optics, then the emulsion, then physical damage,
# then the transfer
FILM_ORDER = ['softfocus', 'grain', 'scratch', 'dust', 'flicker', 'weave', 'telecine', 'compress']


# weave and flicker are a global warp and a global gain. Both have standard classical solutions, so a
# real pipeline removes them before any restoration model runs -- and weave alone carries 3.49M of
# film_all's 3.94M squared error, so leaving it in makes the composed chain look far more
# range-addressable than the part a restorer is actually responsible for. This is the honest chain.
PREPROCESSING = ['weave', 'flicker']
STABILISED_ORDER = [n for n in FILM_ORDER if n not in PREPROCESSING]


def film_all(x, rng):
    for name in FILM_ORDER:
        x = COMPONENTS[name](x, rng)
    return x


def film_stabilised(x, rng):
    for name in STABILISED_ORDER:
        x = COMPONENTS[name](x, rng)
    return x


def shares(gt, obs):
    """Fraction of squared error in the range of D, and the absolute errors, for one clip."""
    gr, gn = split(gt)
    yr, yn = split(obs)
    er = float((yr - gr).square().sum())
    en = float((yn - gn).square().sum())
    return er, en


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=12)
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    D_ = ROOT / a.eval_set
    names = json.loads((D_ / 'manifest.json').read_text())['clips'][:a.clips]

    acc = {k: {'er': 0.0, 'en': 0.0}
           for k in list(COMPONENTS) + ['film_all', 'film_stabilised', 'current_chain']}
    rng = np.random.default_rng(a.seed)
    per_clip = []
    for name in names:
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        ts = list(range(0, T, STRIDE))
        gt = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in ts])
        row = {'clip': name, 'degradation': meta['degradation'].get('kind')}
        for k in list(COMPONENTS) + ['film_all', 'film_stabilised']:
            f = {'film_all': film_all, 'film_stabilised': film_stabilised}.get(k, COMPONENTS.get(k))
            obs = f(gt, np.random.default_rng(a.seed + 1))
            er, en = shares(gt, obs)
            acc[k]['er'] += er; acc[k]['en'] += en
            row[k] = er / max(er + en, 1e-12)
        # the chain this project already trains on, as the reference point for the 43.5% figure
        up = F.interpolate(
            torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in ts]).permute(0, 3, 1, 2),
            size=gt.shape[1:3], mode='bicubic', align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
        er, en = shares(gt, up)
        acc['current_chain']['er'] += er; acc['current_chain']['en'] += en
        row['current_chain'] = er / max(er + en, 1e-12)
        per_clip.append(row)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in row.items()}), flush=True)

    out = {'clips': len(names), 'stride': STRIDE,
           'readout_rule': 'compare each component against the current_chain row, measured identically; '
                           'the 0.20 figure from error_decomposition.py is a restorer-error share and '
                           'does not apply to a degradation-error share',
           'preprocessing_not_restoration': ['weave', 'flicker'],
           'per_clip': per_clip, 'pooled': {}}
    for k, v in acc.items():
        tot = v['er'] + v['en']
        out['pooled'][k] = {'range_share': v['er'] / max(tot, 1e-12),
                            'range_mse_rel': v['er'] / max(tot, 1e-12),
                            'total_sq_error': tot}
    p = ROOT / 'results/film_premise.json'
    p.write_text(json.dumps(out, indent=1))

    print()
    ref = out['pooled']['current_chain']['range_share']
    print('%-14s %12s %10s %12s  %s'
          % ('component', 'range share', 'vs chain', 'total sq err', 'note'))
    print('-' * 76)
    for k in ['current_chain'] + FILM_ORDER + ['film_all', 'film_stabilised']:
        sh = out['pooled'][k]['range_share']
        tot = out['pooled'][k]['total_sq_error']
        if k == 'current_chain':
            note = 'reference: the chain we train on'
        elif k == 'film_stabilised':
            note = 'the chain a restorer is actually responsible for'
        elif k in ('weave', 'flicker'):
            note = 'preprocessing (registration / gain), not restoration'
        elif sh >= ref:
            note = 'more range-addressable than our chain'
        else:
            note = 'null-space dominated: the prior is the lever'
        print('%-14s %11.3f %10s %12.1f  %s'
              % (k, sh, '-' if k == 'current_chain' else '%+.3f' % (sh - ref), tot, note))
    print()
    print('wrote', p)


if __name__ == '__main__':
    main()
