"""A no-reference score for every arm, with its pristine model fitted on this set's own ground truth.

The question this answers is whether our lead over the published baselines survives a judge that never
sees the ground truth -- because for genuine real-world restoration the reference does not exist, and a
reviewer will ask whether 0.1929 LPIPS was bought by blurring.

Why not pyiqa: the environment is offline (HF_HUB_OFFLINE=1) so MUSIQ/CLIPIQA weights cannot be fetched.
That turns out to be the better design anyway. NIQE-family metrics score "do the local statistics look
like a natural image", and the usual implementations fit that model on LIVE/Koniq pristine photographs --
a different domain from 480x832 AudioSet video frames, which is exactly the domain shift that makes NR
numbers hard to trust. Here the pristine model is fitted on the ground-truth frames of this very set, so
the only thing being measured is distance from THIS corpus's natural statistics.

The algorithm is NIQE's: per 96x96 patch, compute MSCN coefficients, fit a GGD to them and an AGGD to the
four paired products, at two scales -> 36 features. Fit a multivariate Gaussian (mu, Sigma) to the
pristine patches, then score an arm by the Mahalanobis-style distance of its own patch mean to that
Gaussian. Lower is more natural.

The split matters: pristine is fitted on the GT of odd-indexed clips and every arm is scored on the
even-indexed ones, so no arm is scored against statistics fitted on its own clip. GT is scored on the
even clips too, which is the floor the metric can reach, and bicubic is the ceiling -- without those two
anchors a single NR number means nothing.

What to expect, and why the expectation is the point: FlashVSR puts 0.8619 of the ground truth's
null-band energy back while only 0.3704 of it points the right way, against our 0.5441 at 0.4876. An NR
metric cannot tell those apart, because invented texture is still natural-looking texture. If FlashVSR
scores BETTER here while losing on all four reference metrics, that is a measurement of the failure mode,
not a defeat -- and it is the cleanest argument available for why this project reports reference numbers.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import special

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
PATCH = 96
STRIDE_FRAMES = 16


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def to_gray(x):
    # luma only, the band NIQE is defined on
    return (0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2]) * 255.0


def gauss_kernel(sigma=7 / 6, size=7):
    a = torch.arange(size, dtype=torch.float64) - (size - 1) / 2
    g = torch.exp(-(a ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return torch.outer(g, g)[None, None]


_K = gauss_kernel()


def mscn(img):
    """Mean subtracted contrast normalised coefficients of a [H,W] float64 luma image."""
    x = img[None, None]
    pad = (_K.shape[-1] - 1) // 2
    mu = F.conv2d(F.pad(x, (pad,) * 4, mode='reflect'), _K)
    sig = (F.conv2d(F.pad(x * x, (pad,) * 4, mode='reflect'), _K) - mu * mu).clamp_min(0).sqrt()
    return ((x - mu) / (sig + 1.0))[0, 0], sig[0, 0]


def ggd_fit(v):
    """Shape and sigma of a generalised Gaussian fitted by moment matching."""
    v = v.flatten()
    sigma_sq = float((v ** 2).mean())
    if sigma_sq < 1e-12:
        return 0.0, 0.0
    E = float(v.abs().mean())
    rho = sigma_sq / (E ** 2 + 1e-12)
    # invert rho(gamma) = G(1/g)G(3/g)/G(2/g)^2 on the precomputed grid
    gamma = float(_G[np.argmin(np.abs(_R_GGD - rho))])
    return gamma, float(np.sqrt(sigma_sq))


def aggd_fit(v):
    """Shape, left/right sigma and mean of an asymmetric generalised Gaussian."""
    v = v.flatten()
    left = v[v < 0]
    right = v[v >= 0]
    if left.numel() < 2 or right.numel() < 2:
        return 0.0, 0.0, 0.0, 0.0
    sl = float((left ** 2).mean()) ** 0.5
    sr = float((right ** 2).mean()) ** 0.5
    if sl < 1e-12 or sr < 1e-12:
        return 0.0, 0.0, 0.0, 0.0
    gh = sl / sr
    gr = float((v.abs().mean()) ** 2) / float((v ** 2).mean() + 1e-12)
    R = gr * (gh ** 3 + 1) * (gh + 1) / ((gh ** 2 + 1) ** 2)
    i = int(np.argmin(np.abs(_R_AGGD - R)))
    alpha = float(_G[i])
    ratio = np.sqrt(_GAM1[i] / _GAM3[i])
    beta_l, beta_r = sl * ratio, sr * ratio
    mean = float((beta_r - beta_l) * (_GAM2[i] / _GAM1[i]))
    return alpha, float(beta_l), float(beta_r), mean


# The two moment-inversion tables. These depend only on the grid, not on any image, but the first
# version recomputed both inside every GGD/AGGD fit -- ten gamma-function sweeps over 9801 points per
# patch. Hoisting them is what makes a corpus large enough to condition a 36x36 covariance affordable.
_G = np.arange(0.2, 10.001, 0.001)
_R_GGD = special.gamma(1 / _G) * special.gamma(3 / _G) / (special.gamma(2 / _G) ** 2)
_R_AGGD = (special.gamma(2 / _G) ** 2) / (special.gamma(1 / _G) * special.gamma(3 / _G))
_GAM1 = special.gamma(1 / _G)
_GAM2 = special.gamma(2 / _G)
_GAM3 = special.gamma(3 / _G)

SHIFTS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def patch_features(p):
    """The 36 NIQE features of one patch, two scales x (1 GGD + 4 AGGD)."""
    out = []
    cur = p
    for _ in range(2):
        m, _s = mscn(cur)
        a, s = ggd_fit(m)
        out += [a, s ** 2]
        for dy, dx in SHIFTS:
            pr = m * torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            al, bl, br, mn = aggd_fit(pr)
            out += [al, mn, bl ** 2, br ** 2]
        cur = F.avg_pool2d(cur[None, None], 2)[0, 0]
    return np.asarray(out, dtype=np.float64)


def sharp_patches(gray, thresh=0.25):
    """Patch grid coordinates whose local contrast is above thresh*max, as NIQE selects.

    NIQE's own threshold is 0.75, but it fits its pristine model on a large photograph corpus. Here the
    corpus is one eval set, and at 0.6 the selector kept only ~3.4 patches per frame -- 54 in total for
    two clips, against 36 feature dimensions, so the pristine covariance was rank-starved and the metric
    inverted: bicubic scored BETTER than ground truth, which cannot be true. This is loose enough to
    condition the covariance on this corpus size.
    """
    _m, sig = mscn(gray)
    h, w = gray.shape
    coords, scores = [], []
    for y in range(0, h - PATCH + 1, PATCH):
        for x in range(0, w - PATCH + 1, PATCH):
            coords.append((y, x))
            scores.append(float(sig[y:y + PATCH, x:x + PATCH].mean()))
    if not coords:
        return []
    mx = max(scores)
    return [c for c, s in zip(coords, scores) if s >= thresh * mx]


def arm_frame(root, D_, name, t, hw):
    if root == 'bicubic':
        lr = load(D_ / name / 'lr' / f'{t:06d}.png')
        return F.interpolate(lr.permute(2, 0, 1)[None], size=hw, mode='bicubic',
                             align_corners=False).clamp(0, 1)[0].permute(1, 2, 0)
    if root == 'gt':
        return load(D_ / name / 'hr' / f'{t:06d}.png')
    for p in (Path(root) / name / 'stream' / f'{t:06d}.png', Path(root) / name / f'{t:06d}.png'):
        if p.exists():
            return load(p)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', action='append', required=True, help='tag=path | tag=bicubic | tag=gt')
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clips', type=int, default=0)
    ap.add_argument('--out', default='results/nr_score.json')
    a = ap.parse_args()

    D_ = ROOT / a.eval_set
    arms = dict(x.split('=', 1) for x in a.arm)
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    names = [n for n in names
             if json.loads((D_ / n / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    if a.clips:
        names = names[:a.clips]
    fit_names = names[1::2]      # pristine model
    test_names = names[0::2]     # every arm scored here
    print(json.dumps({'fit_clips': len(fit_names), 'test_clips': len(test_names)}), flush=True)

    # 1. pristine multivariate Gaussian from ground-truth frames of the fit clips
    feats = []
    for i, name in enumerate(fit_names):
        m = json.loads((D_ / name / 'meta.json').read_text())
        T = m['frames']
        for t in range(0, T, STRIDE_FRAMES):
            g = to_gray(load(D_ / name / 'hr' / f'{t:06d}.png')).double()
            for (y, x) in sharp_patches(g):
                feats.append(patch_features(g[y:y + PATCH, x:x + PATCH]))
        if (i + 1) % 5 == 0:
            print(json.dumps({'fit_clips_done': i + 1, 'patches': len(feats)}), flush=True)
    Fm = np.stack(feats)
    keep = np.isfinite(Fm).all(1)
    Fm = Fm[keep]
    mu_p = Fm.mean(0)
    cov_p = np.cov(Fm, rowvar=False)
    print(json.dumps({'pristine_patches': int(Fm.shape[0]), 'dim': int(Fm.shape[1])}), flush=True)

    # 2. score each arm on the test clips
    out = {'eval_set': a.eval_set, 'patch': PATCH, 'frame_stride': STRIDE_FRAMES,
           'fit_clips': fit_names, 'test_clips': test_names,
           'pristine_patches': int(Fm.shape[0]), 'arms': {}}
    for tag, root in arms.items():
        per_clip = []
        for name in test_names:
            m = json.loads((D_ / name / 'meta.json').read_text())
            T, hw = m['frames'], (m['hr_wh'][1], m['hr_wh'][0])
            fe = []
            for t in range(0, T, STRIDE_FRAMES):
                y_ = arm_frame(root, D_, name, t, hw)
                if y_ is None:
                    continue
                g = to_gray(y_).double()
                for (yy, xx) in sharp_patches(g):
                    fe.append(patch_features(g[yy:yy + PATCH, xx:xx + PATCH]))
            if not fe:
                continue
            Fa = np.stack(fe)
            Fa = Fa[np.isfinite(Fa).all(1)]
            if Fa.shape[0] < 2:
                continue
            mu_a = Fa.mean(0)
            cov_a = np.cov(Fa, rowvar=False)
            d = mu_p - mu_a
            C = (cov_p + cov_a) / 2.0
            try:
                score = float(np.sqrt(max(d @ np.linalg.pinv(C) @ d, 0.0)))
            except np.linalg.LinAlgError:
                continue
            per_clip.append({'clip': name, 'niqe_like': score, 'patches': int(Fa.shape[0])})
        if per_clip:
            s = [r['niqe_like'] for r in per_clip]
            out['arms'][tag] = {'mean': float(np.mean(s)), 'median': float(np.median(s)),
                                'clips': len(per_clip), 'per_clip': per_clip}
            print(json.dumps({'arm': tag, 'mean': round(float(np.mean(s)), 4),
                              'clips': len(per_clip)}), flush=True)

    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print()
    print('pristine fitted on GT of %d clips, every arm scored on the other %d. LOWER is more natural.'
          % (len(fit_names), len(test_names)))
    print('%-18s %10s %10s %7s' % ('arm', 'NR mean', 'NR median', 'clips'))
    for tag, v in sorted(out['arms'].items(), key=lambda kv: kv[1]['mean']):
        print('%-18s %10.4f %10.4f %7d' % (tag, v['mean'], v['median'], v['clips']))
    print()
    print('gt is the floor this metric can reach and bicubic the ceiling; an arm between them is')
    print('only as meaningful as that spread is wide.')
    print('wrote', ROOT / a.out)


if __name__ == '__main__':
    main()
