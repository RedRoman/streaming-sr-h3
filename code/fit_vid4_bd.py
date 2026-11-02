"""Fit Vid4's BD degradation from its own pairs instead of guessing its parameters.

sp_bd scores +15.174 dB on the synthetic BD data it was trained on and -12.708 dB on Vid4 BDx4, uniformly
across all four clips. The diagnostic that names the cause is the distance from the observation to the
target: Vid4's BD sits 38.19 dB from D(x) while the synthetic one sits 30.56 dB away. So the synthetic
chain is far harsher, the specialist learned a correspondingly large correction, and applying it to a
milder degradation overcorrects by 12 dB. That is a severity mismatch, not a difference in kind -- and it
is the 4KLSDB failure repeating: there, a claimed +4.68 dB "native-4K prior" was fully explained by a
+5.98 dB degradation mismatch. The parameters have to be fitted to the target, not assumed.

Vid4 ships GT and BDx4 together, so the operator can be identified directly rather than inferred:
search over the Gaussian width, the kernel support, and the subsampling rule, and report the combination
that reproduces the released BD frames most closely. If some combination reaches near-exactness the
operator is pinned and the training chain can be rebuilt around it; if the best fit is poor, the released
BD was made with something outside this family and that is worth knowing before training anything else.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

V = Path('/cache/zihaohe/benchmarks/vid4/Vid4')
OUT = Path('/cache/zihaohe/streaming_sr0/results/vid4_bd_fit.json')
CLIPS = ['calendar', 'city', 'foliage', 'walk']
FRAMES_PER_CLIP = 4


def load(p):
    return np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float(((a - b) ** 2).mean()), 1e-12)))


def blur(x, sigma, ksize):
    return cv2.GaussianBlur(x, (ksize, ksize), sigma, borderType=cv2.BORDER_REFLECT)


def subsample(x, rule, s=4):
    if rule == 'stride0':
        return x[::s, ::s]
    if rule == 'stride_center':
        return x[s // 2::s, s // 2::s]
    if rule == 'stride_center_m1':
        return x[s // 2 - 1::s, s // 2 - 1::s]
    if rule == 'area':
        return cv2.resize(x, (x.shape[1] // s, x.shape[0] // s), interpolation=cv2.INTER_AREA)
    raise ValueError(rule)


def main():
    pairs = []
    for c in CLIPS:
        gt = sorted((V / 'GT' / c).glob('*.png'))
        bd = sorted((V / 'BDx4' / c).glob('*.png'))
        idx = np.linspace(0, len(gt) - 1, FRAMES_PER_CLIP).astype(int)
        for i in idx:
            g, b = load(gt[i]), load(bd[i])
            h, w = (g.shape[0] // 4) * 4, (g.shape[1] // 4) * 4
            pairs.append((c, g[:h, :w], b[:h // 4, :w // 4]))
    print(json.dumps({'pairs': len(pairs), 'gt_hw': list(pairs[0][1].shape[:2]),
                      'bd_hw': list(pairs[0][2].shape[:2])}), flush=True)

    rows = []
    for rule in ('stride0', 'stride_center', 'stride_center_m1', 'area'):
        for ksize in (7, 13, 21):
            for sigma in np.arange(0.6, 2.61, 0.1):
                vals = []
                for _, g, b in pairs:
                    y = subsample(blur(g, float(sigma), ksize), rule)
                    if y.shape != b.shape:
                        y = y[:b.shape[0], :b.shape[1]]
                    vals.append(psnr(y, b))
                rows.append({'rule': rule, 'ksize': ksize, 'sigma': round(float(sigma), 2),
                             'psnr_vs_released_BD': float(np.mean(vals))})
    rows.sort(key=lambda r: -r['psnr_vs_released_BD'])
    best = rows[0]
    print()
    print('%-18s %6s %7s %22s' % ('subsample rule', 'ksize', 'sigma', 'PSNR vs released BD'))
    for r in rows[:10]:
        print('%-18s %6d %7.2f %22.3f' % (r['rule'], r['ksize'], r['sigma'], r['psnr_vs_released_BD']))

    # how far the released BD and the fitted reconstruction each sit from D(x), the range target
    def Dx(g):
        t = torch.from_numpy(g).permute(2, 0, 1)[None].double()
        d = F.interpolate(t, size=(g.shape[0] // 4, g.shape[1] // 4), mode='bicubic',
                          align_corners=False, antialias=True)
        return d[0].permute(1, 2, 0).float().numpy()

    rel = []
    for _, g, b in pairs:
        d = Dx(g)
        y = subsample(blur(g, best['sigma'], best['ksize']), best['rule'])[:d.shape[0], :d.shape[1]]
        rel.append({'released_vs_Dx': psnr(b, d), 'fitted_vs_Dx': psnr(y, d),
                    'fitted_vs_released': psnr(y, b[:d.shape[0], :d.shape[1]])})
    agg = {k: float(np.mean([r[k] for r in rel])) for k in rel[0]}
    out = {'best': best, 'top10': rows[:10], 'distance_to_range_target': agg,
           'synthetic_bd_base_for_reference': 30.559,
           'note': ('released_vs_Dx is the number the training chain has to match: a specialist trained '
                    'on pairs whose observation sits a different distance from D(x) will mis-scale its '
                    'correction, which is exactly the -12.7 dB sp_bd showed on Vid4')}
    OUT.write_text(json.dumps(out, indent=1))
    print()
    print(json.dumps(agg, indent=1))
    print()
    print('best fit: %s, ksize %d, sigma %.2f -> %.3f dB against the released BD'
          % (best['rule'], best['ksize'], best['sigma'], best['psnr_vs_released_BD']))
    print('released BD sits %.2f dB from D(x); our synthetic bd sat %.2f dB away'
          % (agg['released_vs_Dx'], 30.559))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
