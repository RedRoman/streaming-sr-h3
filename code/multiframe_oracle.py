"""How much of the null component is actually recoverable from neighbouring observations?

This decides whether a multi-frame range/null decomposition is worth building, and it is designed to
be able to kill the idea cheaply.

The project's decomposition is per-frame: range_t = U(D x_t), null_t = x_t - U(D x_t), and every bit of
null is left to the DiT prior. Measured over 77 clips the prior delivers null energy 0.544 of the truth
at a cosine of 0.4876 -- so roughly half of what it invents points the wrong way. The question here is
whether some of that was never a guess at all: motion changes the sampling phase between frames, so a
high frequency destroyed in frame t may survive in frame t+-k, which would make it OBSERVED evidence
rather than something to hallucinate.

The literature leaves this specific gap open. Range/null decompositions (DDNM and descendants) are all
single-measurement; diffusion VSR temporal work (DC-VSR, SDATC, DiffST) enforces consistency with
attention and losses and does not use the decomposition. The nearest argument is FFF-VDI in video
inpainting -- a shifting mask reveals true pixels, so pulling in future latents reduces object
hallucination -- which is the same shape of claim for a different operator.

The measurement, with an oracle flow computed on ground truth so the answer is an upper bound and not a
statement about any particular flow estimator:

  n_t            = x_t - U(D x_t)                   what the prior is currently asked to invent
  r_t^(k)        = W_{t+k -> t}( U(D x_{t+k}) )      a neighbour's OBSERVATION, warped into frame t
  explained      = the part of n_t that linear regression on {r_t^(k)} can predict

Reported per radial frequency band, because the answer is expected to depend on band: the neighbour's
observation is itself band-limited by D, so whatever it contributes must come from the sampling phase
shift rather than from new bandwidth.

A zero here kills the idea. A large value means part of what we call hallucination is simply evidence
that the per-frame operator threw away, and it should be in the range space.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SCALE = 4
NB = 8
TRUE_SIGMA = 1.771   # the blur recorded in these clips' meta.json


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=SCALE, operator='true'):
    """Range and null component. Which operator is used decides whether this question is even askable.

    The project defines the decomposition with an ANTIALIASED bicubic. That operator removes high
    frequency before subsampling, so no sampling phase can recover it and a multi-frame decomposition
    is dead on arrival -- which is exactly the +0.001 this script first measured.

    The degradation that actually produced the eval set is different: degrade.py applies a Gaussian
    (sigma 1.77 on these clips) and then cv2.resize with INTER_LINEAR, recorded as interp=1 in every
    meta.json. INTER_LINEAR does not antialias, so content above the LR Nyquist folds down instead of
    being deleted, and folded energy IS recoverable from several sampling phases. That is the entire
    basis of classical multi-frame super-resolution, and it is only available under the true operator.

    operator='canonical' reproduces the project's definition, so the two can be compared directly.
    """
    import cv2 as _cv2
    h, w = x.shape[:2]
    if operator == 'canonical':
        t = x.permute(2, 0, 1)[None].double()
        d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
        u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
        r = u[0].permute(1, 2, 0).float()
        return r, x - r
    a = x.numpy()
    k = int(2 * round(3 * TRUE_SIGMA) + 1)
    b = _cv2.GaussianBlur(a, (k, k), TRUE_SIGMA, borderType=_cv2.BORDER_REFLECT)
    lo = _cv2.resize(b, (w // s, h // s), interpolation=_cv2.INTER_LINEAR)
    up = _cv2.resize(lo, (w, h), interpolation=_cv2.INTER_CUBIC)
    r = torch.from_numpy(up)
    return r, x - r


def flow(a, b):
    """Oracle optical flow from ground-truth frames a -> b, Farneback on luma."""
    ga = cv2.cvtColor((a.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor((b.numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 4, 25, 5, 7, 1.5, 0)


def warp(x, fl):
    h, w = x.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + fl[..., 0], gy + fl[..., 1]
    out = cv2.remap(x.numpy(), mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return torch.from_numpy(out)


def band_masks(h, w, nb=NB):
    fy = fft.fftfreq(h)[:, None]
    fx = fft.rfftfreq(w)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2)
    r = (r / float(r.max())).clamp(0, 1 - 1e-6)
    idx = (r * nb).long().clamp(0, nb - 1)
    return [(idx == b) for b in range(nb)]


def band_energy(x, masks):
    """Energy of an [H,W,3] signal per radial band."""
    g = x.permute(2, 0, 1).double()
    P = (fft.rfft2(g).abs() ** 2).sum(0)
    return torch.tensor([float(P[m].sum()) for m in masks])


def explained_fraction(target, preds, masks):
    """Per band: fraction of target energy that least squares on preds can explain.

    Solved in the Fourier domain per band, which keeps it a few scalars per band instead of a dense
    solve over pixels, and makes the band-wise answer exact rather than a projection of a pixel fit.
    """
    t = fft.rfft2(target.permute(2, 0, 1).double())
    Ps = [fft.rfft2(p.permute(2, 0, 1).double()) for p in preds]
    out, res = [], []
    for m in masks:
        y = t[:, m].reshape(-1)
        X = torch.stack([P[:, m].reshape(-1) for P in Ps], dim=1)
        if y.numel() == 0:
            out.append(float('nan')); res.append(float('nan')); continue
        # complex least squares; add a small ridge for conditioning
        A = X.conj().transpose(0, 1) @ X
        A = A + 1e-8 * torch.eye(A.shape[0], dtype=A.dtype) * A.diagonal().abs().mean().clamp_min(1e-20)
        b = X.conj().transpose(0, 1) @ y
        try:
            c = torch.linalg.solve(A, b)
        except Exception:
            out.append(0.0); res.append(1.0); continue
        pred = X @ c
        num = float((y - pred).abs().square().sum())
        den = float(y.abs().square().sum())
        out.append(1.0 - num / max(den, 1e-20))
        res.append(num / max(den, 1e-20))
    return torch.tensor(out), torch.tensor(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clips', type=int, default=8)
    ap.add_argument('--frames', type=int, default=6, help='frames sampled per clip')
    ap.add_argument('--k', type=int, default=2, help='neighbours each side')
    ap.add_argument('--operator', default='true', choices=['true', 'canonical'],
                    help="'true' is blur+INTER_LINEAR as degrade.py actually applies it; "
                         "'canonical' is the antialiased bicubic the project defines the "
                         "decomposition with. Only the former can carry aliased evidence.")
    ap.add_argument('--out', default='results/multiframe_oracle.json')
    a = ap.parse_args()

    D_ = ROOT / a.eval_set
    names = [c for c in json.loads((D_ / 'manifest.json').read_text())['clips']
             if json.loads((D_ / c / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    names = names[:a.clips]

    per_clip = []
    for ci, name in enumerate(names):
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        ts = list(range(a.k + 1, T - a.k - 1, max(1, (T - 2 * a.k - 2) // a.frames)))[:a.frames]
        acc_self, acc_nb, acc_both = [], [], []
        for t in ts:
            x = load(D_ / name / 'hr' / f'{t:06d}.png')
            h, w = x.shape[:2]
            masks = band_masks(h, w)
            rng, nul = split(x, operator=a.operator)
            # the neighbours' OBSERVATIONS, warped into this frame with oracle flow
            preds = []
            for dk in range(-a.k, a.k + 1):
                if dk == 0:
                    continue
                xn = load(D_ / name / 'hr' / f'{t + dk:06d}.png')
                rn, _ = split(xn, operator=a.operator)
                fl = flow(x, xn)                     # t -> t+dk, so remap pulls t+dk into t
                preds.append(warp(rn, fl))
            # three regressors, to separate what the frame's own observation already explains
            e_self, _ = explained_fraction(nul, [rng], masks)
            e_nb, _ = explained_fraction(nul, preds, masks)
            e_both, _ = explained_fraction(nul, [rng] + preds, masks)
            acc_self.append(e_self); acc_nb.append(e_nb); acc_both.append(e_both)
        s = torch.stack(acc_self).nanmean(0)
        n = torch.stack(acc_nb).nanmean(0)
        b = torch.stack(acc_both).nanmean(0)
        per_clip.append({'clip': name, 'frames': len(ts),
                         'self': [round(float(v), 4) for v in s],
                         'neighbours': [round(float(v), 4) for v in n],
                         'both': [round(float(v), 4) for v in b]})
        print(json.dumps({'clip': name, 'band_mid_self': round(float(s[NB // 2]), 4),
                          'band_mid_nb': round(float(n[NB // 2]), 4),
                          'band_mid_both': round(float(b[NB // 2]), 4)}), flush=True)

    S = torch.tensor([p['self'] for p in per_clip]).nanmean(0)
    N = torch.tensor([p['neighbours'] for p in per_clip]).nanmean(0)
    B = torch.tensor([p['both'] for p in per_clip]).nanmean(0)
    res = {'eval_set': a.eval_set, 'clips': len(per_clip), 'k': a.k, 'bands': NB,
           'operator': a.operator,
           'oracle_flow': 'Farneback on ground-truth frames, so these are upper bounds',
           'fraction_of_null_explained': {
               'by_own_observation': [round(float(v), 4) for v in S],
               'by_warped_neighbour_observations': [round(float(v), 4) for v in N],
               'by_both': [round(float(v), 4) for v in B]},
           'per_clip': per_clip}
    (ROOT / a.out).write_text(json.dumps(res, indent=1))

    print()
    print('Fraction of the NULL component explainable by linear regression, per radial band.')
    print('Band 0 is DC, band %d is Nyquist. The null component lives in the upper bands by' % (NB - 1))
    print('construction, so those are the ones that matter.')
    print()
    print('%-28s' % 'regressor' + ''.join('%8s' % ('b%d' % i) for i in range(NB)))
    print('%-28s' % 'own observation U(D x_t)' + ''.join('%8.3f' % v for v in S))
    print('%-28s' % 'warped neighbours' + ''.join('%8.3f' % v for v in N))
    print('%-28s' % 'both' + ''.join('%8.3f' % v for v in B))
    print()
    hi = slice(NB // 2, NB)
    print('upper half of the spectrum, mean: own %.3f, neighbours %.3f, both %.3f'
          % (float(S[hi].mean()), float(N[hi].mean()), float(B[hi].mean())))
    print('the number that decides it: neighbours minus own = %+.3f'
          % (float(N[hi].mean()) - float(S[hi].mean())))
    print()
    print('near zero -> neighbour observations carry nothing about this null component,')
    print('and a multi-frame decomposition cannot help. Large -> part of what the prior currently')
    print('invents is evidence the per-frame operator discarded.')


if __name__ == '__main__':
    main()
