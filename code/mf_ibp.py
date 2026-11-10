"""Does classical multi-frame reconstruction recover real detail that the prior currently invents?

The first attempt at this question regressed the null component on warped neighbour observations band
by band in the Fourier domain and measured +0.001. That measurement was invalid: a per-band regression
can only predict band b from band b, while U(D x) has no energy in the upper bands at all. Multi-frame
super-resolution works ACROSS frequency -- content above the LR Nyquist folds down into the low bands
when the subsampling does not antialias, and several sampling phases are what unfolds it. A per-band fit
excludes that mechanism by construction, so the near-zero said nothing about the data.

This measures the thing directly instead. Iterative back-projection against the stacked multi-frame
operator, the classical estimator:

    A = [ D W_1 ; ... ; D W_k ],      x <- x + mu * sum_i W_i^T D^T ( y_i - D W_i x )

with oracle flow from ground truth, so the result is an upper bound on what any flow estimator could
deliver. Then the only comparison that matters: the null component of the IBP reconstruction against
the null component of the truth, in energy and in direction.

The reference point is that the DiT prior currently achieves null_cos 0.4876 at null energy 0.544,
measured over 77 clips. So:

    IBP null_cos near 0        neighbours carry nothing; a multi-frame decomposition is pointless
    IBP null_cos below ~0.2    the prior is the better source of null detail; not worth the machinery
    IBP null_cos above 0.4876  reconstruction beats generation on this component, and the range space
                               of the per-frame operator is leaving recoverable evidence on the table

The operator matters and is selectable. degrade.py applies a Gaussian then cv2.resize with
INTER_LINEAR (interp=1 in every meta.json of this set), which does NOT antialias, so aliased evidence
exists. The project's own decomposition uses an antialiased bicubic, under which no sampling phase can
recover anything -- running both is what separates "the idea fails" from "the idea was tested against
an operator that forbids it".
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SCALE = 4
TRUE_SIGMA = 1.771


def load(p):
    return np.asarray(Image.open(p).convert('RGB')).astype(np.float32) / 255


def blur(x, sigma=TRUE_SIGMA):
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(x, (k, k), sigma, borderType=cv2.BORDER_REFLECT)


def D_op(x, operator):
    """Forward: HR -> LR, exactly as the eval set was produced (or the canonical alternative)."""
    h, w = x.shape[:2]
    if operator == 'canonical':
        t = torch.from_numpy(x).permute(2, 0, 1)[None].double()
        import torch.nn.functional as F
        d = F.interpolate(t, size=(h // SCALE, w // SCALE), mode='bicubic',
                          align_corners=False, antialias=True)
        return d[0].permute(1, 2, 0).float().numpy()
    return cv2.resize(blur(x), (w // SCALE, h // SCALE), interpolation=cv2.INTER_LINEAR)


def Dt_op(r, hw, operator):
    """Adjoint-ish: LR residual -> HR, upsample then the same blur. Not the exact transpose, which is
    standard practice for IBP and sufficient for an upper-bound measurement."""
    h, w = hw
    u = cv2.resize(r, (w, h), interpolation=cv2.INTER_LINEAR)
    return u if operator == 'canonical' else blur(u)


def flow(a, b):
    ga = cv2.cvtColor((a * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gb = cv2.cvtColor((b * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return cv2.calcOpticalFlowFarneback(ga, gb, None, 0.5, 5, 21, 5, 7, 1.5, 0)


def warp(x, fl):
    h, w = x.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(x, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT)


def split_np(x, operator):
    lo = D_op(x, operator)
    h, w = x.shape[:2]
    up = cv2.resize(lo, (w, h), interpolation=cv2.INTER_CUBIC)
    return up, x - up


def cos_energy(a, b):
    """Cosine and energy ratio of a against b."""
    A, B = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(A), np.linalg.norm(B)
    return float(A @ B / max(na * nb, 1e-20)), float((na ** 2) / max(nb ** 2, 1e-20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clips', type=int, default=4)
    ap.add_argument('--frames', type=int, default=3)
    ap.add_argument('--k', type=int, default=3, help='neighbours each side')
    ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--mu', type=float, default=0.3)
    ap.add_argument('--operator', default='true', choices=['true', 'canonical'])
    ap.add_argument('--out', default='results/mf_ibp.json')
    a = ap.parse_args()

    D_ = ROOT / a.eval_set
    names = [c for c in json.loads((D_ / 'manifest.json').read_text())['clips']
             if json.loads((D_ / c / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    names = names[:a.clips]

    rows = []
    for name in names:
        T = json.loads((D_ / name / 'meta.json').read_text())['frames']
        ts = list(range(a.k + 1, T - a.k - 1, max(1, (T - 2 * a.k - 2) // a.frames)))[:a.frames]
        for t in ts:
            x = load(D_ / name / 'hr' / f'{t:06d}.png')
            h, w = x.shape[:2]
            _, n_true = split_np(x, a.operator)
            # the observations: this frame plus its neighbours, each with its own warp into frame t
            obs = [(D_op(x, a.operator), None)]
            for dk in range(-a.k, a.k + 1):
                if dk == 0:
                    continue
                xn = load(D_ / name / 'hr' / f'{t + dk:06d}.png')
                fl = flow(x, xn)                  # pulls frame t+dk into frame t's grid
                obs.append((D_op(xn, a.operator), fl))
            # single-frame baseline: plain upsample of this frame's own observation
            single = cv2.resize(obs[0][0], (w, h), interpolation=cv2.INTER_CUBIC)
            # IBP against every observation
            est = single.copy()
            for _ in range(a.iters):
                g = np.zeros_like(est)
                for y_i, fl in obs:
                    xi = est if fl is None else warp(est, fl)
                    r = y_i - D_op(xi, a.operator)
                    bp = Dt_op(r, (h, w), a.operator)
                    if fl is not None:
                        bp = warp(bp, -fl)        # approximate inverse warp
                    g += bp
                est = np.clip(est + a.mu * g / len(obs), 0, 1)
            _, n_single = split_np(single, a.operator)
            n_ibp = est - cv2.resize(D_op(est, a.operator), (w, h), interpolation=cv2.INTER_CUBIC)
            c_s, e_s = cos_energy(n_single, n_true)
            c_i, e_i = cos_energy(n_ibp, n_true)
            ps = 10 * np.log10(1.0 / max(((single - x) ** 2).mean(), 1e-12))
            pi = 10 * np.log10(1.0 / max(((est - x) ** 2).mean(), 1e-12))
            rows.append({'clip': name, 't': t,
                         'single_null_cos': round(c_s, 4), 'single_null_E': round(e_s, 4),
                         'ibp_null_cos': round(c_i, 4), 'ibp_null_E': round(e_i, 4),
                         'single_psnr': round(float(ps), 2), 'ibp_psnr': round(float(pi), 2)})
            print(json.dumps(rows[-1]), flush=True)

    def m(k):
        return float(np.mean([r[k] for r in rows]))
    res = {'eval_set': a.eval_set, 'operator': a.operator, 'clips': len(names), 'samples': len(rows),
           'k_neighbours': a.k, 'iters': a.iters, 'oracle_flow': True,
           'mean': {k: round(m(k), 4) for k in rows[0] if k not in ('clip', 't')},
           'per_sample': rows}
    (ROOT / a.out).write_text(json.dumps(res, indent=1))
    print()
    print('operator=%s, %d samples, oracle flow, %d neighbours each side' % (a.operator, len(rows), a.k))
    print('%-26s %10s %10s %10s' % ('', 'null cos', 'null E', 'PSNR'))
    print('%-26s %10.4f %10.4f %10.2f' % ('single-frame upsample', m('single_null_cos'),
                                          m('single_null_E'), m('single_psnr')))
    print('%-26s %10.4f %10.4f %10.2f' % ('multi-frame IBP', m('ibp_null_cos'),
                                          m('ibp_null_E'), m('ibp_psnr')))
    print()
    print('the DiT prior, measured over 77 clips, sits at null_cos 0.4876 and null_E 0.544')
    print('IBP null_cos %.4f -> %s' % (
        m('ibp_null_cos'),
        'beats generation on this component' if m('ibp_null_cos') > 0.4876 else
        'below generation; the prior remains the better source' if m('ibp_null_cos') > 0.2 else
        'carries essentially nothing, the idea is dead'))


if __name__ == '__main__':
    main()
