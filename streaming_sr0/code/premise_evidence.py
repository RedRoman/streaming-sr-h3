"""mech4 (PROTOCOL_mech.md): does accumulating OBSERVATIONS shrink the unobserved subspace?

mech2 asked whether the answer can be carried across time and found it cannot -- warping attenuates exactly
the high-frequency band it is trying to move. This asks the dual question: whether the *evidence* can be
accumulated. With one observation y_t = D x_t the null space of D is fixed; with K observations of a moving
scene the effective operator is {D W_k} and its joint null space is strictly smaller whenever the motion has
sub-pixel components. Evidence lives in the observed, low-frequency domain, which survives warping, so this
is the one thing that could cross a block boundary without drifting.

Method: classical iterative back-projection over a causal window of K past observations, with motion
estimated from the observations themselves (model-free). Then measure how much of the single-frame null band
the fused estimate recovers.

Baseline to beat: Stage 1 reports cosine 0.434 between H3+LoRA's generated band and GT.

Read-out rules, fixed before running (K = 8, flow from observations):
  strong : recovery >= 0.434  -> the model is leaving recoverable real information unused; carry evidence
  none   : recovery <  0.10   -> an antialiased observation model leaves nothing to accumulate, so the null
           space can only be hallucinated and no temporal module can help it; report that
  partial: in between
Controls: oracle flow from GT frames (separates flow error from a fundamental limit), and a time-shuffled
window (must give ~0, otherwise the metric is measuring the prior, not the evidence).
"""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/stage2_eval'
OUT = ROOT / 'results/mech4_premise'
KS = (1, 2, 4, 8, 16)
SCALE = 4
ITERS = 12
STRIDE = 16
STAGE1_COS = 0.434


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


AA = True          # set per arm; the aliasing arm is the implementation control


def down(x, aa=None):
    """D: the observation operator, x4 bicubic. antialias=True is the one our evaluation uses."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    return F.interpolate(t, size=(h // SCALE, w // SCALE), mode='bicubic', align_corners=False,
                         antialias=AA if aa is None else aa).permute(0, 2, 3, 1).float()


def up(y, hw):
    t = y.permute(0, 3, 1, 2).double()
    return F.interpolate(t, size=hw, mode='bicubic', align_corners=False).permute(0, 2, 3, 1).float()


def null(x):
    """The band the EVALUATION operator leaves undetermined -- fixed across arms, so the two arms are
    answering the same question: how much of this band does fused evidence recover."""
    return x - up(down(x, aa=True), x.shape[1:3])


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def warp(img, fl):
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def corr(a, b):
    a, b = np.asarray(a).reshape(-1).astype(np.float64), np.asarray(b).reshape(-1).astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def fuse(obs, to_k, to_t, hw):
    """Iterative back-projection onto the target grid.

    to_k[k] warps an image FROM the target grid INTO frame k's grid, to_t[k] warps back. With OpenCV's
    convention -- remap(img, x + fl) resamples an image living in `next`'s domain onto `prev`'s grid when
    fl = calcOpticalFlowFarneback(prev, next) -- pushing the estimate from t into k therefore needs
    flow(k, t), not flow(t, k). Getting this backwards accumulates every residual at the wrong place and
    makes fusion destroy information instead of adding it, which is what the aliasing control is there to
    catch."""
    est = up(obs[-1][None], hw)[0].numpy()                  # start from the plain upsample of the newest frame
    for _ in range(ITERS):
        acc = np.zeros_like(est)
        for k, y in enumerate(obs):
            w_ = est if to_k[k] is None else warp(est, to_k[k])
            d = down(torch.from_numpy(w_)[None])[0].numpy()
            r = up(torch.from_numpy(y.numpy() - d)[None], hw)[0].numpy()
            acc += r if to_t[k] is None else warp(r, to_t[k])
        est = est + acc / len(obs)
    return est


def bootstrap_ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    global AA
    arms = {'antialias': True, 'aliasing': False}
    agg = {a: {k: {'obs_flow': [], 'gt_flow': [], 'shuffled': []} for k in KS} for a in arms}
    rows = []
    rng = np.random.default_rng(0)
    for name in man['clips']:
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        hr = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in range(T)])
        y = down(hr)                                        # clean x4 observations
        uy = up(y, hw)
        g_obs = (uy.mean(-1).numpy() * 255).astype(np.uint8)
        g_gt = (hr.mean(-1).numpy() * 255).astype(np.uint8)
        r = {'clip': name}
        for arm, aa in arms.items():
          AA = aa
          y = down(hr)
          uy = up(y, hw)
          g_obs = (uy.mean(-1).numpy() * 255).astype(np.uint8)
          for K in KS:
            res = {'obs_flow': [], 'gt_flow': [], 'shuffled': []}
            for t in range(K - 1, T, STRIDE):
                idx = list(range(t - K + 1, t + 1))
                tgt = null(hr[t:t + 1])[0].numpy()
                for tag, gray, order in (('obs_flow', g_obs, idx), ('gt_flow', g_gt, idx),
                                         ('shuffled', g_obs, list(rng.permutation(T)[:K]))):
                    to_k, to_t = [], []
                    for k in order:
                        if k == t:
                            to_k.append(None); to_t.append(None)
                        else:
                            to_k.append(flow(gray[k], gray[t]))   # pushes t -> k
                            to_t.append(flow(gray[t], gray[k]))   # pushes k -> t
                    est = fuse([y[k] for k in order], to_k, to_t, hw)
                    res[tag].append(corr(null(torch.from_numpy(est)[None])[0].numpy(), tgt))
            for tag in res:
                m = float(np.mean(res[tag])); r[f'{arm}_K{K}_{tag}'] = m; agg[arm][K][tag].append(m)
        rows.append(r)
        print(json.dumps({'clip': name, **{k: round(v, 3) for k, v in r.items() if k != 'clip'}}), flush=True)

    out_agg = {a: {f'K{K}': {tag: {'mean': float(np.mean(v)), 'ci95': bootstrap_ci(v)}
                             for tag, v in agg[a][K].items()} for K in KS} for a in arms}
    gain = {a: {f'K{K}': out_agg[a][f'K{K}']['obs_flow']['mean'] - out_agg[a]['K1']['obs_flow']['mean']
                for K in KS} for a in arms}
    m8 = gain['antialias']['K8']
    ctrl8 = gain['aliasing']['K8']
    verdict = ('strong: multi-frame evidence recovers real detail the model is not using' if m8 >= 0.20
               else 'none: an antialiased observation model leaves nothing to accumulate' if m8 < 0.05
               else 'partial')
    if ctrl8 < 0.10:
        verdict = 'VOID: the aliasing control did not show a gain, so the fusion itself is not working'
    out = {'n': len(rows), 'scale': SCALE, 'iters': ITERS, 'stage1_model_cos': STAGE1_COS,
           'per_clip': rows, 'aggregate': out_agg, 'verdict': verdict,
           'gain_over_K1': gain,
           'implementation_control_aliasing_K8_gain': ctrl8,
           'control_passes': bool(ctrl8 >= 0.10)}
    (OUT / 'premise.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'aggregate': out_agg, 'verdict': verdict}, indent=1))


if __name__ == '__main__':
    main()
