"""Step 3 of the design: estimate the degradation operator from the stream, condition the range branch on it.

The claim this implements, and the measurements behind it:
  - content cannot be accumulated across time: the null space decorrelates to 0.103 across a block stride,
    warping it costs 0.063, and K=1 scored +0.938 dB against K=5's +0.896 in the HR domain (judge 4);
  - the operator CAN be accumulated: our degradation parameters are drawn once per clip, so a stream is
    many observations of one operator, and temporal statistics identify it (AUC 0.714 vs 0.50 for spatial);
  - the blind case is where the prize is: with a clean operator the branch captures 65% of the available
    range-space error, with an unknown one only 23%.

So: a small estimator produces a low-dimensional operator code from short observation windows, the code is
averaged over the stream, and the range branch is conditioned on it through FiLM. Content stays per-frame.

The operator code is supervised directly against the recorded degradation parameters. That is deliberate:
it makes the estimate interpretable, it separates "can the operator be identified" from "does identifying
it help", and it turns judge 1 into two independently measurable curves.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from range_branch import ResBlock, _blur2d, descriptors, load_split

ROOT = Path('/cache/zihaohe/streaming_sr0')
DATA = ROOT / 'data/range_pairs'
OUT = ROOT / 'results/operator_estimator'
PDIM = 8


def params_vector(deg):
    """Normalised operator parameters: [sig_x, sig_y, aniso, noise, second, crf, jpeg_q, codec_kind]."""
    v = np.zeros(PDIM, dtype=np.float32)
    if deg.get('kind') == 'pillow_bicubic':
        return v                                     # the clean operator is the all-zero code
    b = deg.get('blur1')
    if b:
        v[0] = min(1.0, float(b[0]) / 2.5)
        v[1] = min(1.0, float(b[1]) / 2.5) if len(b) > 1 else v[0]
        v[2] = 1.0 if (len(b) > 3 and b[3] == 'aniso') else 0.0
    v[3] = 1.0 if deg.get('noise1') else 0.0
    v[4] = 1.0 if deg.get('second_order') else 0.0
    if deg.get('h264_crf') is not None:
        v[5] = (float(deg['h264_crf']) - 18) / 19.0
        v[7] = 0.5
    elif deg.get('jpeg_q') is not None:
        v[6] = 1.0 - (float(deg['jpeg_q']) - 35) / 61.0
        v[7] = 1.0
    return np.clip(v, 0, 1)


SDIM = 15


def summary(pair):
    """Per-frame-pair evidence about the operator, as a small vector of spatial statistics.

    Accumulate the evidence, not the conclusion. A first version averaged the estimator's per-frame output
    and the error did not fall with stream length (0.3432 -> 0.3447 over a clip), because what the estimator
    gets wrong is a bias and biases do not average out. Averaging these statistics instead makes "more
    frames, better estimate" structurally true: the variance of a mean falls with the number of samples,
    and the operator is constant over the stream by construction.

    pair: [B, 2, 3, H, W] -> [B, SDIM]."""
    cur = pair[:, -1]
    d = descriptors(pair)                                  # temporal: the pair measured at AUC 0.714
    g = cur.mean(dim=-3, keepdim=True)
    fine = (g - _blur2d(g, 0.7)).abs()
    mid = (g - _blur2d(g, 1.5)).abs()
    coarse = (g - _blur2d(g, 3.0)).abs()
    flat = (_blur2d(coarse, 2.0) < coarse.mean(dim=(-2, -1), keepdim=True) * 0.6).float()
    def st(x):
        return torch.stack([x.mean(dim=(-3, -2, -1)),
                            x.std(dim=(-3, -2, -1)),
                            x.flatten(1).quantile(0.9, dim=-1)], dim=-1)
    masked = (fine * flat).sum(dim=(-3, -2, -1)) / flat.sum(dim=(-3, -2, -1)).clamp_min(1.0)
    return torch.cat([st(d[:, :1]), st(d[:, 1:]), st(fine), st(mid / coarse.clamp_min(1e-4)),
                      masked[:, None], cur.std(dim=(-3, -2, -1))[:, None],
                      cur.mean(dim=(-3, -2, -1))[:, None]], dim=-1)


class OperatorEstimator(nn.Module):
    """Accumulated evidence -> operator code. The accumulation happens outside, on `summary` vectors."""

    def __init__(self, sdim=SDIM, hid=96, pdim=PDIM):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(sdim, hid), nn.ReLU(True),
                                 nn.Linear(hid, hid), nn.ReLU(True),
                                 nn.Linear(hid, pdim))

    def forward(self, s):
        return torch.sigmoid(self.net(s))


class FiLM(nn.Module):
    def __init__(self, pdim, ch):
        super().__init__()
        self.fc = nn.Linear(pdim, 2 * ch)
        nn.init.zeros_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, x, theta):
        g, b = self.fc(theta).chunk(2, dim=-1)
        return x * (1 + g[..., None, None]) + b[..., None, None]


class ConditionedRange(nn.Module):
    """Per-frame content estimate, conditioned on the accumulated operator code.

    K is 2 only because the estimator needs a frame pair; judge 4 showed extra frames do not improve the
    content estimate, so they are not given to the content path."""

    def __init__(self, ch=64, blocks=8, pdim=PDIM):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.films = nn.ModuleList([FiLM(pdim, ch) for _ in range(blocks)])
        self.body = nn.ModuleList([ResBlock(ch) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, y, theta):
        h = self.head(y)
        for film, blk in zip(self.films, self.body):
            h = blk(film(h, theta))
        return (y + self.tail(h)).clamp(0, 1)


def batch(ys, ts, ps, rng, n, crop=96, nobs=4):
    """One sample: a crop to restore, plus `nobs` frame pairs of evidence drawn from the same clip.

    Training sees a random number of observations so the estimator has to work at every stream length."""
    xb, gb, tb, sb = [], [], [], []
    for _ in range(n):
        i = rng.integers(len(ys))
        T, H, W, _ = ys[i].shape
        t = rng.integers(1, T)
        y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
        xb.append(ys[i][t, y0:y0 + crop, x0:x0 + crop])
        gb.append(ts[i][t, y0:y0 + crop, x0:x0 + crop])
        tb.append(ps[i])
        k = int(rng.integers(1, nobs + 1))
        pairs = []
        for _ in range(k):
            u = rng.integers(1, T)
            v0, u0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            pairs.append(ys[i][u - 1:u + 1, v0:v0 + crop, u0:u0 + crop])
        pr = torch.from_numpy(np.stack(pairs)).float().permute(0, 1, 4, 2, 3) / 255
        sb.append(summary(pr).mean(0))
    y = torch.from_numpy(np.stack(xb)).float().permute(0, 3, 1, 2) / 255
    g = torch.from_numpy(np.stack(gb)).float().permute(0, 3, 1, 2) / 255
    th = torch.from_numpy(np.stack(tb)).float()
    return torch.stack(sb), y, g, th


def load_params(split, n):
    fs = sorted((DATA / split).glob('*.npz'))[:n] if n else sorted((DATA / split).glob('*.npz'))
    return [params_vector(json.loads(str(np.load(f)['meta']))['degradation']) for f in fs]


@torch.no_grad()
def eval_stream(est, net, ys, ts, ps, stride=4):
    """Gain and operator error as a function of how many frames of the stream have been seen.

    This is judge 1: the operator code is a running mean over the stream, so if the operator really can be
    accumulated the gain should rise with frame index, while the K=5 content-memory model was flat."""
    per_t, errs = {}, {}
    for yv, tv, pv in zip(ys, ts, ps):
        T = yv.shape[0]
        acc, cnt = torch.zeros(SDIM), 0
        for t in range(1, T):
            pair = torch.from_numpy(yv[t - 1:t + 1]).float().permute(0, 3, 1, 2)[None] / 255
            acc = acc + summary(pair)[0]; cnt += 1
            theta = est((acc / cnt)[None])
            if (t - 1) % stride:
                continue
            y = torch.from_numpy(yv[t]).float().permute(2, 0, 1)[None] / 255
            g = torch.from_numpy(tv[t]).float().permute(2, 0, 1)[None] / 255
            out = net(y, theta)
            gain = float(10 * torch.log10((y - g).square().mean().clamp_min(1e-12)
                                          / (out - g).square().mean().clamp_min(1e-12)))
            per_t.setdefault(t, []).append(gain)
            errs.setdefault(t, []).append(float((theta[0] - torch.from_numpy(pv)).abs().mean()))
    ks = sorted(per_t)
    return ([{'frame': k, 'gain_dB': float(np.mean(per_t[k])), 'theta_mae': float(np.mean(errs[k]))}
             for k in ks])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--aux', type=float, default=1.0, help='weight on the operator-code supervision')
    ap.add_argument('--train-clips', type=int, default=None)
    ap.add_argument('--tag', default='op')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr_y, tr_t = load_split('train', a.train_clips)
    va_y, va_t = load_split('val', 24)
    tr_p = load_params('train', a.train_clips)
    va_p = load_params('val', 24)
    est, net = OperatorEstimator(), ConditionedRange()
    n_par = sum(p.numel() for p in list(est.parameters()) + list(net.parameters()))
    opt = torch.optim.AdamW(list(est.parameters()) + list(net.parameters()), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'params': n_par, 'aux': a.aux, 'iters': a.iters,
           'train_samples': len(tr_y), 'val_samples': len(va_y), 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        sv, y, g, th = batch(tr_y, tr_t, tr_p, rng, a.batch)
        th_hat = est(sv)
        out = net(y, th_hat)
        base = (y - g).abs().mean(dim=(1, 2, 3))
        err = (out - g).abs().mean(dim=(1, 2, 3))
        loss = err.mean() + a.aux * F.mse_loss(th_hat, th)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 10) == 0 or it == a.iters:
            est.eval(); net.eval()
            curve = eval_stream(est, net, va_y, va_t, va_p)
            est.train(); net.train()
            e = {'iter': it, 'loss': float(loss), 'theta_mse': float(F.mse_loss(th_hat, th)),
                 'gain_first': curve[0]['gain_dB'], 'gain_last': curve[-1]['gain_dB'],
                 'theta_mae_first': curve[0]['theta_mae'], 'theta_mae_last': curve[-1]['theta_mae'],
                 'rise_dB': curve[-1]['gain_dB'] - curve[0]['gain_dB'],
                 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e); rec['stream'] = curve
            print(json.dumps(e), flush=True)
    torch.save({'est': est.state_dict(), 'net': net.state_dict()}, OUT / f'{a.tag}.pt')
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
