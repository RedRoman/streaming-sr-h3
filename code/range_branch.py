"""The range branch R_phi (PROTOCOL_mech.md, mech7): estimate D(x) from the degraded observation y.

Why this exists, in measured terms:
  mech5  -- 43.5% of the restorer's squared error lies in the range space, the subspace the observation
            determines exactly. innerDC does not remove it, because its projection is applied in latent
            space and the VAE round trip breaks pixel-exact consistency.
  mech5b -- substituting the raw y recovers +0.71 dB of the +3.43 dB available, because y is itself noisy,
            blurred and compressed. The missing 2.7 dB is a 1/4-resolution denoising problem.
  mech6  -- range content stays 0.614-correlated across a block stride (null: 0.103), so temporal context
            is worth having here; but warping HURTS (-0.063), so no optical flow.

The model therefore takes a causal stack of K past observations at 1/4 resolution and predicts a residual
on the newest frame. It never sees the diffusion model and has no generative prior: its whole job is the
part of the problem that is determined.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
DATA = ROOT / 'data/range_pairs'
OUT = ROOT / 'results/range_branch'


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(x), inplace=True))


def _gauss(sigma, dev, dtype):
    r = max(1, int(round(3 * sigma)))
    x = torch.arange(-r, r + 1, device=dev, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def descriptors(stack):
    """Degradation descriptors, selected by measured separation (results/mech7_desc/descriptor_selection.json).

    The model's failure mode was routing, not capacity: trained with plain L1 it applied one average
    correction and destroyed the Pillow-BICUBIC clips, whose observation already sits at 58 dB (mech8). It
    had no explicit signal for "this input is already clean".

    Spatial statistics cannot provide that signal -- measured AUC 0.43-0.55 between the two regimes, i.e.
    none -- because blur lowers the high-frequency content while noise and blocking raise it and the two
    cancel. Temporal statistics can: sensor and codec noise are independent across frames while scene
    structure is not, giving AUC 0.65-0.71. The two kept here are the best of that set. They need at least
    two frames, which is the only reason this branch takes K > 1: judge 4 showed temporal context does not
    improve the content estimate (K=1 scores +0.938 dB against K=5's +0.896), so the frames are here to
    identify the degradation, not to fuse evidence.
    """
    m = stack.mean(dim=-3)                                   # [B, K, H, W]
    y = m[:, -1:, :, :]
    td = (m[:, -1:] - m[:, -2:-1]).abs() if m.shape[1] > 1 else torch.zeros_like(y)
    mid = (y - _blur2d(y, 1.5)).abs()
    tdf = (td - _blur2d(td, 1.5)).abs()
    ratio = tdf / mid.clamp_min(1e-4)                        # temporal_hf_over_mid, AUC 0.714
    return torch.cat([_blur2d(td, 2.0) * 8.0,                # temporal, AUC 0.678
                      _blur2d(ratio, 2.0).clamp(0, 4) * 0.5], dim=-3)


def _blur2d(x, sigma):
    g = _gauss(sigma, x.device, x.dtype)
    r = g.shape[-1] // 2
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2))


class RangeBranch(nn.Module):
    """Causal K-frame -> residual on the newest frame. No warping: mech6 measured that it costs 0.063."""

    def __init__(self, k=5, ch=64, blocks=8, desc=False):
        super().__init__()
        self.k = k
        self.desc = desc
        self.head = nn.Conv2d(3 * k + (2 if desc else 0), ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)   # start as identity

    def forward(self, stack):
        """stack: [B, K, 3, H, W], oldest first. Returns the estimate for stack[:, -1]."""
        b, k, c, h, w = stack.shape
        x = stack.reshape(b, k * c, h, w)
        if self.desc:
            x = torch.cat([x, descriptors(stack)], dim=1)
        return (stack[:, -1] + self.tail(self.body(self.head(x)))).clamp(0, 1)


def load_split(split, limit=None):
    fs = sorted((DATA / split).glob('*.npz'))
    if limit:
        fs = fs[:limit]
    ys, ts = [], []
    for p in fs:
        d = np.load(p)
        ys.append(d['y']); ts.append(d['target'])
    return ys, ts


def windows(ys, ts, k, rng, n, crop=96):
    """Sample n causal K-frame windows with a random spatial crop."""
    xb, yb = [], []
    for _ in range(n):
        i = rng.integers(len(ys))
        T, H, W, _ = ys[i].shape
        t = rng.integers(k - 1, T)
        y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
        s = ys[i][t - k + 1:t + 1, y0:y0 + crop, x0:x0 + crop]
        g = ts[i][t, y0:y0 + crop, x0:x0 + crop]
        xb.append(s); yb.append(g)
    x = torch.from_numpy(np.stack(xb)).float().permute(0, 1, 4, 2, 3) / 255
    y = torch.from_numpy(np.stack(yb)).float().permute(0, 3, 1, 2) / 255
    return x, y


@torch.no_grad()
def evaluate(model, ys, ts, k, dev, stride=4):
    """Full-frame PSNR against D(x), plus the identity baseline PSNR(y, D(x))."""
    pm, pb = [], []
    for yv, tv in zip(ys, ts):
        T = yv.shape[0]
        for t in range(k - 1, T, stride):
            s = torch.from_numpy(yv[t - k + 1:t + 1]).float().permute(0, 3, 1, 2)[None].to(dev) / 255
            g = torch.from_numpy(tv[t]).float().permute(2, 0, 1)[None].to(dev) / 255
            out = model(s)
            pm.append(float(10 * torch.log10(1 / (out - g).square().mean().clamp_min(1e-12))))
            pb.append(float(10 * torch.log10(1 / (s[:, -1] - g).square().mean().clamp_min(1e-12))))
    return float(np.mean(pm)), float(np.mean(pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--ch', type=int, default=64)
    ap.add_argument('--blocks', type=int, default=8)
    ap.add_argument('--iters', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--train-clips', type=int, default=None)
    ap.add_argument('--loss', default='guard', choices=['l1', 'rel', 'guard'],
                    help="rel = per-sample L1 normalised by the identity's own L1. Plain L1 lets the "
                         'degradation mix decide the gradient: a Pillow-BICUBIC clip already sits at 58 dB '
                         'so its absolute error is ~3%% of a realistic clip\'s, the model never learns to '
                         'leave clean inputs alone, and it destroys them (-14 dB measured). rel fixes the '
                         'destruction but over-corrects: with an epsilon floor the near-perfect samples '
                         'dominate and the model collapses to the identity (+0.005 dB measured). guard '
                         'keeps plain L1, which already gave +1.95 dB where it matters, and adds a hinge '
                         'that only fires when the prediction is worse than the identity.')
    ap.add_argument('--guard-w', type=float, default=10.0)
    ap.add_argument('--desc', action='store_true',
                    help='append the degradation descriptors (see descriptors())')
    ap.add_argument('--tag', default=None)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f'k{a.k}'
    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()

    tr_y, tr_t = load_split('train', a.train_clips)
    va_y, va_t = load_split('val', 24)
    assert tr_y and va_y, (len(tr_y), len(va_y))
    model = RangeBranch(a.k, a.ch, a.blocks, desc=a.desc).to(dev)
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': tag, 'k': a.k, 'ch': a.ch, 'blocks': a.blocks, 'desc': a.desc,
           'params': n_par, 'loss': a.loss,
           'guard_w': a.guard_w,
           'train_samples': len(tr_y), 'val_samples': len(va_y), 'iters': a.iters, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)

    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, g = windows(tr_y, tr_t, a.k, rng, a.batch)
        x, g = x.to(dev), g.to(dev)
        pred = model(x)
        if a.loss == 'rel':
            base = (x[:, -1] - g).abs().mean(dim=(1, 2, 3))
            err = (pred - g).abs().mean(dim=(1, 2, 3))
            loss = (err / (base + 2e-3)).mean()
        elif a.loss == 'guard':
            base = (x[:, -1] - g).abs().mean(dim=(1, 2, 3))
            err = (pred - g).abs().mean(dim=(1, 2, 3))
            loss = err.mean() + a.guard_w * (err - base).clamp_min(0).mean()
        else:
            loss = F.l1_loss(pred, g)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 12) == 0 or it == a.iters:
            model.eval(); pm, pb = evaluate(model, va_y, va_t, a.k, dev); model.train()
            e = {'iter': it, 'loss': float(loss), 'val_psnr': pm, 'baseline_psnr': pb,
                 'gain_dB': pm - pb, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e)
            print(json.dumps(e), flush=True)
    rec['final'] = rec['curve'][-1]
    torch.save({'state': model.state_dict(),
                'cfg': {'k': a.k, 'ch': a.ch, 'blocks': a.blocks, 'desc': a.desc}},
               OUT / f'{tag}.pt')
    (OUT / f'{tag}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({'final': rec['final'], 'params': n_par}), flush=True)


if __name__ == '__main__':
    main()
