"""A fair test of whether the degradation operator is identifiable from the stream.

op1 answered no, but it could not have answered anything else: its estimator sees only `summary(pair)`,
15 hand-designed spatial statistics, and descriptor_select plus noise_sep already measured that spatial
statistics separate the two degradation regimes at AUC 0.43-0.55, i.e. chance. Predicting an 8-dimensional
operator code from features known to carry no operator information is not a test of identifiability. The
numbers bear that out: after 5400 steps its code MAE is 0.277 against trivial baselines of 0.287 (predict
zeros), 0.300 (predict the mean) and 0.252 (predict the per-dimension median) -- worse than the median.

This replaces the hand-designed features with a small CNN over the raw frame pair, which is the honest
version of the question: is the operator recoverable from the observation at all, given enough capacity to
look for it? Everything else is held fixed -- the same code definition, the same 8 targets, the same
train/val split, the same running mean over the stream.

Reported against all three trivial predictors, per degradation kind, and as a function of how many frames
of the stream have been seen, because the premise of the whole design is that the operator is constant per
clip and therefore accumulable while content is not.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/operator_estimator'
PDIM = 8


class OpCNN(nn.Module):
    """Frame pair -> operator code. Strided convolutions then global statistics; the code is a property
    of the clip, not of any pixel, so the head pools mean and std over space."""

    def __init__(self, ch=48, pdim=PDIM):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(6, ch, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 2, 3, 1, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 3, 1, 1), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Linear(ch * 8, 256), nn.ReLU(inplace=True), nn.Linear(256, pdim))

    def forward(self, pair):
        """pair [B,2,3,H,W] -> [B,pdim] in [0,1]."""
        b = pair.shape[0]
        x = self.stem(pair.reshape(b, 6, *pair.shape[-2:]))
        f = torch.cat([x.mean((-2, -1)), x.std((-2, -1))], 1)
        return self.head(f).sigmoid()


def batch(ys, ps, rng, n, crop=96):
    xb, yb = [], []
    for _ in range(n):
        i = rng.integers(len(ys))
        T, H, W, _ = ys[i].shape
        t = rng.integers(1, T)
        y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
        xb.append(ys[i][t - 1:t + 1, y0:y0 + crop, x0:x0 + crop])
        yb.append(ps[i])
    x = torch.from_numpy(np.stack(xb)).float().permute(0, 1, 4, 2, 3) / 255
    return x, torch.from_numpy(np.stack(yb)).float()


@torch.no_grad()
def evaluate(model, ys, ps, kinds, dev, frames=(1, 8, 21), chunk=16):
    """Running-mean code over the stream, scored at several stream lengths and per degradation kind.

    Every pair is independent, so the whole clip goes through in batched chunks and the running means are
    read off a cumulative sum; the first version did one forward per frame and spent longer evaluating
    than training."""
    per = {f: {} for f in frames}
    for yv, pv, k in zip(ys, ps, kinds):
        n = yv.shape[0] - 1
        outs = []
        for s0 in range(0, n, chunk):
            idx = range(s0, min(s0 + chunk, n))
            x = torch.from_numpy(np.stack([yv[t:t + 2] for t in idx])).float()
            outs.append(model(x.permute(0, 1, 4, 2, 3).to(dev) / 255))
        th = torch.cat(outs)                                   # [n, pdim], pair t -> frames t,t+1
        cum = th.cumsum(0) / torch.arange(1, n + 1, device=dev, dtype=th.dtype)[:, None]
        tgt = torch.from_numpy(pv).to(dev)
        for f in frames:
            e = float((cum[min(f, n) - 1] - tgt).abs().mean())
            per[f].setdefault(k, []).append(e); per[f].setdefault('ALL', []).append(e)
    return {f: {k: float(np.mean(v)) for k, v in d.items()} for f, d in per.items()}


def load(split, limit=None):
    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    from operator_estimator import params_vector
    fs = sorted(Path('/temp/zihaohe_range_pairs' if not (ROOT / 'data/range_pairs').exists()
                     else ROOT / 'data/range_pairs').joinpath(split).glob('*.npz'))[:limit]
    ys, ps, ks = [], [], []
    for p in fs:
        d = np.load(p)
        meta = json.loads(str(d['meta']))['degradation']
        ys.append(d['y']); ps.append(params_vector(meta)); ks.append(meta.get('kind', '?'))
    return ys, ps, ks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=96)
    ap.add_argument('--ch', type=int, default=48)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--train-clips', type=int, default=None)
    ap.add_argument('--tag', default='opcnn')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        import sys
        sys.path.insert(0, str(ROOT / 'code'))
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()

    tr_y, tr_p, _ = load('train', a.train_clips)
    va_y, va_p, va_k = load('val')
    th = np.stack(va_p)
    base = {'zeros': float(np.abs(th).mean()),
            'mean': float(np.abs(th - th.mean(0)).mean()),
            'median': float(np.abs(th - np.median(th, 0)).mean())}
    model = OpCNN(a.ch).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'params': sum(p.numel() for p in model.parameters()),
           'iters': a.iters, 'batch': a.batch, 'crop': a.crop,
           'train_clips': len(tr_y), 'val_clips': len(va_y), 'trivial_baselines': base, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, th_ = batch(tr_y, tr_p, rng, a.batch, a.crop)
        loss = F.l1_loss(model(x.to(dev)), th_.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 8) == 0 or it == a.iters:
            model.eval(); ev = evaluate(model, va_y, va_p, va_k, dev); model.train()
            e = {'iter': it, 'loss': float(loss), 'val_mae': ev, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e)
            print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(), 'cfg': {'ch': a.ch}}, OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({'trivial': base, 'final': rec['final']['val_mae']}, indent=1), flush=True)


if __name__ == '__main__':
    main()
