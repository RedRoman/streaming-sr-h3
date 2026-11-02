"""Can a severity estimator tell that an input is already clean?

Gate A came apart into two pieces. Conditioning works: given the true severity, one range branch reaches
+2.71 dB on realistic degradation and 0.000 dB on the clean Pillow-BICUBIC regime, against -14.17 dB for
the unconditioned model that motivated all of this. The whole remaining gap is estimation -- predicted
severity gives -2.95 instead of 0.000 -- and it is localised: within the realistic family the estimator
reaches MAE 0.106, but on clean clips it reports ~0.75 where the truth is 0.055.

So the question is narrow: detect that an observation needs nothing done to it. Three candidate fixes,
run as one sweep because they are not exclusive:

  frames=1,L1      the incumbent, for reference
  frames=K         temporal input. descriptor_select and noise_sep both found spatial statistics at
                   chance (AUC 0.43-0.55) and temporal ones the only informative family (AUC 0.669), and
                   a clean clip is temporally clean too -- its frame differences carry scene motion
                   without sensor or codec noise.
  loss=bce         the severity distribution is bimodal (0.055 and ~0.89) and an L1-optimal prediction of
                   a bimodal target is the mass-weighted middle, which is exactly the ~0.75 observed.
                   Binary cross-entropy on the normalised value does not average across modes the same
                   way; `quantile` fits the median instead, which on a bimodal target picks a mode.

Reported per kind, because the whole point is the clean regime: pillow MAE below 0.1 closes the gate.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/range_branch'
PSNR_CEIL, PSNR_SPAN = 60.0, 35.0


def severity_of(y, t):
    d = (y.astype(np.float32) - t.astype(np.float32)) / 255.0
    psnr = 10 * np.log10(1.0 / max(float((d ** 2).mean()), 1e-12))
    return float(np.clip((PSNR_CEIL - psnr) / PSNR_SPAN, 0, 1))


class Est(nn.Module):
    def __init__(self, frames=1, ch=32):
        super().__init__()
        self.frames = frames
        self.f = nn.Sequential(
            nn.Conv2d(3 * frames, ch, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1), nn.ReLU(inplace=True))
        self.o = nn.Linear(ch * 8, 1)

    def forward(self, x):
        """x [B, frames, 3, H, W] -> logit [B,1]."""
        b = x.shape[0]
        h = self.f(x.reshape(b, -1, *x.shape[-2:]))
        return self.o(torch.cat([h.mean((-2, -1)), h.std((-2, -1))], 1))


class Data:
    def __init__(self, root, split, limit=None):
        fs = sorted((Path(root) / split).glob('*.npz'))[:limit]
        self.y, self.kind, self.s = [], [], []
        for p in fs:
            d = np.load(p)
            self.y.append(d['y'])
            self.kind.append(json.loads(str(d['meta']))['degradation'].get('kind', '?'))
            self.s.append(severity_of(d['y'][0], d['target'][0]))
        assert self.y

    def batch(self, rng, n, crop, frames, balance=True):
        by = {}
        for j, k in enumerate(self.kind):
            by.setdefault(k, []).append(j)
        kinds = sorted(by)
        xb, sb = [], []
        for q in range(n):
            pool = by[kinds[q % len(kinds)]] if balance else range(len(self.y))
            i = pool[rng.integers(len(pool))] if balance else rng.integers(len(self.y))
            T, H, W, _ = self.y[i].shape
            t = rng.integers(frames - 1, T)
            y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            xb.append(self.y[i][t - frames + 1:t + 1, y0:y0 + crop, x0:x0 + crop])
            sb.append(self.s[i])
        x = torch.from_numpy(np.stack(xb)).float().permute(0, 1, 4, 2, 3) / 255
        return x, torch.tensor(sb, dtype=torch.float32)[:, None]


@torch.no_grad()
def evaluate(model, data, dev, frames, stride=8, chunk=8):
    acc = {}
    for yv, k, s in zip(data.y, data.kind, data.s):
        idx = list(range(frames - 1, yv.shape[0], stride))
        preds = []
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = torch.from_numpy(np.stack([yv[t - frames + 1:t + 1] for t in sel])).float()
            preds.append(model(x.permute(0, 1, 4, 2, 3).to(dev) / 255).sigmoid())
        p = float(torch.cat(preds).mean())          # the operator is clip-constant: average the stream
        d = acc.setdefault(k, {'e': [], 'p': [], 't': []})
        d['e'].append(abs(p - s)); d['p'].append(p); d['t'].append(s)
    return {k: {'n': len(d['e']), 'mae': float(np.mean(d['e'])),
                'pred_mean': float(np.mean(d['p'])), 'true_mean': float(np.mean(d['t']))}
            for k, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--frames', type=int, default=1)
    ap.add_argument('--loss', default='l1', choices=['l1', 'bce', 'quantile'])
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--tag', required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        import sys
        sys.path.insert(0, str(ROOT / 'code'))
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
    tr, va = Data(a.data, 'train'), Data(a.data, 'val')
    model = Est(a.frames).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'frames': a.frames, 'loss': a.loss, 'iters': a.iters,
           'params': sum(p.numel() for p in model.parameters()), 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, s = tr.batch(rng, a.batch, a.crop, a.frames)
        x, s = x.to(dev), s.to(dev)
        z = model(x)
        if a.loss == 'bce':
            loss = F.binary_cross_entropy_with_logits(z, s)
        elif a.loss == 'quantile':
            loss = (z.sigmoid() - s).abs().mean()      # median fit; on a bimodal target picks a mode
        else:
            loss = F.l1_loss(z.sigmoid(), s)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 8) == 0 or it == a.iters:
            model.eval(); ev = evaluate(model, va, dev, a.frames); model.train()
            e = {'iter': it, 'loss': float(loss), 'val': ev, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e); print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(), 'cfg': {'frames': a.frames}}, OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    p = rec['final']['val'].get('pillow_bicubic', {})
    print(json.dumps({'tag': a.tag, 'pillow_mae': round(p.get('mae', 9), 4),
                      'GATE_CLOSED': p.get('mae', 9) < 0.1}, indent=1), flush=True)


if __name__ == '__main__':
    main()
