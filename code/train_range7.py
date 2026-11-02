"""Stage-7 range branch: retrain R_phi on real footage, with the routing problem addressed by construction.

Three objectives were tried on the generated data and all failed in the same place. Plain L1 learned a
useful mapping (+1.95 dB where the observation is corrupted) but applied it indiscriminately and destroyed
clips whose observation was already near-perfect (-14 dB, measured). A per-sample normalised loss and a
hinge against being worse than the identity both stopped the destruction by collapsing to the identity
(+0.005 dB). Per-frame degradation descriptors helped but not nearly enough (judge 5: bicubic -8.1 against
a required +1.0), because a single frame pair separates the regimes at only AUC 0.714.

What changes here:
  - real footage, whose unobserved band carries ~20x the energy of the generated clips Stage 1 used;
  - a gate predicted from clip-level accumulated evidence rather than from one frame pair. The evidence is
    the mean of the per-pair summary statistics over the window, which is what makes "more frames, better
    estimate" structurally true: their within-clip spread is 0.023-0.184 and the mean of n of them shrinks
    it by sqrt(n). The gate multiplies the residual, so leaving a clean observation alone is one value the
    model can represent exactly, not something it has to approximate.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
DATA = ROOT / 'data/stage7_train'
OUT = ROOT / 'results/stage7_range'


def gauss(sigma, dtype=torch.float32):
    r = max(1, int(round(3 * sigma)))
    t = torch.arange(-r, r + 1, dtype=dtype)
    g = torch.exp(-0.5 * (t / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def blur2d(x, sigma):
    g = gauss(sigma, x.dtype).to(x.device)
    r = g.shape[-1] // 2
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2))


SDIM = 6


def evidence(pair):
    """Per-frame-pair statistics that separate degradation regimes. pair: [B,2,3,H,W] -> [B,SDIM].

    Temporal terms dominate by design: noise and codec artefacts are independent across frames while scene
    structure is not, which is the only property measured to separate the regimes (AUC 0.65-0.71 temporal
    against 0.43-0.55 for every spatial statistic tried)."""
    m = pair.mean(dim=-3)
    cur = m[:, -1:]
    td = (m[:, -1:] - m[:, -2:-1]).abs()
    fine = (cur - blur2d(cur, 0.7)).abs()
    mid = (cur - blur2d(cur, 1.5)).abs()
    tdf = (td - blur2d(td, 1.5)).abs()
    f = lambda x: x.mean(dim=(-3, -2, -1))
    return torch.stack([f(td), f(tdf), f(tdf) / f(mid).clamp_min(1e-4),
                        f(fine), f(fine) / f(mid).clamp_min(1e-4), cur.std(dim=(-3, -2, -1))], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a, self.b = nn.Conv2d(c, c, 3, padding=1), nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return x + self.b(F.relu(self.a(x), inplace=True))


class RangeBranch7(nn.Module):
    def __init__(self, ch=64, blocks=8, sdim=SDIM):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.Sequential(*[ResBlock(ch) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        # the gate decides how much correction this observation needs, from accumulated evidence alone
        self.gate = nn.Sequential(nn.Linear(sdim, 64), nn.ReLU(True), nn.Linear(64, 64), nn.ReLU(True),
                                  nn.Linear(64, 1))
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
        nn.init.zeros_(self.gate[-1].weight); nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, y, ev):
        g = torch.sigmoid(self.gate(ev))[..., None, None]
        return (y + g * self.tail(self.body(self.head(y)))).clamp(0, 1), g.flatten()


def load(split, limit=None):
    fs = sorted((DATA / split).glob('*.npz'))
    if limit:
        fs = fs[:limit]
    out = []
    for p in fs:
        d = np.load(p)
        out.append((d['lr'], d['dx'], json.loads(str(d['meta']))['degradation'].get('kind', '?')))
    return out


def sample_batch(items, rng, n, crop=96, nobs=6):
    ys, gs, evs = [], [], []
    for _ in range(n):
        lr, dx, _ = items[rng.integers(len(items))]
        T, H, W, _ = lr.shape
        t = rng.integers(1, T)
        y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
        ys.append(lr[t, y0:y0 + crop, x0:x0 + crop])
        gs.append(dx[t, y0:y0 + crop, x0:x0 + crop])
        pr = []
        for _ in range(nobs):
            u = rng.integers(1, T)
            v0, u0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            pr.append(lr[u - 1:u + 1, v0:v0 + crop, u0:u0 + crop])
        t_pr = torch.from_numpy(np.stack(pr)).float().permute(0, 1, 4, 2, 3) / 255
        evs.append(evidence(t_pr).mean(0))
    y = torch.from_numpy(np.stack(ys)).float().permute(0, 3, 1, 2) / 255
    g = torch.from_numpy(np.stack(gs)).float().permute(0, 3, 1, 2) / 255
    return y, g, torch.stack(evs)


@torch.no_grad()
def evaluate(model, items, stride=6):
    by = {}
    for lr, dx, kind in items:
        T = lr.shape[0]
        ev = evidence(torch.from_numpy(lr[:, None][1:]).float().permute(0, 1, 4, 2, 3).reshape(-1, 1, 3, *lr.shape[1:3]).repeat(1, 2, 1, 1, 1) / 255) if False else None
        pr = torch.stack([torch.from_numpy(lr[t - 1:t + 1]).float().permute(0, 3, 1, 2) / 255
                          for t in range(1, T)])
        e = evidence(pr).mean(0, keepdim=True)
        for t in range(1, T, stride):
            y = torch.from_numpy(lr[t]).float().permute(2, 0, 1)[None] / 255
            g = torch.from_numpy(dx[t]).float().permute(2, 0, 1)[None] / 255
            out, gate = model(y, e)
            p_m = float(10 * torch.log10(1 / (out - g).square().mean().clamp_min(1e-12)))
            p_b = float(10 * torch.log10(1 / (y - g).square().mean().clamp_min(1e-12)))
            a = by.setdefault(kind, {'model': [], 'base': [], 'gate': []})
            a['model'].append(p_m); a['base'].append(p_b); a['gate'].append(float(gate))
    res = {k: {'model': float(np.mean(v['model'])), 'base': float(np.mean(v['base'])),
               'gain': float(np.mean(v['model'])) - float(np.mean(v['base'])),
               'gate': float(np.mean(v['gate'])), 'n': len(v['model'])} for k, v in by.items()}
    am = [x for v in by.values() for x in v['model']]; ab = [x for v in by.values() for x in v['base']]
    res['ALL'] = {'model': float(np.mean(am)), 'base': float(np.mean(ab)),
                  'gain': float(np.mean(am)) - float(np.mean(ab)), 'n': len(am)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=8000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--train-clips', type=int, default=None)
    ap.add_argument('--tag', default='r7')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tr, va = load('train', a.train_clips), load('val', 40)
    model = RangeBranch7()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'params': sum(p.numel() for p in model.parameters()),
           'train': len(tr), 'val': len(va), 'iters': a.iters, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        y, g, ev = sample_batch(tr, rng, a.batch)
        out, gate = model(y, ev)
        # Two objectives on two scales, supervised separately. Six attempts to balance them inside one
        # loss all failed the same way: plain L1 gives a 58 dB observation ~3% of the gradient so the model
        # damages it (-14 dB), and every normalisation that fixed that removed the reason to act at all
        # (collapse to identity, +0.000 measured here at iter 4800). The gate does not have to infer
        # whether to act -- how much correction an observation needs is computable from the data, so it
        # gets a label; the residual then only has to learn how to correct, on the samples where it is open.
        base = (y - g).abs().mean(dim=(1, 2, 3))
        want = (base / 0.02).clamp(0, 1)             # 0 for a near-perfect observation, 1 for a bad one
        loss_gate = F.mse_loss(gate, want)
        w = want.detach()
        loss_res = ((out - g).abs().mean(dim=(1, 2, 3)) * w).sum() / w.sum().clamp_min(1e-6)
        loss = loss_res + 0.5 * loss_gate
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 10) == 0 or it == a.iters:
            model.eval(); r = evaluate(model, va); model.train()
            e = {'iter': it, 'loss': float(loss), 'sec': round(time.perf_counter() - t0, 1),
                 **{k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in r.items()}}
            rec['curve'].append(e); print(json.dumps(e), flush=True)
    torch.save({'state': model.state_dict(), 'cfg': {}}, OUT / f'{a.tag}.pt')
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
