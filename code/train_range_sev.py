"""Gate A: can one range branch serve both degradation regimes, if it is told how degraded the input is?

The omni premise is one model over many degradations, and we have a measured counterexample: R_phi trained
on the mixture loses 14.17 dB on the clean Pillow-BICUBIC regime while gaining 1.98 on the realistic one,
and every loss-shaping fix for that (relative, hinge, one-sided deficit) collapsed it to the identity. The
shipped model sidesteps the problem by training on realistic only, which is avoidance, not a solution.

Conditioning is the solution the omni plan needs, but op1 and op_cnn settled that the generative
parameters are not recoverable: a CNN reading raw frames reaches 0.254 code MAE against 0.252 for the
per-dimension median, five of eight dimensions score exactly 0.0% over that baseline, and the noise flag
-- a 70/30 binary -- has rank correlation 0.003. Later stages of the chain erase the signature of earlier
ones, so the chain's own parameters are not in its output.

So the conditioning target changes from "what was done" to "how much is left to fix":

    s = clamp((PSNR_ceiling - PSNR(y, D(x))) / span, 0, 1)

which is a property of the observation and the target, not of the generative history, and is therefore
identifiable by construction. A small head predicts it; the range branch is FiLM-conditioned on the
prediction. Training is on the FULL mixture, which is the whole point.

Gate: gain_dB >= 0 on BOTH kinds. Beating the incumbent on realistic is not required here -- not
destroying the clean regime is.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/range_branch'
PSNR_CEIL, PSNR_SPAN = 60.0, 35.0      # clean sits at ~58 dB, realistic at ~29; s ~ 0.06 vs ~0.89


def severity_of(y, t):
    """s in [0,1] from the observation's own distance to the evaluation target."""
    d = (y.astype(np.float32) - t.astype(np.float32)) / 255.0
    psnr = 10 * np.log10(1.0 / max(float((d ** 2).mean()), 1e-12))
    return float(np.clip((PSNR_CEIL - psnr) / PSNR_SPAN, 0, 1)), float(psnr)


class SevHead(nn.Module):
    """Frame -> scalar severity. Pools mean and std: the statistic is global, not per pixel."""

    def __init__(self, ch=32):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.ReLU(inplace=True))
        self.o = nn.Linear(ch * 8, 1)

    def forward(self, x):
        h = self.f(x)
        return self.o(torch.cat([h.mean((-2, -1)), h.std((-2, -1))], 1)).sigmoid()


class FiLMBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.f = nn.Linear(1, 2 * c)
        nn.init.zeros_(self.f.weight); nn.init.zeros_(self.f.bias)   # starts as an unconditioned block

    def forward(self, x, s):
        g, b = self.f(s).chunk(2, dim=1)
        h = F.relu(self.a(x), inplace=True)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        return x + self.b(h)


class SevRange(nn.Module):
    def __init__(self, ch=96, blocks=12):
        super().__init__()
        self.head = nn.Conv2d(3, ch, 3, padding=1)
        self.body = nn.ModuleList([FiLMBlock(ch) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)
        self.sev = SevHead()

    def forward(self, y, s=None):
        s_hat = self.sev(y) if s is None else s
        h = self.head(y)
        for blk in self.body:
            h = blk(h, s_hat)
        return (y + self.tail(h)).clamp(0, 1), s_hat


class Data:
    def __init__(self, root, split, limit=None):
        fs = sorted((Path(root) / split).glob('*.npz'))[:limit]
        self.y, self.t, self.kind, self.s = [], [], [], []
        for p in fs:
            d = np.load(p)
            s, _ = severity_of(d['y'][0], d['target'][0])
            self.y.append(d['y']); self.t.append(d['target'])
            self.kind.append(json.loads(str(d['meta']))['degradation'].get('kind', '?'))
            self.s.append(s)
        assert self.y

    def batch(self, rng, n, crop, balance=False):
        """balance draws each degradation kind equally often.

        Gate A failed with the true severity fed in (pillow -13.5 dB), which rules out missing
        information as the cause: with oracle conditioning the model knows exactly which regime it is in
        and still destroys the clean one. What it does not have is gradient. A clean Pillow window sits at
        58 dB, so its L1 is about 3% of a realistic window's, and at 20% of the data it contributes ~0.6%
        of the batch gradient. Equal sampling fixes the weight without making the loss scale-invariant,
        which is what sank the earlier relative and hinge attempts (both collapsed to the identity)."""
        xb, yb, sb = [], [], []
        by_kind = {}
        if balance:
            for j, k in enumerate(self.kind):
                by_kind.setdefault(k, []).append(j)
            kinds = sorted(by_kind)
        for q in range(n):
            if balance:
                pool = by_kind[kinds[q % len(kinds)]]
                i = pool[rng.integers(len(pool))]
            else:
                i = rng.integers(len(self.y))
            T, H, W, _ = self.y[i].shape
            t = rng.integers(T)
            y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            xb.append(self.y[i][t, y0:y0 + crop, x0:x0 + crop])
            yb.append(self.t[i][t, y0:y0 + crop, x0:x0 + crop])
            sb.append(self.s[i])
        f = lambda a: torch.from_numpy(np.stack(a)).float().permute(0, 3, 1, 2) / 255
        return f(xb), f(yb), torch.tensor(sb, dtype=torch.float32)[:, None]


@torch.no_grad()
def evaluate(model, data, dev, stride=4, chunk=8):
    acc = {}
    for yv, tv, k, s in zip(data.y, data.t, data.kind, data.s):
        idx = list(range(0, yv.shape[0], stride))
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = torch.from_numpy(yv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            g = torch.from_numpy(tv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            o, sh = model(x)
            d = acc.setdefault(k, {'m': [], 'b': [], 'se': []})
            for i in range(len(sel)):
                d['m'].append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                d['b'].append(float(10 * torch.log10(1 / (x[i] - g[i]).square().mean().clamp_min(1e-12))))
            d['se'].append(float((sh.mean() - s).abs()))
    return {k: {'n': len(d['m']), 'psnr': float(np.mean(d['m'])), 'base': float(np.mean(d['b'])),
                'gain_dB': float(np.mean(d['m']) - np.mean(d['b'])),
                'sev_mae': float(np.mean(d['se']))} for k, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--ch', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--sev-w', type=float, default=1.0)
    ap.add_argument('--balance', action='store_true',
                    help='sample degradation kinds equally; see Data.batch')
    ap.add_argument('--oracle-sev', action='store_true',
                    help='feed the true severity instead of the predicted one: separates "is the '
                         'conditioning useful" from "is it estimable"')
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--train-clips', type=int, default=None)
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

    tr = Data(a.data, 'train', a.train_clips)
    va = Data(a.data, 'val')
    model = SevRange(a.ch, a.blocks).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'params': sum(p.numel() for p in model.parameters()),
           'oracle_sev': a.oracle_sev, 'balance': a.balance, 'sev_w': a.sev_w, 'iters': a.iters,
           'train_clips': len(tr.y), 'val_clips': len(va.y),
           'kinds_train': {k: tr.kind.count(k) for k in set(tr.kind)}, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, g, s = tr.batch(rng, a.batch, a.crop, balance=a.balance)
        x, g, s = x.to(dev), g.to(dev), s.to(dev)
        o, sh = model(x, s if a.oracle_sev else None)
        loss = F.l1_loss(o, g) + a.sev_w * F.l1_loss(sh, s)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 12) == 0 or it == a.iters:
            model.eval(); ev = evaluate(model, va, dev); model.train()
            e = {'iter': it, 'loss': float(loss), 'val': ev, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e); print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(),
                        'cfg': {'ch': a.ch, 'blocks': a.blocks}}, OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    g = rec['final']['val']
    ok = all(v['gain_dB'] >= 0 for v in g.values())
    print(json.dumps({'tag': a.tag, 'GATE_A_PASS': ok,
                      'per_kind': {k: round(v['gain_dB'], 3) for k, v in g.items()}}, indent=1), flush=True)


if __name__ == '__main__':
    main()
