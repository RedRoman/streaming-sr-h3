"""The control I should have run first: with labels, can anything tell the two regimes apart?

Seven attempts have now failed to separate clean Pillow-BICUBIC clips from realistically degraded ones --
hand statistics (AUC 0.49), an 8-dim parameter regressor, a scalar severity regressor in four variants
(best AUC 0.736), and a contrastive representation whose InfoNCE loss fell from 2.6 to 0.278 while its
probe stayed at chance, meaning the encoder solved the pretext task through content and never needed the
degradation. Each failure is a statement about a method. None of them bounds the problem.

A supervised binary classifier does. It is handed the labels, so it has strictly more information than
any unsupervised or regression route, and whatever it reaches is an upper bound on all of them:

  AUC ~0.95   the signal is plainly there and every previous failure was optimisation, not information
  AUC ~0.75   weak signal; matches the severity regressor, and a representation route cannot do much more
  AUC ~0.55   the regimes really are near-indistinguishable from the observation, and blind conditioning
              on regime has to be abandoned rather than re-attempted

Deliberately generous: balanced sampling, clip-level aggregation over 16 patches, and a validation set
the model never sees. If this cannot separate them, nothing in this family will.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/range_branch'


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--iters', type=int, default=3000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--tag', default='kindclf')
    a = ap.parse_args()
    dev = torch.device(a.device)
    import torch_npu  # noqa: F401
    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    from sev_est import Est, Data

    tr, va = Data(a.data, 'train'), Data(a.data, 'val')
    ytr = np.array([k == 'realistic' for k in tr.kind], dtype=np.float32)
    yva = np.array([k == 'realistic' for k in va.kind], dtype=np.float32)
    print(json.dumps({'train': {'realistic': int(ytr.sum()), 'clean': int((1 - ytr).sum())},
                      'val': {'realistic': int(yva.sum()), 'clean': int((1 - yva).sum())}}), flush=True)
    by = {0: np.where(ytr == 0)[0], 1: np.where(ytr == 1)[0]}
    model = Est(frames=1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    curve = []

    def sample(n):
        xs, ys = [], []
        for q in range(n):
            c = q % 2
            i = by[c][rng.integers(len(by[c]))]
            T, H, W, _ = tr.y[i].shape
            t = rng.integers(T)
            y0, x0 = rng.integers(0, H - a.crop + 1), rng.integers(0, W - a.crop + 1)
            xs.append(tr.y[i][t, y0:y0 + a.crop, x0:x0 + a.crop]); ys.append(float(c))
        x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2)[:, None] / 255
        return x, torch.tensor(ys)[:, None]

    @torch.no_grad()
    def val(n_patch=16):
        model.eval(); r = np.random.default_rng(1); ps = []
        for yv in va.y:
            T, H, W, _ = yv.shape
            xs = []
            for _ in range(n_patch):
                t = r.integers(T)
                y0, x0 = r.integers(0, H - a.crop + 1), r.integers(0, W - a.crop + 1)
                xs.append(yv[t, y0:y0 + a.crop, x0:x0 + a.crop])
            x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2)[:, None].to(dev) / 255
            ps.append(float(model(x).sigmoid().mean()))
        model.train()
        p = np.array(ps)
        return auc(p[yva == 1], p[yva == 0]), float(p[yva == 1].mean()), float(p[yva == 0].mean())

    for it in range(1, a.iters + 1):
        x, y = sample(a.batch)
        loss = F.binary_cross_entropy_with_logits(model(x.to(dev)), y.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 6) == 0 or it == a.iters:
            A, pr, pc = val()
            curve.append({'iter': it, 'loss': float(loss), 'auc': A,
                          'p_realistic': pr, 'p_clean': pc, 'sec': round(time.perf_counter() - t0, 1)})
            print(json.dumps(curve[-1]), flush=True)
    best = max(c['auc'] for c in curve)
    (OUT / f'{a.tag}.json').write_text(json.dumps({'curve': curve, 'best_auc': best}, indent=1))
    print(json.dumps({'SUPERVISED_UPPER_BOUND_AUC': round(best, 3),
                      'prior_best_unsupervised': 0.736}, indent=1), flush=True)


if __name__ == '__main__':
    main()
