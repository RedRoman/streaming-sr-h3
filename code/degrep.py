"""A DASR-style contrastive degradation representation, instead of regressing the degradation.

Six supervised-regression attempts failed to recover the operator: the 8-dim generative parameter vector
(hand features, then a CNN on raw frames) lands at the per-dimension median baseline, and a scalar
severity lands at AUC <= 0.736 with an optimistically-calibrated MAE of 0.247 where the gate needs 0.1.
I read that as the information being absent. The blind-SR literature says otherwise, and says it about
exactly this failure: DASR (CVPR 2021) opens by noting that explicit degradation estimation "can cause SR
failure due to large estimation errors", and drops estimation entirely in favour of a representation
trained by contrast. AirNet, PromptIR and DAIR all avoid naming the degradation too.

The reason to expect this to work where regression did not: regression demands a calibrated output (the
clean regime must emit 0.055), while contrast only demands that clean clips embed away from degraded
ones. That is strictly less information, and the regression network already reached AUC 0.736 as a
by-product, so some discriminative signal is present and simply was not what the loss optimised.

DASR's sampling assumption maps onto this project better than onto its own setting: the degradation
parameters here are drawn once per clip, so two patches from DIFFERENT FRAMES and different positions of
one clip share the operator exactly while sharing almost no content, and patches from other clips differ
in operator. No degradation label is used anywhere in training.

Two judgements, in order:
  AUC    does the embedding separate the clean regime from the realistic one -- the thing a scalar
         regressor could only do at 0.736
  probe  a ridge probe from the embedding to the true severity, FITTED ON TRAIN and scored on val, to
         see whether severity is recoverable from a representation that was never asked for it
The real test is downstream (feed it to FiLM in place of the scalar and see whether pillow returns to
0.000 dB), and that comes next if these two pass.
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


class Encoder(nn.Module):
    """Patch -> degradation embedding. Global mean+std pooling: the operator is a property of the whole
    patch, not of any location, and std is what carries noise and blocking."""

    def __init__(self, ch=32, dim=128, proj=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, ch, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch * 2, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(ch * 4, ch * 4, 3, 2, 1), nn.ReLU(inplace=True))
        self.emb = nn.Linear(ch * 8, dim)
        self.head = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, proj))

    def forward(self, x, project=True):
        h = self.f(x)
        e = self.emb(torch.cat([h.mean((-2, -1)), h.std((-2, -1))], 1))
        return F.normalize(self.head(e), dim=1) if project else e


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

    def pair_batch(self, rng, n_clips, crop):
        """n_clips clips, two patches each from different frames and different positions."""
        xs, ids = [], []
        pick = rng.choice(len(self.y), size=min(n_clips, len(self.y)), replace=False)
        for c, i in enumerate(pick):
            T, H, W, _ = self.y[i].shape
            ts = rng.choice(T, size=2, replace=(T < 2))
            for t in ts:
                y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
                xs.append(self.y[i][t, y0:y0 + crop, x0:x0 + crop]); ids.append(c)
        x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2) / 255
        return x, torch.tensor(ids)


def info_nce(z, ids, tau=0.07):
    """Positives are the other patch with the same clip id; everything else is a negative."""
    sim = z @ z.t() / tau
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e4)
    pos = (ids[:, None] == ids[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    return -(log_prob[pos].mean())


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


@torch.no_grad()
def embed_clips(model, data, dev, crop, n_patch=8, seed=0):
    rng = np.random.default_rng(seed)
    E = []
    for yv in data.y:
        T, H, W, _ = yv.shape
        xs = []
        for _ in range(n_patch):
            t = rng.integers(T)
            y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            xs.append(yv[t, y0:y0 + crop, x0:x0 + crop])
        x = torch.from_numpy(np.stack(xs)).float().permute(0, 3, 1, 2).to(dev) / 255
        E.append(model(x, project=False).mean(0).cpu().numpy())
    return np.stack(E)


def judge(Etr, dtr, Eva, dva, ridge=1.0):
    """Probe fitted on train, scored on val.

    The first version fitted the probe on the validation embeddings themselves. With a 128-dim embedding
    and 89 validation clips that system is underdetermined, so least squares interpolated it exactly and
    reported AUC 1.0 with MAE 3e-13 -- a property of the fit, not of the representation.
    """
    sv_, kv = np.array(dva.s), np.array(dva.kind)
    real, pill = kv == 'realistic', kv != 'realistic'
    # most discriminative single direction found without any label: first principal component of val
    Ec = Eva - Eva.mean(0)
    _, _, vt = np.linalg.svd(Ec, full_matrices=False)
    pc = Ec @ vt[0]
    a_pc = max(auc(pc[real], pc[pill]), auc(pc[pill], pc[real]))
    # ridge probe fitted on train only
    st = np.array(dtr.s)
    A = np.concatenate([Etr, np.ones((len(Etr), 1))], 1)
    d = A.shape[1]
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(d), A.T @ st)
    pred = np.concatenate([Eva, np.ones((len(Eva), 1))], 1) @ coef
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    return {'n_val': len(Eva), 'n_train': len(Etr), 'auc_pc1': a_pc,
            'auc_probe': max(auc(pred[real], pred[pill]), auc(pred[pill], pred[real])),
            'probe_mae': float(np.abs(pred - sv_).mean()),
            'probe_mae_pillow': float(np.abs(pred[pill] - sv_[pill]).mean()),
            'probe_rho_realistic': float(np.corrcoef(rk(pred[real]), rk(sv_[real]))[0, 1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--iters', type=int, default=4000)
    ap.add_argument('--clips-per-batch', type=int, default=16)
    ap.add_argument('--crop', type=int, default=112)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--tau', type=float, default=0.07)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--tag', default='degrep')
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
    model = Encoder().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    rng = np.random.default_rng(0)
    rec = {'tag': a.tag, 'iters': a.iters, 'clips_per_batch': a.clips_per_batch, 'crop': a.crop,
           'params': sum(p.numel() for p in model.parameters()),
           'note': 'no degradation label used in training', 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, ids = tr.pair_batch(rng, a.clips_per_batch, a.crop)
        loss = info_nce(model(x.to(dev)), ids.to(dev), a.tau)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 8) == 0 or it == a.iters:
            model.eval()
            j = judge(embed_clips(model, tr, dev, a.crop), tr,
                      embed_clips(model, va, dev, a.crop), va)
            model.train()
            e = {'iter': it, 'loss': float(loss), 'val': j, 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e); print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(), 'cfg': {}}, OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    j = rec['final']['val']
    print(json.dumps({'tag': a.tag, 'auc_pc1': round(j['auc_pc1'], 3),
                      'auc_probe': round(j['auc_probe'], 3),
                      'beats_regression_0.736': j['auc_probe'] > 0.736}, indent=1), flush=True)


if __name__ == '__main__':
    main()
