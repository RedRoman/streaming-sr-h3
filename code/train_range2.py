"""Range branch v2: stop optimising the low band for L2 and start optimising it for what the pipeline does
with it.

What v1 got wrong is not accuracy. Measured against D(GT) at LR resolution on the 19 realistic clips of
the real-footage set, R_phi(y) scores 30.57 dB where the raw observation scores 29.47 and InfVSR's implied
low band scores 27.73 -- we are 2.95 dB ahead of the judge. And we lose LPIPS there, 0.2309 against 0.2183.
cross_band.py explains it: substitute the truth's detail into both and ours still scores worse (0.0849 vs
0.0820), while the mid-frequency RMS of our low band is 0.01407 against the truth's 0.01959, a 28% deficit
(InfVSR -13%). An L1-optimal estimate of D(x) given y is a conditional median, and conditional medians are
smooth. The deficit then compounds, because the quota gate scales the generated detail by the guide's own
local std: a smoother guide admits less of the detail that would have covered for it.

So the objective gains two terms that a pointwise distance cannot express, both moment constraints rather
than distances, which is the point -- they ask the output to have the right local contrast even where the
model cannot know the phase:

  hf    L1 between high-pass(pred) and high-pass(target), re-weighting the band the pointwise term
        under-counts;
  std   L1 between the 8x8 local standard deviations, which is literally the statistic the gate reads.

Both are evaluated after the same bicubic x4 upsample the pipeline applies, not at LR resolution. The
deficit is defined at HR -- sigma 2 there is sigma 0.5 at LR, and the gate's 8x8 window is 2x2 at LR -- so
measuring at LR would target a different band than the one that was measured to be missing.

No pretrained network is involved. An lpips term is available behind --perc for comparison, but it is the
evaluation metric's own network and is reported as a diagnostic, never as the shipped configuration.

The degradation mixture is handled by selection rather than by loss shaping. 25% of the training clips are
clean Pillow-BICUBIC downsamples sitting at 58 dB, where the correct output is the input; v1 trained on the
mixture and lost 14.17 dB on them while gaining 1.98 on the realistic ones, and both the relative and the
hinge loss that were tried against that collapsed the model to the identity (gain 0.000). noise_sep.py then
measured that the regime is not estimable from a single frame at all -- Immerkaer's sigma separates the two
at AUC 0.490, chance -- so an unconditioned network cannot route and there is nothing to condition it on.
--kinds realistic therefore trains the estimator on the regime it is for; which estimator to run is a
deployment-time decision, and on the clean regime the identity is already within 0.3 dB of optimal.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/range_branch'


def gk(sigma, dev, dtype):
    r = max(1, int(round(3 * sigma)))
    a = torch.arange(-r, r + 1, device=dev, dtype=dtype)
    g = torch.exp(-0.5 * (a / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def blur(x, sigma):
    c = x.shape[1]
    g = gk(sigma, x.device, x.dtype).expand(c, 1, 1, -1)
    r = g.shape[-1] // 2
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g, groups=c)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2), groups=c)


def local_std(x, k=8):
    """sqrt is not differentiable at 0 and flat 8x8 windows land exactly there, which is where the first
    attempt went to NaN on step one; the floor is below any variance the gate can distinguish."""
    y = x.mean(1, keepdim=True)
    m = F.avg_pool2d(y, k, k)
    return (F.avg_pool2d(y * y, k, k) - m * m).clamp_min(1e-8).sqrt()


def mid_rms(x, sigma=2.0):
    return float((x - blur(x, sigma)).square().mean().sqrt())


def up4(x):
    h, w = x.shape[-2:]
    return F.interpolate(x, size=(h * 4, w * 4), mode='bicubic', align_corners=False)


class Data:
    """All windows in RAM as uint8; the arrays are 2.6 GB for the full split and the host has 1.5 TB."""

    def __init__(self, root, split, kinds, limit=None):
        fs = sorted((Path(root) / split).glob('*.npz'))
        if limit:
            fs = fs[:limit]
        self.y, self.t, self.kind = [], [], []
        for p in fs:
            d = np.load(p)
            k = json.loads(str(d['meta']))['degradation'].get('kind', '?')
            if kinds != 'all' and k != kinds:
                continue
            self.y.append(d['y']); self.t.append(d['target']); self.kind.append(k)
        assert self.y, f'no {kinds} windows in {root}/{split}'

    def batch(self, rng, n, crop, k=1):
        """Returns x [B,K,3,h,w] and the target for the newest frame, [B,3,h,w].

        K>1 was previously only reachable through train_range7.py, so passing --k 3 here built a 9-channel
        stem and then fed it one frame: "weight of size [128,9,3,3], expected input to have 9 channels,
        but got 3". The stack is causal -- frames t-K+1..t, clamped at the clip start by repeating the
        first frame, exactly as run_deploy and deploy_eval's rangesub do at inference -- so training and
        deployment see the same construction.
        """
        xb, yb = [], []
        for _ in range(n):
            i = rng.integers(len(self.y))
            T, H, W, _ = self.y[i].shape
            t = rng.integers(T)
            y0, x0 = rng.integers(0, H - crop + 1), rng.integers(0, W - crop + 1)
            idx = [max(0, t - j) for j in range(k - 1, -1, -1)]
            xb.append(self.y[i][idx, y0:y0 + crop, x0:x0 + crop])          # [K,h,w,3]
            yb.append(self.t[i][t, y0:y0 + crop, x0:x0 + crop])
        x = torch.from_numpy(np.stack(xb)).float().permute(0, 1, 4, 2, 3) / 255
        y = torch.from_numpy(np.stack(yb)).float().permute(0, 3, 1, 2) / 255
        return x, y


@torch.no_grad()
def evaluate(model, data, dev, stride=4, chunk=8, k=1):
    """Per-kind PSNR against D(x) and the identity's, plus the two statistics the objective targets.

    `k` must match the checkpoint: with K>1 the model consumes a causal stack, so scoring it one frame at
    a time would feed the wrong channel count. The stack is clamped at the clip start the same way
    batch() and the deployed estimator do.
    """
    acc = {}
    for yv, tv, kind in zip(data.y, data.t, data.kind):
        T = yv.shape[0]
        idx = list(range(0, T, stride))
        for s in range(0, len(idx), chunk):
            sel = idx[s:s + chunk]
            x = torch.from_numpy(yv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            g = torch.from_numpy(tv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            if k == 1:
                xin = x[:, None]
            else:
                st = np.stack([[max(0, t - j) for j in range(k - 1, -1, -1)] for t in sel])
                xin = torch.from_numpy(yv[st]).float().permute(0, 1, 4, 2, 3).to(dev) / 255
            o = model(xin)
            d = acc.setdefault(kind, {'m': [], 'b': [], 'mid_o': [], 'mid_g': [], 'mid_y': []})
            for i in range(len(sel)):
                d['m'].append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                d['b'].append(float(10 * torch.log10(1 / (x[i] - g[i]).square().mean().clamp_min(1e-12))))
            # the band the deficit was measured in: HR, sigma 2, after the pipeline's own x4 upsample
            d['mid_o'].append(mid_rms(up4(o))); d['mid_g'].append(mid_rms(up4(g)))
            d['mid_y'].append(mid_rms(up4(x)))
    out = {}
    for k, d in acc.items():
        out[k] = {'n': len(d['m']), 'psnr': float(np.mean(d['m'])), 'base': float(np.mean(d['b'])),
                  'gain_dB': float(np.mean(d['m']) - np.mean(d['b'])),
                  'mid_ratio': float(np.mean(d['mid_o']) / np.mean(d['mid_g'])),
                  'mid_ratio_y': float(np.mean(d['mid_y']) / np.mean(d['mid_g']))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--kinds', default='realistic', choices=['realistic', 'pillow_bicubic', 'all'])
    ap.add_argument('--k', type=int, default=1)
    ap.add_argument('--ch', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--crop', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--hf-w', type=float, default=0.0, help='L1 on the sigma=2 high pass, at HR')
    ap.add_argument('--std-w', type=float, default=0.0, help='L1 on the 8x8 local std at HR, the gate stat')
    ap.add_argument('--hf-sigma', type=float, default=2.0)
    ap.add_argument('--std-mode', default='l1', choices=['l1', 'deficit'],
                    help="deficit penalises only std(pred) < std(target). An L1 between the two maps is "
                         'still a distance, and a distance is minimised in part by being conservative; '
                         'the first pair of runs moved the gain to +2.1 dB and left mid_ratio at 0.81, '
                         'unchanged from the incumbent, which is what that predicts. The one-sided form '
                         'has no such fixed point.')
    ap.add_argument('--hf-mode', default='l1', choices=['l1', 'energy'],
                    help='energy matches the local RMS of the high pass instead of the high pass itself, '
                         'for the same reason')
    ap.add_argument('--perc', type=float, default=0.0, help='lpips-alex; diagnostic only, see docstring')
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--train-clips', type=int, default=None)
    ap.add_argument('--init', default=None, help='warm start from a checkpoint')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds the batch sampler AND torch init. The omni-minus-specialist deltas are '
                         '0.04-0.24 dB and no seed spread has ever been measured for either side, so '
                         'the signs of the per-operator deltas are uninterpretable until it is.')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        import sys
        sys.path.insert(0, str(ROOT / 'code'))
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
        # Hold the device before the dataset is read: see patches/claim_card_early.py
        _claim = torch.zeros(1 << 28, device=dev)
        del _claim
    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    from range_branch import RangeBranch

    tr = Data(a.data, 'train', a.kinds, a.train_clips)
    va = Data(a.data, 'val', 'all')
    model = RangeBranch(a.k, a.ch, a.blocks).to(dev)
    if a.init:
        st = torch.load(a.init, map_location='cpu')
        if st['cfg'] == {'k': a.k, 'ch': a.ch, 'blocks': a.blocks, 'desc': False}:
            model.load_state_dict(st['state'])
        else:
            print(json.dumps({'init_skipped': st['cfg']}), flush=True)
    perc = None
    if a.perc > 0:
        import lpips
        perc = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().to(dev)
        for p in perc.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rec = {'tag': a.tag, 'seed': a.seed, 'kinds': a.kinds, 'k': a.k, 'ch': a.ch, 'blocks': a.blocks,
           'params': sum(p.numel() for p in model.parameters()), 'iters': a.iters, 'batch': a.batch,
           'crop': a.crop, 'lr': a.lr, 'hf_w': a.hf_w, 'std_w': a.std_w, 'hf_sigma': a.hf_sigma,
           'std_mode': a.std_mode, 'hf_mode': a.hf_mode, 'perc': a.perc,
           'train_windows': len(tr.y), 'val_windows': len(va.y), 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)

    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, g = tr.batch(rng, a.batch, a.crop, a.k)
        x, g = x.to(dev), g.to(dev)
        pred = model(x)
        loss = F.l1_loss(pred, g)
        parts = {'l1': float(loss)}
        if a.hf_w or a.std_w:
            pu, gu = up4(pred), up4(g)
            if a.hf_w:
                hp, hg = pu - blur(pu, a.hf_sigma), gu - blur(gu, a.hf_sigma)
                if a.hf_mode == 'energy':
                    ep = F.avg_pool2d(hp.square().mean(1, keepdim=True), 8, 8).clamp_min(1e-8).sqrt()
                    eg = F.avg_pool2d(hg.square().mean(1, keepdim=True), 8, 8).clamp_min(1e-8).sqrt()
                    h = (eg - ep).clamp_min(0).mean()
                else:
                    h = F.l1_loss(hp, hg)
                loss = loss + a.hf_w * h; parts['hf'] = float(h)
            if a.std_w:
                sp, sg = local_std(pu), local_std(gu)
                st_ = (sg - sp).clamp_min(0).mean() if a.std_mode == 'deficit' else F.l1_loss(sp, sg)
                loss = loss + a.std_w * st_; parts['std'] = float(st_)
        if perc is not None:
            p = perc(pred * 2 - 1, g * 2 - 1).mean()
            loss = loss + a.perc * p; parts['perc'] = float(p)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % max(1, a.iters // 12) == 0 or it == a.iters:
            model.eval(); ev = evaluate(model, va, dev, k=a.k); model.train()
            e = {'iter': it, 'loss': float(loss), **parts, 'val': ev,
                 'sec': round(time.perf_counter() - t0, 1)}
            rec['curve'].append(e)
            print(json.dumps(e), flush=True)
            torch.save({'state': model.state_dict(),
                        'cfg': {'k': a.k, 'ch': a.ch, 'blocks': a.blocks, 'desc': False}},
                       OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    print(json.dumps({'tag': a.tag, 'final': rec['final']}), flush=True)


if __name__ == '__main__':
    main()
