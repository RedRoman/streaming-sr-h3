"""Omni stage 1: one range branch over several operators, told which operator it is on.

Two gates were cleared to get here. Conditioning works when the conditioning signal is known -- with the
true severity a single branch reaches +2.71 dB on realistic degradation and 0.000 dB on the clean regime,
against -14.17 dB for the unconditioned model. And the branch is not x4-specific: retrained under x2 and
x8 it gains +2.27 and +1.93 dB, against +2.53 at x4.

Blind conditioning is not attempted, and that is a measured decision rather than a shortcut. Eight
independent attempts to recover the degradation from the observation -- three hand statistics, two
parameter regressors, four severity regressors, three contrastive encoders -- top out at AUC 0.736, and a
SUPERVISED binary classifier handed the labels reaches only 0.61-0.68, which bounds every unsupervised
route above. So the operator is supplied, as it is in the all-in-one literature before PromptIR.

What varies here that did not before: the operator itself. The branch sees x2, x4 and x8 data in one
training run and is told the scale through the same FiLM path that carried severity. Two conditioning
signals are compared because they answer different questions:

  scale     the operator identity, a property of the deployment, always known
  scale+sev the operator plus how degraded this particular clip is; severity is NOT known blind, so this
            arm is an oracle that upper-bounds what a future estimator could buy

Gate: per-operator gain must be non-negative everywhere, and the multi-operator model must not lose
materially to the three single-operator models it replaces (+2.27 / +2.53 / +1.93 dB).
"""
import os
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from operator_desc import DIM as MTF_DIM, descriptor as _mtf_descriptor

ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/omni'
# Per-operator pair directories. OMNI_DATA_SUFFIX switches the whole family to the regenerated
# sets: the original ones hold ~600 realistic windows per operator, and the x4 line has since measured
# that this is the binding constraint -- 8.7x the data moved the estimate +0.457 dB and the deployed
# LPIPS -0.0178, while capacity only started paying once the data was there. The omni deficits on
# deblur (-0.19 dB) and denoise (-0.26 dB) were therefore measured under the same starvation.
_SUF = os.environ.get('OMNI_DATA_SUFFIX', '')
DATA = {2: f'/cache/zihaohe/range_pairs_s2{_SUF}',
        4: '/temp/zihaohe_range_pairs' if not _SUF else f'/cache/zihaohe/range_pairs_x4{_SUF}',
        8: f'/cache/zihaohe/range_pairs_s8{_SUF}',
        'denoise': f'/cache/zihaohe/range_pairs_denoise{_SUF}',
        'deblur': f'/cache/zihaohe/range_pairs_deblur{_SUF}',
        # Evaluation-only operators. x6 is never trained on by any arm: it exists to separate the two
        # conditioning signals, because a one-hot over OPS has no slot for it at all while an MTF simply
        # reads a knee at 0.167 of Nyquist and interpolates between the x4 and x8 the model did see.
        6: f'/cache/zihaohe/range_pairs_s6{_SUF}',
        3: f'/cache/zihaohe/range_pairs_s3{_SUF}'}
# Operator identity as a one-hot. The first three are one operator at three strengths -- D is always a
# bicubic downsample -- so transfer between them could be explained by a single shared structure. The
# last two change what the null space IS: the band a Gaussian kernel attenuates, and nothing at all.
OPS = [2, 4, 8, 'deblur', 'denoise']
# Conditioning width per signal. onehot is |OPS| slots plus severity; mtf is operator_desc.DIM, a radial
# MTF of U(D(.)) plus the same severity slot, and is independent of OPS -- which is the whole difference.
COND_DIM = {'onehot': len(OPS) + 1, 'mtf': MTF_DIM}
_MTF_CACHE = {}


def mtf_cond(op, sev, use_sev):
    """The operator measured rather than named, plus severity when the arm may see it.

    Cached per operator: the probe is a 256x256 rfft2 and the sampler asks for it once per sample, which
    is 30 times an iteration for 12000 iterations, while the answer depends only on the operator.
    """
    if op not in _MTF_CACHE:
        _MTF_CACHE[op] = [float(x) for x in _mtf_descriptor(op)[:-1]]
    return _MTF_CACHE[op] + [float(sev) if use_sev else 0.0]
SINGLE = {2: 2.269, 4: 2.526, 8: 1.925}      # superseded by omni_compare.py, kept for the log header
PSNR_CEIL, PSNR_SPAN = 60.0, 35.0


def severity_of(y, t):
    d = (y.astype(np.float32) - t.astype(np.float32)) / 255.0
    return float(np.clip((PSNR_CEIL - 10 * np.log10(1.0 / max(float((d ** 2).mean()), 1e-12)))
                         / PSNR_SPAN, 0, 1))


class FiLMBlock(nn.Module):
    def __init__(self, c, cdim):
        super().__init__()
        self.a = nn.Conv2d(c, c, 3, padding=1)
        self.b = nn.Conv2d(c, c, 3, padding=1)
        self.f = nn.Linear(cdim, 2 * c)
        nn.init.zeros_(self.f.weight); nn.init.zeros_(self.f.bias)

    def forward(self, x, c):
        g, b = self.f(c).chunk(2, dim=1)
        h = F.relu(self.a(x), inplace=True)
        h = h * (1 + g[..., None, None]) + b[..., None, None]
        return x + self.b(h)


class OmniRange(nn.Module):
    """One range branch over several operators, FiLM-conditioned and optionally causal over K frames.

    K>1 stacks frames t-K+1..t on the channel axis and predicts a residual on the NEWEST frame, clamping
    at the clip start by repeating the first frame. That is the construction train_range2.py, run_deploy's
    range_estimate and deploy_eval's rangesub all use, so a checkpoint from here loads in the deployed
    pipeline without a second code path -- the x4 line already measured +0.31 dB from K=3 over K=1, and
    the omni arm had no way to collect it.
    """

    def __init__(self, ch=96, blocks=12, cdim=len(OPS) + 1, k=1):
        super().__init__()
        self.k = k
        self.head = nn.Conv2d(3 * k, ch, 3, padding=1)
        self.body = nn.ModuleList([FiLMBlock(ch, cdim) for _ in range(blocks)])
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight); nn.init.zeros_(self.tail.bias)

    def forward(self, y, cond):
        if y.dim() == 4:                                  # [B,3,H,W] from a K=1 caller
            y = y[:, None]
        b, kk, c, h_, w_ = y.shape
        if kk != self.k:
            raise RuntimeError(
                'this model was built for k=%d but was handed a stack of %d frames; the stem is a '
                '%d-channel conv and would fail with a bare channel-count error' % (self.k, kk, 3 * self.k))
        h = self.head(y.reshape(b, kk * c, h_, w_))
        for blk in self.body:
            h = blk(h, cond)
        return (y[:, -1] + self.tail(h)).clamp(0, 1)


class Multi:
    """All scales in one object; the operator is part of the sample, not of the dataset."""

    def __init__(self, split, scales, limit=None, mode='onehot', k=1, persample=False):
        self.mode, self.k, self.persample = mode, k, persample
        self.item = []
        for s in scales:
            kept = 0
            for p in sorted((Path(DATA[s]) / split).glob('*.npz')):
                if limit is not None and kept >= limit:
                    break
                d = np.load(p)
                if json.loads(str(d['meta']))['degradation'].get('kind') != 'realistic':
                    continue
                # blur_sigma is kept because the operator of a deblur sample is the KERNEL, not the
                # word "deblur". Dropping it is what made mtf_cond fall back to operator_desc's default
                # sigma=1.75 for every deblur pair, collapsing a 17-dim measurement of U(D(.)) into one
                # constant vector -- a higher-dimensional one-hot carrying a prior that is wrong for all
                # but one kernel. ||MTF(1.0) - MTF(2.5)|| is 0.8851, so the discarded signal was large.
                self.item.append({'scale': s, 'y': d['y'], 't': d['target'],
                                  'blur_sigma': json.loads(str(d['meta']))['degradation'].get('blur_sigma'),
                                  'sev': severity_of(d['y'][0], d['target'][0])})
                kept += 1
        assert self.item
        self.by_scale = {}
        for i, it in enumerate(self.item):
            self.by_scale.setdefault(it['scale'], []).append(i)

    def cond(self, scale, sev, use_sev, blur_sigma=None):
        """The conditioning vector for one sample, in whichever signal this arm was built with.

        persample=True resolves a deblur sample to its own kernel ('deblur:1.234') instead of the bare
        'deblur'. It is opt-in so every number published before stays reproducible bit for bit, and so
        the two settings are a controlled pair rather than a replacement.
        """
        if self.mode == 'mtf':
            if self.persample and blur_sigma is not None and scale == 'deblur':
                return mtf_cond('deblur:%.4f' % float(blur_sigma), sev, use_sev)
            return mtf_cond(scale, sev, use_sev)
        if scale not in OPS:
            raise SystemExit(
                'one-hot conditioning has no slot for operator %r (OPS is %s). This is not a bug to code '
                'around: it is the limitation the mtf arm exists to remove, and an operator outside the '
                'vocabulary is unrepresentable rather than merely untrained. Use --cond mtf to evaluate '
                'one.' % (scale, OPS))
        v = [0.0] * (len(OPS) + 1)
        v[OPS.index(scale)] = 1.0
        v[-1] = sev if use_sev else 0.0
        return v

    def batch(self, rng, n, crop, use_sev):
        scales = sorted(self.by_scale, key=lambda s: OPS.index(s) if s in OPS else len(OPS))
        xb, yb, cb = [], [], []
        for q in range(n):
            s = scales[q % len(scales)]            # equal operators per batch
            i = self.by_scale[s][rng.integers(len(self.by_scale[s]))]
            it = self.item[i]
            T, H, W, _ = it['y'].shape
            t = rng.integers(T)
            # Every operator in the batch must yield the same crop, or np.stack fails with a bare
            # "all input arrays must have the same shape". The pair resolution differs per operator --
            # at 480x832 HR the x2 pairs are 240x416, x4/deblur/denoise are 120x208 and x8 is 60x104 --
            # so a crop larger than the smallest operator's LR silently became per-operator before, and
            # now raises something that names the cause.
            if crop > min(H, W):
                raise SystemExit(
                    'crop %d exceeds operator %r pair size %dx%d; either lower --crop or drop that '
                    'operator from --scales (a per-operator crop would change the gradient scale '
                    'per operator and break the comparison)' % (crop, s, H, W))
            c = crop
            y0, x0 = rng.integers(0, H - c + 1), rng.integers(0, W - c + 1)
            idx = [max(0, t - j) for j in range(self.k - 1, -1, -1)]
            xb.append(it['y'][idx, y0:y0 + c, x0:x0 + c])                 # [K,c,c,3]
            yb.append(it['t'][t, y0:y0 + c, x0:x0 + c])
            cb.append(self.cond(s, it['sev'], use_sev, it.get('blur_sigma')))
        fx = lambda a: torch.from_numpy(np.stack(a)).float().permute(0, 1, 4, 2, 3) / 255
        fy = lambda a: torch.from_numpy(np.stack(a)).float().permute(0, 3, 1, 2) / 255
        return fx(xb), fy(yb), torch.tensor(cb, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, data, dev, use_sev, stride=8, chunk=4):
    acc = {}
    for it in data.item:
        idx = list(range(0, it['y'].shape[0], stride))
        c = torch.tensor([data.cond(it['scale'], it['sev'], use_sev, it.get('blur_sigma'))],
                         dtype=torch.float32, device=dev)
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            st = np.stack([[max(0, t - j) for j in range(data.k - 1, -1, -1)] for t in sel])
            x = torch.from_numpy(it['y'][st]).float().permute(0, 1, 4, 2, 3).to(dev) / 255
            g = torch.from_numpy(it['t'][sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            o = model(x, c.expand(len(sel), -1))
            d = acc.setdefault(it['scale'], {'m': [], 'b': []})
            for i in range(len(sel)):
                d['m'].append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                # the identity's score is the NEWEST frame, not the oldest in the stack
                d['b'].append(float(10 * torch.log10(
                    1 / (x[i, -1] - g[i]).square().mean().clamp_min(1e-12))))
    return {s: {'n': len(d['m']), 'base': float(np.mean(d['b'])), 'psnr': float(np.mean(d['m'])),
                'gain_dB': float(np.mean(d['m']) - np.mean(d['b'])),
                'vs_single': float(np.mean(d['m']) - np.mean(d['b']) - SINGLE[s]) if s in SINGLE
                else None}
            for s, d in sorted(acc.items(),
                               key=lambda kv: OPS.index(kv[0]) if kv[0] in OPS else len(OPS))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='2,4,8',
                    help='comma list over %s' % (OPS,))
    ap.add_argument('--use-sev', action='store_true', help='oracle arm: also condition on severity')
    ap.add_argument('--train-limit', type=int, default=0,
                    help='cap the TRAINING windows per operator (0 = every file in the set). The arms '
                         'that produced the published mtf-vs-onehot deltas read ~590 windows per '
                         'operator, and this code already records that data, not capacity, is the '
                         'binding constraint on this family: 8.7x the data moved the x4 estimate '
                         '+0.457 dB. A 17-dim conditioning vector is more data-hungry than a 6-dim one, '
                         'so a null result at 590 windows cannot be read as a null result in general. '
                         'This makes the data axis sweepable instead of fixed.')
    ap.add_argument('--mtf-persample', action='store_true',
                    help="condition a deblur sample on its OWN kernel ('deblur:1.234') rather than the "
                         'bare operator name. Without it mtf_cond resolves every deblur pair to '
                         "operator_desc's default sigma=1.75, so the MTF arm sees one constant vector "
                         'across a kernel range of sigma 1.0-2.5 and cannot be better than a one-hot '
                         'there by construction. Off by default so earlier runs stay reproducible.')
    ap.add_argument('--cond', default='onehot', choices=['onehot', 'mtf'],
                    help='onehot names the operator, mtf measures it. Both are known at deployment time '
                         '-- D is given in the non-blind setting -- so neither is an oracle, and they '
                         'cost the same at inference. They differ in what they can express: a one-hot is '
                         'a vocabulary and an operator outside it has no representation, while the MTF '
                         'is a continuous function of D and lands x6 between the x4 and x8 the model '
                         'trained on. --holdout-scales 6 is the experiment that separates them.')
    ap.add_argument('--k', type=int, default=1,
                    help='causal frame stack, frames t-K+1..t. The x4 line measured +0.31 dB from K=3; '
                         'the omni branch was single-frame only and could not collect it.')
    ap.add_argument('--holdout-scales', default='',
                    help='operators EVALUATED but never trained on. With --cond onehot this is expected '
                         'to fail loudly rather than score badly, which is the finding.')
    ap.add_argument('--log-every', type=int, default=0,
                    help='0 = iters//12. Card contention has killed three runs at 60-70%% of their '
                         'iterations with the last checkpoint 2000 iterations behind; a small value here '
                         'bounds that loss to minutes.')
    ap.add_argument('--iters', type=int, default=12000)
    ap.add_argument('--batch', type=int, default=30)
    ap.add_argument('--crop', type=int, default=48)
    ap.add_argument('--ch', type=int, default=96)
    ap.add_argument('--blocks', type=int, default=12)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--val-limit', type=int, default=24,
                    help='validation clips per operator. Full-resolution evaluation over all 213 val '
                         'items did not finish in 600 s on CPU, and every NPU on every host is taken; '
                         '24 per operator keeps the estimate usable while the run fits overnight.')
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
    parse = lambda s: [int(x) if x.isdigit() else x for x in s.split(',') if x]
    scales, hold = parse(a.scales), parse(a.holdout_scales)
    if set(scales) & set(hold):
        raise SystemExit('operators %s are in both --scales and --holdout-scales; a held-out operator '
                         'that was trained on measures nothing' % sorted(set(scales) & set(hold)))
    cdim = COND_DIM[a.cond]
    tr = Multi('train', scales, limit=(a.train_limit or None), mode=a.cond, k=a.k,
               persample=a.mtf_persample)
    va = Multi('val', scales, limit=a.val_limit, mode=a.cond, k=a.k, persample=a.mtf_persample)
    ho = (Multi('val', hold, limit=a.val_limit, mode=a.cond, k=a.k, persample=a.mtf_persample)
          if hold else None)
    model = OmniRange(a.ch, a.blocks, cdim=cdim, k=a.k).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    rec = {'tag': a.tag, 'seed': a.seed, 'scales': [str(s) for s in scales], 'use_sev': a.use_sev,
           'crop': a.crop, 'cond': a.cond, 'cdim': cdim, 'k': a.k,
           'holdout_scales': [str(s) for s in hold],
           'params': sum(p.numel() for p in model.parameters()),
           'train_items': {str(s): len(v) for s, v in tr.by_scale.items()},
           'val_items': {str(s): len(v) for s, v in va.by_scale.items()},
           'holdout_items': ({str(s): len(v) for s, v in ho.by_scale.items()} if ho else {}),
           'single_scale_reference': SINGLE, 'curve': []}
    print(json.dumps({k: v for k, v in rec.items() if k != 'curve'}), flush=True)
    every = a.log_every or max(1, a.iters // 12)
    t0 = time.perf_counter()
    for it in range(1, a.iters + 1):
        x, g, c = tr.batch(rng, a.batch, a.crop, a.use_sev)
        loss = F.l1_loss(model(x.to(dev), c.to(dev)), g.to(dev))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % every == 0 or it == a.iters:
            model.eval()
            ev = evaluate(model, va, dev, a.use_sev)
            hv = evaluate(model, ho, dev, a.use_sev) if ho else None
            model.train()
            e = {'iter': it, 'loss': float(loss), 'val': ev, 'sec': round(time.perf_counter() - t0, 1)}
            if hv:
                e['holdout'] = hv
            rec['curve'].append(e); print(json.dumps(e), flush=True)
            # cfg carries the conditioning signal and K, because run_deploy rebuilds the module from it
            torch.save({'state': model.state_dict(),
                        'cfg': {'ch': a.ch, 'blocks': a.blocks, 'k': a.k, 'cond': a.cond, 'cdim': cdim,
                                'omni': True, 'ops': [str(o) for o in OPS]}}, OUT / f'{a.tag}.pt')
    rec['final'] = rec['curve'][-1]
    (OUT / f'{a.tag}.json').write_text(json.dumps(rec, indent=1))
    v = rec['final']['val']
    out = {'tag': a.tag, 'cond': a.cond, 'k': a.k,
           'gain': {str(s): round(x['gain_dB'], 3) for s, x in v.items()},
           'vs_single': {str(s): (round(x['vs_single'], 3) if x['vs_single'] is not None else None)
                         for s, x in v.items()},
           'ALL_NONNEGATIVE': all(x['gain_dB'] >= 0 for x in v.values())}
    if rec['final'].get('holdout'):
        out['holdout_gain'] = {str(s): round(x['gain_dB'], 3)
                               for s, x in rec['final']['holdout'].items()}
        out['HOLDOUT_NONNEGATIVE'] = all(x['gain_dB'] >= 0 for x in rec['final']['holdout'].values())
    print(json.dumps(out, indent=1), flush=True)


if __name__ == '__main__':
    main()
