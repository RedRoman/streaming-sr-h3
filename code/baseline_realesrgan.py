"""A published baseline on our own 12 evaluation clips: Real-ESRGAN x4 (RRDBNet, 23 blocks, 16.7M params).

Every number in this project so far has been self-referential -- our outputs against our own ablations, and
one comparison against a classical bicubic projection that we lost by 2.55 dB. Without an external floor we
do not know where we stand, so this runs a standard real-world SR baseline on the exact same LR frames, with
the exact same metrics, and reports it alongside the H3 pipeline.

Real-ESRGAN is the right floor for this setting: it is trained on a second-order blur/noise/compression
chain, which is what our `realistic` degradation is. It is a per-frame model, so it also serves as the
"no temporal modelling at all" reference.

Architecture is reimplemented here rather than pulled from basicsr, which is not installed; the state dict's
own key layout (702 tensors, body.0..22, conv_first 64ch) fixes every hyperparameter.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, '/cache/zihaohe/streaming_sr0/code')
from revisit_metrics import lpips_dist

ROOT = Path('/cache/zihaohe/streaming_sr0')
import os
D_ = ROOT / os.environ.get('EVAL_SET', 'data/stage2_eval')
W = ROOT / 'assets/baselines/RealESRGAN_x4.pth'
OUTD = ROOT / ('results/baseline_realesrgan_' + os.environ.get('EVAL_TAG', 'stage2'))
STRIDE = 8


class RDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        for i in range(1, 6):
            setattr(self, f'conv{i}', nn.Conv2d(nf + (i - 1) * gc, gc if i < 5 else nf, 3, 1, 1))

    def forward(self, x):
        t = [x]
        for i in range(1, 6):
            c = getattr(self, f'conv{i}')(torch.cat(t, 1))
            t.append(F.leaky_relu(c, 0.2, inplace=True) if i < 5 else c)
        return t[-1] * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.rdb1, self.rdb2, self.rdb3 = RDB(nf, gc), RDB(nf, gc), RDB(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, nf=64, nb=23, gc=32):
        super().__init__()
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, 3, 3, 1, 1)

    def forward(self, x):
        f = self.conv_first(x)
        f = f + self.conv_body(self.body(f))
        f = F.leaky_relu(self.conv_up1(F.interpolate(f, scale_factor=2, mode='nearest')), 0.2, True)
        f = F.leaky_relu(self.conv_up2(F.interpolate(f, scale_factor=2, mode='nearest')), 0.2, True)
        return self.conv_last(F.leaky_relu(self.conv_hr(f), 0.2, True))


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save-frames', action='store_true')
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    sd = torch.load(W, map_location='cpu', weights_only=True)
    sd = sd.get('params_ema', sd.get('params', sd))
    net = RRDBNet().eval()
    missing, unexpected = net.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (len(missing), len(unexpected), missing[:3], unexpected[:3])
    print(json.dumps({'params': sum(p.numel() for p in net.parameters()),
                      'loaded': 'strict'}), flush=True)

    man = json.loads((D_ / 'manifest.json').read_text())
    rows = []
    with torch.no_grad():
        for name in man['clips']:
            meta = json.loads((D_ / name / 'meta.json').read_text())
            T = meta['frames']
            kind = meta['degradation'].get('kind', '?')
            ps, lp = [], []
            # temporal metrics need consecutive pairs, so each sampled index is paired with the next
            idx = sorted({t for u in range(0, T - 1, STRIDE) for t in (u, u + 1)})
            for t in idx:
                lr = load(D_ / name / 'lr' / f'{t:06d}.png').permute(2, 0, 1)[None]
                g = load(D_ / name / 'hr' / f'{t:06d}.png')[None]
                out = net(lr).clamp(0, 1)
                y = out[0].permute(1, 2, 0)[None]
                assert y.shape == g.shape, (y.shape, g.shape)
                ps.append(psnr(y, g)); lp.append(lpips_dist(y[0], g[0]))
                if a.save_frames:
                    d = ROOT / 'results/baseline_frames' / name
                    d.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.uint8(y[0].numpy().clip(0, 1) * 255 + .5)).save(d / f'{t:06d}.png')
            r = {'clip': name, 'degradation': kind, 'psnr': float(np.mean(ps)), 'lpips': float(np.mean(lp)),
                 'frames': len(ps)}
            rows.append(r)
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    def agg(sel):
        s = [r for r in rows if sel(r)]
        return {'n': len(s), 'psnr': {'mean': float(np.mean([r['psnr'] for r in s])), 'ci95': ci([r['psnr'] for r in s])},
                'lpips': {'mean': float(np.mean([r['lpips'] for r in s])), 'ci95': ci([r['lpips'] for r in s])}} if s else None
    out = {'per_clip': rows, 'all': agg(lambda r: True),
           'realistic': agg(lambda r: r['degradation'] == 'realistic'),
           'bicubic': agg(lambda r: r['degradation'] != 'realistic')}
    (OUTD / 'metrics.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
