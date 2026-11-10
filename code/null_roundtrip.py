"""Can the frozen H3 VAE decoder emit a pure-detail image?

The mapper line needs a latent z whose decode is the null component n = x - U(D x) of a frame, so that the
final output can be assembled as U(R_phi(y)) + decode(z). Before training any mapper, ask whether such a z
is even reachable: the encoder provides the best cheaply-findable candidate, so round-trip the target through
encode/decode and see how much of it survives.

Arms, all on GT frames of the held-out set:
  gt        control: round-trip x itself, then split. Measures how much null the VAE preserves when the
            detail rides on a natural image -- the regime the weights were trained for.
  n05       round-trip n + 0.5 (the only way to put a zero-mean signal in the [0,1] pixel range).
  n05x2     round-trip 2n + 0.5, i.e. use twice the dynamic range, then halve. Tests whether the loss in
            n05 is a contrast/quantisation effect rather than a manifold effect.
Reported per arm: energy ratio E|n_hat|^2 / E|n|^2, cosine <n_hat, n>, and PSNR of n_hat against n in the
same units as the frame (so it is comparable to the fidelity numbers everywhere else).
"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
from h3_comfy import H3

D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/heldout')
NCLIP = int(os.environ.get('NCLIP', 4))
T = int(os.environ.get('T', 22))


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x):
    """x [T,H,W,3] -> (range, null) with range = U(D x), x4 bicubic, the decomposition used everywhere."""
    t = x.permute(0, 3, 1, 2).double()
    d = F.interpolate(t, scale_factor=.25, mode='bicubic', align_corners=False, antialias=True)
    r = F.interpolate(d, size=t.shape[-2:], mode='bicubic', align_corners=False, antialias=True)
    r = r.permute(0, 2, 3, 1).float()
    return r, x - r


def stats(nh, n):
    a, b = nh.reshape(-1).double(), n.reshape(-1).double()
    return {'energy_ratio': float((a @ a) / (b @ b)),
            'cos': float((a @ b) / (a.norm() * b.norm() + 1e-12)),
            'psnr': float(-10 * torch.log10(((a - b) ** 2).mean() + 1e-12))}


def main():
    h3 = H3(load_dit=False)
    clips = sorted(p.name for p in D_.iterdir() if (p / 'hr').is_dir())[:NCLIP]
    acc = {}
    per = {}
    for c in clips:
        f = sorted((D_ / c / 'hr').glob('*.png'))[:T]
        x = torch.stack([load(p) for p in f])                     # [T,H,W,3]
        r, n = split(x)
        out = {}
        # control: the VAE's own fidelity on a natural image, and what it does to that image's null band
        xr = h3.decode(h3.encode(x)).clamp(0, 1)[:T]
        out['gt'] = stats(split(xr)[1], n)
        out['gt']['frame_psnr'] = float(-10 * torch.log10(((xr - x) ** 2).mean() + 1e-12))
        for tag, k in (('n05', 1.0), ('n05x2', 2.0)):
            p = (n * k + 0.5).clamp(0, 1)
            q = h3.decode(h3.encode(p)).float()[:T]
            out[tag] = stats((q - 0.5) / k, n)
            out[tag]['encode_clip_fraction'] = float(((n * k + 0.5) != p).float().mean())
        per[c] = out
        for kk, vv in out.items():
            acc.setdefault(kk, []).append(vv)
        print(c, json.dumps(out), flush=True)
    agg = {k: {m: float(np.mean([d[m] for d in v])) for m in v[0]} for k, v in acc.items()}
    res = {'clips': clips, 'frames_per_clip': T, 'per_clip': per, 'mean': agg}
    (ROOT / 'results/deploy/null_roundtrip.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(agg, indent=1))


main()
