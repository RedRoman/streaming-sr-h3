"""Vid4 as the first public benchmark, with the BD arm as the one that can show anything.

Vid4 ships three parallel sets: GT, BIx4 (antialiased bicubic) and BDx4 (Gaussian blur then direct
subsampling). The distinction decides what each arm can demonstrate:

  BIx4   y = D(x) for the very operator the range/null split is defined by, so R_phi's optimal output is
         the identity and the projection target makes no difference by construction. This arm can only
         show that the method does not break on clean input -- worth reporting, worth stating plainly,
         but it is not evidence for the contribution.
  BDx4   blur-then-subsample is a different operator, so y != D(x): the observation is not the clean
         low-resolution image, which is exactly the condition the method exists for. This is the arm
         where a gain is expected, and it is a published degradation rather than our own chain -- which
         answers the main objection to an evaluation built entirely on a self-made degradation.

Vid4 is 171 frames over four clips at 180x144 (BD/BI) against 720x576 GT. Our pipeline is built for
480x832 blocks, so each clip is padded to the block geometry and cropped back; the padding is recorded
and the metrics are computed on the original extent only.

Scored with the same metric code as the held-out table (revisit_metrics.lpips_dist, the same PSNR, the
same range/null decomposition), so the numbers sit in the same units as everything else in the paper.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
V = Path('/cache/zihaohe/benchmarks/vid4/Vid4')
CLIPS = ['calendar', 'city', 'foliage', 'walk']


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=4):
    """(range, null) under the x4 antialiased bicubic operator -- the same split as everywhere else."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u.permute(0, 2, 3, 1).float()
    return r, x - r


def psnr(a, b):
    return float(10 * torch.log10(1 / (a - b).square().mean().clamp_min(1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--deg', default='BDx4', choices=['BDx4', 'BIx4'])
    ap.add_argument('--rphi', default='results/range_branch/r6_xl_k3.pt')
    ap.add_argument('--stride', type=int, default=4)
    ap.add_argument('--out', default=None)
    a = ap.parse_args()

    import torch_npu  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    from revisit_metrics import lpips_dist
    from run_deploy import range_estimate

    rows = []
    for clip in CLIPS:
        lo = sorted((V / a.deg / clip).glob('*.png'))
        hi = sorted((V / 'GT' / clip).glob('*.png'))
        assert len(lo) == len(hi) and lo, (clip, len(lo), len(hi))
        ts = list(range(0, len(lo), a.stride))
        y = torch.stack([load(p) for p in (lo[t] for t in ts)])
        gt = torch.stack([load(p) for p in (hi[t] for t in ts)])
        # GT is 4x the LR extent; crop both to a multiple of 4 so the decomposition is exact
        H, W = (gt.shape[1] // 4) * 4, (gt.shape[2] // 4) * 4
        gt = gt[:, :H, :W]
        y = y[:, :H // 4, :W // 4]

        est = range_estimate(y.permute(0, 3, 1, 2), ROOT / a.rphi,
                             dev=torch.device('npu', 0)).permute(0, 2, 3, 1)
        up = lambda z: F.interpolate(z.permute(0, 3, 1, 2), size=(H, W), mode='bicubic',
                                     align_corners=False).clamp(0, 1).permute(0, 2, 3, 1)
        bic, rph = up(y), up(est)
        gr, _ = split(gt)

        # What R_phi is asked for here is D(x) at low resolution. Measuring it directly separates the
        # estimator's own quality from whatever the sampler later does with it, which is the quantity the
        # whole scaling study moved. The full pipeline comparison needs a sampling run and is separate.
        dx = F.interpolate(gt.permute(0, 3, 1, 2).double(), size=(H // 4, W // 4), mode='bicubic',
                           align_corners=False, antialias=True).float().permute(0, 2, 3, 1)
        rows.append({
            'clip': clip, 'frames': len(ts), 'hw': [H, W],
            'psnr_y_vs_Dx': psnr(y, dx),
            'psnr_rphi_vs_Dx': psnr(est, dx),
            'gain_dB': psnr(est, dx) - psnr(y, dx),
            'psnr_bicubic_vs_gt': psnr(bic, gt),
            # no range-substitution row here: that arm needs a restorer's null band, which only a
            # sampling run produces. U(R_phi(y)) alone is reported instead, as the upsampled estimate.
            'psnr_rphi_up_vs_gt': psnr(rph, gt),
            # lpips_dist takes one HxWx3 frame, not a batch: passing [T,H,W,3] reached a permute
            # built for 3 dims and failed inside the metric rather than here.
            'lpips_bicubic': float(np.mean([lpips_dist(bic[i], gt[i]) for i in range(len(ts))])),
            'lpips_rphi_up': float(np.mean([lpips_dist(rph[i], gt[i]) for i in range(len(ts))])),
        })
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v)
                          for k, v in rows[-1].items()}), flush=True)

    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if isinstance(rows[0][k], float)}
    out = {'benchmark': 'Vid4', 'degradation': a.deg, 'rphi': a.rphi, 'stride': a.stride,
           'note': ('BIx4 is the operator the range/null split is defined by, so y = D(x) and the '
                    'range branch has nothing to recover; BDx4 is a different operator, so y != D(x) '
                    'and a gain is expected'),
           'per_clip': rows, 'mean': agg}
    p = ROOT / (a.out or f'results/vid4_{a.deg}.json')
    p.write_text(json.dumps(out, indent=1))
    print()
    print('%-22s %10s' % ('metric (mean over 4 clips)', a.deg))
    for k in ['psnr_y_vs_Dx', 'psnr_rphi_vs_Dx', 'gain_dB', 'psnr_bicubic_vs_gt',
              'psnr_rphi_up_vs_gt', 'lpips_bicubic', 'lpips_rphi_up']:
        print('%-22s %10.4f' % (k, agg[k]))
    print()
    print('wrote', p)


if __name__ == '__main__':
    main()
