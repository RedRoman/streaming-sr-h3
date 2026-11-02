"""Is a low-resolution latent a downscaled high-resolution latent, or a different object?

Everything about the latent-upscaler route rests on an assumption nobody checked: that E(LR) relates to
E(HR) the way LR relates to HR, so a network can learn the map between them. If instead the VAE's latent
encodes "the content of an image of this size", then E(LR) and E(HR) live in different spaces and upscaling
between them is ill-posed no matter how good the upscaler is -- which would make the 19.3 dB measured for
LBH a property of the problem, not of LBH.

Three ways to arrive at a 56x32 latent for the same ground truth, all decoded through the same VAE:

  direct        E(HR)                              the reference
  latent_pool   E(HR) -> avgpool /4 -> trilinear x4    round-trips entirely inside latent space
  lr_encode     HR -> pixels /4 -> E -> trilinear x4   the route's actual input path

  latent_pool vs direct  : what latent-space down/up-sampling alone costs
  lr_encode  vs latent_pool : what is lost because E(LR) is not a downscaled E(HR)

If latent_pool decodes well and lr_encode does not, the gap is the semantic mismatch, and no upscaler
trained on this VAE can close it from LR latents alone. If both are bad, latent-space resampling itself is
destructive. Either way the answer is about the representation rather than about any particular model.

A fourth arm, lr_encode_pixelup, upsamples in pixels before encoding -- the current pipeline's path -- as
the yardstick both latent routes have to beat.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from h3_comfy import H3, ROOT

D_ = ROOT / 'data/real_eval/const'
OUT = ROOT / 'results/latent_semantics'
H, W = 480, 832
LH, LW = 128, 224
UH, UW = LH * 4, LW * 4
PAD = (32, 32, 16, 16)      # l, r, t, b: 832->896, 480->512


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def crop(px):
    if px.shape[1] == H and px.shape[2] == W:
        return px
    return px[:, PAD[2]:PAD[2] + H, PAD[0]:PAD[0] + W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=6)
    ap.add_argument('--frames', type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    from revisit_metrics import lpips_dist

    h3 = H3(load_dit=False)
    clips = json.loads((D_ / 'manifest.json').read_text())['clips'][:a.clips]
    rows = []
    for name in clips:
        hr = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in range(a.frames)])
        t = hr.permute(0, 3, 1, 2)
        # pad HR to the x4 grid so every arm lands on the same latent size
        # 480x832 is not a multiple of 16, so it cannot be encoded, and resizing to 512x896 would change
        # the content scale (that bug put the VAE round trip at 15 dB instead of ~34). Replicate-pad instead:
        # LR and HR then differ by exactly x4 and the crop gives back the original pixels.
        hr_pad = F.pad(t, PAD, mode='replicate')
        lo = F.interpolate(hr_pad.double(), size=(LH, LW), mode='bicubic', align_corners=False,
                           antialias=True).clamp(0, 1).float()
        with torch.no_grad():
            z_hi = h3.encode(hr_pad.permute(0, 2, 3, 1))                       # [1,24,T,32,56]
            z_lo = h3.encode(lo.permute(0, 2, 3, 1))                           # [1,24,T,8,14]
            tgt = tuple(z_hi.shape[2:])
            # latent-space round trip: pool the HR latent down, push it back up
            z_pool = F.avg_pool3d(z_hi.float(), (1, 4, 4))
            z_pool_up = F.interpolate(z_pool, size=tgt, mode='trilinear', align_corners=False)
            z_lo_up = F.interpolate(z_lo.float(), size=tgt, mode='trilinear', align_corners=False)
            up_pix = F.interpolate(lo.double(), size=(UH, UW), mode='bicubic',
                                   align_corners=False).clamp(0, 1).float()
            z_pix = h3.encode(up_pix.permute(0, 2, 3, 1))
            dec = {'direct': h3.decode(z_hi),
                   'latent_pool': h3.decode(z_pool_up.to(z_hi.dtype)),
                   'lr_encode': h3.decode(z_lo_up.to(z_hi.dtype)),
                   'lr_encode_pixelup': h3.decode(z_pix)}
            # how far the two low-res latents are from the pooled HR latent, in latent space itself
            lat_gap = {'pool_vs_lo': float((z_pool - z_lo.float()).square().mean().sqrt()),
                       'pool_rms': float(z_pool.square().mean().sqrt()),
                       'lo_rms': float(z_lo.float().square().mean().sqrt())}
        r = {'clip': name, **{f'latent_{k}': v for k, v in lat_gap.items()}}
        for tag, px in dec.items():
            px = crop(px)
            r[tag + '_psnr'] = float(np.mean([psnr(px[i], hr[i]) for i in range(a.frames)]))
            r[tag + '_lpips'] = float(np.mean([lpips_dist(px[i], hr[i]) for i in range(0, a.frames, 8)]))
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    arms = ['direct', 'latent_pool', 'lr_encode', 'lr_encode_pixelup']
    agg = {t: {m: float(np.mean([r[f'{t}_{m}'] for r in rows])) for m in ('psnr', 'lpips')} for t in arms}
    agg_lat = {k: float(np.mean([r['latent_' + k] for r in rows])) for k in ('pool_vs_lo', 'pool_rms', 'lo_rms')}
    out = {'clips': len(rows), 'arms': agg, 'latent_gap': agg_lat}
    (OUT / 'semantics.json').write_text(json.dumps({**out, 'per_clip': rows}, indent=1))
    print()
    print('%-20s %8s %8s' % ('arm', 'PSNR', 'LPIPS'))
    for t in arms:
        print('%-20s %8.3f %8.4f' % (t, agg[t]['psnr'], agg[t]['lpips']))
    print()
    print('latent space: ||pool(E(HR)) - E(LR)|| rms %.4f  (pool rms %.4f, E(LR) rms %.4f)'
          % (agg_lat['pool_vs_lo'], agg_lat['pool_rms'], agg_lat['lo_rms']))
    print('cost of latent-space resampling alone : %.2f dB' % (agg['direct']['psnr'] - agg['latent_pool']['psnr']))
    print('additional cost of starting from E(LR): %.2f dB' % (agg['latent_pool']['psnr'] - agg['lr_encode']['psnr']))


if __name__ == '__main__':
    main()
