"""Does the latent-upscaler route have a ceiling worth chasing? Measured before building anything.

The route runs the DiT on the native-resolution latent (784 tokens, 0.81 s/step against 2.11) and recovers
the output resolution with a latent upscaler. Its cost is known; its quality is not, and the quality does
not depend on the DiT at all -- it is bounded by what the upscaler can reconstruct. Sol-H3 lost exactly
here: a frozen UltraFlash upsampler fed a perfect input still decoded at 17.5 dB, worse than bicubic.

So the DiT is removed from the question. The upscaler is handed the best input it could ever receive --
the ground truth itself, encoded at the low resolution the DiT would work at -- and whatever comes out is
the hard ceiling of the whole route:

    arm            what it measures
    ceiling_lbh    GT -> 224x128 -> VAE_enc -> LBH x4 -> VAE_dec       the route's ceiling
    ceiling_tri    same, trilinear instead of LBH                       no-learning floor
    vae_roundtrip  GT -> VAE_enc (full res) -> VAE_dec                  VAE's own ceiling, route-independent
    bicubic_lr     GT -> 224x128 -> bicubic x4 in pixels                what you get without any latent model

PRE-REGISTERED READOUT, fixed before the numbers exist:
  * ceiling_lbh PSNR below 26.8 dB (the deployed pipeline's own PSNR) -> the upscaler is a hard bottleneck
    and the route is dead, because no DiT can push past its own post-processing ceiling.
  * ceiling_lbh above 30 dB -> worth building.
  * in between -> decide on LPIPS, reported alongside.
`vae_roundtrip` is the sanity rail: if the ceiling sits close to it, the loss is the VAE's, not the
upscaler's, and the verdict would be about resolution rather than about LBH.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from h3_comfy import H3, ROOT

sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / 'data/real_eval/const'
OUT = ROOT / 'results/latent_sr_ceiling'
LBH = Path('/temp/h3_training_assets_20260921/H3-LBH-Latent-Upscaler/minimax_h3_latent_upscaler_3d_bf16.safetensors')
H, W = 480, 832
LH, LW = 128, 224                 # padded native resolution; /16 -> 8x14 latent, both even for the 2x2 patch
UH, UW = LH * 4, LW * 4
PAD = (32, 32, 16, 16)      # l, r, t, b: 832->896, 480->512           # 512x896 after x4, crops back to 480x832


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=8)
    ap.add_argument('--frames', type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import latent_upscaler as LU
    from latent_upscaler import LatentResizer3D
    from safetensors.torch import load_file
    from revisit_metrics import lpips_dist

    # latent_upscaler.py transcribes TemporalConv from the published config as norm -> dwconv -> pwconv, but
    # the released node code applies SiLU after the norm. Without it the block's output is several times too
    # large and the decoded result collapses (measured: 3-7 dB, below trilinear). The fix was found on
    # 2026-10-16 but never landed in the file, so it is applied here rather than left to chance, and the
    # published check -- scale=1.0 must be close to identity -- is asserted before anything is measured.
    def _temporal_forward(self, x, emb=None):
        return x + self.pwconv(self.dwconv(F.silu(self.norm(x))))
    LU.TemporalConv.forward = _temporal_forward

    h3 = H3(load_dit=False)
    up = LatentResizer3D()
    sd = load_file(str(LBH))
    missing, unexpected = up.load_state_dict({k: v.float() for k, v in sd.items()}, strict=False)
    # mean/std are locally registered normalisation buffers, not published weights (latent_upscaler.py
    # exempts them the same way); anything else missing would mean the architecture does not match.
    missing = [m for m in missing if m not in ("mean", "std")]
    assert not missing and not unexpected, (len(missing), len(unexpected), missing[:3], unexpected[:3])
    up = up.eval()
    n_par = sum(p.numel() for p in up.parameters())

    # identity check at scale=1: the upscaler must return what it was given, in raw latent space
    with torch.no_grad():
        g = torch.Generator().manual_seed(0)
        mu = up.mean.view(1, -1, 1, 1, 1)
        sg = up.std.view(1, -1, 1, 1, 1)
        zc = torch.randn(1, 24, 7, 8, 14, generator=g) * sg + mu
        ident = up(zc, scale=1.0, size=(8, 14))
        rel = float((ident - zc).abs().mean() / zc.abs().mean())
    print(json.dumps({'identity_scale1_rel_err': round(rel, 4),
                      'in_rms': round(float(zc.square().mean().sqrt()), 4),
                      'out_rms': round(float(ident.float().square().mean().sqrt()), 4),
                      'params': n_par}), flush=True)
    assert rel < 0.35, f'scale=1 is not near-identity (rel {rel:.3f}); the architecture still does not match'

    clips = json.loads((D_ / 'manifest.json').read_text())['clips'][:a.clips]
    rows = []
    for name in clips:
        hr = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in range(a.frames)])   # [T,H,W,3]
        t = hr.permute(0, 3, 1, 2)
        # the low resolution the DiT would operate at, with the same antialiased kernel used everywhere here
        hr_pad = F.pad(t, PAD, mode='replicate')
        lo = F.interpolate(hr_pad.double(), size=(LH, LW), mode='bicubic', align_corners=False,
                           antialias=True).clamp(0, 1).float()
        with torch.no_grad():
            z_lo = h3.encode(lo.permute(0, 2, 3, 1))
            z_hi = h3.encode(hr_pad.permute(0, 2, 3, 1))
            dec = {}
            dec['vae_roundtrip'] = h3.decode(z_hi)
            z_up = up(z_lo.float(), scale=4.0, size=(UH // 16, UW // 16)).to(z_lo.dtype)
            dec['ceiling_lbh'] = h3.decode(z_up)
            z_tri = F.interpolate(z_lo.float(), size=(z_lo.shape[2], UH // 16, UW // 16),
                                  mode='trilinear', align_corners=False).to(z_lo.dtype)
            dec['ceiling_tri'] = h3.decode(z_tri)
        dec['bicubic_lr'] = F.interpolate(lo.double(), size=(H, W), mode='bicubic',
                                          align_corners=False).clamp(0, 1).float().permute(0, 2, 3, 1)
        r = {'clip': name}
        for tag, px in dec.items():
            if px.shape[1] != H or px.shape[2] != W:                 # undo the pad on the 512x896 arms
                px = px[:, PAD[2]:PAD[2] + H, PAD[0]:PAD[0] + W]
            assert px.shape == (a.frames, H, W, 3), (tag, px.shape)
            r[tag + '_psnr'] = float(np.mean([psnr(px[i], hr[i]) for i in range(a.frames)]))
            r[tag + '_lpips'] = float(np.mean([lpips_dist(px[i], hr[i]) for i in range(0, a.frames, 8)]))
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    arms = ['vae_roundtrip', 'ceiling_lbh', 'ceiling_tri', 'bicubic_lr']
    agg = {t: {m: float(np.mean([r[f'{t}_{m}'] for r in rows])) for m in ('psnr', 'lpips')} for t in arms}
    ceil_psnr = agg['ceiling_lbh']['psnr']
    verdict = ('DEAD: upscaler ceiling below the deployed pipeline PSNR 26.8' if ceil_psnr < 26.8 else
               'BUILD: ceiling above 30 dB' if ceil_psnr > 30.0 else 'MARGINAL: decide on LPIPS')
    out = {'clips': len(rows), 'frames': a.frames, 'low_res': [LH, LW], 'upscaler_params': n_par,
           'per_clip': rows, 'arms': agg, 'deployed_reference': {'psnr': 26.821, 'lpips': 0.2030},
           'verdict': verdict}
    (OUT / 'ceiling.json').write_text(json.dumps(out, indent=1))
    print()
    print('%-16s %8s %8s' % ('arm', 'PSNR', 'LPIPS'))
    for t in arms:
        print('%-16s %8.3f %8.4f' % (t, agg[t]['psnr'], agg[t]['lpips']))
    print('%-16s %8.3f %8.4f   <- deployed pipeline, for reference' % ('(deployed)', 26.821, 0.2030))
    print()
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
