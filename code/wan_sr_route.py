"""Plan A: keep our range/null machinery, replace the generator with the colleague's one-step stack.

Measured on one 910B, warm: their 1.419B SR-DiT is 0.0249 s/frame at 832x512 and the 23 MB Flash-VAED
decoder 0.0082, against 1.789 s/frame for our 20.11B 4-step pipeline. This runs their 'sr_wan' route --
Wan VAE encode of the LR, latent upsample, one SR-DiT evaluation, streaming decode -- on OUR evaluation
set, at OUR geometry, so the output drops straight into the existing 24-clip protocol.

Geometry. neighbor8.partition asserts the patch grid divides by 4, i.e. the latent divides by 8, i.e. the
RGB divides by 64; 480 does not. So the LR is replicate-padded 120->128 and the 512x832 output is cropped
back to 480x832. Wan's temporal layout needs T = 1 mod 4 and every chunk needs exactly 2 latents, so
frames 0..120 (the stride-8 evaluation covers exactly those) are extended by repeating frame 120 four
times: 125 frames -> 32 latents -> 16 chunks -> 5 + 15*8 = 125 decoded frames -> keep 121.

Condition. Their upsampler is trained at x3 and we need x4, and x3 can never land on a latent divisible
by 8 from an integer LR. Three constructions, because which one costs least is a measurement:
  up3_interp  Wan(LR 128x208) -> 16x26 -> trained x3 -> 48x78 -> bicubic x4/3 -> 64x104.
              The learned module does exactly the x3 it was trained for; only a mild 1.33x interpolation
              is added, and the DiT regenerates from the condition anyway.
  bicubic     Wan(bicubic x4 of LR, at 512x832) -> 64x104 directly. No upsampler, no scale constraint,
              but a condition the SR-DiT never saw in training.
  interp_only Wan(LR) -> bicubic x4 in latent -> 64x104. The control that isolates what the trained
              upsampler contributes.

Prompt. The SR-DiT takes a UMT5 embedding and their handoff says an empty prompt is not a substitute.
A deployed stream has no caption, so whether that matters is itself a result: --prompt zeros runs the
degenerate case, --prompt-file supplies a real embedding.
"""
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np

PKG = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/ascend_eval')
WEIGHTS = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/weights')
sys.path.insert(0, str(PKG))

import torch
import torch_npu  # noqa: F401
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / 'data/real_eval/const'
OUT = Path(os.environ.get('WAN_OUT', str(ROOT / 'results/wan_route')))
H, W = 480, 832                 # evaluation geometry
PH = 512                        # padded height: 64 | 512, 480 is not
PAD_T = (PH - H) // 2           # 16 px of HR = 4 px of LR on each side
LAST = 120                      # frames 0..120 inclusive; stride-8 evaluation covers exactly these
NPAD = 4                        # repeats of frame 120 to reach 125 = 1 + 4*31


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def save(p, x):
    Image.fromarray(np.uint8(x.clamp(0, 1).numpy() * 255 + .5)).save(p)


def psnr(a, b):
    return float(10 * np.log10(1.0 / max(float((a - b).square().mean()), 1e-12)))


def build(dev, cond_kind):
    import common
    common.MODELS = WEIGHTS
    from wan.modules.vae import WanVAE
    from model import StreamingSR
    from flash_decoder import FlashDecoder
    from sr.latent_upsampler.ultra_latent_up_v1 import UltraLatentUpsampler

    vae = WanVAE(vae_pth=str(WEIGHTS / 'Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=dev)
    vae.model.eval().requires_grad_(False)

    up = None
    if cond_kind == 'up3_interp':
        ck = torch.load(WEIGHTS / 'conditioning2000.pt', map_location='cpu', weights_only=False, mmap=True)
        st = {k[len('upsampler.'):]: v for k, v in ck['model'].items() if k.startswith('upsampler.')}
        assert st, sorted(ck['model'])[:5]
        up = UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128, num_blocks=8,
                                  upsample_scale=3, activation='relu', residual=False,
                                  memory_init='replicate', zero_init_final=False, default_parallel=True)
        up.load_state_dict(st, strict=True)
        up = up.eval().requires_grad_(False).to(device=dev, dtype=torch.bfloat16)
        del ck, st

    sr = StreamingSR()
    ck = torch.load(WEIGHTS / 'srdit_h3_step2000.pt', map_location='cpu', weights_only=False, mmap=True)
    sr.load_state_dict(ck['model'], strict=True)
    del ck
    sr = sr.to(dev).eval().requires_grad_(False)
    sr.release_eval_cache = True

    dec = FlashDecoder.__new__(FlashDecoder)
    FlashDecoder.__init__(dec, vae, segment=2, weights=WEIGHTS / 'Flash-VAED/Flash_VAED_Wan.pth')
    dec = dec.to(dev).eval().requires_grad_(False)
    return vae, up, sr, dec


def condition(vae, up, lr_pad, cond_kind, dev):
    """lr_pad [T,3,128,208] in [0,1] -> normalized Wan HR condition [1,16,Tl,64,104]."""
    def enc(x):                                      # x [T,3,h,w] in [0,1]
        vae.model.clear_cache()
        z = vae.encode([x.permute(1, 0, 2, 3).to(dev) * 2 - 1])[0][None]
        vae.model.clear_cache()
        return z.bfloat16()

    if cond_kind == 'bicubic':
        big = F.interpolate(lr_pad.double(), size=(PH, W), mode='bicubic', align_corners=False)
        return enc(big.clamp(0, 1).float())
    z = enc(lr_pad)                                  # [1,16,Tl,16,26]
    if cond_kind == 'interp_only':
        return F.interpolate(z.float(), size=(z.shape[2], PH // 8, W // 8), mode='trilinear',
                             align_corners=False).to(z.dtype)
    with torch.autocast('npu', dtype=torch.bfloat16):
        z3 = up(z, parallel=True)                    # [1,16,Tl,48,78]
    return F.interpolate(z3.float(), size=(z3.shape[2], PH // 8, W // 8), mode='trilinear',
                         align_corners=False).to(z.dtype)


def run_clip(name, vae, up, sr, dec, prompt, cond_kind, dev, seed=101):
    d = OUT / cond_kind / name / 'stream'
    d.mkdir(parents=True, exist_ok=True)
    lr = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(LAST + 1)]).permute(0, 3, 1, 2)
    lr = torch.cat([lr, lr[-1:].repeat(NPAD, 1, 1, 1)])                  # 121 -> 125
    lr = F.pad(lr, (0, 0, (PH - H) // 8, (PH - H) // 8), mode='replicate')  # 120 -> 128 rows
    assert lr.shape[1:] == (3, 128, 208), lr.shape

    t0 = time.perf_counter()
    with torch.no_grad():
        cond = condition(vae, up, lr, cond_kind, dev)
        tl = cond.shape[2]
        assert tl % 2 == 0 and cond.shape[-2:] == (PH // 8, W // 8), cond.shape
        gen = torch.Generator(device=dev).manual_seed(seed)
        noise = torch.randn(cond.shape, device=dev, dtype=torch.bfloat16, generator=gen)
        cache = memory = None
        frames = []
        for off in range(0, tl, 2):
            with torch.autocast('npu', dtype=torch.bfloat16):
                z, cache = sr(noise[:, :, off:off + 2], cond[:, :, off:off + 2], prompt, off, cache)
                rgb, memory = dec(z.float(), memory, first=off == 0)
            frames.append(rgb.float().cpu())
    px = torch.cat(frames, 2)[0].permute(1, 0, 2, 3)                     # [T,3,PH,W]
    sec = time.perf_counter() - t0
    px = px[:LAST + 1, :, PAD_T:PAD_T + H].clamp(0, 1).permute(0, 2, 3, 1)
    for t in range(px.shape[0]):
        save(d / f'{t:06d}.png', px[t])
    g = torch.stack([load(D_ / name / 'hr' / f'{t:06d}.png') for t in range(0, LAST + 1, 8)])
    p = float(np.mean([psnr(px[t], g[i]) for i, t in enumerate(range(0, LAST + 1, 8))]))
    rec = {'clip': name, 'cond': cond_kind, 'frames': int(px.shape[0]), 'latents': tl,
           'seconds': sec, 's_per_frame': sec / px.shape[0], 'psnr_stride8': p}
    (OUT / cond_kind / name / 'stream_result.json').write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cond', default='up3_interp', choices=['up3_interp', 'bicubic', 'interp_only'])
    ap.add_argument('--clips', default='')
    ap.add_argument('--n', type=int, default=2)
    ap.add_argument('--prompt', default='zeros', choices=['zeros', 'randn'])
    ap.add_argument('--prompt-file', default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    dev = torch.device('npu:0')
    torch.npu.set_device(dev)
    names = a.clips.split(',') if a.clips else json.loads((D_ / 'manifest.json').read_text())['clips'][:a.n]
    names = [n for i, n in enumerate(names) if i % a.nshards == a.shard]
    vae, up, sr, dec = build(dev, a.cond)
    if a.prompt_file:
        prompt = torch.load(a.prompt_file, map_location='cpu', weights_only=True)
        prompt = (prompt if prompt.dim() == 3 else prompt[None]).to(dev).bfloat16()
    elif a.prompt == 'randn':
        prompt = torch.randn(1, 512, 4096, device=dev, dtype=torch.bfloat16,
                             generator=torch.Generator(device=dev).manual_seed(0))
    else:
        prompt = torch.zeros(1, 512, 4096, device=dev, dtype=torch.bfloat16)
    for n in names:
        print(json.dumps(run_clip(n, vae, up, sr, dec, prompt, a.cond, dev), default=float), flush=True)


if __name__ == '__main__':
    main()
