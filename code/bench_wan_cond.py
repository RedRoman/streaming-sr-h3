"""The condition path of the colleague's stack, timed on the same card as the generator.

bench_wan_chain measured the two expensive halves: the 1.419B one-step SR-DiT at 0.0249 s/frame and the
23 MB Flash-VAED decoder at 0.0082 s/frame, both at 832x512. What is left is what produces the condition.

Two routes exist in their own evaluate.py and they differ by a lot:
  h3    H3 encoder -> Sol-H3 latent x2 -> temporal alignment -> 188.78M Mapper -> UltraFlash x3
  wan   Wan VAE encode of the LR -> UltraFlash x3                       (their 'sr_wan' control arm)
The Mapper only exists to carry an H3 latent into Wan space. If we are replacing our generator anyway, the
wan route drops it, and with it the one module their own handoff flags as non-causal.

Timed per 85-frame clip and per frame, at their LR (448x256) and at ours (208x120 -> the x4 grid).
"""
import argparse, json, sys, time
from pathlib import Path

PKG = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/ascend_eval')
WEIGHTS = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/weights')
sys.path.insert(0, str(PKG))

import torch
import torch_npu  # noqa: F401


def timeit(fn, warm=1, runs=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lr', default='256x448,120x208')
    ap.add_argument('--frames', type=int, default=85)
    ap.add_argument('--out', default='/cache/zihaohe/streaming_sr0/results/deploy/bench_wan_cond.json')
    a = ap.parse_args()
    import common
    common.MODELS = WEIGHTS
    dev = torch.device('npu:0'); torch.npu.set_device(dev)
    from wan.modules.vae import WanVAE
    from sr.latent_upsampler.ultra_latent_up_v1 import UltraLatentUpsampler

    vae = WanVAE(vae_pth=str(WEIGHTS / 'Wan2.1-T2V-1.3B/Wan2.1_VAE.pth'), dtype=torch.bfloat16, device=dev)
    vae.model.eval().requires_grad_(False)
    up = UltraLatentUpsampler(in_channels=16, out_channels=16, mid_channels=128, num_blocks=8,
                              upsample_scale=3, activation='relu', residual=False,
                              memory_init='replicate', zero_init_final=True, default_parallel=True)
    up = up.eval().requires_grad_(False).to(device=dev, dtype=torch.bfloat16)
    out = {'frames': a.frames, 'upsampler_params': sum(p.numel() for p in up.parameters()), 'lr': {}}

    for g in a.lr.split(','):
        h, w = (int(x) for x in g.split('x'))
        rec = {'lr_hw': [h, w]}
        x = torch.rand(3, a.frames, h, w, device=dev) * 2 - 1

        def enc():
            vae.model.clear_cache()
            z = vae.encode([x])[0]
            vae.model.clear_cache()
            return z
        with torch.no_grad():
            rec['wan_encode_s'] = timeit(enc)
            z = enc()[None].bfloat16()
            rec['latent_shape'] = list(z.shape)

            def run_up():
                with torch.autocast('npu', dtype=torch.bfloat16):
                    return up(z, parallel=True)
            rec['upsampler_s'] = timeit(run_up)
            rec['up_shape'] = list(run_up().shape)
        rec['cond_s_per_frame'] = (rec['wan_encode_s'] + rec['upsampler_s']) / a.frames
        out['lr'][g] = rec
        print(json.dumps({g: rec}, default=float), flush=True)

    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print()
    print('%-10s %12s %12s %14s' % ('LR hxw', 'wan_enc s', 'upsampler s', 'cond s/frame'))
    for g, r in out['lr'].items():
        print('%-10s %12.3f %12.3f %14.4f'
              % (g, r['wan_encode_s'], r['upsampler_s'], r['cond_s_per_frame']))


if __name__ == '__main__':
    main()
