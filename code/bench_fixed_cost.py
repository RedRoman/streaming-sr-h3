"""Where does the 8.7 s/block of non-sampling overhead go?

At 1 step the sampler is 7.7 s but guide encoding + decode + other is 8.7 s, already the majority. Step
compression is nearly exhausted as a lever. This breaks that 8.7 s into its parts so the next optimisation
target is chosen by measured cost, not by assumption.

Profiled on one block, warm, with the merged turbo + stage1 lora weights. Each phase is timed in isolation
and repeated; the first call is dropped as compilation warm-up. The guide path and the decode path are the
two fixed costs that survive any step reduction, so they are reported separately.
"""
import os
import time
import torch
import torch_npu  # noqa: F401

from h3_comfy import H3, ROOT, sync

H, W = 480, 832
FRAMES = 22


def main():
    import turbo as TB
    import lora as L
    from safetensors.torch import load_file

    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    TB.merge(model, ROOT / 'assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors')
    L.inject(model, 32)
    L.load(model, load_file(str(ROOT / 'results/stage1_lora/lora_step002000.safetensors')))
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]

    lr = torch.rand(FRAMES, 3, H // 4, W // 4).clamp(0, 1)
    from range_branch import RangeBranch
    st = torch.load(ROOT / os.environ.get('RPHI_CKPT', 'results/range_branch/r3_big24.pt'),
                    map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])

    import h3_comfy
    h3_comfy.SIGMAS = TB.turbo_sigmas(8, 12.0)
    phases = {}
    with torch.no_grad():
        # 1. R_phi range estimation
        sync(); t0 = time.perf_counter()
        for _ in range(3):
            est = torch.cat([rb(lr[t:t+1][None]) for t in range(FRAMES)]).clamp(0, 1)
        sync(); phases['R_phi'] = (time.perf_counter() - t0) / 3

        # 2. bicubic upsample of observation
        sync(); t0 = time.perf_counter()
        for _ in range(3):
            up = torch.nn.functional.interpolate(lr.permute(0,3,1,2), size=(H,W), mode='bicubic', align_corners=False, antialias=True)
        sync(); phases['bicubic_up'] = (time.perf_counter() - t0) / 3

        # 3. VAE encode (guide)
        for _ in range(2):
            g = h3.encode_guide(lr, [H, W])
        sync(); t0 = time.perf_counter()
        for _ in range(3):
            g = h3.encode_guide(lr, [H, W])
        sync(); phases['vae_encode_guide'] = (time.perf_counter() - t0) / 3

        # 4. one DiT step (warm + timed)
        for _ in range(2):
            h3.sample(g, 'innerdc', low=est, hw=[H,W], seed=101)
        sync(); t0 = time.perf_counter()
        for _ in range(3):
            h3.sample(g, 'innerdc', low=est, hw=[H,W], seed=101)
        sync(); phases['dit_8step'] = (time.perf_counter() - t0) / 3

        # 5. VAE decode -- use a synthetic latent of the right shape; decode time is content-independent
        z_synth = torch.randn(1, 24, FRAMES, H//16, W//16, device=h3.dev, dtype=torch.float16) * 0.5
        for _ in range(2):
            h3.decode(z_synth)
        sync(); t0 = time.perf_counter()
        for _ in range(3):
            h3.decode(z_synth)
        sync(); phases['vae_decode'] = (time.perf_counter() - t0) / 3

    import json
    total_fixed = phases['R_phi'] + phases['bicubic_up'] + phases['vae_encode_guide'] + phases['vae_decode']
    phases['total_fixed'] = total_fixed
    phases['dit_per_step'] = phases['dit_8step'] / 8
    print(json.dumps({k: round(v, 3) for k, v in phases.items()}, indent=1))
    print()
    print('%-22s %8s' % ('phase', 'seconds'))
    for k in ['R_phi', 'bicubic_up', 'vae_encode_guide', 'vae_decode', 'total_fixed', 'dit_8step', 'dit_per_step']:
        print('%-22s %8.3f' % (k, phases[k]))


if __name__ == '__main__':
    main()
