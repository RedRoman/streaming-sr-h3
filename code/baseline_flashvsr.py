"""FlashVSR v1.1 Tiny (CVPR 2026) on the 24-clip real-footage set -- the second judge.

Official repo, official weights, official pipeline call. Two deviations, both forced by our setting and
both verifiable:

  attention kernel  Block-Sparse-Attention is CUDA-only, so block_sparse_attn.py reimplements its entry
                    point on masked SDPA. The mask is preserved exactly (self-test: an all-true mask equals
                    dense attention to 0.0 max-abs; a restrictive mask changes the output), so quality is
                    comparable. LATENCY THROUGH THIS PATH IS NOT COMPARABLE and is not reported as such.
  geometry          FlashVSR requires a 128-multiple output and 8n+1 frames. Our 208x120 LR would be
                    centre-cropped to 768x384, which does not align with our 832x480 ground truth, so the
                    LR is replicate-padded to 224x128 (-> 896x512, a clean 128 multiple, no crop inside
                    FlashVSR) and the output is cropped back to the original field of view. Frames are
                    padded to 129 by repeating the last one and truncated back to 124.

Everything else -- sparse_ratio, kv_ratio, local_range, color_fix, one denoising step -- is left at the
values the authors ship in infer_flashvsr_v1.1_tiny.py.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
REPO = Path('/temp/zihaohe/baselines/FlashVSR-main')
WEIGHTS = Path(os.environ.get('FLASHVSR_WEIGHTS', '/cache/zihaohe/streaming_sr0/assets/baselines/flashvsr'))
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
OUT = ROOT / 'results/baseline_flashvsr'
PAD_LR = (8, 8, 4, 4)          # l, r, t, b on the 208x120 LR -> 224x128
SCALE = 4
NEED = 124
FRAMES_IN = 129                # 8n+1 >= NEED+4


def main():
    sys.path[:0] = [str(ROOT / 'code'), str(REPO), str(REPO / 'examples/WanVSR')]
    import block_sparse_attn as BSA
    # diffsynth's package __init__ imports every model it ships, including a StepVideo text encoder that
    # wants transformers' pre-5.x export of PretrainedConfig. Nothing on the FlashVSR path touches it, so
    # restore the alias rather than editing the vendored repo.
    import transformers.modeling_utils as _mu
    if not hasattr(_mu, 'PretrainedConfig'):
        from transformers.configuration_utils import PretrainedConfig as _PC
        _mu.PretrainedConfig = _PC
    if not hasattr(_mu, 'PreTrainedModel'):
        from transformers import PreTrainedModel as _PM
        _mu.PreTrainedModel = _PM
    from diffsynth import ModelManager, FlashVSRTinyPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder

    # the pipeline loads its cached prompt tensor via a path relative to examples/WanVSR; every path this
    # script owns is absolute, so moving the working directory there is safe.
    os.chdir(REPO / 'examples/WanVSR')
    t0 = time.perf_counter()
    mm = ModelManager(torch_dtype=torch.bfloat16, device='cpu')
    mm.load_models([str(WEIGHTS / 'diffusion_pytorch_model_streaming_dmd.safetensors')])
    pipe = FlashVSRTinyPipeline.from_model_manager(mm, device='cuda')
    dm = pipe.denoising_model()
    dm.LQ_proj_in = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).to('cuda', dtype=torch.bfloat16)
    dm.LQ_proj_in.load_state_dict(torch.load(WEIGHTS / 'LQ_proj_in.ckpt', map_location='cpu'), strict=True)
    dm.LQ_proj_in.to('cuda')
    pipe.TCDecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    miss = pipe.TCDecoder.load_state_dict(torch.load(WEIGHTS / 'TCDecoder.ckpt'), strict=False)
    pipe.to('cuda')
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    pipe.init_cross_kv()
    pipe.load_models_to_device(['dit', 'vae'])
    load_s = time.perf_counter() - t0

    names = os.environ.get('CLIPS', '').split(',') or None
    if not names or names == ['']:
        names = json.loads((D_ / 'manifest.json').read_text())['clips']
    shard, nshards = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARDS', 1))
    names = [c for i, c in enumerate(names) if i % nshards == shard]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f'runtime_shard{shard}.json').write_text(json.dumps(
        {'repo': str(REPO), 'weights': str(WEIGHTS), 'load_seconds': load_s,
         'missing_tcdecoder_keys': len(miss.missing_keys), 'unexpected': len(miss.unexpected_keys),
         'kernel': 'block_sparse_attn shim on masked SDPA (quality comparable, latency NOT)',
         'pad_lr_lrtb': PAD_LR, 'frames_in': FRAMES_IN, 'sparse_ratio': 2.0, 'kv_ratio': 3.0,
         'local_range': 11, 'steps': 1, 'clips': names}, indent=1))

    for name in names:
        d = OUT / name
        if (d / 'result.json').exists():
            continue
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        assert T == NEED, T
        lr = torch.stack([torch.from_numpy(np.asarray(Image.open(D_ / name / 'lr' / f'{t:06d}.png').convert('RGB')).copy())
                          for t in range(T)]).float().div_(255)          # [T,h,w,3]
        x = lr.permute(0, 3, 1, 2)
        x = F.pad(x, PAD_LR, mode='replicate')                            # [T,3,128,224]
        x = torch.cat([x, x[-1:].repeat(FRAMES_IN - T, 1, 1, 1)])         # repeat last frame to 8n+1
        H4, W4 = x.shape[-2] * SCALE, x.shape[-1] * SCALE
        up = F.interpolate(x, size=(H4, W4), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        LQ = (up * 2 - 1).permute(1, 0, 2, 3).unsqueeze(0).to('cuda', torch.bfloat16)   # [1,C,F,H,W]

        BSA._SHIM_CHECK['calls'] = 0
        BSA._SHIM_CHECK['dense_calls'] = 0
        BSA._SHIM_CHECK['mask_density'] = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        video = pipe(prompt='', negative_prompt='', cfg_scale=1.0, num_inference_steps=1, seed=0,
                     LQ_video=LQ, num_frames=FRAMES_IN, height=H4, width=W4,
                     is_full_block=False, if_buffer=True,
                     topk_ratio=2.0 * 768 * 1280 / (H4 * W4), kv_ratio=3.0, local_range=11, color_fix=True)
        torch.cuda.synchronize(); dt = time.perf_counter() - t1

        v = video.float().add(1).div(2).clamp(0, 1)                       # [C,F,H,W] in [0,1]
        v = v.permute(1, 2, 3, 0)[:T]                                     # [T,H,W,3]
        y0, x0 = PAD_LR[2] * SCALE, PAD_LR[0] * SCALE
        v = v[:, y0:y0 + meta['hr_wh'][1], x0:x0 + meta['hr_wh'][0]]
        assert v.shape == (T, meta['hr_wh'][1], meta['hr_wh'][0], 3), v.shape
        d.mkdir(parents=True, exist_ok=True)
        for t in range(T):
            Image.fromarray(np.uint8(v[t].cpu().numpy() * 255 + .5)).save(d / f'{t:06d}.png')
        dens = BSA._SHIM_CHECK['mask_density']
        (d / 'result.json').write_text(json.dumps(
            {'clip': name, 'model': 'flashvsr_v1.1_tiny', 'seconds_total': dt, 'frames': T,
             'peak_GB': torch.cuda.max_memory_allocated() / 1e9,
             'shim_calls': BSA._SHIM_CHECK['calls'], 'shim_dense_calls': BSA._SHIM_CHECK['dense_calls'],
             'mask_density_mean': float(np.mean(dens)) if dens else None,
             'latency_comparable': False}, indent=1))
        print(json.dumps({'done': name, 'sec': round(dt, 1),
                          'peak_GB': round(torch.cuda.max_memory_allocated() / 1e9, 1),
                          'shim_calls': BSA._SHIM_CHECK['calls'],
                          'mask_density': round(float(np.mean(dens)), 4) if dens else None}), flush=True)
    print(json.dumps({'shard': shard, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
