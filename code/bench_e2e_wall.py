"""End-to-end wall time per clip, including the phases the block timer never saw.

run_deploy computes the R_phi estimate once per clip, before the block loop, and `total_s` starts inside
that loop. `sec` is just the sum of those block times. So every latency figure reported from those runs
excludes R_phi entirely -- which is also why moving it to the accelerator produced no visible change in
`total_s`, and why comparing a 1-process run against a 4-process one looked like a speedup when it was
contention.

This times the whole per-clip path on one process and one card, with R_phi on each device, so the two
numbers differ in exactly one variable and both include every phase a deployed stream would pay for.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image

from h3_comfy import H3, ROOT, sync

D_ = ROOT / 'data/real_eval/const'
H, W = 480, 832


def load_png(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default=None)
    ap.add_argument('--steps', type=int, default=4)
    a = ap.parse_args()

    import h3_comfy
    import lora as L
    import turbo as TB
    import lora_merge as LM
    from safetensors.torch import load_file
    from range_branch import RangeBranch

    man = json.loads((D_ / 'manifest.json').read_text())
    name = a.clip or man['clips'][0]
    blocks = man['blocks']
    T = json.loads((D_ / name / 'meta.json').read_text())['frames']

    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    TB.merge(model, ROOT / 'assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors')
    L.inject(model, 32)
    L.load(model, load_file(str(ROOT / 'results/stage1_lora/lora_step002000.safetensors')))
    LM.merge_(model)
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
    h3_comfy.SIGMAS = TB.turbo_sigmas(a.steps, 12.0)

    # the shipped estimator, not the first one that existed: r3_big24 is what the quality table uses
    st = torch.load(ROOT / os.environ.get('RPHI_CKPT', 'results/range_branch/r3_big24.pt'),
                    map_location='cpu')
    lr = torch.stack([load_png(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)]).permute(0, 3, 1, 2)

    out = {'clip': name, 'frames': T, 'steps': a.steps, 'new_frames_per_block': 17,
           'blocks': len(blocks), 'devices': {}}

    for devname in ('cpu', 'npu'):
        rb = RangeBranch(**st['cfg']).eval()
        rb.load_state_dict(st['state'])
        x = lr
        if devname == 'npu':
            rb = rb.to(h3.dev); x = lr.to(h3.dev)
        # warm
        with torch.no_grad():
            rb(x[:2, None])
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            est = rb(x[:, None]).clamp(0, 1).cpu()
        sync()
        out['devices'][devname] = {'rphi_s_per_clip': time.perf_counter() - t0}
        del rb
        torch.npu.empty_cache()

    # one block of the sampling path, to express R_phi as a share of the real total
    est = est.to(lr.dtype)
    s, e = blocks[1]['range']
    sync(); t0 = time.perf_counter()
    guide = h3.encode_guide(lr[s:e], [H, W])
    sampled, _ = h3.sample(guide, 'innerdc', low=est[s:e], hw=[H, W], seed=101)
    px = h3.decode(sampled['video'])
    sync()
    out['block_s'] = time.perf_counter() - t0
    out['sampling_total_s_per_clip'] = out['block_s'] * len(blocks)

    for d, v in out['devices'].items():
        total = out['sampling_total_s_per_clip'] + v['rphi_s_per_clip']
        v['e2e_s_per_clip'] = total
        v['e2e_s_per_frame'] = total / T
        v['realtime_factor'] = total / T * 24
        v['rphi_share'] = v['rphi_s_per_clip'] / total

    print(json.dumps(out, indent=1, default=float))
    print()
    print('%-6s %14s %14s %12s %10s' % ('R_phi', 'rphi s/clip', 'e2e s/clip', 's/frame', 'x realtime'))
    for d, v in out['devices'].items():
        print('%-6s %14.3f %14.2f %12.3f %10.1f' % (d, v['rphi_s_per_clip'], v['e2e_s_per_clip'],
                                                    v['e2e_s_per_frame'], v['realtime_factor']))
    cpu, npu = out['devices']['cpu'], out['devices']['npu']
    print()
    print('R_phi on NPU saves %.2f s/clip = %.3f s/frame (%.1f%% of end-to-end)'
          % (cpu['rphi_s_per_clip'] - npu['rphi_s_per_clip'],
             (cpu['e2e_s_per_frame'] - npu['e2e_s_per_frame']),
             100 * (cpu['e2e_s_per_clip'] - npu['e2e_s_per_clip']) / cpu['e2e_s_per_clip']))


if __name__ == '__main__':
    main()
