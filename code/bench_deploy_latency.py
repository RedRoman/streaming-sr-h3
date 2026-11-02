"""Latency of the deployed path only: the real block loop with the evaluation harness removed.

The instrumented profile of one clip through run_deploy came out as, for 124 frames over 7 blocks:

    sample (DiT)   126.54 s   67.1%
    png_write       30.66 s   16.2%
    guide            20.29 s   10.8%
    decode            7.35 s    3.9%
    to_u8             3.04 s    1.6%
    stream_write     24.71 s   (after the block loop)

so 55.4 s of the 213.4 s total -- 26% -- is PNG encoding for the judge. A deployed stream writes no PNGs
and computes no sha256, and every latency figure this project has reported includes that work. This runs
the identical block path with the IO deleted, so the number is what a deployment would actually pay.

Both numbers are printed, because the harness figure is the one every earlier result was measured with
and the two must not be silently swapped. R_phi is included: it runs once per clip before the block loop,
which is why the block timer never saw it.

No quality output is produced on purpose -- this cannot be mistaken for an eval run.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image

from h3_comfy import H3, ROOT, sync, to_u8

H, W = 480, 832


def load_png(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clip', default=None)
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--rphi', default='results/range_branch/r5_k3.pt')
    ap.add_argument('--lora', default='results/stage1_lora/lora_step002000.safetensors')
    ap.add_argument('--out', default='results/deploy_latency.json')
    a = ap.parse_args()

    import h3_comfy
    import lora as L
    import lora_merge as LM
    import turbo as TB
    from safetensors.torch import load_file
    from run_deploy import range_estimate, TURBO_PATH

    D_ = ROOT / a.eval_set
    man = json.loads((D_ / 'manifest.json').read_text())
    name = a.clip or man['clips'][0]
    blocks = man['blocks']
    T = json.loads((D_ / name / 'meta.json').read_text())['frames']

    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    TB.merge(model, TURBO_PATH)
    L.inject(model, 32)
    L.load(model, load_file(str(ROOT / a.lora)))
    LM.merge_(model)
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
    h3_comfy.SIGMAS = TB.turbo_sigmas(a.steps, 12.0)

    lr = torch.stack([load_png(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)]).permute(0, 3, 1, 2)

    # one warm block first: the first call on Ascend includes operator compilation and has measured two
    # orders of magnitude over steady state, so an unwarmed figure is not a latency figure
    s0, e0 = blocks[0]['range']
    est_w = range_estimate(lr[s0:e0], ROOT / a.rphi, dev=h3.dev)
    g = h3.encode_guide(lr[s0:e0], [H, W])
    sm, _ = h3.sample(g, 'innerdc', low=est_w, hw=[H, W], seed=101)
    h3.decode(sm['video'])
    sync()

    ph = {k: 0.0 for k in ('rphi', 'guide', 'sample', 'decode', 'to_u8', 'png')}
    sync(); t_clip = time.perf_counter()
    t0 = time.perf_counter()
    est = range_estimate(lr, ROOT / a.rphi, dev=h3.dev)
    sync(); ph['rphi'] = time.perf_counter() - t0

    import tempfile
    tmp = tempfile.mkdtemp()
    frames_out = 0
    for b in blocks:
        s, e = b['range']
        sync(); t0 = time.perf_counter()
        guide = h3.encode_guide(lr[s:e], [H, W])
        sync(); ph['guide'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        sampled, _ = h3.sample(guide, 'innerdc', low=est[s:e], hw=[H, W], seed=101)
        sync(); ph['sample'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        px = h3.decode(sampled['video'])
        sync(); ph['decode'] += time.perf_counter() - t0
        t0 = time.perf_counter()
        u8 = to_u8(px)
        ph['to_u8'] += time.perf_counter() - t0
        # the harness cost, timed but separable: write the same PNGs run_deploy would
        t0 = time.perf_counter()
        for i, f in enumerate(range(s, e)):
            Image.fromarray(u8[i]).save(f'{tmp}/{f:06d}.png')
        ph['png'] += time.perf_counter() - t0
        frames_out += b['new'][1] - b['new'][0] + 1 if isinstance(b.get('new'), (list, tuple)) else 0
    sync()
    wall = time.perf_counter() - t_clip

    deployed = wall - ph['png']
    out = {'clip': name, 'frames': T, 'blocks': len(blocks), 'steps': a.steps,
           'phases_s': {k: round(v, 2) for k, v in ph.items()},
           'harness_wall_s': round(wall, 2),
           'deployed_wall_s': round(deployed, 2),
           'harness_s_per_frame': round(wall / T, 4),
           'deployed_s_per_frame': round(deployed / T, 4),
           'harness_x_realtime_24fps': round(wall / T * 24, 1),
           'deployed_x_realtime_24fps': round(deployed / T * 24, 1),
           'png_share_of_harness': round(ph['png'] / wall, 4),
           'dit_share_of_deployed': round(ph['sample'] / deployed, 4),
           'note': ('deployed = harness minus PNG encoding only; the sha256 and the second pass over the '
                    'stitched stream that run_deploy also does are not included here at all, so the real '
                    'harness overhead in those runs is larger than png_share suggests')}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    print()
    print('%-22s %10s %10s' % ('', 'harness', 'deployed'))
    print('%-22s %10.2f %10.2f' % ('wall s / clip', wall, deployed))
    print('%-22s %10.4f %10.4f' % ('s / frame', wall / T, deployed / T))
    print('%-22s %10.1f %10.1f' % ('x realtime @24fps', wall / T * 24, deployed / T * 24))
    print()
    print('DiT is %.1f%% of the deployed path; PNG was %.1f%% of the harness path'
          % (100 * ph['sample'] / deployed, 100 * ph['png'] / wall))


if __name__ == '__main__':
    main()
