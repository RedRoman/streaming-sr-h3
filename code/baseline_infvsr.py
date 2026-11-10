"""InfVSR (ICML 2026) on the 24-clip real-footage set: the first baseline that is a judge, not a floor.

Real-ESRGAN is a per-frame ISR and cannot speak to any streaming claim. InfVSR is the direct comparison --
causal DiT with a rolling KV cache, one diffusion step -- so it is scored on exactly our protocol: our LR
frames, our stitched-stream output layout, and the metric pass in deploy_eval.py.

Official pipeline, unmodified: diffsynth_wan.pipeline.InfVSR with the published weights, chunked 9 then 12
pixel frames as diffsynth_wan.video.frame_batches does, scale=4 (208x120 -> 832x480, our HR size), and
scene detection off, which is what `--no_scene_detection` does: one reference frame and an unbroken KV
cache for the whole clip. That is the streaming setting; letting it cut scenes would reset the cache and
quietly hand it a per-shot advantage our pipeline does not take.

The repo is CUDA-written, so this runs under npurun.py (torch_npu + transfer_to_npu).
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
B = Path('/temp/zihaohe_stage/baselines/InfVSR-main')
W = Path('/temp/zihaohe_stage/weights/infvsr')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
# Suffix so a second eval set cannot land in the directory built for the first. FlashVSR+ already
# had this; InfVSR did not, and a smoke run on MVSR4x wrote clip 003 straight into the heldout
# baseline tree. judge looks clips up by name so nothing was overwritten, but one directory
# holding two input scales is exactly the mistake that made an earlier x6 table meaningless.
OUT = ROOT / ('results/baseline_infvsr' + os.environ.get('INFVSR_OUT_SUFFIX', ''))
FIRST, REST = 9, 12                       # diffsynth_wan.video.frame_batches


def chunk_bounds(T):
    b, s = [], 0
    n = FIRST
    while s < T:
        e = min(s + n, T)
        b.append((s, e))
        s, n = e, REST
    return b


def main():
    sys.path[:0] = [str(B)]
    from diffsynth_wan.pipeline import InfVSR

    only = os.environ.get('CLIPS', '')
    names = only.split(',') if only else json.loads((D_ / 'manifest.json').read_text())['clips']
    shard, nshards = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARDS', 1))
    names = [c for i, c in enumerate(names) if i % nshards == shard]
    OUT.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = InfVSR(W / 'InfVSR.ckpt', B / 'diffsynth_wan/config.json', W / 'Wan2.1_VAE.pth',
                   W / 'ram_swin_large_14m.pth', W / 'DAPE.pth', device='cuda')
    load_s = time.perf_counter() - t0
    (OUT / f'runtime_shard{shard}.json').write_text(json.dumps(
        {'weights': str(W), 'repo': str(B), 'load_seconds': load_s, 'scale': 4,
         'chunks': f'{FIRST} then {REST} pixel frames', 'scene_detection': False,
         'dtype': 'bfloat16', 'clips': names}, indent=1))

    for name in names:
        d = OUT / name
        if (d / 'result.json').exists():
            continue
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        lr = np.stack([np.asarray(Image.open(D_ / name / 'lr' / f'{t:06d}.png').convert('RGB'))
                       for t in range(T)])
        d.mkdir(parents=True, exist_ok=True)
        bounds = chunk_bounds(T)

        def gen():
            for s, e in bounds:
                v = torch.from_numpy(lr[s:e]).permute(3, 0, 1, 2)[None].float().div_(255).mul_(2).sub_(1)
                yield v

        torch.npu.reset_peak_memory_stats()
        recs, written = [], 0
        tc = time.perf_counter()
        for (s, e), frames in zip(bounds, model(gen(), scale=4)):
            torch.npu.synchronize()
            dt = time.perf_counter() - tc
            x = frames[0].permute(1, 2, 3, 0).float().add(1).div(2).clamp(0, 1).numpy()
            assert x.shape[0] == e - s, (x.shape, s, e)
            for i, t in enumerate(range(s, e)):
                Image.fromarray(np.uint8(x[i] * 255 + .5)).save(d / f'{t:06d}.png')
                written += 1
            recs.append({'range': [s, e], 'seconds': dt, 'frames': e - s,
                         'peak_GB': torch.npu.max_memory_allocated() / 1e9})
            tc = time.perf_counter()
        assert written == T, (written, T)
        (d / 'result.json').write_text(json.dumps(
            {'clip': name, 'model': 'infvsr', 'chunks': recs, 'frames': written,
             'total_seconds': sum(r['seconds'] for r in recs)}, indent=1))
        print(json.dumps({'done': name, 'frames': written,
                          'sec': round(sum(r['seconds'] for r in recs), 1),
                          'peak_GB': round(max(r['peak_GB'] for r in recs), 1)}), flush=True)
    print(json.dumps({'shard': shard, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
