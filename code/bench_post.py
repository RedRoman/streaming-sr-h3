"""Latency of the post-processing chain, so the deployable configuration reports an honest end-to-end cost.

Sampling is measured per block by run_deploy.py. The chain that turns a sampled block into the shipped
frames -- range substitution and the spatial quota -- was developed offline on 24 CPU workers, which says
nothing about its per-frame cost. Both stages are per frame and tiny next to a diffusion step, but "tiny"
has to be a number before it goes in a table.

Measured single-threaded on one frame at a time, which is the streaming case; batching would be faster.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
import deploy_eval as DE                                     # noqa: E402

N = 40


def main():
    torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
    from range_branch import RangeBranch
    st = torch.load(ROOT / 'results/range_branch/k1.pt', map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    name = DE.clips()[0]
    m = DE.meta(name)
    hw = (m['hr_wh'][1], m['hr_wh'][0])
    lr = torch.stack([DE.load(DE.D_ / name / 'lr' / f'{t:06d}.png') for t in range(N)])
    out = torch.stack([DE.load(ROOT / 'results/deploy_runs/t8' / name / 'stream' / f'{t:06d}.png')
                       for t in range(N)])
    rec = {'threads': torch.get_num_threads(), 'frames': N, 'hr': list(hw)}
    with torch.no_grad():
        for _ in range(3):                                   # warm up
            rb(lr[:1].permute(0, 3, 1, 2)[None])
        t0 = time.perf_counter()
        est = [rb(lr[t:t + 1].permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0) for t in range(N)]
        rec['rphi_ms'] = 1e3 * (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        guides = [DE.up(e, hw).clamp(0, 1) for e in est]
        rec['upsample_ms'] = 1e3 * (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        nulls = [DE.split(out[t])[1] for t in range(N)]
        rec['split_ms'] = 1e3 * (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        ws = [(DE._local_std(g) / 0.02).clamp(0, 1) for g in guides]
        rec['quota_gate_ms'] = 1e3 * (time.perf_counter() - t0) / N
        t0 = time.perf_counter()
        _ = [(guides[t] + nulls[t] * ws[t]).clamp(0, 1) for t in range(N)]
        rec['compose_ms'] = 1e3 * (time.perf_counter() - t0) / N
    rec['total_per_frame_ms'] = sum(rec[k] for k in
                                    ('rphi_ms', 'upsample_ms', 'split_ms', 'quota_gate_ms', 'compose_ms'))
    rec['total_per_block_s'] = rec['total_per_frame_ms'] * 17 / 1e3       # 17 new frames per block
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}, indent=1))
    (ROOT / 'results/deploy/bench_post.json').write_text(json.dumps(rec, indent=1))


if __name__ == '__main__':
    main()
