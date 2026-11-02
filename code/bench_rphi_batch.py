"""R_phi is called once per frame; with K=1 the frame loop is pure overhead.

bench_fixed_cost measured R_phi at 3.445 s for a 22-frame block, against 3.6 ms for a single frame on NPU
measured earlier -- a factor of 400. The cause is the call shape: run_deploy and gen_teacher both do

    torch.cat([rb(lr[t:t+1][None]) for t in range(T)])

which launches the whole network T times with batch 1. RangeBranch.forward already takes [B, K, 3, H, W],
and the k1 checkpoint has K=1, so every frame is independent and the loop can collapse to one batched call.

Equivalence is not assumed: the batched output is compared against the looped output element-wise before
any timing is reported, because a silent mismatch here would corrupt both the range substitution and the
quota gate, which read the same estimate.
"""
import json
import time

import torch
import torch_npu  # noqa: F401

from h3_comfy import ROOT, sync

H, W = 480, 832
FRAMES = 22


def main():
    from range_branch import RangeBranch
    st = torch.load(ROOT / 'results/range_branch/k1.pt', map_location='cpu')
    K = st['cfg']['k']
    assert K == 1, f'batching over frames is only valid for K=1, got K={K}'
    rb = RangeBranch(**st['cfg']).eval()
    rb.load_state_dict(st['state'])
    dev = torch.device('npu', 0)
    rb = rb.to(dev)

    lr = torch.rand(FRAMES, 3, H // 4, W // 4, device=dev).clamp(0, 1)
    out = {}

    with torch.no_grad():
        # correctness first
        looped = torch.cat([rb(lr[t:t + 1][None]) for t in range(FRAMES)]).clamp(0, 1)
        batched = rb(lr[:, None]).clamp(0, 1)
        assert looped.shape == batched.shape, (looped.shape, batched.shape)
        diff = (looped - batched).abs()
        out['max_abs_diff'] = float(diff.max())
        out['mean_abs_diff'] = float(diff.mean())
        out['shape'] = list(batched.shape)
        assert out['max_abs_diff'] < 1e-4, f"batched output differs: {out['max_abs_diff']}"

        # warm up both paths
        for _ in range(3):
            torch.cat([rb(lr[t:t + 1][None]) for t in range(FRAMES)])
            rb(lr[:, None])
        sync()

        t0 = time.perf_counter()
        for _ in range(5):
            torch.cat([rb(lr[t:t + 1][None]) for t in range(FRAMES)]).clamp(0, 1)
        sync()
        out['looped_s'] = (time.perf_counter() - t0) / 5

        t0 = time.perf_counter()
        for _ in range(5):
            rb(lr[:, None]).clamp(0, 1)
        sync()
        out['batched_s'] = (time.perf_counter() - t0) / 5

    out['speedup'] = out['looped_s'] / out['batched_s']
    out['saved_s_per_block'] = out['looped_s'] - out['batched_s']
    out['saved_s_per_frame'] = out['saved_s_per_block'] / 17     # 17 new frames per block
    print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in out.items()}, indent=1))


if __name__ == '__main__':
    main()
