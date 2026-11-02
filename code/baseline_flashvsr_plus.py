"""FlashVSR+ (arXiv 2610.17552) on our evaluation protocol.

The last published baseline missing from this project. FlashVSR+ is nine days old at the time of writing
and is the successor to the FlashVSR arm already measured here, which turned out to be the stronger of
the two published baselines on the held-out set (LPIPS 0.2253 against InfVSR's 0.2344). Whether its two
changes -- hybrid local/sparse attention and reward-weighted distillation -- move quality as well as speed
is the open question its own paper does not answer on real degraded footage.

Run through their own CLI rather than a reimplementation, so the arm is the published method:
  --attn_kernel sdpa    their Triton kernel cannot build on Ascend; their package already ships an exact
                        SDPA expansion of the same block mask, selected here rather than written by us
  --hardware-profile    bypasses the CUDA capability probe
The turbo variant is the one-step model, which is what compares against our four-step configuration.

Input is written as a video because the CLI takes files; frames are read back and scored by deploy_eval
exactly like every other arm, so the comparison stays on one ruler.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
REPO = Path('/cache/zihaohe/baselines/FlashVSR_Plus')
W = Path('/cache/zihaohe/baselines/flashvsr_plus_weights')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
# one output tree per scale: the x4 tree is the published-baseline arm and must not be mixed with
# off-design-point runs at x2/x8
OUT = ROOT / ('results/baseline_flashvsr_plus' + os.environ.get('FVP_OUT_SUFFIX', ''))
VARIANT = os.environ.get('FVP_VARIANT', 'turbo')
# FlashVSR+ validates that the geometry divides by 128, and 480x832 does not. The LR is replicate-padded
# 120x208 -> 128x224 so the x4 output lands on 512x896, then cropped back to the evaluation geometry.
# Same padding the FlashVSR arm already uses, so the two baselines see identical pixels.
# The output geometry is the same 480x832 for every operator in our eval sets -- only the LR side
# changes -- so running this x4 method at x2 or x8 means feeding a different input resolution and asking
# for the same output. That is out of its design point, and that is exactly the measurement that says
# whether a scale-conditioned estimator is worth having: a fixed-scale baseline used off its scale is
# the honest alternative to one model covering the family.
#
# run_stream requires both input and output sides divisible by 128, so the pad is per scale: at x2 the LR
# is 416x240 -> 512x256 (output 1024x512, cropped back to 480x832); at x8 it is 104x60 -> 128x64.
SCALE = int(os.environ.get('FVP_SCALE', '4'))
PAD_LR = {4: (8, 8, 4, 4), 2: (48, 48, 8, 8), 8: (12, 12, 2, 2)}[SCALE]


def write_mp4(frames, path, fps=24):
    import av
    with av.open(str(path), 'w') as c:
        s = c.add_stream('libx264', rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], 'yuv420p'
        s.options = {'crf': '10', 'preset': 'veryfast'}      # near-lossless: the arm must see our LR
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format='rgb24')):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def read_mp4(path):
    import av
    with av.open(str(path)) as c:
        return np.stack([f.to_ndarray(format='rgb24') for f in c.decode(video=0)])


def main():
    names = os.environ.get('CLIPS', '').split(',') if os.environ.get('CLIPS') else \
        json.loads((D_ / 'manifest.json').read_text())['clips']
    shard, nsh = int(os.environ.get('SHARD', 0)), int(os.environ.get('NSHARDS', 1))
    names = [c for i, c in enumerate(names) if i % nsh == shard]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = ROOT / 'tmp' / f'fvp{shard}'
    tmp.mkdir(parents=True, exist_ok=True)

    for name in names:
        d = OUT / name
        if (d / 'result.json').exists():
            continue
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        lr = np.stack([np.asarray(Image.open(D_ / name / 'lr' / f'{t:06d}.png').convert('RGB'))
                       for t in range(T)])
        l, r_, t_, b = PAD_LR
        lr = np.pad(lr, ((0, 0), (t_, b), (l, r_), (0, 0)), mode='edge')
        src, dst = tmp / 'in.mp4', tmp / 'out.mp4'
        for p in (src, dst):
            if p.exists():
                p.unlink()
        write_mp4(lr, src)
        # run_stream.py takes a different option set from cli.py: --mode selects the variant,
        # --ckpt-dir points at the weight tree, and the OUTPUT geometry is required rather than inferred.
        H0, W0 = meta['hr_wh'][1], meta['hr_wh'][0]
        H, W_ = lr.shape[1] * SCALE, lr.shape[2] * SCALE        # 512 x 896
        cmd = [sys.executable, str(ROOT / 'code' / 'fvp_run_stream.py'),
               '--input', str(src), '--output', str(dst),
               '--height', str(H), '--width', str(W_),
               # --ckpt-dir is the VARIANT directory: presets.py joins dit.safetensors and lq_proj_in
               # onto it directly, while TCDecoder and the prompt tensor resolve relative to the
               # repo (symlinked into REPO/ckpts).
               '--mode', VARIANT, '--ckpt-dir', str(W / VARIANT),
               '--attn-kernel', 'sdpa', '--hardware-profile', 'generic',
               # NHWC storage for the TinyDecoder convolutions is a CUDA layout optimisation;
               # `x.contiguous(memory_format=torch.channels_last)` raises "NPU contiguous operator only
               # supported contiguous memory format" on Ascend. Their own CLI exposes the switch, so the
               # arm stays the published code path and only the convolution storage order changes -- the
               # values the decoder computes are unaffected.
               '--no-tiny-decoder-channels-last',
               '--device', 'npu:0']
        t0 = time.perf_counter()
        r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
        sec = time.perf_counter() - t0
        if r.returncode != 0 or not dst.exists():
            (OUT / f'FAILED_{name}.log').write_text((r.stdout or '')[-4000:] + '\n---\n'
                                                    + (r.stderr or '')[-8000:])
            print(json.dumps({'clip': name, 'failed': True, 'rc': r.returncode}), flush=True)
            continue
        hr = read_mp4(dst)
        y0, x0 = PAD_LR[2] * SCALE, PAD_LR[0] * SCALE
        hr = hr[:, y0:y0 + H0, x0:x0 + W0]                      # back to the evaluation geometry
        assert hr.shape[1:3] == (H0, W0), hr.shape
        d.mkdir(parents=True, exist_ok=True)
        n = min(len(hr), T)
        for t in range(n):
            Image.fromarray(hr[t]).save(d / f'{t:06d}.png')
        (d / 'result.json').write_text(json.dumps(
            {'clip': name, 'variant': VARIANT, 'frames': n, 'seconds': sec,
             'out_hw': [H0, W0], 'model_hw': [H, W_], 'pad_lr_lrtb': list(PAD_LR),
             'attn_kernel': 'sdpa (their own exact expansion; Triton unavailable on Ascend)',
             'out_shape': list(hr.shape)}, indent=1))
        print(json.dumps({'clip': name, 'frames': n, 'shape': list(hr.shape[1:3]),
                          'sec': round(sec, 1)}), flush=True)


if __name__ == '__main__':
    main()
