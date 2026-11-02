"""Stage-7 pre-encoding: latents for residual-parameterised training, on real footage.

Two guides are encoded per window, so the ceiling of the idea can be measured before R_phi is good enough
to supply it:

  guide_raw     E(U(lr))   -- what Stage 1 trained on: the degraded observation, upsampled
  guide_oracle  E(U(dx))   -- the same pipeline fed the true D(x). The range branch can never beat this,
                              so training on it bounds what residual parameterisation can buy. If the
                              bound is small the direction is not worth pursuing, and that is cheaper to
                              find out now than after R_phi converges.

Targets are the HR latents, unchanged from Stage 1, so the two runs differ in exactly one input.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from h3_comfy import H3, ROOT

DATA = ROOT / 'data/stage7_train'
OUT = ROOT / 'data/stage7_latents'
H, W = 480, 832


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    files = sorted((DATA / a.split).glob('*.npz'))
    files = [f for i, f in enumerate(files) if i % a.nshards == a.shard]
    d = OUT / a.split
    d.mkdir(parents=True, exist_ok=True)
    h3 = H3(load_dit=False)                       # VAE only: no DiT needed to encode
    done, t0 = 0, time.perf_counter()
    for f in files:
        out = d / (f.stem + '.safetensors')
        if out.exists():
            done += 1
            continue
        z = np.load(f)
        hr = torch.from_numpy(z['hr']).float() / 255                 # [22,H,W,3]
        lr = torch.from_numpy(z['lr']).float().permute(0, 3, 1, 2) / 255
        dx = torch.from_numpy(z['dx']).float().permute(0, 3, 1, 2) / 255
        with torch.no_grad():
            target = h3.encode(hr).float().contiguous()
            g_raw = h3.encode_guide(lr, [H, W]).float().contiguous()
            g_oracle = h3.encode_guide(dx, [H, W]).float().contiguous()
        assert target.shape == g_raw.shape == g_oracle.shape, (target.shape, g_raw.shape)
        save_file({'target': target, 'guide_raw': g_raw, 'guide_oracle': g_oracle},
                  str(out), metadata={'meta': str(z['meta'])})
        done += 1
        if done % 20 == 0:
            print(json.dumps({'shard': a.shard, 'done': done, 'of': len(files),
                              'sec': round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps({'shard': a.shard, 'done': done, 'of': len(files), 'finished': True,
                      'sec': round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
