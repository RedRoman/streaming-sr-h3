"""Stage-7 training data: real footage, all-degradation, at both resolutions the system needs.

Why real footage replaces the generated clips we trained Stage 1 on: measured during evaluation-set
construction, generated video carries roughly 20x less energy in the band our task has to reproduce
(band_ratio 0.005 vs 0.09-0.10 on the sharpest real clips). Training and evaluating on it made the whole
pipeline look better than it is -- on real footage our LPIPS falls behind a 2021 baseline.

Each window produces three aligned tensors:
  hr      832x480 uint8, the target for the H3 branch
  lr      208x120 uint8, the degraded observation (degrade.py, clip-consistent parameters)
  dx      208x120 uint8, D(hr) under the evaluation operator -- the range branch's target, and the
          range component the H3 branch no longer has to reproduce under residual parameterisation
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import degrade as DG

ROOT = Path('/cache/zihaohe/streaming_sr0')
CAND = ROOT / 'data/real_av/candidates_big.json'
SRC = ROOT / 'data/real_av/_cand'
OUT = ROOT / 'data/stage7_train'
T, H, W = 22, 480, 832
FPS, SCALE = 24.0, 4
SEED = 20261020


def read_window(path, start_s, n=T):
    r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start_s}', '-i', str(path),
                        '-vf', f'fps={FPS},crop={W}:{H}', '-frames:v', str(n),
                        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    if a.size < n * H * W * 3:
        return None
    return a[:n * H * W * 3].reshape(n, H, W, 3).copy()


def eval_down(hr_u8):
    t = torch.from_numpy(hr_u8).float().permute(0, 3, 1, 2) / 255
    d = F.interpolate(t.double(), size=(H // SCALE, W // SCALE), mode='bicubic',
                      align_corners=False, antialias=True).clamp(0, 1)
    return (d.permute(0, 2, 3, 1).float().numpy() * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=290)
    ap.add_argument('--windows', type=int, default=3)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--val-frac', type=float, default=0.08)
    a = ap.parse_args()
    sel = json.loads(CAND.read_text())['selected'][:a.clips]
    made, skipped = 0, 0
    for i, c in enumerate(sel):
        if i % a.nshards != a.shard:
            continue
        split = 'val' if (i % max(2, int(1 / a.val_frac))) == 0 else 'train'
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        name = Path(c['file']).stem
        for w in range(a.windows):
            out = d / f'{name}_w{w}.npz'
            if out.exists():
                made += 1
                continue
            rng = np.random.default_rng(SEED + i * 17 + w)
            dur = float(c['duration'])
            span = max(0.2, dur - T / FPS - 0.3)
            start = float(rng.uniform(0.1, 0.1 + span)) if dur > 2 else 0.0
            hr = read_window(SRC / c['file'], start)
            if hr is None:
                skipped += 1
                continue
            lr, params = DG.degrade(hr, rng)
            dx = eval_down(hr)
            if lr.shape != dx.shape:
                skipped += 1
                continue
            np.savez_compressed(out, hr=hr, lr=lr, dx=dx,
                                meta=json.dumps({'id': name, 'start_s': start, 'split': split,
                                                 'band_ratio': c['band_ratio'], 'degradation': params}))
            made += 1
        if made % 40 == 0 and made:
            print(json.dumps({'made': made, 'skipped': skipped}), flush=True)
    print(json.dumps({'shard': a.shard, 'made': made, 'skipped': skipped, 'done': True}), flush=True)


if __name__ == '__main__':
    main()
