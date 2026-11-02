"""Training pairs for the range branch (PROTOCOL_mech.md, mech7). CPU only; the diffusion model is not used.

The range branch has one job: from the degraded observation y, estimate D(x) -- the clean low-resolution
image the evaluation operator would produce from the ground truth. mech5 measured that 43.5% of the
restorer's error lives in exactly that subspace, and mech5b measured that substituting the raw y recovers
only 39% of it because y is itself noisy, blurred and compressed.

Both tensors live at 1/4 resolution (208x120 for our 832x480 crops), so this is a sixteenth of the pixels
of the full problem and needs no generative prior.

Target operator note: D must be the *evaluation* operator, torch bicubic with antialias=True, because that
is what defines the range/null split everywhere else in this project. degrade.py's Pillow-BICUBIC path is a
different kernel and is only ever used to build the input.
"""
import argparse, hashlib, json, os, time
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F

from degrade import degrade

ROOT = Path('/cache/zihaohe/streaming_sr0')
MAN = Path('/temp/h3_unified_dataset_20260918/manifests')
OUT = Path(os.environ.get('RANGE_OUT', str(ROOT / 'data/range_pairs')))
# the x4 set lives on the shared /temp mount which is 91% full; other scales go to /cache
T, H, W = 22, 480, 832
SCALE = int(os.environ.get('SR_SCALE', 4))


def read_window(path, start_s):
    """uint8 [T,H0,W0,3] at 24 fps from start_s, or None. Same protocol as preencode.py."""
    try:
        c = av.open(str(path))
    except Exception:
        return None
    vs = c.streams.video[0]
    want = [start_s + i / 24.0 for i in range(T)]
    frames, j = [], 0
    try:
        for fr in c.decode(video=0):
            ts = float(fr.pts * vs.time_base) if fr.pts is not None else None
            if ts is None:
                continue
            while j < T and ts >= want[j] - 0.5 / 24:
                frames.append(fr.to_ndarray(format='rgb24')); j += 1
            if j >= T:
                break
    finally:
        c.close()
    return np.stack(frames) if len(frames) == T else None


def centre_crop(fr):
    h, w = fr.shape[1:3]
    if h < H or w < W:
        return None
    y0, x0 = (h - H) // 2, (w - W) // 2
    return fr[:, y0:y0 + H, x0:x0 + W]


def eval_down(hr_u8):
    """D: the evaluation operator. uint8 [T,H,W,3] -> uint8 [T,H/4,W/4,3]."""
    t = torch.from_numpy(hr_u8).float().permute(0, 3, 1, 2) / 255
    d = F.interpolate(t.double(), size=(H // SCALE, W // SCALE), mode='bicubic',
                      align_corners=False, antialias=True).clamp(0, 1)
    return (d.permute(0, 2, 3, 1).float().numpy() * 255 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--clips', type=int, default=400)
    ap.add_argument('--windows', type=int, default=2, help='windows per clip, at different offsets')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    d = OUT / a.split
    d.mkdir(parents=True, exist_ok=True)

    # the train manifest contains at least one truncated line; skip rather than abort the whole shard
    rows, bad = [], 0
    for l in (MAN / f'eligible.{a.split}.jsonl').read_text().splitlines():
        if not l.strip():
            continue
        try:
            rows.append(json.loads(l))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(json.dumps({'manifest_lines_skipped': bad}), flush=True)
    # keep the Stage-2 evaluation clips out of training
    held = set(json.loads((ROOT / 'data/stage2_eval/manifest.json').read_text())['clips'])
    rows = [r for r in rows if r['id'] not in held and r.get('original_id') not in held][:a.clips]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]

    made, skipped, t0 = 0, 0, time.perf_counter()
    for r in rows:
        dur = float(r.get('duration', 0) or 0)
        for w in range(a.windows):
            out = d / f"{r['id']}_w{w}.npz"
            if out.exists():
                made += 1
                continue
            rng = np.random.default_rng(abs(hash((r['id'], w))) % (2 ** 32))
            start = float(rng.uniform(0.5, max(0.6, dur - T / 24.0 - 0.5))) if dur > 2 else 0.0
            fr = read_window(r['video_path'], start)
            if fr is None:
                skipped += 1
                continue
            hr = centre_crop(fr)
            if hr is None:
                skipped += 1
                continue
            lr, rec = degrade(hr, rng, SCALE)
            tgt = eval_down(hr)
            if lr.shape != tgt.shape:
                skipped += 1
                continue
            np.savez_compressed(out, y=lr, target=tgt,
                                meta=json.dumps({'id': r['id'], 'start_s': start, 'degradation': rec}))
            made += 1
        if made % 50 == 0 and made:
            print(json.dumps({'made': made, 'skipped': skipped,
                              'sec': round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps({'split': a.split, 'shard': a.shard, 'made': made, 'skipped': skipped,
                      'sec': round(time.perf_counter() - t0, 1), 'done': True}), flush=True)


if __name__ == '__main__':
    main()
