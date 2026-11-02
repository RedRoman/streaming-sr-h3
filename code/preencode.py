"""Stage-1 pilot data: decode clips from /temp/h3_unified_dataset_20260918 (read-only), HR = 480p 832x480 22-frame
window at 24 fps, LR = clip-consistent degradation x4 (degrade.py), then the frozen H3 VAE encodes the HR target
latent and the official-preprocessed LR guide latent. Outputs go to our own /cache. Sharded across NPUs."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from degrade import degrade
from h3_comfy import H3, ROOT

MAN = Path('/temp/h3_unified_dataset_20260918/manifests')
T, H, W = 22, 480, 832


def read_window(path, start_s, rng_off):
    """Return uint8 [T,H0,W0,3] frames sampled at 24 fps starting at start_s seconds, or None."""
    c = av.open(str(path))
    vs = c.streams.video[0]
    want = [start_s + i / 24.0 for i in range(T)]
    frames, j = [], 0
    for fr in c.decode(video=0):
        ts = float(fr.pts * vs.time_base) if fr.pts is not None else None
        if ts is None:
            continue
        while j < T and ts >= want[j] - 0.5 / 24:
            frames.append(fr.to_ndarray(format='rgb24')); j += 1
        if j >= T:
            break
    c.close()
    return np.stack(frames) if len(frames) == T else None


def to_hr(frames, rng):
    """Downscale to height 480 (Lanczos, never upscale), random horizontal crop to 832."""
    h0, w0 = frames.shape[1:3]
    s = H / h0
    if h0 < H or round(w0 * s) < W:
        return None
    nw = round(w0 * s)
    x0 = int(rng.integers(0, nw - W + 1))
    out = [np.asarray(Image.fromarray(f).resize((nw, H), Image.LANCZOS))[:, x0:x0 + W] for f in frames]
    return np.stack(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--split', choices=['train', 'val'], required=True)
    p.add_argument('--windows', type=int, default=2)
    p.add_argument('--out', type=Path, default=ROOT / 'data/pilot_latents')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    rows = [json.loads(l) for l in open(MAN / f'eligible.{a.split}.jsonl')]
    rows.sort(key=lambda r: r['id'])
    if a.limit:
        rows = rows[:a.limit]
    rows = [r for i, r in enumerate(rows) if i % a.nshards == a.shard]
    out = a.out / a.split
    out.mkdir(parents=True, exist_ok=True)
    h3 = H3(load_dit=False)
    n_ok = n_skip = 0
    t0 = time.time()
    for r in rows:
        for w in range(a.windows):
            name = f"{r['id']}__w{w}"
            if (out / f'{name}.json').exists():
                continue
            seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            dur = float(r.get('duration') or 0)
            if dur < T / 24 + 0.2:
                n_skip += 1; continue
            # windows spread over the clip, deterministic jitter
            span = dur - T / 24 - 0.1
            start = span * (w + rng.uniform(0.1, 0.9)) / a.windows
            try:
                frames = read_window(r['video_path'], start, rng)
            except Exception as e:  # corrupt stream etc.
                frames = None
            hr = to_hr(frames, rng) if frames is not None else None
            if hr is None:
                n_skip += 1; continue
            lr, dp = degrade(hr, rng)
            hr_t = torch.from_numpy(hr).float() / 255
            low = torch.from_numpy(lr).permute(0, 3, 1, 2).float() / 255
            with torch.no_grad():
                target = h3.encode(hr_t).cpu().float().contiguous()
                guide = h3.encode_guide(low, [H, W]).float().contiguous()
            assert target.shape == guide.shape == (1, 24, 7, H // 16, W // 16), (target.shape, guide.shape)
            save_file({'target': target, 'guide': guide, 'lr_u8': torch.from_numpy(lr).contiguous()}, str(out / f'{name}.safetensors'))
            (out / f'{name}.json').write_text(json.dumps({'id': name, 'source_id': r['id'], 'source': r.get('source'), 'split': a.split,
                                                          'video_sha256': r.get('video_sha256'), 'start_s': start, 'degradation': dp,
                                                          'source_group': r.get('source_group'), 'audio_present': r.get('audio_present')}))
            n_ok += 1
        if n_ok and n_ok % 50 == 0:
            print(json.dumps({'shard': a.shard, 'ok': n_ok, 'skip': n_skip, 'sec': round(time.time() - t0)}), flush=True)
    print(json.dumps({'shard': a.shard, 'ok': n_ok, 'skip': n_skip, 'sec': round(time.time() - t0), 'done': True}), flush=True)


if __name__ == '__main__':
    main()
