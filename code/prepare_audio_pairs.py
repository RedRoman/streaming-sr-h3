"""Audio latent pairs: (degraded, clean) in the audio VAE's latent space.

Designed to mirror the video side rather than to invent a second architecture. The video contribution is
innerDC projecting the sampler onto R_phi(y), where R_phi is a small network trained standalone on
(observation, D(x)) pairs -- no DiT training at all. The audio analogue is the same shape: a small branch
on (degraded latent, clean latent) pairs, trained standalone, which the sampler's audio stream can then be
conditioned on. That keeps the DiT frozen, which matters because train_stage1 gives the audio stream
"noise input, no loss" and changing that is a full retrain.

The gate that justified building this at all, measured in degrade_audio.py over 8 held-out clips:

    degradation damage   mel 26.543 dB      the signal to recover
    codec damage         mel  2.768 dB      what the latent cannot express
    ratio                9.59x              -> worth training

and the no-training baseline a trained branch has to beat is the codec round trip of the degraded audio,
mel 16.34 dB, because the VAE already recovers 43% of the gap by pulling toward its own prior.

Windows match the video pairs' protocol: 22 frames at 24 fps from the same start, so an audio pair and a
video pair describe the same instant and a joint model could later use both. The audio VAE downscales by
800 samples, so a 22-frame window is 29333 samples and 37 latent frames, which is exactly the audio_t
h3_comfy.sample computes for one block.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
MAN = Path(os.environ.get('RANGE_MAN', '/temp/h3_unified_dataset_20260918/manifests'))
SRC_PREFIX = os.environ.get('RANGE_SRC_PREFIX', '')
OUT = Path(os.environ.get('AUDIO_OUT', '/cache/zihaohe/audio_pairs'))
SR, FPS, T = 32000, 24, 22


def resolve(p):
    if SRC_PREFIX and str(p).startswith('/temp/'):
        return SRC_PREFIX.rstrip('/') + '/' + str(p)[len('/temp/'):]
    return str(p)


def read_audio(path, start_s, dur_s):
    """[2,N] float32 at SR, or None. Same ffmpeg protocol prepare_real_eval uses for audio_gt."""
    try:
        raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-ss', str(start_s),
                              '-t', str(dur_s), '-i', str(path), '-f', 'f32le',
                              '-ac', '2', '-ar', str(SR), '-'],
                             stdout=subprocess.PIPE, timeout=180).stdout
    except Exception:
        return None
    if len(raw) < 4 * 2 * int(SR * dur_s * 0.9):
        return None
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='train', choices=['train', 'val'])
    ap.add_argument('--clips', type=int, default=2182)
    ap.add_argument('--windows', type=int, default=2)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()

    import torch_npu  # noqa: F401
    import h3_comfy  # noqa: F401
    from degrade_audio import degrade_audio
    import comfy.sd
    import comfy.utils

    dev = torch.device('npu:0')
    _c = torch.zeros(1 << 28, device=dev); del _c
    sd, meta = comfy.utils.load_torch_file(
        str(ROOT / 'assets/comfy_h3/minimax_h3_audio_vae_fp32.safetensors'), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, device=dev, dtype=torch.float32, metadata=meta)
    vae.first_stage_model.eval().requires_grad_(False)

    def enc(w):
        with torch.no_grad():
            return vae.encode(w[None].movedim(1, -1).to(dev)).float().cpu()

    d = OUT / a.split
    d.mkdir(parents=True, exist_ok=True)
    held = set()
    hp = ROOT / 'data/stage2_eval/manifest.json'
    if hp.exists():
        held = set(json.loads(hp.read_text())['clips'])
    lines = (MAN / f'eligible.{a.split}.jsonl').read_text().splitlines()[:a.clips]
    n_ok = n_skip = 0
    for i, ln in enumerate(lines):
        if i % a.nshards != a.shard:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        if r.get('id') in held or not r.get('audio_present'):
            n_skip += 1; continue
        src = resolve(r['video_path'])
        if not Path(src).exists():
            n_skip += 1; continue
        for w in range(a.windows):
            name = '%s_w%d' % (r['id'], w)
            f = d / f'{name}.npz'
            if f.exists():
                n_ok += 1; continue
            start = 1.0 + w * (T / FPS)
            au = read_audio(src, start, T / FPS)
            if au is None:
                n_skip += 1; continue
            # Python's str hash is salted per process (PYTHONHASHSEED), so abs(hash(name)) gave a
            # different draw in every shard of the same run and on every host. The pairs were therefore
            # same-distribution but not reproducible, which was measured rather than assumed: the first
            # train window drew lowpass 7308.1/snr 19.91/clip 0.9917/aac 48 on one host and
            # 7310.3/15.02/0.9108/64 on the other. sha256 of the name is stable everywhere.
            seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
            rng = np.random.default_rng(seed)
            dg, p = degrade_audio(au, rng)
            zc, zd = enc(au), enc(dg)
            np.savez_compressed(
                f, clean=zc.numpy().astype(np.float32), degraded=zd.numpy().astype(np.float32),
                meta=json.dumps({'id': r['id'], 'window': w, 'start_s': start, 'sr': SR,
                                 'frames': T, 'degradation': p,
                                 'latent_shape': list(zc.shape)}))
            n_ok += 1
            if n_ok % 25 == 0:
                print(json.dumps({'shard': a.shard, 'ok': n_ok, 'skip': n_skip,
                                  'latent': list(zc.shape)}), flush=True)
    print(json.dumps({'shard': a.shard, 'done': n_ok, 'skipped': n_skip,
                      'dir': str(d), 'files': len(list(d.glob('*.npz')))}), flush=True)


if __name__ == '__main__':
    main()
