"""Audio pairs for the omni stage, on the same real clips the video stages use.

The project's target is omni restoration and the audio side has produced nothing so far. mech1 showed why
it cannot be prepared on the material we had: on generated soundtracks the audio/video event correlation
flips sign with content, so no audio claim survives there. Every clip here is a real recording with a real
track, selected by the same filter as the video data.

Degradation mirrors degrade.py's structure -- parameters drawn once per clip, so a streaming restorer never
sees the operator change under it mid-clip -- using the chain measured as realistic for audio: bandwidth
limit, lossy codec, additive noise, clipping, with a second-order pass on some clips.

Targets are stored at the H3 audio VAE's native 32 kHz, so no resampling sits between the data and the model.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path

import numpy as np

ROOT = Path('/cache/zihaohe/streaming_sr0')
CAND = ROOT / 'data/real_av/candidates_big.json'
SRC = ROOT / 'data/real_av/_cand'
OUT = ROOT / 'data/stage8_audio'
SR = 32000
DUR = 124 / 24.0
SEED = 20261020


def ff(args, stdin=None):
    r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', *args],
                       input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError(' '.join(args) + '\n' + r.stderr.decode(errors='replace')[:300])
    return r.stdout


def read(path, start):
    raw = ff(['-ss', f'{start}', '-i', str(path), '-map', '0:a', '-t', f'{DUR}',
              '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'])
    a = np.frombuffer(raw, dtype=np.float32)
    return a.reshape(-1, 2).T.copy() if a.size > SR else None


def through(x, filt=None, codec=None):
    pcm = np.clip(x.T, -1, 1).astype(np.float32).tobytes()
    base = ['-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0']
    if codec:
        # AAC muxes as 'adts' but demuxes as 'aac'; this build has no libmp3lame and its native Opus
        # encoder is flagged experimental, hence -strict -2.
        name, kbps = codec
        mux, demux, enc, extra = {'aac': ('adts', 'aac', 'aac', []),
                                  'opus': ('ogg', 'ogg', 'opus', ['-strict', '-2'])}[name]
        blob = ff([*base, *(['-af', filt] if filt else []), '-c:a', enc, *extra,
                   '-b:a', f'{kbps}k', '-f', mux, 'pipe:1'], stdin=pcm)
        raw = ff(['-f', demux, '-i', 'pipe:0', '-f', 'f32le', '-ac', '2', '-ar', str(SR), 'pipe:1'], stdin=blob)
    else:
        raw = ff([*base, *(['-af', filt] if filt else []), '-f', 'f32le', '-ac', '2', '-ar', str(SR), 'pipe:1'],
                 stdin=pcm)
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def pink(n, rng):
    f = np.fft.rfftfreq(n, 1 / SR)
    s = np.ones_like(f); s[1:] = 1 / np.sqrt(f[1:])
    z = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) * s
    y = np.fft.irfft(z, n)
    return y / max(float(np.abs(y).max()), 1e-12)


def degrade(x, rng):
    rec = {}
    rate = int(rng.choice([8000, 12000, 16000, 24000])); rec['band_hz'] = rate
    y = through(x, filt=f'aresample={rate},aresample={SR}')
    name = str(rng.choice(['aac', 'opus'])); kbps = int(rng.integers(16, 65))
    rec['codec'] = [name, kbps]
    y = through(y, codec=(name, kbps))
    snr = float(rng.uniform(10, 30)); colour = str(rng.choice(['white', 'pink']))
    n = np.stack([(rng.normal(size=y.shape[1]) if colour == 'white' else pink(y.shape[1], rng)) for _ in range(2)])
    sig, nz = float(np.sqrt((y ** 2).mean())), float(np.sqrt((n ** 2).mean()))
    y = y + n * (sig / max(nz, 1e-12)) * 10 ** (-snr / 20)
    rec |= {'snr_dB': snr, 'noise': colour}
    frac = float(rng.uniform(0.3, 0.9)); peak = float(np.abs(y).max())
    y = np.clip(y, -frac * peak, frac * peak); rec['clip_frac'] = frac
    return y.astype(np.float32), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=290)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    sel = json.loads(CAND.read_text())['selected'][:a.clips]
    made, skipped = 0, 0
    for i, c in enumerate(sel):
        if i % a.nshards != a.shard:
            continue
        split = 'val' if i % 12 == 0 else 'train'
        d = OUT / split
        d.mkdir(parents=True, exist_ok=True)
        name = Path(c['file']).stem
        out = d / f'{name}.npz'
        if out.exists():
            made += 1
            continue
        rng = np.random.default_rng(SEED + i)
        start = max(0.0, (float(c['duration']) - DUR) / 2)
        gt = read(SRC / c['file'], start)
        if gt is None:
            skipped += 1
            continue
        lr, rec = degrade(gt, rng)
        if rng.random() < 0.30:
            lr, rec2 = degrade(lr, rng)
            rec['second_order'] = rec2
        n = min(gt.shape[1], lr.shape[1])
        np.savez_compressed(out, gt=gt[:, :n].astype(np.float32), lr=lr[:, :n].astype(np.float32),
                            meta=json.dumps({'id': name, 'start_s': start, 'split': split,
                                             'sample_rate': SR, 'degradation': rec}))
        made += 1
    print(json.dumps({'shard': a.shard, 'made': made, 'skipped': skipped, 'done': True}), flush=True)


if __name__ == '__main__':
    main()
