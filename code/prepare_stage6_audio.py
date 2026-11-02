"""Stage-6 data prep (PROTOCOL_stage6.md): ground-truth and degraded audio for the 12 Stage-2 eval clips.

Degradation parameters are drawn once per clip and reused for every block of that clip, mirroring the video
pipeline in degrade.py -- a streaming restorer must not see the degradation change under it mid-clip.
"""
import hashlib, json, subprocess, tempfile
from pathlib import Path
import numpy as np
# CPU-only: h3_comfy is deliberately not imported, it pulls in torch_npu and wants a card
ROOT = Path('/cache/zihaohe/streaming_sr0')

D = ROOT / 'data/stage2_eval'
OUT = ROOT / 'data/stage6_audio'
SR = 32000
SEED = 20261018


def ff(args, stdin=None):
    r = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', *args],
                       input=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode:
        raise RuntimeError('ffmpeg failed: ' + ' '.join(args) + '\n' + r.stderr.decode(errors='replace'))
    return r.stdout


def read_audio(src):
    raw = ff(['-i', str(src), '-map', '0:a', '-f', 'f32le', '-ac', '2', '-ar', str(SR), '-'])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def write_wav(path, x):
    pcm = np.clip(x.T, -1, 1).astype(np.float32).tobytes()
    ff(['-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0', '-c:a', 'pcm_s16le', '-y', str(path)], stdin=pcm)


def through_ffmpeg(x, filt=None, codec=None):
    """Round-trip raw float32 stereo through an ffmpeg filter and/or a lossy codec, back to float32."""
    pcm = np.clip(x.T, -1, 1).astype(np.float32).tobytes()
    base = ['-f', 'f32le', '-ar', str(SR), '-ac', '2', '-i', 'pipe:0']
    if codec is not None:
        name, kbps = codec
        # (muxer, demuxer, encoder, extra): AAC muxes as 'adts' but demuxes as 'aac'; this build has no
        # libmp3lame, and its native Opus encoder is flagged experimental, hence -strict -2 after -c:a.
        mux, demux, enc_name, extra = {'aac': ('adts', 'aac', 'aac', []),
                                       'opus': ('ogg', 'ogg', 'opus', ['-strict', '-2'])}[name]
        enc = ff([*base, *(['-af', filt] if filt else []), '-c:a', enc_name, *extra,
                  '-b:a', f'{kbps}k', '-f', mux, 'pipe:1'], stdin=pcm)
        raw = ff(['-f', demux, '-i', 'pipe:0', '-f', 'f32le', '-ac', '2', '-ar', str(SR), 'pipe:1'], stdin=enc)
    else:
        raw = ff([*base, *(['-af', filt] if filt else []), '-f', 'f32le', '-ac', '2', '-ar', str(SR), 'pipe:1'],
                 stdin=pcm)
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy()


def pink(n, rng):
    """1/f noise via spectral shaping; pink is the realistic case for room and device noise."""
    f = np.fft.rfftfreq(n, 1 / SR)
    s = np.ones_like(f); s[1:] = 1 / np.sqrt(f[1:])
    z = (rng.normal(size=f.shape) + 1j * rng.normal(size=f.shape)) * s
    y = np.fft.irfft(z, n)
    return y / max(float(np.abs(y).max()), 1e-12)


def degrade(x, rng):
    rec = {}
    y = x
    # 1) bandwidth limit: resample down and back up
    rate = int(rng.choice([8000, 12000, 16000, 24000]))
    rec['band_hz'] = rate
    y = through_ffmpeg(y, filt=f'aresample={rate},aresample={SR}')
    # 2) lossy codec
    name = str(rng.choice(['aac', 'opus']))
    kbps = int(rng.integers(16, 65))
    rec['codec'] = [name, kbps]
    y = through_ffmpeg(y, codec=(name, kbps))
    # 3) additive noise at a drawn SNR
    snr = float(rng.uniform(10, 30))
    colour = str(rng.choice(['white', 'pink']))
    n = np.stack([(rng.normal(size=y.shape[1]) if colour == 'white' else pink(y.shape[1], rng)) for _ in range(2)])
    sig = float(np.sqrt((y ** 2).mean())); nz = float(np.sqrt((n ** 2).mean()))
    y = y + n * (sig / max(nz, 1e-12)) * 10 ** (-snr / 20)
    rec |= {'snr_dB': snr, 'noise': colour}
    # 4) hard clipping at a fraction of the peak
    frac = float(rng.uniform(0.3, 0.9))
    peak = float(np.abs(y).max())
    y = np.clip(y, -frac * peak, frac * peak)
    rec['clip_frac'] = frac
    return y.astype(np.float32), rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    rows = []
    for i, name in enumerate(man['clips']):
        meta = json.loads((D / name / 'meta.json').read_text())
        rng = np.random.default_rng(SEED + i)
        gt = read_audio(meta['video'])
        lr, rec = degrade(gt, rng)
        second = bool(rng.random() < 0.30)          # 30% of clips get a second-order pass
        if second:
            lr, rec2 = degrade(lr, rng)
            rec['second_order'] = rec2
        n = min(gt.shape[1], lr.shape[1])
        gt, lr = gt[:, :n], lr[:, :n]
        d = OUT / name; d.mkdir(exist_ok=True)
        write_wav(d / 'gt.wav', gt); write_wav(d / 'lr.wav', lr)
        r = {'clip': name, 'samples': int(n), 'sample_rate': SR, 'second_order': second, 'degradation': rec,
             'gt_rms': float(np.sqrt((gt ** 2).mean())), 'lr_rms': float(np.sqrt((lr ** 2).mean())),
             'gt_sha256': hashlib.sha256((d / 'gt.wav').read_bytes()).hexdigest(),
             'lr_sha256': hashlib.sha256((d / 'lr.wav').read_bytes()).hexdigest()}
        rows.append(r)
        print(json.dumps(r), flush=True)
    (OUT / 'manifest.json').write_text(json.dumps({'n': len(rows), 'seed': SEED, 'clips': rows}, indent=1))


if __name__ == '__main__':
    main()
