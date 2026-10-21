"""Build a real-recording audio-visual evaluation pool (for Stage 6 / E2 / the mech1 retest).

Everything with an audio track on this cluster is model-generated, and mech1 showed that matters: on
generated soundtracks the audio-video event correlation flips sign with content. Any claim about audio
needs real recordings.

Source: AudioSet balanced segments that are mirrored as actual mp4 (10 s YouTube excerpts, real diegetic
audio, mixed quality).

Container resolution is not enough to accept a clip. LaSOT taught us that a nominally-720p source can be a
compressed upload whose ground truth is already soft, which makes a restorer's correct detail score as
error. So every candidate is scored on the band our task actually has to reproduce -- the unobserved band
N(x) = x - U(D(x)) -- and must clear the bar set by the VIDGEN clips we already accepted. The same logic is
applied to audio: a clip whose soundtrack is already low-passed below the degradation cutoff has nothing
above the cutoff to restore, so it is useless as an audio reference.
"""
import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
REPO = 'chikuwa-AI/AudioSet_balanced_video'
BASE = f'https://hf-mirror.com/datasets/{REPO}/resolve/main/'
API = f'https://hf-mirror.com/api/datasets/{REPO}/tree/main/'
OUT = ROOT / 'data/real_av'
SR = 32000
NEED_FRAMES = 124          # same clip length as every other stage
FPS = 24.0


def sh(cmd, **kw):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)


def probe(p, stream, fields):
    r = sh(['ffprobe', '-v', 'error', '-select_streams', stream, '-show_entries',
            f'stream={",".join(fields)}', '-of', 'csv=p=0', str(p)])
    return r.stdout.decode().strip()


def duration(p):
    r = sh(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', str(p)])
    try:
        return float(r.stdout.decode().strip())
    except ValueError:
        return 0.0


def frames_at(p, n, w, h, start=0.0):
    """Decode n frames at 24 fps, centre-cropped to w x h, as float [n,h,w,3]."""
    r = sh(['ffmpeg', '-nostdin', '-v', 'error', '-ss', f'{start}', '-i', str(p),
            '-vf', f'fps={FPS},crop={w}:{h}', '-frames:v', str(n), '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-'])
    a = np.frombuffer(r.stdout, dtype=np.uint8)
    if a.size < n * h * w * 3:
        return None
    return torch.from_numpy(a[:n * h * w * 3].reshape(n, h, w, 3).copy()).float() / 255


def band_ratio(x, s=4):
    """Energy the x s observation does not determine, relative to the frame: our sharpness bar."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    n = (t - u)
    return float(n.square().mean() / max(float(t.square().mean()), 1e-12))


def audio_stats(p):
    r = sh(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(p), '-map', '0:a', '-f', 'f32le',
            '-ac', '1', '-ar', str(SR), '-'])
    w = np.frombuffer(r.stdout, dtype=np.float32)
    if w.size < SR:
        return None
    x = torch.from_numpy(w.copy())
    S = torch.stft(x, 2048, 800, window=torch.hann_window(2048), return_complex=True).abs()
    p_tot = float(S.square().mean())
    k8 = int(8000 / (SR / 2) * (S.shape[0] - 1))       # energy above 8 kHz: what a 16 kHz observation loses
    return {'rms': float(x.square().mean().sqrt()),
            'above_8k_ratio': float(S[k8:].square().mean() / max(p_tot, 1e-12))}


BAR_MEASURED = 0.00504   # median band_ratio of the VIDGEN clips this project already accepted


def reference_bar():
    """The sharpness bar, measured once on the accepted VIDGEN clips (median 0.00504, range 0.0023-0.019).

    Recomputing it needs data/stage2_eval, which only exists on the machine where it was first built; the
    value is a property of the accepted set, not of the host, so it is recorded here instead."""
    D = ROOT / 'data/stage2_eval'
    if not (D / 'manifest.json').exists():
        return BAR_MEASURED, []
    D = ROOT / 'data/stage2_eval'
    man = json.loads((D / 'manifest.json').read_text())
    from PIL import Image
    vals = []
    for name in man['clips'][:6]:
        fr = torch.stack([torch.from_numpy(np.asarray(Image.open(D / name / 'hr' / f'{t:06d}.png').convert('RGB')).copy())
                          for t in (0, 40, 80)]).float() / 255
        vals.append(band_ratio(fr))
    return float(np.median(vals)), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', type=int, default=3)
    ap.add_argument('--per-shard', type=int, default=120)
    ap.add_argument('--keep', type=int, default=24)
    ap.add_argument('--min-width', type=int, default=1280)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / '_cand'; tmp.mkdir(exist_ok=True)

    bar, ref = reference_bar()
    print(json.dumps({'vidgen_band_ratio_median': bar, 'vidgen_samples': [round(v, 5) for v in ref]}), flush=True)

    paths = []
    for s in range(int(os.environ.get('SHARD_START','0')), int(os.environ.get('SHARD_START','0'))+a.shards):
        r = sh(['curl', '-s', '--max-time', '40', f'{API}balanced_train_segments/{s:05d}?recursive=true'])
        try:
            paths += [x['path'] for x in json.loads(r.stdout) if x.get('type') == 'file' and x['path'].endswith('.mp4')][:a.per_shard]
        except Exception as e:
            print('listing failed', s, e, flush=True)
    print(json.dumps({'candidates_listed': len(paths)}), flush=True)

    rows = []
    for i, rel in enumerate(paths):
        f = tmp / Path(rel).name
        if not f.exists():
            sh(['curl', '-sL', '--max-time', '120', '-o', str(f), BASE + rel])
        if f.stat().st_size < 200_000:
            f.unlink(missing_ok=True); continue
        v = probe(f, 'v:0', ['width', 'height'])
        au = probe(f, 'a:0', ['codec_name', 'sample_rate'])
        if not v or not au:
            f.unlink(missing_ok=True); continue
        try:
            W, H = (int(z) for z in v.split(',')[:2])
        except ValueError:
            f.unlink(missing_ok=True); continue
        dur = duration(f)
        if W < a.min_width or H < 720 or dur < NEED_FRAMES / FPS + 0.5:
            f.unlink(missing_ok=True); continue
        fr = frames_at(f, 6, 832, 480, start=max(0.0, (dur - NEED_FRAMES / FPS) / 2))
        if fr is None:
            f.unlink(missing_ok=True); continue
        st = audio_stats(f)
        if st is None:
            f.unlink(missing_ok=True); continue
        br = band_ratio(fr)
        rows.append({'file': f.name, 'rel': rel, 'wh': [W, H], 'duration': dur,
                     'band_ratio': br, 'sharp_ok': br >= bar, **st})
        print(json.dumps({**rows[-1], 'band_ratio': round(br, 5),
                          'above_8k_ratio': round(st['above_8k_ratio'], 6)}), flush=True)

    ok = [r for r in rows if r['sharp_ok'] and r['rms'] > 0.005 and r['above_8k_ratio'] > 1e-4]
    ok.sort(key=lambda r: -r['band_ratio'])
    sel = ok[:a.keep]
    (OUT / 'candidates.json').write_text(json.dumps(
        {'bar_band_ratio': bar, 'n_probed': len(rows), 'n_pass': len(ok),
         'selected': sel, 'all': rows}, indent=1))
    print(json.dumps({'probed': len(rows), 'passed': len(ok), 'selected': len(sel),
                      'bar': round(bar, 5)}, indent=1))


if __name__ == '__main__':
    main()
