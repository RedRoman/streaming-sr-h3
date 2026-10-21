"""E1/E2 (PROTOCOL_mech.md): how long is the null space correlated, per modality and per degradation strength?

Range-null decomposition says data consistency pins only the range space, so in streaming restoration the
only thing memory has to carry is the null space. mech2 measured its correlation at one degradation factor
(x4) for video. This sweeps the factor and adds the audio modality, to turn that single number into a
design rule for how much history a streaming restorer should keep -- per modality.

Prediction being tested: a stronger degradation leaves a larger null space that reaches down into coarser,
more slowly varying structure, so the correlation time should grow with the factor. If it does not, the
budget is a property of content alone and the rule is simpler but less interesting; either way it is reported.

No model is involved; GT frames and GT audio only.
"""
import json, subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D = ROOT / 'data/stage2_eval'
A = ROOT / 'data/stage6_audio'
OUT = ROOT / 'results/history_budget'
SCALES = (2, 4, 8)
VID_LAGS = (1, 2, 4, 8, 17, 34)
AUD_LAGS = (1, 2, 4, 8, 17, 34)        # audio latent frames at 40 Hz
FPS, SR, AHZ = 24.0, 32000, 40.0
STRIDE = 6


def null_video(x, s):
    """N_s(x) = x - U(D_s(x)): what an antialiased bicubic x s observation does not determine."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def corr(a, b):
    a, b = a.reshape(-1).astype(np.float64), b.reshape(-1).astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def bootstrap(v, n=10000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    return [float(np.percentile(m, p)) for m, p in
            [(np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)]), 2.5)] ] + \
           [float(np.percentile(np.array([np.random.default_rng(seed + 1).choice(v, v.size, replace=True).mean()
                                          for _ in range(n)]), 97.5))]


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    r = np.random.default_rng(seed)
    m = np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def read_audio(p):
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(p), '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), '-'], stdout=subprocess.PIPE, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    amen = {c['clip']: c for c in json.loads((A / 'manifest.json').read_text())['clips']}
    vid = {s: {d: [] for d in VID_LAGS} for s in SCALES}
    aud = {d: [] for d in AUD_LAGS}
    rows = []
    for name in man['clips']:
        meta = json.loads((D / name / 'meta.json').read_text())
        T = meta['frames']
        hr = torch.stack([torch.from_numpy(np.asarray(Image.open(D / name / 'hr' / f'{t:06d}.png').convert('RGB')).copy())
                          for t in range(T)]).float() / 255
        r = {'clip': name, 'video': {}, 'audio': {}}
        for s in SCALES:
            N = null_video(hr, s).numpy()
            for d in VID_LAGS:
                v = [corr(N[t], N[t + d]) for t in range(0, T - d, STRIDE)]
                r['video'][f's{s}_d{d}'] = float(np.mean(v))
                vid[s][d].append(float(np.mean(v)))
            del N
        # audio: content above the observation's bandwidth cutoff, on a 40 Hz grid (the audio latent rate)
        deg = amen[name]['degradation']
        cut = deg.get('second_order', deg)['band_hz'] / 2
        w = read_audio(A / name / 'gt.wav')
        hop = int(round(SR / AHZ))
        S = torch.stft(w, 2048, hop, window=torch.hann_window(2048), return_complex=True).abs().numpy()
        k = int(cut / (SR / 2) * (S.shape[0] - 1))
        B = S[k:]                                        # [freq_above_cutoff, time]
        for d in AUD_LAGS:
            v = [corr(B[:, t], B[:, t + d]) for t in range(0, B.shape[1] - d, STRIDE)]
            r['audio'][f'd{d}'] = float(np.mean(v))
            aud[d].append(float(np.mean(v)))
        r['audio_cutoff_hz'] = cut
        rows.append(r)
        print(json.dumps({'clip': name,
                          'video_s4': {f'd{d}': round(r['video'][f's4_d{d}'], 3) for d in VID_LAGS},
                          'audio': {f'd{d}': round(r['audio'][f'd{d}'], 3) for d in AUD_LAGS}}), flush=True)

    agg = {'video': {}, 'audio': {}}
    for s in SCALES:
        agg['video'][f'scale{s}'] = {f'd{d}': {'mean': float(np.mean(vid[s][d])), 'ci95': ci(vid[s][d]),
                                               'lag_ms': 1000 * d / FPS} for d in VID_LAGS}
    agg['audio'] = {f'd{d}': {'mean': float(np.mean(aud[d])), 'ci95': ci(aud[d]),
                              'lag_ms': 1000 * d / AHZ} for d in AUD_LAGS}

    def half_life(curve, lags, hz):
        """Lag (ms) at which correlation falls to half its value at the shortest lag; linear interpolation."""
        c0 = curve[lags[0]]
        for i in range(1, len(lags)):
            if curve[lags[i]] <= c0 / 2:
                a, b = curve[lags[i - 1]], curve[lags[i]]
                f = (a - c0 / 2) / max(a - b, 1e-12)
                return 1000 * (lags[i - 1] + f * (lags[i] - lags[i - 1])) / hz
        return None
    budget = {f'video_scale{s}_ms': half_life({d: float(np.mean(vid[s][d])) for d in VID_LAGS}, VID_LAGS, FPS)
              for s in SCALES}
    budget['audio_ms'] = half_life({d: float(np.mean(aud[d])) for d in AUD_LAGS}, AUD_LAGS, AHZ)
    out = {'n': len(rows), 'per_clip': rows, 'aggregate': agg, 'half_life_ms': budget}
    (OUT / 'budget.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'aggregate': agg, 'half_life_ms': budget}, indent=1))


if __name__ == '__main__':
    main()
