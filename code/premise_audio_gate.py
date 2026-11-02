"""Premise test for the audio-gated temporal constraint (PROTOCOL_mech1.md). No model, GT + data only.

The mechanism would use audio onsets to decide when a streaming video restorer may change its generated
detail. That is only worth building if audio actually predicts when detail changes, and if the prediction
survives the audio degradation the model will see. Both are measured here before anything is implemented.

Read-out rules, fixed before running:
  P1 accepted iff mean Pearson r(onset, detail-change) >= 0.20 over clips, bootstrap CI excludes 0,
     AND r exceeds the time-shuffled control by >= 0.15.
  P2 (degradation robustness) accepted iff r computed from the DEGRADED audio retains >= 70% of the r
     computed from GT audio.
  If P1 fails the mechanism is dropped and this file records that it was.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D = ROOT / 'data/stage2_eval'
A = ROOT / 'data/stage6_audio'
OUT = ROOT / 'results/mech1_premise'
SR, FPS = 32000, 24.0
RNG = np.random.default_rng(0)


def read_audio(p):
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(p), '-f', 'f32le', '-ac', '1',
                          '-ar', str(SR), '-'], stdout=subprocess.PIPE, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).copy())


def onset_strength(w, n_frames):
    """Spectral flux resampled onto the video frame grid: half-wave-rectified rise in magnitude spectrum."""
    hop = int(round(SR / FPS / 4))                     # 4 audio hops per video frame
    S = torch.stft(w, 1024, hop, window=torch.hann_window(1024), return_complex=True).abs()
    flux = (S[:, 1:] - S[:, :-1]).clamp_min(0).sum(0)
    flux = flux / max(float(flux.max()), 1e-12)
    # average the 4 hops that fall inside each video frame
    o = F.adaptive_avg_pool1d(flux[None, None], n_frames)[0, 0]
    return o


def unobserved(x):
    """N(x) = x - U(D(x)) with the project's x4 antialiased bicubic. x: [T,H,W,3] in [0,1]."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(float(a.norm() * b.norm()), 1e-12))


def bootstrap(v, n=10000, seed=0):
    v = np.asarray(v, dtype=float)
    r = np.random.default_rng(seed)
    m = np.array([r.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    rows = []
    for name in man['clips']:
        meta = json.loads((D / name / 'meta.json').read_text())
        T = meta['frames']
        frames = torch.stack([torch.from_numpy(np.asarray(Image.open(D / name / 'hr' / f'{t:06d}.png').convert('RGB')).copy())
                              for t in range(T)]).float() / 255
        N = unobserved(frames)
        # detail-change rate between consecutive frames, in the band the observation does not determine
        c = torch.tensor([float((N[t + 1] - N[t]).square().mean()) for t in range(T - 1)])
        r = {'clip': name, 'frames': T}
        for tag, wav in (('gt', 'gt.wav'), ('lr', 'lr.wav')):
            o = onset_strength(read_audio(A / name / wav), T - 1)
            r[f'r_{tag}'] = pearson(o, c)
            sh = torch.tensor(RNG.permutation(o.numpy()).copy())
            r[f'r_{tag}_shuffled'] = pearson(sh, c)
        r['retention'] = r['r_lr'] / r['r_gt'] if abs(r['r_gt']) > 1e-9 else float('nan')
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    agg = {}
    for k in ('r_gt', 'r_lr', 'r_gt_shuffled', 'r_lr_shuffled'):
        v = [x[k] for x in rows]
        agg[k] = {'mean': float(np.mean(v)), 'ci95': bootstrap(v)}
    gain = [x['r_lr'] - x['r_lr_shuffled'] for x in rows]
    agg['r_lr_minus_shuffled'] = {'mean': float(np.mean(gain)), 'ci95': bootstrap(gain)}
    p1 = (agg['r_lr']['mean'] >= 0.20 and agg['r_lr']['ci95'][0] > 0
          and agg['r_lr_minus_shuffled']['mean'] >= 0.15)
    ret = float(np.mean([x['retention'] for x in rows if np.isfinite(x['retention'])]))
    p2 = ret >= 0.70
    out = {'n': len(rows), 'per_clip': rows, 'aggregate': agg, 'mean_retention': ret,
           'P1_audio_predicts_detail_change': bool(p1), 'P2_survives_degradation': bool(p2),
           'verdict': ('premise holds, build the mechanism' if p1 and p2 else
                       'premise fails on this data, drop the mechanism as specified')}
    (OUT / 'premise.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))


if __name__ == '__main__':
    main()
