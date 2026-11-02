"""Audio metrics for Stage 6 (PROTOCOL_stage6.md), and the headroom they imply.

Run standalone to print the two bounds that bracket any result the audio branch can produce:
  floor   -- the degraded input itself (`passthrough`), which a restorer must beat;
  ceiling -- the VAE's own encode/decode round trip, which no latent-space restorer can beat.
"""
import json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
A = ROOT / 'data/stage6_audio'
SR = 32000
GROUP_DELAY = -144          # constant codec delay measured by code/audio_probe.py


def read_wav(p):
    raw = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(p), '-f', 'f32le', '-ac', '2',
                          '-ar', str(SR), '-'], stdout=subprocess.PIPE, check=True).stdout
    return torch.from_numpy(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T.copy())


def spec(x, n_fft=1024):
    return torch.stft(x, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs()


def lsd(a, b, n_fft=1024):
    """Log-spectral distance in dB, floored at 1e-5 of the reference peak so inaudible bins cannot dominate."""
    A_, B_ = spec(a, n_fft), spec(b, n_fft)
    fl = B_.amax() * 1e-5
    A_, B_ = A_.clamp_min(fl), B_.clamp_min(fl)
    return float(((20 * (A_.log10() - B_.log10())) ** 2).mean(-2).sqrt().mean())


def si_sdr(est, ref):
    est, ref = est - est.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / ref.square().sum(-1, keepdim=True).clamp_min(1e-12) * ref
    return float((10 * torch.log10(s.square().sum(-1).clamp_min(1e-12)
                                   / (est - s).square().sum(-1).clamp_min(1e-12))).mean())


def band(y, g, cutoff_hz, n_fft=1024):
    """Content above the observation's cutoff: the audio analogue of the video band N(x).

    Energy ratio alone is not enough here, unlike in the video case. The degradation adds full-band noise and
    clipping harmonics *after* the bandwidth limit, so the degraded input already has far more out-of-band
    energy than GT (measured: 152x) while carrying none of GT's content. Ratio ~1 is therefore necessary but
    not sufficient; the cosine says whether the out-of-band content is the right content."""
    Y, G = spec(y, n_fft), spec(g, n_fft)
    k = int(cutoff_hz / (SR / 2) * (Y.shape[-2] - 1))
    a, b = Y[..., k:, :].flatten(), G[..., k:, :].flatten()
    return {'band_energy_ratio': float(a.square().mean() / max(float(b.square().mean()), 1e-12)),
            'band_cos': float((a * b).sum() / max(float(a.square().sum().sqrt() * b.square().sum().sqrt()), 1e-12))}


def evaluate(est, ref, cutoff_hz=None, align=True):
    n = min(est.shape[-1], ref.shape[-1])
    est, ref = est[:, :n], ref[:, :n]
    e = torch.roll(est, -GROUP_DELAY, dims=-1) if align else est
    r = {'si_sdr_dB': si_sdr(e, ref), 'lsd_dB': lsd(e, ref)}
    if cutoff_hz:
        r |= band(e, ref, cutoff_hz)
    return r


def main():
    man = json.loads((A / 'manifest.json').read_text())
    rows = []
    for c in man['clips']:
        name = c['clip']
        gt, lr = read_wav(A / name / 'gt.wav'), read_wav(A / name / 'lr.wav')
        cut = c['degradation'].get('second_order', c['degradation'])['band_hz'] / 2
        # the degraded input is not delayed relative to GT, so no alignment here
        r = {'clip': name, 'cutoff_hz': cut, 'passthrough': evaluate(lr, gt, cut, align=False)}
        rows.append(r)
        print(json.dumps(r), flush=True)
    agg = {k: float(np.mean([r['passthrough'][k] for r in rows])) for k in rows[0]['passthrough']}
    out = {'n': len(rows), 'per_clip': rows, 'passthrough_mean': agg}
    (ROOT / 'results/stage6_passthrough.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({'passthrough_mean': agg}, indent=1))


if __name__ == '__main__':
    main()
