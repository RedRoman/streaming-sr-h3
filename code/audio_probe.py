"""Stage-6 feasibility probe: does the MiniMax-H3 audio VAE run on the 910B, and is its round trip faithful?

Blocking question for the audio branch. The decoder is BigVGAN, which uses anti-aliased activations (upsample ->
tanh/snake -> downsample); if any of that falls back to CPU or breaks under NPU float32, the branch is blocked
before any model work. Nothing here touches the DiT.
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from h3_comfy import ASSETS, ROOT, patch_module_to, set_precision, sync

SRC = Path('/temp/liwei/vidgen1m-h3-5s-selected-493-20261017/videos')
OUT = ROOT / 'results/stage6_probe'
CLIPS = ['VIDGEN5S_0769265', 'VIDGEN5S_0112364', 'VIDGEN5S_0130423']
SR = 32000


def read_audio(mp4, sr=SR):
    """Decode the soundtrack to float32 stereo [2, L] at the VAE's native rate (the clips are already 32 kHz)."""
    p = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(mp4), '-map', '0:a',
                        '-f', 'f32le', '-ac', '2', '-ar', str(sr), '-'],
                       stdout=subprocess.PIPE, check=True)
    w = np.frombuffer(p.stdout, dtype=np.float32).reshape(-1, 2).T.copy()
    return torch.from_numpy(w)


def lsd(a, b, n_fft=1024):
    """Log-spectral distance in dB; the standard scalar for speech/audio restoration."""
    win = torch.hann_window(n_fft)
    A = torch.stft(a, n_fft, n_fft // 4, window=win, return_complex=True).abs().clamp_min(1e-8)
    B = torch.stft(b, n_fft, n_fft // 4, window=win, return_complex=True).abs().clamp_min(1e-8)
    return float(((20 * (A.log10() - B.log10())) ** 2).mean(0).sqrt().mean())


def si_sdr(est, ref):
    est, ref = est - est.mean(-1, keepdim=True), ref - ref.mean(-1, keepdim=True)
    s = (est * ref).sum(-1, keepdim=True) / ref.square().sum(-1, keepdim=True).clamp_min(1e-12) * ref
    return float((10 * torch.log10(s.square().sum(-1) / (est - s).square().sum(-1).clamp_min(1e-12))).mean())


def main():
    patch_module_to(); set_precision()
    import comfy.sd, comfy.utils, comfy.model_management
    rec = {'vae_path': str(ASSETS / 'minimax_h3_audio_vae_fp32.safetensors')}
    t0 = time.perf_counter()
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(ASSETS / 'minimax_h3_audio_vae_fp32.safetensors')))
    rec |= {'load_seconds': time.perf_counter() - t0, 'sample_rate': vae.audio_sample_rate,
            'latent_channels': vae.latent_channels, 'downscale_ratio': vae.downscale_ratio}
    print(json.dumps(rec), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in CLIPS:
        w = read_audio(SRC / f'{name}.mp4')                      # [2, L]
        t0 = time.perf_counter()
        z = vae.encode(w.T.unsqueeze(0))                         # encode wants [B, samples, 2]
        sync(); t_enc = time.perf_counter() - t0
        t0 = time.perf_counter()
        y = vae.decode(z)
        sync(); t_dec = time.perf_counter() - t0
        y = y.float().cpu()
        y = y[0].T if y.ndim == 3 and y.shape[-1] == 2 else y.reshape(2, -1)
        n = min(y.shape[-1], w.shape[-1])
        r = {'clip': name, 'samples': int(w.shape[-1]), 'latent_shape': list(z.shape),
             'encode_seconds': t_enc, 'decode_seconds': t_dec,
             'latent_finite': bool(torch.isfinite(z).all()), 'out_finite': bool(torch.isfinite(y).all()),
             'in_rms': float(w.square().mean().sqrt()), 'out_rms': float(y.square().mean().sqrt()),
             'roundtrip_lsd_dB': lsd(y[:, :n], w[:, :n]), 'roundtrip_si_sdr_dB': si_sdr(y[:, :n], w[:, :n])}
        rows.append(r)
        print(json.dumps(r), flush=True)
    (OUT / 'audio_vae_probe.json').write_text(json.dumps({'vae': rec, 'clips': rows}, indent=1))


if __name__ == '__main__':
    main()
