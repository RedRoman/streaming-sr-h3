"""Audio, step 1: find the ceiling before measuring anything against it.

The audio branch has been reported as 0% complete and the reason is structural, not missing effort. Two
facts about the current pipeline:

  * h3_comfy.sample builds the audio latent as torch.zeros(1, 32, 2, audio_t) and denoises it, so the
    model GENERATES audio rather than restoring any. There is no degraded audio input anywhere.
  * H3 exposes decode() for the video VAE only. The audio latent is returned by sample() and asserted
    finite, then dropped on the floor. The 605 MB audio VAE has never been loaded in this project.

So before asking "how good is our audio restoration", two bounds have to exist, and one of them can
invalidate the whole line:

  ceiling   GT audio -> audio VAE encode -> decode. No latent-space method can beat this round trip. If
            it is already poor, a restorer working in that latent cannot be good no matter what it does.
  floor     silence, and the GT against itself. These bracket the metric so a number like "LSD 2.1 dB"
            can be read at all.

Only the ceiling needs a sampler; it needs none. That is why it is step 1 and why it is cheap.

LSD and the -144 sample group delay come from audio_metrics.py, which measured that delay rather than
assuming zero.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
ASSETS = ROOT / 'assets/comfy_h3'
AUDIO_VAE = ASSETS / 'minimax_h3_audio_vae_fp32.safetensors'
SR = 32000


def spec(x, n_fft=1024):
    return torch.stft(x, n_fft, n_fft // 4, window=torch.hann_window(n_fft, device=x.device),
                      return_complex=True).abs()


def lsd(a, b, n_fft=1024):
    """Log-spectral distance in dB, floored at 1e-5 of the reference peak, as audio_metrics does."""
    A_, B_ = spec(a, n_fft), spec(b, n_fft)
    n = min(A_.shape[-1], B_.shape[-1])
    A_, B_ = A_[..., :n], B_[..., :n]
    fl = 1e-5 * float(B_.max())
    la, lb = torch.log10(A_.clamp_min(fl)), torch.log10(B_.clamp_min(fl))
    return float((20 * (la - lb)).square().mean(dim=-2).sqrt().mean())


def align(est, ref, max_shift=4096):
    """Shift est to maximise cross-correlation with ref before any waveform metric.

    Necessary, not cosmetic. audio_metrics.py records a MEASURED codec group delay of -144 samples, and
    the VAE round trip returns 164800 samples for a 165333-sample input, so est and ref are neither
    phase- nor length-aligned. SI-SDR is destroyed by sub-millisecond misalignment -- the first pass of
    this script reported -6.8 to -48.3 dB with plain truncation, which says nothing about the VAE and
    everything about a 533-sample offset. LSD is computed on magnitude spectrograms with a 256-sample hop
    and is far more shift-tolerant, which is why it was the usable number even before this.
    """
    e = est.double(); r = ref.double()
    a = e.mean(0) - e.mean(); b = r.mean(0) - r.mean()
    n = min(a.shape[-1], b.shape[-1])
    fa = torch.fft.rfft(a[:n], 2 * n); fb = torch.fft.rfft(b[:n], 2 * n)
    cc = torch.fft.irfft(fa * fb.conj(), 2 * n)
    k = int(torch.cat([cc[:max_shift], cc[-max_shift:]]).argmax())
    lag = k if k < max_shift else k - 2 * max_shift
    if lag > 0:
        e = e[..., lag:]
    elif lag < 0:
        e = torch.nn.functional.pad(e, (-lag, 0))
    m = min(e.shape[-1], r.shape[-1])
    return e[..., :m], r[..., :m], lag


def si_sdr(est, ref):
    """Scale-invariant SDR in dB; insensitive to the gain the VAE may apply."""
    n = min(est.shape[-1], ref.shape[-1])
    e, r = est[..., :n].double(), ref[..., :n].double()
    e = e - e.mean(-1, keepdim=True); r = r - r.mean(-1, keepdim=True)
    s = (e * r).sum(-1, keepdim=True) / r.square().sum(-1, keepdim=True).clamp_min(1e-12) * r
    return float((10 * torch.log10(s.square().sum(-1) / (e - s).square().sum(-1).clamp_min(1e-12))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--clips', type=int, default=8)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--out', default='results/audio_bounds.json')
    a = ap.parse_args()

    if not AUDIO_VAE.exists():
        raise SystemExit('audio VAE missing at %s' % AUDIO_VAE)
    import torch_npu  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    sys.path.insert(0, str(ROOT / 'code'))
    import comfy.sd
    import comfy.utils

    dev = torch.device(a.device)
    _claim = torch.zeros(1 << 28, device=dev); del _claim
    sd, meta = comfy.utils.load_torch_file(str(AUDIO_VAE), return_metadata=True)
    print(json.dumps({'audio_vae_tensors': len(sd),
                      'metadata_keys': sorted(meta)[:8] if meta else [],
                      'example_keys': sorted(sd)[:4]}), flush=True)
    vae = comfy.sd.VAE(sd=sd, device=dev, dtype=torch.float32, metadata=meta)
    vae.first_stage_model.eval().requires_grad_(False)

    D_ = ROOT / a.eval_set
    clips = json.loads((D_ / 'manifest.json').read_text())['clips']
    clips = [c for c in clips
             if json.loads((D_ / c / 'meta.json').read_text()).get('audio_present')][:a.clips]
    rows = []
    for c in clips:
        gt = torch.from_numpy(np.load(D_ / c / 'audio_gt.npy'))          # [2, N]
        x = gt[None].to(dev)                                             # [1, 2, N]
        with torch.no_grad():
            z = vae.encode(x.movedim(1, -1) if x.shape[1] == 2 else x)
            y = vae.decode(z)
        y = y.float().cpu()
        if y.ndim == 3 and y.shape[1] != 2 and y.shape[-1] == 2:
            y = y.movedim(-1, 1)
        y = y.reshape(2, -1) if y.ndim == 3 else y
        rec = {'clip': c, 'gt_samples': int(gt.shape[-1]), 'rt_samples': int(y.shape[-1]),
               'latent_shape': list(z.shape) if hasattr(z, 'shape') else None,
               'finite': bool(torch.isfinite(y).all()),
               'lsd_roundtrip_dB': lsd(y, gt),
               'si_sdr_raw_unaligned_dB': si_sdr(y, gt),
               'lsd_silence_dB': lsd(torch.zeros_like(gt), gt),
               'lsd_self_dB': lsd(gt, gt),
               'gt_rms': float(gt.square().mean().sqrt()), 'rt_rms': float(y.square().mean().sqrt())}
        ya, ga, lag = align(y, gt)
        rec['alignment_lag_samples'] = lag
        rec['si_sdr_aligned_dB'] = si_sdr(ya, ga)
        rec['lsd_aligned_dB'] = lsd(ya, ga)
        rows.append(rec)
        print(json.dumps(rec), flush=True)

    out = {'eval_set': a.eval_set, 'sr': SR, 'clips': len(rows),
           'ceiling_lsd_dB': float(np.mean([r['lsd_roundtrip_dB'] for r in rows])),
           'ceiling_si_sdr_aligned_dB': float(np.mean([r['si_sdr_aligned_dB'] for r in rows])),
           'ceiling_si_sdr_unaligned_dB': float(np.mean([r['si_sdr_raw_unaligned_dB'] for r in rows])),
           'ceiling_lsd_aligned_dB': float(np.mean([r['lsd_aligned_dB'] for r in rows])),
           'alignment_lags': [r['alignment_lag_samples'] for r in rows],
           'floor_silence_lsd_dB': float(np.mean([r['lsd_silence_dB'] for r in rows])),
           'self_lsd_dB': float(np.mean([r['lsd_self_dB'] for r in rows])),
           'per_clip': rows,
           'note': ('ceiling is the audio VAE round trip on ground-truth audio: no latent-space restorer '
                    'can beat it. floor is silence. A generated-audio LSD must be read between them, and '
                    'the current pipeline denoises an all-zero audio latent, so it generates rather than '
                    'restores.')}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print()
    print('audio VAE round-trip ceiling over %d clips:' % len(rows))
    print('  LSD    %.3f dB   (silence floor %.3f dB, GT vs itself %.3f dB)'
          % (out['ceiling_lsd_dB'], out['floor_silence_lsd_dB'], out['self_lsd_dB']))
    print('  LSD aligned %.3f dB' % out['ceiling_lsd_aligned_dB'])
    print('  SI-SDR %.2f dB aligned, %.2f dB if alignment is skipped; lags %s'
          % (out['ceiling_si_sdr_aligned_dB'], out['ceiling_si_sdr_unaligned_dB'],
             out['alignment_lags']))
    print('  -> if aligned SI-SDR is still far below 0, the VAE is not waveform-preserving and no '
          'waveform metric can be used for this line at all')
    print()
    print('wrote', ROOT / a.out)


if __name__ == '__main__':
    main()
