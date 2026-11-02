"""An audio degradation operator, and the gate that decides whether restoring it is worth training.

Why this file did not exist: degrade.py never mentions audio, so the project has clean GT audio in every
eval clip and no degraded counterpart anywhere. train_stage1.py says so in its own docstring -- "audio
stream gets noise and no loss (pilot: video only)" -- and LoRA targets only the shared attn/mlp blocks.
The audio branch was therefore never trained on anything, which is the whole explanation for its output,
not a capability limit.

Before building the training, one number has to be checked, because it can kill the idea outright. The
audio VAE is a PERCEPTUAL codec at 25x compression -- 32 kHz stereo is 64000 values per second and the
latent is 40 Hz x 32 ch x 2 = 2560 -- so its round trip is LSD 9.6 dB and SI-SDR +4.0 dB on real
recordings. That is normal for the rate and not a defect, but it means a latent-space restorer is capped
by it. The gate is therefore comparative, not absolute:

    is the damage the degradation does MUCH larger than the damage the codec does?

If yes, there is signal to recover inside what the latent can express, and training is worth it. If the
two are comparable, a restorer would be chasing the codec's own error and the honest deliverable is
audio-visual generation with the ceiling reported. Absolute LSD against speech-enhancement numbers
(0.9-1.5 dB) is the wrong comparison and is not used here.

The operators target the compressed/archival regime the restoration story is about, each with a measured
reason to exist rather than a guess:
  band    a lowpass at 3.5-8 kHz. AudioSet real recordings already have above_8k_ratio recorded per clip,
          so the eval set can say how much energy this actually removes per clip.
  codec   re-encode through a low-bitrate AAC and back. This is the artefact family the video side
          already models with h264_crf, applied to the same container.
  noise   broadband hiss at a controlled SNR, the analogue of the video chain's noise1.
  clip    hard clipping at a quantile of the waveform, which an archival transfer with wrong gain does.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
SR = 32000


def _ffmpeg_roundtrip(x, sr, args):
    """Push [2,N] float32 through ffmpeg with `args` and read it back at the same rate and length."""
    with tempfile.TemporaryDirectory() as d:
        raw = Path(d) / 'in.f32'
        mid = Path(d) / ('mid.' + args['ext'])
        raw.write_bytes(np.ascontiguousarray(x.T.numpy().astype(np.float32)).tobytes())
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'f32le', '-ar', str(sr), '-ac', '2',
                        '-i', str(raw), *args['enc'], str(mid)], check=True)
        out = subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(mid),
                              '-f', 'f32le', '-ar', str(sr), '-ac', '2', '-'],
                             stdout=subprocess.PIPE, check=True).stdout
    y = torch.from_numpy(np.frombuffer(out, dtype=np.float32).reshape(-1, 2).T.copy())
    n = x.shape[-1]
    return torch.nn.functional.pad(y, (0, max(0, n - y.shape[-1])))[..., :n]


def lowpass(x, cutoff, sr=SR):
    """Zero-phase brick-wall lowpass in the rfft domain, so the band limit is exact and reproducible."""
    F = torch.fft.rfft(x.double())
    f = torch.fft.rfftfreq(x.shape[-1], 1.0 / sr)
    F[..., f > cutoff] = 0
    return torch.fft.irfft(F, n=x.shape[-1]).float()


def degrade_audio(x, rng, kind='realistic'):
    """[2,N] float32 in roughly [-1,1] -> (degraded [2,N], params dict). Deterministic given rng."""
    p = {'kind': kind}
    y = x.clone()
    if kind == 'clean':
        return y, p
    cut = float(rng.uniform(3500, 8000))
    y = lowpass(y, cut); p['lowpass_hz'] = round(cut, 1)
    snr = float(rng.uniform(12, 30))
    sig = float(y.square().mean().sqrt())
    n = torch.from_numpy(rng.standard_normal(y.shape).astype(np.float32))
    y = y + n * (sig / (10 ** (snr / 20)) / max(float(n.square().mean().sqrt()), 1e-9))
    p['snr_dB'] = round(snr, 2)
    q = float(rng.uniform(0.90, 1.0))
    if q < 0.999:
        thr = float(torch.quantile(y.abs().flatten().double(), q))
        y = y.clamp(-thr, thr); p['clip_quantile'] = round(q, 4)
    br = int(rng.choice([24, 32, 48, 64]))
    y = _ffmpeg_roundtrip(y, SR, {'ext': 'm4a', 'enc': ['-c:a', 'aac', '-b:a', f'{br}k']})
    p['aac_kbps'] = br
    return y.clamp(-1, 1), p


# ---------------------------------------------------------------- metrics
def _spec(x, n_fft):
    return torch.stft(x, n_fft, n_fft // 4, window=torch.hann_window(n_fft), return_complex=True).abs()


def lsd(a, b, n_fft=1024):
    A, B = _spec(a.float(), n_fft), _spec(b.float(), n_fft)
    n = min(A.shape[-1], B.shape[-1]); A, B = A[..., :n], B[..., :n]
    fl = 1e-5 * float(B.max())
    return float((20 * (torch.log10(A.clamp_min(fl)) - torch.log10(B.clamp_min(fl))))
                 .square().mean(dim=-2).sqrt().mean())


def _mel_fb(n_mels=64, n_fft=1024, sr=SR):
    f = torch.fft.rfftfreq(n_fft, 1.0 / sr)
    m = lambda h: 2595 * torch.log10(1 + h / 700)
    mi = lambda q: 700 * (10 ** (q / 2595) - 1)
    edges = mi(torch.linspace(float(m(torch.tensor(40.0))), float(m(torch.tensor(sr / 2))), n_mels + 2))
    fb = torch.zeros(n_mels, len(f))
    for i in range(n_mels):
        lo, ce, hi = edges[i], edges[i + 1], edges[i + 2]
        up = (f - lo) / max(float(ce - lo), 1e-9)
        dn = (hi - f) / max(float(hi - ce), 1e-9)
        fb[i] = torch.minimum(up, dn).clamp_min(0)
    return fb


_MEL = None


def mel_dist(a, b, n_fft=1024):
    """Mean absolute log-mel distance in dB -- the metric a perceptual codec can actually express.

    LSD weights every linear bin equally, so it is dominated by near-silent high-frequency bins where a
    25x codec legitimately invents content. A mel-weighted log distance is the comparison the codec was
    designed against, which is why the gate below is read on this and LSD is reported beside it.
    """
    global _MEL
    if _MEL is None:
        _MEL = _mel_fb(n_fft=n_fft)
    # align() upcasts to double, so a float filterbank would fail with "expected m1 and m2 to have the
    # same dtype". Cast both sides here rather than at every call site.
    A, B = _spec(a.float(), n_fft), _spec(b.float(), n_fft)
    n = min(A.shape[-1], B.shape[-1])
    fb = _MEL.to(A.dtype)
    MA = (fb @ A[..., :n]).clamp_min(1e-8)
    MB = (fb @ B[..., :n]).clamp_min(1e-8)
    return float((20 * (torch.log10(MA) - torch.log10(MB))).abs().mean())


if __name__ == '__main__':
    import torch_npu  # noqa: F401
    import h3_comfy  # noqa: F401
    from audio_line1 import align, si_sdr
    import comfy.sd
    import comfy.utils

    dev = torch.device('npu:0')
    _c = torch.zeros(1 << 28, device=dev); del _c
    sd, meta = comfy.utils.load_torch_file(
        str(ROOT / 'assets/comfy_h3/minimax_h3_audio_vae_fp32.safetensors'), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, device=dev, dtype=torch.float32, metadata=meta)
    vae.first_stage_model.eval().requires_grad_(False)

    def rt(w):
        with torch.no_grad():
            y = vae.decode(vae.encode(w[None].movedim(1, -1).to(dev)))
        y = y.float().cpu()
        y = y[0] if y.ndim == 3 else y
        return (y.T if y.shape[0] != 2 else y).reshape(2, -1)

    D_ = ROOT / 'data/real_eval/heldout'
    clips = [c for c in json.loads((D_ / 'manifest.json').read_text())['clips']
             if json.loads((D_ / c / 'meta.json').read_text()).get('audio_present')][:int(
                 os.environ.get('N', '8'))]
    rows = []
    for i, c in enumerate(clips):
        gt = torch.from_numpy(np.load(D_ / c / 'audio_gt.npy'))
        rng = np.random.default_rng(20261101 + i)
        dg, p = degrade_audio(gt, rng)
        ce = rt(gt)                       # the codec's own damage
        dr = rt(dg)                       # the degraded signal seen through the codec
        a1, b1, _ = align(ce, gt)
        r = {'clip': c, 'params': p,
             'degradation_damage': {'lsd_dB': round(lsd(dg, gt), 3), 'mel_dB': round(mel_dist(dg, gt), 3),
                                    'si_sdr_dB': round(si_sdr(*align(dg, gt)[:2]), 2)},
             'codec_damage': {'lsd_dB': round(lsd(a1, b1), 3), 'mel_dB': round(mel_dist(a1, b1), 3),
                              'si_sdr_dB': round(si_sdr(a1, b1), 2)},
             'degraded_through_codec': {'lsd_dB': round(lsd(*align(dr, gt)[:2]), 3),
                                        'mel_dB': round(mel_dist(*align(dr, gt)[:2]), 3)}}
        r['headroom_mel_dB'] = round(r['degradation_damage']['mel_dB'] - r['codec_damage']['mel_dB'], 3)
        r['ratio_mel'] = round(r['degradation_damage']['mel_dB'] / max(r['codec_damage']['mel_dB'], 1e-9), 2)
        rows.append(r); print(json.dumps(r), flush=True)

    dm = float(np.mean([x['degradation_damage']['mel_dB'] for x in rows]))
    cm = float(np.mean([x['codec_damage']['mel_dB'] for x in rows]))
    out = {'clips': len(rows), 'per_clip': rows,
           'degradation_mel_dB': dm, 'codec_mel_dB': cm,
           'headroom_mel_dB': dm - cm, 'ratio': dm / max(cm, 1e-9),
           'gate': ('restoration is worth training if the degradation damage is much larger than the '
                    'codec damage in the mel metric, because only then is there signal to recover that '
                    'the latent can represent'),
           'verdict': 'WORTH TRAINING' if dm > 2 * cm else 'NOT CLEARLY WORTH IT'}
    (ROOT / 'results/audio_gate.json').write_text(json.dumps(out, indent=1))
    print()
    print('%-26s %10s %10s %10s' % ('', 'LSD dB', 'mel dB', 'SI-SDR dB'))
    print('%-26s %10.3f %10.3f %10.2f' % ('degradation does',
          float(np.mean([x['degradation_damage']['lsd_dB'] for x in rows])), dm,
          float(np.mean([x['degradation_damage']['si_sdr_dB'] for x in rows]))))
    print('%-26s %10.3f %10.3f %10.2f' % ('the codec itself does',
          float(np.mean([x['codec_damage']['lsd_dB'] for x in rows])), cm,
          float(np.mean([x['codec_damage']['si_sdr_dB'] for x in rows]))))
    print()
    print('mel headroom %.3f dB, ratio %.2fx  -> %s' % (dm - cm, dm / max(cm, 1e-9), out['verdict']))
