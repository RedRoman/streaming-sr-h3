"""The joint experiment: audio restoration inside the video sampler, and what it costs the video.

This is the one run that makes "omni audio-video restoration" a single claim instead of two separate
results. Until now the audio branch R_phi^audio was standalone -- it closes 72.3% of the gap between the
codec round trip and the ceiling, replicated independently on a second host at 72.9% under a different
degradation draw -- while the sampler's audio stream started at exact zeros, received no loss in training,
and produced output that is spectrally in the right region but temporally unrelated to the truth
(SI-SDR -36.3 dB against a +5.8 ceiling, rms ratio 0.343 over a 0.005-1.487 range).

Three arms, because two of the three differences have to be separated:

  A  zeros   + no projection   the shipped behaviour, bit-exact. The control.
  B  randn   + no projection   the init fix alone. train_stage1 fed the audio stream torch.randn and this
                               pipeline feeds torch.zeros, so B isolates that mismatch from the mechanism.
  C  randn   + projection      innerDC applied to audio: step the audio z0 toward R_phi^audio's output at
                               the same POINTS the video projection uses.

Both outputs are scored every time, and the VIDEO score matters as much as the audio one. The DiT is
shared between the two streams, so changing what the audio stream carries can move the video. A run that
restores audio and costs video quality is not a joint restoration, it is a trade, and this project has
already been wrong once by reading an estimate-level gain as a delivered one.

Audio bounds for reading the numbers, in mel dB on the held-out set's own audio:
  26.54 passthrough, 16.34 codec round trip of the degraded signal, 2.77 ceiling.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
H, W = 480, 832


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=4)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--rphi', default='results/omni/omni_m5.pt')
    ap.add_argument('--op', default='4')
    ap.add_argument('--arphi', default='results/audio_range/ar_31984.pt')
    ap.add_argument('--lora', default='results/stage1_lora/lora_step002000.safetensors')
    ap.add_argument('--arms', default='A,B,C')
    ap.add_argument('--strength', type=float, default=1.0)
    ap.add_argument('--out', default='results/audio_joint.json')
    ap.add_argument('--tag', default='', help='suffix for the output file when sharding')
    a = ap.parse_args()

    import torch_npu  # noqa: F401
    import h3_comfy
    from h3_comfy import H3, sync
    import comfy.sd
    import comfy.utils
    import lora as L
    import lora_merge as LM
    import turbo as TB
    from PIL import Image
    from safetensors.torch import load_file
    from audio_line1 import align, si_sdr
    from degrade_audio import degrade_audio, lsd, mel_dist
    from run_deploy import TURBO_PATH, range_estimate
    from train_audio_range import AudioRange

    D_ = ROOT / 'data/real_eval/heldout'
    man = json.loads((D_ / 'manifest.json').read_text())
    blocks = man['blocks']
    clips = [c for c in sorted(man['clips'])
             if json.loads((D_ / c / 'meta.json').read_text()).get('audio_present')
             and json.loads((D_ / c / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    # --clips 0 means every eligible clip; sharding is by stride so each card gets a mix of lengths
    if a.clips:
        clips = clips[:a.clips]
    clips = [c for i, c in enumerate(clips) if i % a.nshards == a.shard]

    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    TB.merge(model, TURBO_PATH)
    L.inject(model, 32)
    L.load(model, load_file(str(ROOT / a.lora)))
    LM.merge_(model)
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
    h3_comfy.SIGMAS = TB.turbo_sigmas(a.steps, 12.0)

    sd_, meta_ = comfy.utils.load_torch_file(
        str(ROOT / 'assets/comfy_h3/minimax_h3_audio_vae_fp32.safetensors'), return_metadata=True)
    avae = comfy.sd.VAE(sd=sd_, device=h3.dev, dtype=torch.float32, metadata=meta_)
    avae.first_stage_model.eval().requires_grad_(False)
    st = torch.load(ROOT / a.arphi, map_location='cpu')
    ar = AudioRange(**st['cfg']).eval(); ar.load_state_dict(st['state']); ar = ar.to(h3.dev)

    def aenc(w):
        # avae.encode moves data to h3.dev internally but returns on CPU -- this is ComfyUI's convention
        with torch.no_grad():
            return avae.encode(w[None].movedim(1, -1).to(h3.dev))   # returns CPU tensor

    def adec(z):
        with torch.no_grad():
            y = avae.decode(z.to(h3.dev)).float().cpu()   # z may be cpu or dev
        y = y[0] if y.ndim == 3 else y
        return (y.T if y.shape[0] != 2 else y).reshape(2, -1)

    ARMS = {'A': ('zeros', False), 'B': ('randn', False), 'C': ('randn', True)}
    want = [x for x in a.arms.split(',') if x in ARMS]
    rows = []
    for c in clips:
        meta = json.loads((D_ / c / 'meta.json').read_text())
        T = meta['frames']
        lr = torch.stack([torch.from_numpy(np.asarray(Image.open(D_ / c / 'lr' / f'{t:06d}.png')
                                                      .convert('RGB')).copy()).float() / 255
                          for t in range(T)]).permute(0, 3, 1, 2)
        hr = torch.stack([torch.from_numpy(np.asarray(Image.open(D_ / c / 'hr' / f'{t:06d}.png')
                                                      .convert('RGB')).copy()).float() / 255
                          for t in range(T)])
        gt_a = torch.from_numpy(np.load(D_ / c / 'audio_gt.npy'))
        # the degradation that audio_lr.npy records on the other host; the seed is now stable
        import hashlib
        seed = int.from_bytes(hashlib.sha256(c.encode()).digest()[:4], 'little')
        dg_a, dpar = degrade_audio(gt_a, np.random.default_rng(seed))
        est = range_estimate(lr, ROOT / a.rphi, dev=h3.dev, op=a.op)

        rec = {'clip': c, 'frames': T, 'audio_degradation': dpar, 'arms': {}}
        # the three audio bounds, this clip's own
        zd = aenc(dg_a)                                  # cpu
        with torch.no_grad():
            zr = ar(zd.to(h3.dev)).cpu()                     # ar lives on dev; result back to cpu
        rec['bounds'] = {
            'passthrough_mel': round(mel_dist(*align(dg_a, gt_a)[:2]), 3),
            'vae_only_mel': round(mel_dist(*align(adec(zd), gt_a)[:2]), 3),
            'rphi_audio_mel': round(mel_dist(*align(adec(zr), gt_a)[:2]), 3),
            'ceiling_mel': round(mel_dist(*align(adec(aenc(gt_a)), gt_a)[:2]), 3)}
        print(json.dumps({'clip': c, 'bounds': rec['bounds']}), flush=True)

        for arm in want:
            init, use_proj = ARMS[arm]
            vid, aud, t0 = [], [], time.perf_counter()
            for b in blocks:
                s, e = b['range']
                guide = h3.encode_guide(lr[s:e], [H, W])
                tgt = None
                if use_proj:
                    # R_phi^audio on the degraded audio of exactly this block's time span
                    n0 = int(round(s / 24 * 32000)); n1 = int(round(e / 24 * 32000))
                    zb = aenc(dg_a[..., n0:n1])          # cpu
                    with torch.no_grad():
                        tgt = ar(zb.to(h3.dev)).cpu()        # move to dev for ar, return cpu for h3.sample
                sampled, info = h3.sample(guide, 'innerdc', low=est[s:e], hw=[H, W], seed=101,
                                          audio_target=tgt, audio_strength=a.strength,
                                          audio_init=init)
                vid.append(h3.decode(sampled['video']))
                aud.append(sampled['audio'])
            sync()
            # Each block spans 22 frames but only `new` of them are its own; the first 5 are the
            # previous block's context, re-generated. Concatenating whole blocks therefore repeats 5
            # frames every 17 and shifts everything after block 0. The video here used to do exactly
            # that and the audio still did, which stretched the waveform by 25.3% (259 latent frames
            # against the correct 207) -- a periodic time-base error that align() cannot undo because
            # it only removes a constant delay. Cut both streams to each block's own `new` span, at the
            # offset that span has inside the block.
            vp, ap = [], []
            for v, aa, b in zip(vid, aud, blocks):
                s, _e = b['range']
                nl, nh = b['new']
                vp.append(v[nl - s:nh - s])
                ap.append(aa[..., int(round((nl - s) / 24 * 40)):int(round((nh - s) / 24 * 40))])
            px = torch.cat(vp)[:T]
            wav = adec(torch.cat(ap, dim=-1)[..., :int(round(T / 24 * 40))])
            # video: PSNR against the ground-truth frames that exist on disk
            n = min(px.shape[0], hr.shape[0])
            mse = (px[:n] - hr[:n]).square().mean()
            rec['arms'][arm] = {
                'audio_init': init, 'projection': use_proj, 'strength': a.strength if use_proj else 0.0,
                'audio_mel_dB': round(mel_dist(*align(wav, gt_a)[:2]), 3),
                'audio_lsd_dB': round(lsd(*align(wav, gt_a)[:2]), 3),
                'audio_si_sdr_dB': round(si_sdr(*align(wav, gt_a)[:2]), 2),
                'audio_rms_ratio': round(float(wav.square().mean().sqrt() / gt_a.square().mean().sqrt()), 4),
                'video_psnr_dB': round(float(10 * torch.log10(1 / mse.clamp_min(1e-12))), 3),
                'sec': round(time.perf_counter() - t0, 1)}
            print(json.dumps({'clip': c, 'arm': arm, **rec['arms'][arm]}), flush=True)
            del vid, aud, px, wav, mse
            torch.npu.empty_cache() if hasattr(torch, 'npu') else torch.cuda.empty_cache()
        rows.append(rec)

    out = {'clips': len(rows), 'steps': a.steps, 'strength': a.strength, 'per_clip': rows}
    for arm in want:
        g = [r['arms'][arm] for r in rows if arm in r['arms']]
        if g:
            out[arm] = {k: round(float(np.mean([x[k] for x in g])), 4)
                        for k in ('audio_mel_dB', 'audio_lsd_dB', 'audio_si_sdr_dB',
                                  'audio_rms_ratio', 'video_psnr_dB')}
    (ROOT / a.out).write_text(json.dumps(out, indent=1))
    print()
    print('%-28s %10s %10s %10s %10s %11s' % ('arm', 'audio mel', 'audio LSD', 'SI-SDR', 'rms ratio', 'VIDEO PSNR'))
    lab = {'A': 'A zeros, no projection', 'B': 'B randn, no projection', 'C': 'C randn + innerDC audio'}
    for arm in want:
        if arm in out:
            o = out[arm]
            print('%-28s %10.3f %10.3f %10.2f %10.4f %11.3f'
                  % (lab[arm], o['audio_mel_dB'], o['audio_lsd_dB'], o['audio_si_sdr_dB'],
                     o['audio_rms_ratio'], o['video_psnr_dB']))
    bm = lambda k: float(np.mean([r['bounds'][k] for r in rows]))
    print()
    print('audio bounds on these clips: passthrough %.3f  vae_only %.3f  R_phi^audio alone %.3f  ceiling %.3f'
          % (bm('passthrough_mel'), bm('vae_only_mel'), bm('rphi_audio_mel'), bm('ceiling_mel')))
    if 'A' in out and 'C' in out:
        print()
        print('C vs A:  audio mel %+.3f dB    VIDEO PSNR %+.3f dB'
              % (out['C']['audio_mel_dB'] - out['A']['audio_mel_dB'],
                 out['C']['video_psnr_dB'] - out['A']['video_psnr_dB']))
        print('a video PSNR drop here means the shared DiT was disturbed: a trade, not a joint restoration')


if __name__ == '__main__':
    main()
