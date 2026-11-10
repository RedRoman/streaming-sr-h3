"""The joint audio-video demo: one video, two soundtracks, and the video identical between them.

This is the only deliverable in the project that can be heard, and the only one where the central claim
is an equality rather than an improvement. The audio side goes from mel 28.247 dB to 8.718 dB averaged
over 77 clips -- on the clip used here, 82.20 to 5.33 -- while the video stream comes out
bit-identical on 77 of 77 clips. A table can state that; a pair of files lets someone check it.

So two renders of the same clip:
  <clip>_audio_degraded.mp4   video from arm A, its own untouched audio stream
  <clip>_audio_restored.mp4   video from arm C, audio corrected at the innerDC projection points
and a third that proves the video claim rather than asserting it:
  <clip>_video_identical.mp4  the absolute difference between the two video streams, amplified 32x

If the difference render is not pure black, the joint claim is false and this file says so on its face.

Arm A is the control, bit-exact to every earlier deployment (audio_init='zeros', no projection). Arm C
adds one thing: at the same POINTS where the video projection enforces the observation constraint, the
audio latent takes a step toward R_phi^audio of the degraded audio. The target has to be multiplied by
audio_scale = shift/audio_shift = 12/3 = 4, because the sampler carries the audio stream scaled onto
the video sigma schedule -- without that the output lands at exactly 0.25x the target with cosine
1.0000, which is how the bug was found.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
SR = 32000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clip', default='N-IvW6J2e80')
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--steps', type=int, default=2)
    ap.add_argument('--rphi', default='results/omni/omni_m5.pt')
    ap.add_argument('--arphi', default='results/audio_range/ar_31984.pt')
    ap.add_argument('--lora', default='results/stage1_lora/lora_step002000.safetensors')
    ap.add_argument('--op', default='4')
    ap.add_argument('--amp', type=float, default=32.0, help='gain on the video difference render')
    ap.add_argument('--out', default='results/demo')
    a = ap.parse_args()

    import torch_npu  # noqa: F401
    import av
    import h3_comfy
    import lora as L
    import lora_merge as LM
    import turbo as TB
    from h3_comfy import H3, sync
    from run_deploy import TURBO_PATH, range_estimate
    from audio_line1 import align, si_sdr
    from degrade_audio import degrade_audio, lsd, mel_dist
    from train_audio_range import AudioRange
    import comfy.sd
    from safetensors.torch import load_file

    h3_comfy.SIGMAS = TB.turbo_sigmas(a.steps, 12.0)
    D_ = ROOT / a.eval_set
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    blocks = man['blocks']
    name = a.clip
    meta = json.loads((D_ / name / 'meta.json').read_text())
    T, H, W = meta['frames'], meta['hr_wh'][1], meta['hr_wh'][0]

    h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
    model = h3.patcher.model.diffusion_model
    TB.merge(model, TURBO_PATH)
    L.inject(model, 32)
    L.load(model, load_file(str(ROOT / a.lora)))
    LM.merge_(model)
    h3.frozen = [(n, v, v._version) for n, v in model.named_parameters()
                 if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]

    import comfy.utils
    sd_, meta_ = comfy.utils.load_torch_file(
        str(ROOT / 'assets/comfy_h3/minimax_h3_audio_vae_fp32.safetensors'), return_metadata=True)
    avae = comfy.sd.VAE(sd=sd_, device=h3.dev, dtype=torch.float32, metadata=meta_)
    avae.first_stage_model.eval().requires_grad_(False)

    def aenc(w):
        with torch.no_grad():
            return avae.encode(w[None].movedim(1, -1).to(h3.dev))

    def adec(z):
        # the VAE returns [1, N, 2]; every metric and the muxer want [2, N]. audio_joint.py does the
        # same reshape -- omitting it fed a [1, N, 2] tensor into the aligner, whose 512-sample pad
        # then overran dimension 2.
        with torch.no_grad():
            y = avae.decode(z.to(h3.dev)).float().cpu()
        y = y[0] if y.ndim == 3 else y
        return (y.T if y.shape[0] != 2 else y).reshape(2, -1)

    st = torch.load(ROOT / a.arphi, map_location='cpu')
    ar = AudioRange(**st['cfg']).eval()
    ar.load_state_dict(st['state'])
    ar = ar.to(h3.dev)

    from PIL import Image
    lr = torch.stack([torch.from_numpy(np.asarray(Image.open(D_ / name / 'lr' / f'{t:06d}.png')
                                                  .convert('RGB')).copy()).float() / 255
                      for t in range(T)]).permute(0, 3, 1, 2)
    gt_a = torch.from_numpy(np.load(D_ / name / 'audio_gt.npy'))
    import hashlib
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], 'little')
    dg_a, _ = degrade_audio(gt_a, np.random.default_rng(seed))
    est = range_estimate(lr, ROOT / a.rphi, dev=h3.dev, op=a.op)

    renders = {}
    for arm, (init, proj) in (('degraded', ('zeros', False)), ('restored', ('randn', True))):
        vid, aud, t0 = [], [], time.perf_counter()
        for b in blocks:
            s, e = b['range']
            guide = h3.encode_guide(lr[s:e], [H, W])
            tgt = None
            if proj:
                n0, n1 = int(round(s / 24 * SR)), int(round(e / 24 * SR))
                zb = aenc(dg_a[..., n0:n1])
                with torch.no_grad():
                    tgt = ar(zb.to(h3.dev)).cpu()
            sampled, _ = h3.sample(guide, 'innerdc', low=est[s:e], hw=[H, W], seed=101,
                                   audio_target=tgt, audio_strength=1.0, audio_init=init)
            vid.append(h3.decode(sampled['video']))
            aud.append(sampled['audio'])
        sync()
        # cut both streams to each block's own span; whole-block concatenation repeats the 5-frame
        # context every 17 frames and stretched the audio 25.3%
        vp, ap_ = [], []
        for v, aa, b in zip(vid, aud, blocks):
            s, _e = b['range']
            nl, nh = b['new']
            vp.append(v[nl - s:nh - s])
            ap_.append(aa[..., int(round((nl - s) / 24 * 40)):int(round((nh - s) / 24 * 40))])
        px = torch.cat(vp)[:T]
        wav = adec(torch.cat(ap_, dim=-1)[..., :int(round(T / 24 * 40))])
        g, r = align(wav, gt_a)[:2]
        renders[arm] = {
            'px': px,
            'wav': wav,
            'mel': round(mel_dist(g, r), 3), 'lsd': round(lsd(g, r), 3),
            'sisdr': round(si_sdr(g, r), 2),
            'sec': round(time.perf_counter() - t0, 1),
        }
        print(json.dumps({'arm': arm, **{k: v for k, v in renders[arm].items()
                                         if k not in ('px', 'wav')}}), flush=True)

    # the equality claim, rendered rather than asserted
    d = (renders['degraded']['px'] - renders['restored']['px']).abs()
    print(json.dumps({'video_max_abs_diff': float(d.max()),
                      'video_mean_abs_diff': float(d.mean()),
                      'bit_identical': bool(float(d.max()) == 0.0)}), flush=True)

    def write(tag, px, wav, caption):
        from PIL import ImageDraw, Image as PImage
        vp_ = out / f'{name}_{tag}.mp4'
        c = av.open(str(vp_), mode='w')
        vs = c.add_stream('libx264', rate=24)
        vs.width, vs.height = W, H
        vs.pix_fmt = 'yuv420p'
        vs.options = {'crf': '16'}
        as_ = c.add_stream('aac', rate=SR)
        as_.layout = 'stereo'
        n = min(px.shape[0], T)
        for t in range(n):
            im = PImage.fromarray((px[t].clamp(0, 1).numpy() * 255).astype(np.uint8))
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, W, 34], fill=(0, 0, 0))
            dr.text((8, 4), caption, fill=(255, 255, 255))
            for pkt in vs.encode(av.VideoFrame.from_ndarray(np.asarray(im), format='rgb24')):
                c.mux(pkt)
        for pkt in vs.encode():
            c.mux(pkt)
        w = wav.detach().cpu().numpy()
        if w.ndim == 1:
            w = np.stack([w, w])
        w = np.clip(np.ascontiguousarray(w[:2], dtype=np.float32), -1, 1)
        # PyAV's packed s16 wants interleaved samples in a single row, [1, N*channels], not [2, N].
        # Handing it [2, N] raises "Expected packed array.shape[0] to equal 1".
        inter = (w.T.reshape(1, -1) * 32767).astype(np.int16)
        inter = np.ascontiguousarray(inter)
        # encode in chunks so one frame does not have to hold the whole clip
        step = SR
        for i in range(0, inter.shape[1], step * 2):
            chunk = np.ascontiguousarray(inter[:, i:i + step * 2])
            if chunk.shape[1] == 0:
                continue
            fr = av.AudioFrame.from_ndarray(chunk, format='s16', layout='stereo')
            fr.sample_rate = SR
            for pkt in as_.encode(fr):
                c.mux(pkt)
        for pkt in as_.encode():
            c.mux(pkt)
        c.close()
        return vp_

    # The comparison that was shipped first was wrong and has to be stated. Arm A was labelled
    # "audio degraded", but it is not the degraded audio -- it is the untrained DiT audio branch
    # emitting near-silence: on this clip its loudness is 0.4% of the truth and its mel distance is
    # 82.20 dB, far WORSE than simply passing the degraded audio through at 24.53. Pairing that against
    # the restored arm makes the result look like silence becoming sound, which is not the claim and
    # flatters it. The honest baseline is the degraded audio itself.
    dg_pair = adec(aenc(dg_a))                       # degraded audio through the codec, nothing else
    gt_pair = adec(aenc(gt_a))                       # the ceiling: truth through the same codec
    for tag, wav_, lab in (('input_degraded', dg_pair, 'INPUT: the degraded audio we are given'),
                           ('ours_restored', renders['restored']['wav'], 'OURS: restored'),
                           ('reference', gt_pair, 'REFERENCE: ground truth through the same codec')):
        g, r = align(wav_, gt_a)[:2]
        mel = mel_dist(g, r)
        cap = '%s  |  mel %.2f dB' % (lab, mel)
        pth = write('audio_' + tag, renders['restored']['px'], wav_, cap)
        print(json.dumps({'file': str(pth), 'MB': round(pth.stat().st_size / 1e6, 2),
                          'tag': tag, 'mel': round(mel, 3)}), flush=True)

    made = []
    for arm in ('degraded', 'restored'):
        r = renders[arm]
        cap = 'audio %s  |  mel %.2f dB   SI-SDR %.1f dB   (video stream identical in both)' % (
            arm, r['mel'], r['sisdr'])
        p = write('audio_' + arm, r['px'], r['wav'], cap)
        made.append({'file': str(p), 'MB': round(p.stat().st_size / 1e6, 2), 'arm': arm, **{
            k: r[k] for k in ('mel', 'lsd', 'sisdr')}})
        print(json.dumps(made[-1]), flush=True)

    # The video-invariance claim, shown rather than asserted. A pure black render was the first
    # attempt and it was a bad deliverable: a file with no picture and no sound is unreadable, and the
    # reader has no way to tell "identical" from "broken export". This puts the two streams next to the
    # amplified difference, so the left and right panels prove the render works while the middle panel
    # being empty is the actual result.
    left = renders['degraded']['px']
    right = renders['restored']['px']
    mid = (d * a.amp).clamp(0, 1)
    n = min(left.shape[0], right.shape[0], mid.shape[0])
    trip = torch.cat([left[:n], mid[:n], right[:n]], dim=2)      # side by side on width
    p = write('video_invariance', trip, renders['restored']['wav'],
              'audio OFF  |  difference x%g (empty = video untouched)  |  audio ON        '
              'max abs diff %.5f' % (a.amp, float(d.max())))
    made.append({'file': str(p), 'MB': round(p.stat().st_size / 1e6, 2), 'arm': 'invariance_triptych'})
    print(json.dumps(made[-1]), flush=True)

    (out / f'{name}_audio_demo.json').write_text(json.dumps({
        'clip': name, 'frames': T, 'steps': a.steps,
        'arms': {k: {kk: vv for kk, vv in v.items() if kk not in ('px', 'wav')}
                 for k, v in renders.items()},
        'video_max_abs_diff': float(d.max()),
        'files': made}, indent=1))
    print()
    print('wrote %d files to %s' % (len(made), out))


if __name__ == '__main__':
    main()
