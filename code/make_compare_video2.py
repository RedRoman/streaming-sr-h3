"""Comparison videos over every arm that exists on the held-out set, plus a zoomed sheet.

Eight tiles in a 4x2 grid, all native 480x832 so nothing is resampled for display:

    LR (bicubic x4)   InfVSR            FlashVSR            FlashVSR+
    ours dc=y         ours 4-step       ours 2-step         ground truth

`ours dc=y` is the ablation control -- innerDC projecting onto the raw observation instead of onto
R_phi(y) -- so the figure shows the mechanism's own effect next to the published baselines rather than
only the final number. `ours 4-step` vs `ours 2-step` is the step change that turned out to improve
quality as well as latency.

A second sheet writes a 256x256 zoom of the same tiles. The full frame at 480x832 is too small on screen
to judge faces or text, and this project has a standing rule that eye artefacts are unacceptable -- that
cannot be checked on a thumbnail. The crop is chosen by a rule, not by eye: the window with the most
ground-truth null-band energy, i.e. the most fine detail that no method can read off the observation.

Clip choice is also a rule. --pick worst takes the clips where our LPIPS is furthest ABOVE FlashVSR+'s,
so the cases that argue against us are the ones shown.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / 'data/real_eval/heldout'
OUT = ROOT / 'results/compare_video'

ARMS = [
    ('LR (bicubic x4)', None, None),
    ('InfVSR', ROOT / 'results/baseline_infvsr', 'infvsr'),
    ('FlashVSR (approx)', ROOT / 'results/baseline_flashvsr', 'flashvsr'),
    ('FlashVSR+ (approx)', ROOT / 'results/baseline_flashvsr_plus', 'flashvsr_plus'),
    ('ours: innerDC->y', ROOT / 'results/deploy_runs/t4_dcy_ho', 'dcy'),
    ('ours 4-step', ROOT / 'results/deploy_runs/t4_k3_ho', 'dck3'),
    ('ours 2-step (shipped)', ROOT / 'results/deploy_runs/t4_plain2', 'plain2'),
    ('ground truth', None, None),
]
COLS, ROWS = 4, 2
CROP = 256


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def arm_frame(root, name, t, hw):
    if root is None:
        return None
    for p in (root / name / 'stream' / f'{t:06d}.png', root / name / f'{t:06d}.png'):
        if p.exists():
            im = Image.open(p).convert('RGB')
            return np.asarray(im if im.size == (hw[1], hw[0]) else im.resize((hw[1], hw[0]), Image.BICUBIC))
    return None


def label(img, text, sub=None, scale=1):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    h = 34 * scale
    d.rectangle([0, 0, im.width, h], fill=(0, 0, 0))
    d.text((8, 4 * scale), text, fill=(255, 255, 255))
    if sub:
        d.text((8, 19 * scale), sub, fill=(170, 220, 170))
    return np.asarray(im)


def null_energy_window(gt, crop):
    """Top-left corner of the crop with the most GT null-band energy, on a coarse grid."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(gt).permute(2, 0, 1)[None].float() / 255
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    n = (t - u).square().mean(1)[0]
    best, by, bx = -1.0, 0, 0
    for y in range(0, h - crop + 1, 32):
        for x in range(0, w - crop + 1, 32):
            e = float(n[y:y + crop, x:x + crop].mean())
            if e > best:
                best, by, bx = e, y, x
    return by, bx


def grid(tiles, cols, rows):
    return np.concatenate([np.concatenate(tiles[i * cols:(i + 1) * cols], axis=1) for i in range(rows)],
                          axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', default=str(ROOT / 'results/deploy/judge_plain2.json'))
    ap.add_argument('--ref', default='plain2', help='our arm used to rank clips')
    ap.add_argument('--pick', default='worst', choices=['worst', 'best', 'median'])
    ap.add_argument('--clips', type=int, default=2)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--crf', type=int, default=16)
    ap.add_argument('--audio', action='store_true', help='mux the ground-truth soundtrack')
    ap.add_argument('--zoom', action='store_true', help='also write the 256x256 null-energy crop sheet')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    j = json.loads(Path(a.judge).read_text())
    have = set(j['arms'])
    rows = [r for r in j['per_clip'] if r['degradation'] == 'realistic'
            and r['arms'].get(a.ref) and r['arms'].get('flashvsr_plus')]
    for r in rows:
        r['_d'] = r['arms'][a.ref]['lpips'] - r['arms']['flashvsr_plus']['lpips']
    rows.sort(key=lambda r: -r['_d'])
    sel = rows[:a.clips] if a.pick == 'worst' else (
        rows[-a.clips:][::-1] if a.pick == 'best' else rows[len(rows) // 2:][:a.clips])
    print(json.dumps({'pick': a.pick, 'arms_in_judge': sorted(have), 'clips': [
        {'clip': r['clip'], 'ours': round(r['arms'][a.ref]['lpips'], 4),
         'fvp': round(r['arms']['flashvsr_plus']['lpips'], 4),
         'ours_minus_fvp': round(r['_d'], 4)} for r in sel]}, indent=1), flush=True)

    import av
    import torch
    import torch.nn.functional as F
    for r in sel:
        name = r['clip']
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T, (H, W) = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        gt0 = load(D_ / name / 'hr' / '000000.png')
        cy, cx = null_energy_window(gt0, CROP) if a.zoom else (0, 0)

        for sheet in (['full'] + (['zoom'] if a.zoom else [])):
            z = sheet == 'zoom'
            th, tw = (CROP, CROP) if z else (H, W)
            dst = OUT / f'{a.pick}_{name}_{sheet}.mp4'
            c = av.open(str(dst), mode='w')
            st = c.add_stream('libx264', rate=a.fps)
            st.width, st.height = COLS * tw, ROWS * th
            st.pix_fmt = 'yuv420p'
            st.options = {'crf': str(a.crf)}
            # Ground-truth audio on the track, so the clips can be watched with sound. This is NOT an
            # audio restoration arm: the sampler does emit an audio latent, but run_deploy discards it
            # and the audio channel is conditioned on zeros, so whatever it produces is unconditional
            # generation rather than a restoration of this clip's soundtrack. Labelled in the filename.
            ast = None
            ag = D_ / name / 'audio_gt.npy'
            if a.audio and ag.exists():
                wav = np.load(ag)                                  # [2, L] float32 at 32 kHz
                ast = c.add_stream('aac', rate=32000)
                ast.layout = 'stereo'
            n_ok = 0
            for t in range(T):
                tiles = []
                for title, root, key in ARMS:
                    if title.startswith('LR'):
                        lr = torch.from_numpy(load(D_ / name / 'lr' / f'{t:06d}.png')).float().permute(2, 0, 1)[None] / 255
                        img = (F.interpolate(lr, size=(H, W), mode='bicubic', align_corners=False)
                               .clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    elif title.startswith('ground'):
                        img = load(D_ / name / 'hr' / f'{t:06d}.png')
                    else:
                        img = arm_frame(root, name, t, (H, W))
                    if img is None:
                        img = np.zeros((H, W, 3), np.uint8)
                    if z:
                        img = img[cy:cy + CROP, cx:cx + CROP]
                    sub = None
                    if key and key in r['arms'] and r['arms'][key]:
                        sub = 'LPIPS %.4f' % r['arms'][key]['lpips']
                    tiles.append(label(img, title, sub))
                for pkt in st.encode(av.VideoFrame.from_ndarray(grid(tiles, COLS, ROWS), format='rgb24')):
                    c.mux(pkt)
                n_ok += 1
            for pkt in st.encode():
                c.mux(pkt)
            if ast is not None:
                dur = T / a.fps
                cut = wav[:, :int(dur * 32000)]
                fr = av.AudioFrame.from_ndarray(
                    np.ascontiguousarray(cut.T.reshape(-1)[None].astype('float32')),
                    format='flt', layout='stereo')
                fr.sample_rate = 32000
                for pkt in ast.encode(fr):
                    c.mux(pkt)
                for pkt in ast.encode():
                    c.mux(pkt)
            c.close()
            print(json.dumps({'clip': name, 'sheet': sheet, 'frames': n_ok,
                              'canvas': [ROWS * th, COLS * tw],
                              'crop_yx': [cy, cx] if z else None,
                              'out': str(dst), 'bytes': dst.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
