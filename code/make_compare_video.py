"""Side-by-side comparison videos of the current arm against both baselines.

The numbers say we lead FlashVSR+ on five of six metrics with the sixth (GT-normalised range-band warp)
against us. A still panel cannot show which of those a viewer actually notices, and warp is a temporal
quantity -- it is invisible in a frame. So this writes video, at the real frame rate, with every arm on
the same clip and the same frames.

Layout is a 2x3 grid so each tile keeps its native 480x832 and nothing is resampled for display:

    LR (bicubic x4)   InfVSR            FlashVSR+
    ours (dcxl)       ours (dck3)       ground truth

Clip choice is a rule, not a preference: --pick worst takes the clips where our LPIPS is furthest ABOVE
FlashVSR+'s, so the cases that argue against us are the ones shown. --pick best and --pick median are
there for completeness. Per-clip LPIPS for every arm is burned into the corner of each tile so the video
cannot be read more favourably than the measurement.

Both ours arms are included on purpose: dcxl beat dck3 by only 0.0020 LPIPS, and whether that is
perceptible is exactly the kind of question a table cannot answer.
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
    ('FlashVSR+ (approx)', ROOT / 'results/baseline_flashvsr_plus', 'flashvsr_plus'),
    ('ours 13.3M K=3', ROOT / 'results/deploy_runs/t4_xl_ho', 'dcxl'),
    ('ours 4.7M K=3', ROOT / 'results/deploy_runs/t4_k3_ho', 'dck3'),
    ('ground truth', None, None),
]
COLS, ROWS = 3, 2


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def arm_frame(root, name, t, hw):
    if root is None:
        return None
    for p in (root / name / 'stream' / f'{t:06d}.png', root / name / f'{t:06d}.png'):
        if p.exists():
            a = Image.open(p).convert('RGB')
            return np.asarray(a if a.size == (hw[1], hw[0]) else a.resize((hw[1], hw[0]), Image.BICUBIC))
    return None


def label(img, text, sub=None):
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    d.text((8, 4), text, fill=(255, 255, 255))
    if sub:
        d.text((8, 19), sub, fill=(170, 220, 170))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--judge', default=str(ROOT / 'results/deploy/judge_xl.json'))
    ap.add_argument('--pick', default='worst', choices=['worst', 'best', 'median'])
    ap.add_argument('--clips', type=int, default=3)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--crf', type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    j = json.loads(Path(a.judge).read_text())
    rows = [r for r in j['per_clip'] if r['degradation'] == 'realistic'
            and 'dcxl' in r['arms'] and 'flashvsr_plus' in r['arms']]
    # ours minus FlashVSR+: positive means we are worse on this clip
    for r in rows:
        r['_d'] = r['arms']['dcxl']['lpips'] - r['arms']['flashvsr_plus']['lpips']
    rows.sort(key=lambda r: -r['_d'])
    if a.pick == 'best':
        sel = rows[-a.clips:][::-1]
    elif a.pick == 'median':
        m = len(rows) // 2
        sel = rows[max(0, m - a.clips // 2):][:a.clips]
    else:
        sel = rows[:a.clips]
    print(json.dumps({'picked': a.pick, 'clips': [
        {'clip': r['clip'], 'lpips_ours': round(r['arms']['dcxl']['lpips'], 4),
         'lpips_fvp': round(r['arms']['flashvsr_plus']['lpips'], 4),
         'ours_minus_fvp': round(r['_d'], 4)} for r in sel]}, indent=1), flush=True)

    written = []
    for r in sel:
        name = r['clip']
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T, hw = meta['frames'], (meta['hr_wh'][1], meta['hr_wh'][0])
        gt_dir, lr_dir = D_ / name / 'hr', D_ / name / 'lr'
        h, w = hw
        canvas_hw = (ROWS * h, COLS * w)
        dst = OUT / f'{a.pick}_{name}.mp4'
        # Encode through PyAV rather than the ffmpeg binary: the system build on this host ships no
        # libx264 (only h264_v4l2m2m, a V4L2 hardware wrapper that is not usable here) and rejects
        # -preset outright, while PyAV's bundled libx264 is present and is already what degrade.py and
        # the FlashVSR+ driver use. Frames are fed one at a time; a 6-tile 960x2496 canvas over 124
        # frames would be about 2 GB if staged as PNGs.
        import av
        import torch
        import torch.nn.functional as F
        container = av.open(str(dst), mode='w')
        stream = container.add_stream('libx264', rate=a.fps)
        stream.width, stream.height = canvas_hw[1], canvas_hw[0]
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': str(a.crf)}
        n_ok = 0
        for t in range(T):
            tiles = []
            for title, root, key in ARMS:
                if title.startswith('LR'):
                    z = torch.from_numpy(load(lr_dir / f'{t:06d}.png')).float().permute(2, 0, 1)[None] / 255
                    img = (F.interpolate(z, size=hw, mode='bicubic', align_corners=False)
                           .clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                elif title.startswith('ground'):
                    img = load(gt_dir / f'{t:06d}.png')
                else:
                    img = arm_frame(root, name, t, hw)
                if img is None:
                    img = np.zeros((h, w, 3), np.uint8)
                sub = None
                if key and key in r['arms']:
                    sub = 'LPIPS %.4f  PSNR %.2f' % (r['arms'][key]['lpips'], r['arms'][key]['psnr'])
                tiles.append(label(img, title, sub))
            grid = np.concatenate([np.concatenate(tiles[i * COLS:(i + 1) * COLS], axis=1)
                                   for i in range(ROWS)], axis=0)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(grid, format='rgb24')):
                container.mux(pkt)
            n_ok += 1
        for pkt in stream.encode():
            container.mux(pkt)
        container.close()
        print(json.dumps({'clip': name, 'frames': n_ok, 'canvas': list(canvas_hw),
                          'out': str(dst),
                          'bytes': dst.stat().st_size if dst.exists() else 0}), flush=True)
        written.append(str(dst))
    print()
    for p in written:
        print(p)


if __name__ == '__main__':
    main()
