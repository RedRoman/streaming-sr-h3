"""Side-by-side crops where we lose to InfVSR, to name the defect instead of guessing at it.

The metric table says we trail InfVSR by LPIPS +0.067 on realistic footage while producing *more*
unobserved-band energy (median 0.62 against 0.44) and a slightly better direction (cos 0.282 vs 0.338 --
theirs is better) . "More detail, worse perceptual score" has a small number of possible causes, and they
look different: wrong-texture hallucination, ringing around edges, blocking inherited from the compressed
observation, or temporal smear. One look at the worst clips settles it.

Crops are picked by local LPIPS gap rather than by hand, so the panel cannot be accused of cherry-picking:
for each clip the frame and the 256x256 window with the largest (ours - infvsr) LPIPS are shown.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
from revisit_metrics import lpips_dist                       # noqa: E402
import deploy_eval as DE                                     # noqa: E402

OUT = ROOT / 'results/judge_panels'
CROP = 256
ARMS = [('lr', None), ('esr', ROOT / 'results/baseline_frames'),
        ('infvsr', ROOT / 'results/baseline_infvsr'),
        ('ours', ROOT / 'results/deploy/s1_blend_rs'), ('gt', None)]


def crop_of(x, y0, x0):
    return x[y0:y0 + CROP, x0:x0 + CROP]


def main():
    clips = sys.argv[1].split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in clips:
        m = DE.meta(name)
        T = m['frames']
        # frames every 8 are the ones both arms were scored on; esr only has those
        best = None
        for t in range(0, T, 8):
            g = DE.load(DE.D_ / name / 'hr' / f'{t:06d}.png')
            o = DE.arm_frame(str(ROOT / 'results/deploy/s1_blend_rs'), name, t)
            iv = DE.arm_frame(str(ROOT / 'results/baseline_infvsr'), name, t)
            if o is None or iv is None:
                continue
            H, W = g.shape[:2]
            for y0 in range(0, H - CROP + 1, CROP // 2):
                for x0 in range(0, W - CROP + 1, CROP // 2):
                    gc, oc, ic = crop_of(g, y0, x0), crop_of(o, y0, x0), crop_of(iv, y0, x0)
                    d = lpips_dist(oc, gc) - lpips_dist(ic, gc)
                    if best is None or d > best[0]:
                        best = (d, t, y0, x0)
        if best is None:
            print(json.dumps({'clip': name, 'skip': 'no frames'})); continue
        d, t, y0, x0 = best
        tiles = []
        for tag, p in ARMS:
            if tag == 'gt':
                im = DE.load(DE.D_ / name / 'hr' / f'{t:06d}.png')
            elif tag == 'lr':
                im = DE.arm_frame('bicubic', name, t)
            else:
                im = DE.arm_frame(str(p), name, t)
            tiles.append((tag, crop_of(im, y0, x0) if im is not None else None))
        wpx = CROP * len(tiles)
        canvas = Image.new('RGB', (wpx, CROP + 18), (255, 255, 255))
        dr = ImageDraw.Draw(canvas)
        for i, (tag, im) in enumerate(tiles):
            if im is None:
                dr.text((i * CROP + 4, CROP + 4), f'{tag}: missing', fill=(200, 0, 0)); continue
            canvas.paste(Image.fromarray(np.uint8(im.numpy().clip(0, 1) * 255 + .5)), (i * CROP, 0))
            dr.text((i * CROP + 4, CROP + 4), tag, fill=(0, 0, 0))
        q = OUT / f'{name}_f{t:03d}_y{y0}_x{x0}.png'
        canvas.save(q)
        print(json.dumps({'clip': name, 'frame': t, 'crop': [y0, x0],
                          'lpips_gap_in_crop': round(float(d), 4), 'panel': str(q)}), flush=True)


if __name__ == '__main__':
    main()
