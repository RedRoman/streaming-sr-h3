"""Stage-5 visual check: the accumulation itself.

One row per mode, one column per block, each cell the LAST frame that block newly emits, cropped to a fixed
textured window. Reading left to right along the `cache` row is the accumulation; the other rows are the
controls that do not accumulate. A GT row is included so over-sharpening is judged against the real texture.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
# CPU-only: h3_comfy is deliberately not imported, it pulls in torch_npu and wants a card
ROOT = Path('/cache/zihaohe/streaming_sr0')

D = ROOT / 'data/stage2_eval'
MODES = ['GT', 'indep', 'cache', 'cache_anchor']


def mdir(m):
    return {'cache': ROOT / 'results/stage5_blocks/cache/indep',
            'cache_anchor': ROOT / 'results/stage5_blocks/cache_anchor/indep',
            'causal': ROOT / 'results/stage4_blocks/causal/indep',
            'indep': ROOT / 'results/stage2_blocks/indep'}[m]


p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=ROOT / 'results/stage5_panels')
p.add_argument('--only', nargs='*')
p.add_argument('--win', type=int, default=256, help='side of the crop window in HR pixels')
p.add_argument('--cell', type=int, default=200)
a = p.parse_args()
a.out.mkdir(parents=True, exist_ok=True)
man = json.loads((D / 'manifest.json').read_text())
blocks = man['blocks']

for name in man['clips']:
    if a.only and name not in a.only:
        continue
    modes = [m for m in MODES if m == 'GT' or (mdir(m) / name / 'stream_result.json').exists()]
    if len(modes) < 3:
        continue
    g0 = np.asarray(Image.open(D / name / 'hr' / '000000.png').convert('RGB'))
    H, W = g0.shape[:2]
    # pick the window with the most high-frequency energy: accumulation shows up in texture, not flat sky
    gr = g0.astype(np.float32).mean(-1)
    best, bxy = -1, (0, 0)
    for y in range(0, H - a.win + 1, 64):
        for x in range(0, W - a.win + 1, 64):
            v = float(np.abs(np.diff(gr[y:y + a.win, x:x + a.win], axis=0)).mean()
                      + np.abs(np.diff(gr[y:y + a.win, x:x + a.win], axis=1)).mean())
            if v > best:
                best, bxy = v, (x, y)
    x0, y0 = bxy
    rows = []
    for m in modes:
        cells = []
        for b in blocks:
            f = b['new'][1] - 1                      # last frame this block newly emits
            src = (D / name / 'hr' / f'{f:06d}.png') if m == 'GT' else (mdir(m) / name / f"block{b['k']:03d}" / f'{f:06d}.png')
            im = Image.open(src).convert('RGB').crop((x0, y0, x0 + a.win, y0 + a.win)).resize((a.cell, a.cell), Image.LANCZOS)
            cells.append(np.asarray(im))
        rows.append(np.concatenate(cells, axis=1))
    grid = np.concatenate(rows, axis=0)
    pad_l, pad_t = 96, 20
    canvas = Image.new('RGB', (grid.shape[1] + pad_l, grid.shape[0] + pad_t), 'white')
    canvas.paste(Image.fromarray(grid), (pad_l, pad_t))
    d = ImageDraw.Draw(canvas)
    for i, b in enumerate(blocks):
        d.text((pad_l + i * a.cell + 4, 4), f"block {b['k']}", fill='black')
    for j, m in enumerate(modes):
        d.text((4, pad_t + j * a.cell + a.cell // 2), m, fill='black')
    canvas.save(a.out / f'{name}.png')
    print(json.dumps({'clip': name, 'window_xy': [x0, y0], 'modes': modes}), flush=True)
