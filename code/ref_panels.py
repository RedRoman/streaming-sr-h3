"""Step 1 panels: target crops on fixed frames of kpost (first, middle, last evaluable frame), columns
GT | fl2va | r2v_none | r2v_gt | r2v_self | r2v_wrong (those that exist), plus the reference image used for gt."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from revisit_metrics import ROOT, load, region

E = ROOT / 'data/events'
ORDER = ['GT', 'fl2va', 'r2v_none', 'r2v_gt', 'r2v_self', 'r2v_wrong']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/step1_panels')
    p.add_argument('--only', nargs='*')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((E / 'selection.json').read_text())['events']
    for e in sel:
        seq = e['sequence']
        if a.only and seq not in a.only:
            continue
        ev = json.loads((E / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        frames = [f for f in range(17 * k + 5, 17 * k + 22)
                  if not ev['frame_records'][str(f)]['absent'] and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
        if not frames:
            continue
        pick = sorted({frames[0], frames[len(frames) // 2], frames[-1]})
        dirs = {'fl2va': ROOT / 'results/step0_blocks/innerdc' / seq / f'block{k:03d}'}
        for c in ORDER[2:]:
            dirs[c] = ROOT / 'results/step1_blocks' / c / seq / f'block{k:03d}'
        cols = ['GT'] + [c for c in ORDER[1:] if (dirs[c] / 'result.json').exists()]
        rows = []
        for f in pick:
            g = load(E / seq / 'hr' / f'{f:06d}.png')
            x0, y0, x1, y1 = region(ev['frame_records'][str(f)]['bbox_hr_xywh'], g.shape)
            crops = [g[y0:y1, x0:x1]] + [load(dirs[c] / f'{f:06d}.png')[y0:y1, x0:x1] for c in cols[1:]]
            h = 240; crops = [np.asarray(Image.fromarray(np.uint8(c * 255 + .5)).resize((max(1, round(c.shape[1] * h / c.shape[0])), h), Image.LANCZOS)) for c in crops]
            rows.append(np.concatenate(crops, axis=1))
        width = max(r.shape[1] for r in rows)
        canvas = Image.new('RGB', (width, 24 + sum(r.shape[0] for r in rows)), 'white')
        d = ImageDraw.Draw(canvas)
        cw = rows[0].shape[1] // len(cols)
        for i, c in enumerate(cols):
            d.text((i * cw + 4, 6), c, fill='black')
        y = 24
        for r in rows:
            canvas.paste(Image.fromarray(r), (0, y)); y += r.shape[0]
        canvas.save(a.out / f'{seq}.png')
        print(json.dumps({'panel': seq, 'frames': pick, 'cols': cols}), flush=True)


if __name__ == '__main__':
    main()
