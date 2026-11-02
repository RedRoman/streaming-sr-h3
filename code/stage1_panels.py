"""Stage-1 visual check: target crops, columns GT | classic_none | fl2va(frozen) | lora_innerdc.
Also a 4x zoom of a fixed sub-window so texture (not just sharpness) can be judged."""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from revisit_metrics import ROOT, load, region
from stage1_metrics import folder
E = ROOT / 'data/events'
COLS = ['GT', 'classic_none', 'fl2va', 'lora_innerdc']
p = argparse.ArgumentParser()
p.add_argument('--out', type=Path, default=ROOT / 'results/stage1_panels')
p.add_argument('--only', nargs='*'); p.add_argument('--zoom', type=int, default=0)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
for e in json.loads((E / 'selection.json').read_text())['events']:
    seq = e['sequence']
    if a.only and seq not in a.only: continue
    ev = json.loads((E / seq / 'event.json').read_text()); k = ev['blocks']['kpost']
    fr = [f for f in range(17*k+5, 17*k+22) if not ev['frame_records'][str(f)]['absent']
          and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
    if not fr: continue
    f = fr[len(fr)//2]
    g = load(E / seq / 'hr' / f'{f:06d}.png')
    box = ev['frame_records'][str(f)]['bbox_hr_xywh']
    x0, y0, x1, y1 = region(box, g.shape)
    if a.zoom:  # centre sub-window, nearest-upscaled, to inspect texture
        cx, cy = (x0+x1)//2, (y0+y1)//2; h = a.zoom//2
        x0, x1 = max(0, cx-h), min(g.shape[1], cx+h); y0, y1 = max(0, cy-h), min(g.shape[0], cy+h)
    crops = []
    for c in COLS:
        img = g if c == 'GT' else load(folder(c, seq, k) / f'{f:06d}.png')
        crop = np.uint8(np.clip(img[y0:y1, x0:x1], 0, 1)*255 + .5)
        H = 360
        im = Image.fromarray(crop)
        im = im.resize((max(1, round(crop.shape[1]*H/crop.shape[0])), H), Image.NEAREST if a.zoom else Image.LANCZOS)
        crops.append(np.asarray(im))
    w = min(c.shape[1] for c in crops)
    row = np.concatenate([c[:, :w] for c in crops], axis=1)
    canvas = Image.new('RGB', (row.shape[1], row.shape[0]+22), 'white')
    d = ImageDraw.Draw(canvas)
    for i, c in enumerate(COLS): d.text((i*w+4, 5), c, fill='black')
    canvas.paste(Image.fromarray(row), (0, 22))
    canvas.save(a.out / f'{seq}{"_zoom" if a.zoom else ""}.png')
    print(json.dumps({'panel': seq, 'frame': f, 'crop_wh': [x1-x0, y1-y0]}), flush=True)
