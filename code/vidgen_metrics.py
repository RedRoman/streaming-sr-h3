"""Metrics + panels on the VIDGEN eval set (full-detail ground truth)."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from ref_metrics import bootstrap
from revisit_metrics import ROOT, load, lpips_dist, psnr, unobserved
D = ROOT / 'data/vidgen_eval'; B = ROOT / 'results/vidgen_blocks'
CONDS = ['classic', 'frozen', 'lora']
clips = json.loads((D/'manifest.json').read_text())['clips']
rows = []
for name in clips:
    m = json.loads((D/name/'meta.json').read_text()); fr = m['frames']
    pick = [fr[0], fr[len(fr)//2], fr[-1]]
    r = {'clip': name, 'category': m['category'], 'deg': m['degradation'].get('kind')}
    for c in CONDS:
        if not (B/c/name/'result.json').exists(): continue
        ps, lp, en, cs = [], [], [], []
        for f in fr:
            g = load(D/name/'hr'/f'{f:06d}.png'); y = load(B/c/name/f'{f:06d}.png')
            ps.append(psnr(y, g))
            ng, ny = unobserved(g), unobserved(y)
            en.append(float((ny**2).mean() / max(float((ng**2).mean()), 1e-12)))
            cs.append(float((ny*ng).sum() / max(np.sqrt((ny**2).sum()*(ng**2).sum()), 1e-12)))
            if f in pick: lp.append(lpips_dist(y, g))
        r[c] = {'psnr': float(np.mean(ps)), 'lpips': float(np.mean(lp)),
                'energy_ratio': float(np.mean(en)), 'cos': float(np.mean(cs))}
    rows.append(r); print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv,4) for kk,vv in v.items()}) for k,v in r.items()}), flush=True)
out = {'n': len(rows), 'per_clip': rows, 'means': {}}
for c in CONDS:
    rs = [r for r in rows if c in r]
    if rs: out['means'][c] = {k: float(np.mean([r[c][k] for r in rs])) for k in ['psnr','lpips','energy_ratio','cos']} | {'n': len(rs)}
for x, y in [('lora','frozen'), ('lora','classic'), ('frozen','classic')]:
    d = {}
    rs = [r for r in rows if x in r and y in r]
    for k in ['psnr','lpips','energy_ratio','cos']:
        diffs = [r[x][k]-r[y][k] for r in rs]
        better = sum((v<0) if k=='lpips' else (v>0) for v in diffs)
        d[k] = {'mean': float(np.mean(diffs)), 'ci95': bootstrap(diffs), 'n_better': better, 'n': len(diffs)}
    out.setdefault('comparisons', {})[f'{x} - {y}'] = d
(ROOT/'results/vidgen_metrics.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))
# panels: GT | classic | frozen | lora, centre 300px crop, nearest-upscaled
P = ROOT/'results/vidgen_panels'; P.mkdir(exist_ok=True)
for name in clips:
    m = json.loads((D/name/'meta.json').read_text()); f = m['frames'][len(m['frames'])//2]
    g = load(D/name/'hr'/f'{f:06d}.png'); H, W = g.shape[:2]
    cy, cx = H//2, W//2; s = 150
    y0, y1, x0, x1 = cy-s, cy+s, cx-s, cx+s
    cols, imgs = ['GT'], [g]
    for c in CONDS:
        if (B/c/name/'result.json').exists(): cols.append(c); imgs.append(load(B/c/name/f'{f:06d}.png'))
    crops = [np.asarray(Image.fromarray(np.uint8(np.clip(im[y0:y1, x0:x1],0,1)*255+.5)).resize((300,300), Image.NEAREST)) for im in imgs]
    row = np.concatenate(crops, axis=1)
    canvas = Image.new('RGB', (row.shape[1], row.shape[0]+22), 'white'); d = ImageDraw.Draw(canvas)
    for i, c in enumerate(cols): d.text((i*300+4, 5), c, fill='black')
    canvas.paste(Image.fromarray(row), (0, 22)); canvas.save(P/f'{name}.png')
print('panels', len(clips))
