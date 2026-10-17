"""Temporal check (the GAN lesson): did the extra detail come with extra flicker?
Within the target box, frame-to-frame difference of the generated band, normalised by GT's own
frame-to-frame difference. ~1 means the output changes between frames as much as the real scene does;
>>1 means added flicker. No registration: consecutive frames of the same block."""
import json
import numpy as np
from revisit_metrics import ROOT, load, unobserved
from stage1_metrics import folder
E = ROOT / 'data/events'
CONDS = ['classic_none', 'fl2va', 'lora_innerdc', 'lora_base']
rows = []
for e in json.loads((E / 'selection.json').read_text())['events']:
    seq = e['sequence']
    ev = json.loads((E / seq / 'event.json').read_text()); k = ev['blocks']['kpost']
    fs = [f for f in range(17*k+5, 17*k+22) if not ev['frame_records'][str(f)]['absent']
          and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
    pairs = [(a, b) for a, b in zip(fs, fs[1:]) if b == a + 1]
    if len(pairs) < 3: continue
    r = {'sequence': seq, 'n_pairs': len(pairs)}
    acc = {c: [] for c in CONDS}; gtacc = []
    for a, b in pairs:
        ga, gb = load(E/seq/'hr'/f'{a:06d}.png'), load(E/seq/'hr'/f'{b:06d}.png')
        x, y, w, h = ev['frame_records'][str(a)]['bbox_hr_xywh']
        m = np.zeros(ga.shape[:2], bool)
        m[max(0,int(y)):min(ga.shape[0],int(y+h)), max(0,int(x)):min(ga.shape[1],int(x+w))] = True
        dgt = float(((unobserved(ga) - unobserved(gb))[m] ** 2).mean()); gtacc.append(dgt)
        for c in CONDS:
            ya, yb = load(folder(c,seq,k)/f'{a:06d}.png'), load(folder(c,seq,k)/f'{b:06d}.png')
            acc[c].append(float(((unobserved(ya) - unobserved(yb))[m] ** 2).mean()))
    gt = float(np.mean(gtacc))
    for c in CONDS: r[c] = float(np.mean(acc[c])) / max(gt, 1e-12)
    rows.append(r); print(json.dumps({k2: (round(v,3) if isinstance(v,float) else v) for k2,v in r.items()}), flush=True)
out = {'n_events': len(rows), 'flicker_ratio_vs_gt': {c: float(np.mean([r[c] for r in rows])) for c in CONDS}}
(ROOT/'results/stage1_metrics'/'temporal.json').write_text(json.dumps({**out, 'per_event': rows}, indent=1))
print(json.dumps(out, indent=1))
