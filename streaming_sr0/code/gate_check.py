"""Diagnose the LR gate on bottle-14 (does not change any parameter): distribution of e inside fb-ok target pixels,
and the same statistic when W is the GT of frame t itself (perfect content) -- if e is large even for perfect
content, the gate is mis-specified (a bug); if only for registered content, it reflects true misregistration."""
import json
import numpy as np, cv2
import build_memory_guides as B
from revisit_metrics import load
E = B.E
seq = 'bottle-14'
ev = json.loads((E / seq / 'event.json').read_text()); fr = ev['frame_records']
sel = json.loads((E / 'selection.json').read_text())['events']; e = next(x for x in sel if x['sequence'] == seq)
fref = e['pairs_revisit_a'][len(e['pairs_revisit_a']) // 2]; k = ev['blocks']['kpost']
g_ref = load(E / seq / 'hr' / f'{fref:06d}.png'); hw = g_ref.shape[:2]
for f in [17 * k + 8, 17 * k + 15]:
    lr = load(E / seq / 'lr' / f'{f:06d}.png'); base = B.up(lr, hw); g_t = load(E / seq / 'hr' / f'{f:06d}.png')
    box = fr[str(f)]['bbox_hr_xywh']; ro = B.Reg(g_t, g_ref, box, fr[str(fref)]['bbox_hr_xywh'])
    x0, y0, x1, y1 = ro.rt
    def err(region_img):
        W = base.copy(); W[y0:y1, x0:x1] = np.clip(region_img, 0, 1)
        return cv2.GaussianBlur(np.abs(B.down_up(W) - base).mean(-1), (0, 0), 2.0)[y0:y1, x0:x1]
    m = ro.fb_ok & B.box_mask(box, hw, ro.rt)
    e_reg, e_gt = err(ro.warp(g_ref)), err(g_t[y0:y1, x0:x1])
    q = lambda x: [round(float(np.percentile(x[m], p)), 4) for p in (10, 50, 90)]
    print(f, 'fb-ok px', int(m.sum()), 'e(registered ref) p10/50/90', q(e_reg), 'pass', round(float((e_reg[m] < 0.04).mean()), 3),
          '| e(GT itself)', q(e_gt), 'pass', round(float((e_gt[m] < 0.04).mean()), 3))
