"""Sanity check for the registration code: fb-consistency of Reg between adjacent frames (should be high) vs the
revisit pair (ref -> kpost frame), on GT. Uses only build_memory_guides.Reg as-is."""
import json
import numpy as np
import build_memory_guides as B
from revisit_metrics import load
E = B.E
sel = json.loads((E / 'selection.json').read_text())['events']
for seq in ['leopard-18', 'goldfish-10', 'gecko-3', 'hippo-2', 'bottle-14']:
    e = next(x for x in sel if x['sequence'] == seq)
    ev = json.loads((E / seq / 'event.json').read_text()); fr = ev['frame_records']; k = ev['blocks']['kpost']
    fref = e['pairs_revisit_a'][len(e['pairs_revisit_a']) // 2]
    t = next(f for f in range(17 * k + 8, 17 * k + 22) if not fr[str(f)]['absent'])
    def fb(fa, fb_):
        ga, gb = load(E / seq / 'hr' / f'{fa:06d}.png'), load(E / seq / 'hr' / f'{fb_:06d}.png')
        r = B.Reg(ga, gb, fr[str(fa)]['bbox_hr_xywh'], fr[str(fb_)]['bbox_hr_xywh'])
        m = B.box_mask(fr[str(fa)]['bbox_hr_xywh'], ga.shape, r.rt)
        return round(float((r.fb_ok & m).sum() / max(m.sum(), 1)), 3)
    print(seq, 'adjacent(kpost t,t+1)', fb(t, t + 1), '| adjacent(ref,ref+1)', fb(fref, fref + 1) if str(fref + 1) in fr else None,
          '| revisit(t <- ref)', fb(t, fref), '| gap frames', t - fref)
