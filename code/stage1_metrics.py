"""Stage-1 metrics: trained LoRA vs the frozen baseline on the same 32 held-out kpost blocks (PROTOCOL_stage1.md)."""
import argparse
import json
from pathlib import Path

import numpy as np

from memory_metrics import paired
from ref_metrics import KEYS, frame_metrics
from revisit_metrics import ROOT, load

E = ROOT / 'data/events'
CONDS = ['fl2va', 'lora_innerdc', 'lora_base', 'classic_none']


def folder(cond, seq, k):
    if cond == 'fl2va':
        return ROOT / 'results/step0_blocks/innerdc' / seq / f'block{k:03d}'
    if cond == 'classic_none':
        return ROOT / 'data/step2_guides' / seq / 'classic_none'
    return ROOT / 'results/stage1_blocks' / cond / seq / f'block{k:03d}'


def done(cond, seq, k):
    d = folder(cond, seq, k)
    return (d / f'{17 * k + 21:06d}.png').exists() if cond == 'classic_none' else (d / 'result.json').exists()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/stage1_metrics')
    p.add_argument('--conds', nargs='+', default=CONDS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((E / 'selection.json').read_text())['events']
    per_event = []
    for e in sel:
        seq = e['sequence']
        ev = json.loads((E / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        conds = [c for c in a.conds if done(c, seq, k)]
        if 'lora_innerdc' not in conds or 'fl2va' not in conds:
            continue
        frames = [f for f in range(17 * k + 5, 17 * k + 22)
                  if not ev['frame_records'][str(f)]['absent'] and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
        if not frames:
            continue
        rows = {c: [] for c in conds}
        for f in frames:
            g = load(E / seq / 'hr' / f'{f:06d}.png'); lr = load(E / seq / 'lr' / f'{f:06d}.png')
            box = ev['frame_records'][str(f)]['bbox_hr_xywh']
            for c in conds:
                rows[c].append(frame_metrics(load(folder(c, seq, k) / f'{f:06d}.png'), g, lr, box))
        r = {'sequence': seq, 'frames': frames, 'conds': {c: {kk: float(np.mean([x[kk] for x in rows[c]])) for kk in KEYS} for c in conds}}
        (a.out / f'{seq}.json').write_text(json.dumps({**r, 'per_frame': rows}, indent=1))
        per_event.append(r)
        print(json.dumps({'event': seq, **{c: round(r['conds'][c]['psnr_box'], 3) for c in conds}}), flush=True)
    summary = {'n_events': len(per_event), 'means': {}, 'comparisons': {}}
    for c in a.conds:
        evc = [r for r in per_event if c in r['conds']]
        if evc:
            summary['means'][c] = {kk: float(np.mean([r['conds'][c][kk] for r in evc])) for kk in KEYS} | {'n_events': len(evc)}
    for x, y in [('lora_innerdc', 'fl2va'), ('lora_base', 'fl2va'), ('lora_innerdc', 'lora_base'),
                 ('lora_innerdc', 'classic_none'), ('fl2va', 'classic_none')]:
        summary['comparisons'][f'{x} - {y}'] = paired(per_event, x, y, KEYS)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
