"""Step 2 metrics (PROTOCOL_step2.md). Same-frame comparisons against GT; no registration at evaluation time."""
import argparse
import json
from pathlib import Path

import numpy as np

from ref_metrics import KEYS, bootstrap, frame_metrics
from revisit_metrics import ROOT, load, psnr

E = ROOT / 'data/events'
H3_MODES = ['none_rerun', 'mem_gt_oflow', 'mem_gt_lrflow', 'mem_self_lrflow', 'mem_wrong_oflow', 'mem_gt_oflow_nogate', 'mem_wrong_oflow_nogate']
CLASSIC = ['classic_none', 'classic_gt_oflow', 'classic_gt_lrflow', 'classic_self_lrflow', 'classic_wrong_oflow', 'classic_gt_oflow_nogate', 'classic_wrong_oflow_nogate']
ALL = ['fl2va'] + H3_MODES + CLASSIC
KEYS2 = KEYS + ['psnr_memmask', 'psnr_memmask_nogate']


def folder(cond, seq, k):
    if cond == 'fl2va':
        return ROOT / 'results/step0_blocks/innerdc' / seq / f'block{k:03d}'
    if cond.startswith('classic_'):
        return ROOT / 'data/step2_guides' / seq / cond
    return ROOT / 'results/step2_blocks' / cond / seq / f'block{k:03d}'


def done(cond, seq, k):
    d = folder(cond, seq, k)
    return (d / 'result.json').exists() if not cond.startswith('classic_') else (d / f'{17 * k + 21:06d}.png').exists()


def paired(per_event, a, b, keys):
    out = {}
    ev = [r for r in per_event if a in r['conds'] and b in r['conds']]
    for kk in keys:
        diffs = [r['conds'][a][kk] - r['conds'][b][kk] for r in ev if r['conds'][a][kk] is not None and r['conds'][b][kk] is not None]
        if not diffs:
            continue
        better = sum((x < 0) if kk in ('lpips_box', 'lr_rmse') else (x > 0) for x in diffs)
        out[kk] = {'mean': float(np.mean(diffs)), 'ci95': bootstrap(diffs), 'n_better': int(better), 'n': len(diffs)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/step2_metrics')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((E / 'selection.json').read_text())['events']
    per_event = []
    for e in sel:
        seq = e['sequence']
        ev = json.loads((E / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        conds = [c for c in ALL if done(c, seq, k)]
        if 'none_rerun' not in conds:
            continue
        mask = np.load(ROOT / 'data/step2_guides' / seq / 'mask_gt_oflow.npy')
        mask_ng = np.load(ROOT / 'data/step2_guides' / seq / 'mask_gt_oflow_nogate.npy')
        frames = [f for f in range(17 * k + 5, 17 * k + 22)
                  if not ev['frame_records'][str(f)]['absent'] and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
        if not frames:
            continue
        rows = {c: [] for c in conds}
        for f in frames:
            g = load(E / seq / 'hr' / f'{f:06d}.png'); lr = load(E / seq / 'lr' / f'{f:06d}.png')
            box = ev['frame_records'][str(f)]['bbox_hr_xywh']; m = mask[f - 17 * k]
            for c in conds:
                y = load(folder(c, seq, k) / f'{f:06d}.png')
                r = frame_metrics(y, g, lr, box)
                r['psnr_memmask'] = psnr(y, g, m) if m.sum() >= 64 else None
                mn = mask_ng[f - 17 * k]
                r['psnr_memmask_nogate'] = psnr(y, g, mn) if mn.sum() >= 64 else None
                rows[c].append(r)
        mean = lambda xs, kk: (float(np.mean([x[kk] for x in xs if x[kk] is not None])) if any(x[kk] is not None for x in xs) else None)
        r = {'sequence': seq, 'frames': frames, 'conds': {c: {kk: mean(rows[c], kk) for kk in KEYS2} for c in conds},
             'coverage': json.loads((ROOT / 'data/step2_guides' / seq / 'record.json').read_text())['mean_coverage']}
        (a.out / f'{seq}.json').write_text(json.dumps({**r, 'per_frame': rows}, indent=1))
        per_event.append(r)
        print(json.dumps({'event': seq, **{c: round(r['conds'][c]['psnr_box'], 3) for c in conds}}), flush=True)
    summary = {'n_events': len(per_event), 'means': {}, 'comparisons': {},
               'coverage_mean': {v: float(np.mean([r['coverage'][v] for r in per_event])) for v in per_event[0]['coverage']} if per_event else {}}
    for c in ALL:
        evc = [r for r in per_event if c in r['conds']]
        if evc:
            summary['means'][c] = {kk: (float(np.mean([r['conds'][c][kk] for r in evc if r['conds'][c][kk] is not None]))
                                        if any(r['conds'][c][kk] is not None for r in evc) else None) for kk in KEYS2} | {'n_events': len(evc)}
    pairs = [('none_rerun', 'fl2va'), ('mem_gt_oflow', 'none_rerun'), ('mem_gt_oflow', 'mem_wrong_oflow'), ('mem_gt_oflow', 'classic_gt_oflow'),
             ('mem_gt_lrflow', 'none_rerun'), ('mem_self_lrflow', 'none_rerun'), ('mem_wrong_oflow', 'none_rerun'),
             ('classic_gt_oflow', 'classic_none'), ('none_rerun', 'classic_none'), ('mem_self_lrflow', 'classic_self_lrflow'),
             ('mem_gt_oflow_nogate', 'none_rerun'), ('mem_gt_oflow_nogate', 'mem_wrong_oflow_nogate'), ('mem_gt_oflow_nogate', 'classic_gt_oflow_nogate'),
             ('mem_wrong_oflow_nogate', 'none_rerun'), ('classic_gt_oflow_nogate', 'classic_none')]
    for x, y in pairs:
        summary['comparisons'][f'{x} - {y}'] = paired(per_event, x, y, KEYS2)
    noise = [abs(r['conds']['none_rerun']['psnr_box'] - r['conds']['fl2va']['psnr_box']) for r in per_event if 'fl2va' in r['conds']]
    summary['noise_floor_abs_psnr_box'] = {'mean': float(np.mean(noise)), 'max': float(np.max(noise))} if noise else None
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != 'means'}, indent=1))


if __name__ == '__main__':
    main()
