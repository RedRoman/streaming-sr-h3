"""Step 1 metrics (PROTOCOL_step1.md). Same-frame comparisons against GT only; no registration."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from revisit_metrics import ROOT, load, lpips_dist, psnr, region, unobserved

E = ROOT / 'data/events'
CONDS = ['fl2va', 'r2v_none', 'r2v_gt', 'r2v_self', 'r2v_wrong']
KEYS = ['psnr_box', 'lpips_box', 'energy_ratio_box', 'cos_box', 'psnr_full', 'lr_rmse']


def folder(blocks0, blocks1, cond, seq, k):
    if cond == 'fl2va':
        return blocks0 / 'innerdc' / seq / f'block{k:03d}'
    return blocks1 / cond / seq / f'block{k:03d}'


def lr_rmse(y, lr):
    t = torch.from_numpy(y).permute(2, 0, 1)[None].double()
    d = F.interpolate(t, size=lr.shape[:2], mode='bicubic', align_corners=False, antialias=True)[0].permute(1, 2, 0).numpy()
    return float(np.sqrt(((d - lr) ** 2).mean()))


def frame_metrics(y, g, lr, box):
    H, W = g.shape[:2]
    x, yy, w, h = box
    m = np.zeros((H, W), bool)
    m[max(0, int(yy)):min(H, int(yy + h)), max(0, int(x)):min(W, int(x + w))] = True
    ny, ng = unobserved(y)[m], unobserved(g)[m]
    x0, y0, x1, y1 = region(box, g.shape)
    return {'psnr_box': psnr(y, g, m), 'lpips_box': lpips_dist(y[y0:y1, x0:x1], g[y0:y1, x0:x1]),
            'energy_ratio_box': float((ny ** 2).mean() / max((ng ** 2).mean(), 1e-12)),
            'cos_box': float((ny * ng).sum() / max(np.sqrt((ny ** 2).sum() * (ng ** 2).sum()), 1e-12)),
            'psnr_full': psnr(y, g), 'lr_rmse': lr_rmse(y, lr)}


def bootstrap(v, n=10000, seed=0):
    v = np.asarray(v, float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--blocks0', type=Path, default=ROOT / 'results/step0_blocks')
    p.add_argument('--blocks1', type=Path, default=ROOT / 'results/step1_blocks')
    p.add_argument('--out', type=Path, default=ROOT / 'results/step1_metrics')
    p.add_argument('--conds', nargs='+', default=CONDS)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((E / 'selection.json').read_text())['events']
    per_event = []
    for e in sel:
        seq = e['sequence']
        ev = json.loads((E / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        conds = [c for c in a.conds if (folder(a.blocks0, a.blocks1, c, seq, k) / 'result.json').exists()]
        if 'r2v_none' not in conds:
            continue
        frames = [f for f in range(17 * k + 5, 17 * k + 22)
                  if not ev['frame_records'][str(f)]['absent'] and min(ev['frame_records'][str(f)]['bbox_hr_xywh'][2:]) >= 16]
        rows = {c: [] for c in conds}
        for f in frames:
            g = load(E / seq / 'hr' / f'{f:06d}.png'); lr = load(E / seq / 'lr' / f'{f:06d}.png')
            box = ev['frame_records'][str(f)]['bbox_hr_xywh']
            for c in conds:
                rows[c].append(frame_metrics(load(folder(a.blocks0, a.blocks1, c, seq, k) / f'{f:06d}.png'), g, lr, box))
        if not frames:
            continue
        r = {'sequence': seq, 'frames': frames, 'conds': {c: {kk: float(np.mean([x[kk] for x in rows[c]])) for kk in KEYS} for c in conds}}
        (a.out / f'{seq}.json').write_text(json.dumps({**r, 'per_frame': rows}, indent=1))
        per_event.append(r)
        print(json.dumps({'event': seq, 'n_frames': len(frames), **{c: round(r['conds'][c]['psnr_box'], 3) for c in conds}}), flush=True)
    summary = {'n_events': len(per_event), 'means': {}, 'vs_r2v_none': {}}
    for c in a.conds:
        ev_c = [r for r in per_event if c in r['conds']]
        if not ev_c:
            continue
        summary['means'][c] = {kk: float(np.mean([r['conds'][c][kk] for r in ev_c])) for kk in KEYS} | {'n_events': len(ev_c)}
        if c == 'r2v_none':
            continue
        d = {}
        for kk in KEYS:
            diffs = [r['conds'][c][kk] - r['conds']['r2v_none'][kk] for r in ev_c]
            better = sum((x < 0) if kk in ('lpips_box', 'lr_rmse') else (x > 0) for x in diffs)
            d[kk] = {'mean': float(np.mean(diffs)), 'ci95': bootstrap(diffs), 'n_better': int(better), 'n': len(diffs)}
        summary['vs_r2v_none'][c] = d
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
