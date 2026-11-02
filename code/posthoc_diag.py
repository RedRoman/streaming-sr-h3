"""POST-HOC diagnostics (added after seeing step-0 metrics; not part of the pre-registered protocol).

All comparisons are on identical frames, so no registration is involved:
  overlap : frames 17*kpre .. 17*kpre+4 are rendered by both kctl (as new output) and kpre (as context);
            how differently do two memoryless blocks render the same frame?
  seed    : kpre frames rendered with seed 101 vs 202.
  correct : is the generated unobserved band aligned with GT's (cosine), and is inner-DC's extra band
            (beyond VAE) pointing toward the GT residual?
Masks are the target bbox. N(.) is the same unobserved-band operator as the main metric.
"""
import json
from pathlib import Path

import numpy as np

from revisit_metrics import ROOT, load, unobserved

E, B = ROOT / 'data/events', ROOT / 'results/step0_blocks'


def bbox_mask(ev, f, shape):
    x, y, w, h = ev['frame_records'][str(f)]['bbox_hr_xywh']
    m = np.zeros(shape[:2], bool)
    m[max(0, int(y)):min(shape[0], int(y + h)), max(0, int(x)):min(shape[1], int(x + w))] = True
    return m


def cos(a, b):
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def main():
    sel = json.loads((E / 'selection.json').read_text())['events']
    rows = []
    for e in sel:
        seq = e['sequence']
        ev = json.loads((E / seq / 'event.json').read_text())
        kpre, kctl = ev['blocks']['kpre'], ev['blocks'].get('kctl')
        row = {'sequence': seq}
        folder = lambda mode, k: B / mode / seq / f'block{k:03d}'
        # overlap: same frame, two blocks
        if kctl is not None:
            acc = {m: [] for m in ('vae_only', 'innerdc')}
            for f in range(17 * kpre, 17 * kpre + 5):
                if ev['frame_records'][str(f)]['absent']:
                    continue
                g = load(E / seq / 'hr' / f'{f:06d}.png'); m = bbox_mask(ev, f, g.shape)
                if m.sum() < 64:
                    continue
                for mode in acc:
                    n1 = unobserved(load(folder(mode, kctl) / f'{f:06d}.png'))[m]
                    n2 = unobserved(load(folder(mode, kpre) / f'{f:06d}.png'))[m]
                    acc[mode].append(float(((n1 - n2) ** 2).sum() / max((n2 ** 2).sum(), 1e-12)))
            for mode, v in acc.items():
                row[f'overlap_relchange_{mode}'] = float(np.mean(v)) if v else None
        # seed and correctness on the kpre revisit-a frames
        sv, cv, cd, cg = [], {'vae_only': [], 'innerdc': []}, [], []
        for f in ev['pairs_revisit_a']:
            g = load(E / seq / 'hr' / f'{f:06d}.png'); m = bbox_mask(ev, f, g.shape)
            ng = unobserved(g)[m]
            ny = {mode: unobserved(load(folder(mode, kpre) / f'{f:06d}.png'))[m] for mode in ('vae_only', 'innerdc', 'innerdc_s202')}
            sv.append(float(((ny['innerdc'] - ny['innerdc_s202']) ** 2).sum() / max((ny['innerdc'] ** 2).sum(), 1e-12)))
            for mode in cv:
                cv[mode].append(cos(ny[mode], ng))
            extra = ny['innerdc'] - ny['vae_only']
            cd.append(cos(extra, ng - ny['vae_only']))
            cg.append(float((extra ** 2).sum() / max((ng ** 2).sum(), 1e-12)))
        row.update({'seed_relchange_innerdc': float(np.mean(sv)), 'cos_vs_gt_vae_only': float(np.mean(cv['vae_only'])),
                    'cos_vs_gt_innerdc': float(np.mean(cv['innerdc'])), 'cos_extra_vs_gt_residual': float(np.mean(cd)),
                    'extra_energy_over_gt': float(np.mean(cg))})
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = [k for k in rows[0] if k != 'sequence']
    summary = {'posthoc': True, 'n_events': len(rows),
               'mean': {k: float(np.mean([r[k] for r in rows if r.get(k) is not None])) for k in keys},
               'n_events_per_key': {k: sum(r.get(k) is not None for r in rows) for k in keys}}
    out = ROOT / 'results/step0_posthoc'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'rows.json').write_text(json.dumps(rows, indent=1))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
