"""Stage-2 metrics (PROTOCOL_stage2.md): block-boundary disagreement, cross-block flicker, quality, timing."""
import json
from pathlib import Path
import numpy as np
from ref_metrics import bootstrap
from revisit_metrics import ROOT, load, lpips_dist, psnr, unobserved
D = ROOT / 'data/stage2_eval'
import os
MODES = os.environ.get('S2_MODES', 'indep,carry').split(',')
DIRS = {m: ROOT / ('results/stage2_blocks' if m in ('indep', 'carry') else 'results/stage3_blocks') / m for m in MODES}
OUTJSON = ROOT / os.environ.get('S2_OUT', 'results/stage2_metrics.json')
man = json.loads((D / 'manifest.json').read_text())
blocks = man['blocks']
rows = []
for name in man['clips']:
    m = json.loads((D / name / 'meta.json').read_text())
    r = {'clip': name, 'category': m['category']}
    for mode in MODES:
        base = DIRS[mode] / name
        if not (base / 'stream_result.json').exists(): continue
        sr = json.loads((base / 'stream_result.json').read_text())
        # 1) block-boundary disagreement: frames emitted by block k that block k+1 also renders
        bd, within = [], []
        for k in range(len(blocks) - 1):
            ov = [f for f in range(17 * k + 17, 17 * k + 22)]
            for f in ov:
                a = load(base / f'block{k:03d}' / f'{f:06d}.png')
                b = load(base / f'block{k+1:03d}' / f'{f:06d}.png')
                na, nb = unobserved(a), unobserved(b)
                bd.append(float(((na - nb) ** 2).sum() / max(float((na ** 2).sum()), 1e-12)))
            # within-block scale reference: consecutive frames inside block k
            for f in ov[:-1]:
                a = load(base / f'block{k:03d}' / f'{f:06d}.png')
                b = load(base / f'block{k:03d}' / f'{f+1:06d}.png')
                na, nb = unobserved(a), unobserved(b)
                within.append(float(((na - nb) ** 2).sum() / max(float((na ** 2).sum()), 1e-12)))
        # 2) cross-block flicker on the assembled stream, split by boundary vs interior pairs
        sd = base / 'stream'
        emitted = sorted(int(p.stem) for p in sd.glob('*.png'))
        bnd = {b['new'][0] for b in blocks if b['k'] > 0}
        fl = {'boundary': [], 'interior': []}
        for f in emitted[:-1]:
            if f + 1 not in emitted: continue
            ya, yb = load(sd / f'{f:06d}.png'), load(sd / f'{f+1:06d}.png')
            ga, gb = load(D / name / 'hr' / f'{f:06d}.png'), load(D / name / 'hr' / f'{f+1:06d}.png')
            num = float(((unobserved(ya) - unobserved(yb)) ** 2).mean())
            den = max(float(((unobserved(ga) - unobserved(gb)) ** 2).mean()), 1e-12)
            fl['boundary' if f + 1 in bnd else 'interior'].append(num / den)
        # 3) quality over the assembled stream
        ps, lp, en, cs = [], [], [], []
        for i, f in enumerate(emitted):
            g = load(D / name / 'hr' / f'{f:06d}.png'); y = load(sd / f'{f:06d}.png')
            ps.append(psnr(y, g))
            ng, ny = unobserved(g), unobserved(y)
            en.append(float((ny ** 2).mean() / max(float((ng ** 2).mean()), 1e-12)))
            cs.append(float((ny * ng).sum() / max(np.sqrt((ny ** 2).sum() * (ng ** 2).sum()), 1e-12)))
            if i % 20 == 0: lp.append(lpips_dist(y, g))
        r[mode] = {'boundary_disagreement': float(np.mean(bd)), 'within_block_change': float(np.mean(within)),
                   'bd_over_within': float(np.mean(bd) / max(np.mean(within), 1e-12)),
                   'flicker_boundary': float(np.mean(fl['boundary'])), 'flicker_interior': float(np.mean(fl['interior'])),
                   'flicker_ratio': float(np.mean(fl['boundary']) / max(np.mean(fl['interior']), 1e-12)),
                   'psnr': float(np.mean(ps)), 'lpips': float(np.mean(lp)),
                   'energy_ratio': float(np.mean(en)), 'cos': float(np.mean(cs)),
                   'sec_per_block': float(np.mean([x['total_seconds'] for x in sr['blocks']])),
                   'sample_sec': float(np.mean([x['sample_seconds'] for x in sr['blocks']])),
                   'decode_sec': float(np.mean([x['decode_seconds'] for x in sr['blocks']])),
                   'guide_sec': float(np.mean([x['guide_seconds'] for x in sr['blocks']]))}
    rows.append(r)
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in r.items()}), flush=True)
KEYS = ['boundary_disagreement','within_block_change','bd_over_within','flicker_boundary','flicker_interior',
        'flicker_ratio','psnr','lpips','energy_ratio','cos','sec_per_block','sample_sec','decode_sec','guide_sec']
out = {'n': len(rows), 'per_clip': rows, 'means': {}}
for mode in MODES:
    rs = [r for r in rows if mode in r]
    if rs: out['means'][mode] = {k: float(np.mean([r[mode][k] for r in rs])) for k in KEYS} | {'n': len(rs)}
out['comparisons'] = {}
import itertools
for x, y in itertools.permutations(MODES, 2):
    if MODES.index(x) <= MODES.index(y): continue
    rs = [r for r in rows if x in r and y in r]
    if not rs: continue
    out['comparisons'][f'{x} - {y}'] = {k: {'mean': float(np.mean([r[x][k]-r[y][k] for r in rs])),
                                            'ci95': bootstrap([r[x][k]-r[y][k] for r in rs]), 'n': len(rs)} for k in KEYS}
# is the boundary problem real? ratio of bd to within-block change, per mode
for mode in MODES:
    rs2 = [r[mode]['bd_over_within'] for r in rows if mode in r]
    if rs2: out.setdefault('bd_over_within_ci', {})[mode] = {'mean': float(np.mean(rs2)), 'ci95': bootstrap(rs2)}
OUTJSON.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'per_clip'}, indent=1))
