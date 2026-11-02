#!/usr/bin/env bash
# Collect the Gate B result: per-scale gain of R_phi over the raw observation.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1
python3 - <<'PY'
import json
from pathlib import Path
OUT = Path('/cache/zihaohe/streaming_sr0/results/range_branch')
rows = []
for s, tag in ((2, 'r2_s2'), (4, 'r2_hs'), (8, 'r2_s8')):
    f = OUT / f'{tag}.json'
    if not f.exists():
        rows.append((s, tag, None)); continue
    d = json.loads(f.read_text())
    v = d['final']['val']['realistic']
    rows.append((s, tag, v))
print('%-6s %-8s %7s %9s %9s %9s' % ('scale', 'tag', 'crop', 'obs dB', 'R_phi dB', 'gain dB'))
for s, tag, v in rows:
    if v is None:
        print('%-6s %-8s %7s %9s %9s %9s' % ('x%d' % s, tag, '-', '-', '-', 'pending')); continue
    print('%-6s %-8s %7s %9.3f %9.3f %+9.3f'
          % ('x%d' % s, tag, {2: 112, 4: 112, 8: 48}[s], v['base'], v['psnr'], v['gain_dB']))
print()
print('Gate B passes if the gain stays positive under a change of operator.')
PY
