#!/usr/bin/env bash
# Is R_phi data-limited? Three runs, each differing from an existing one in the data only.
#
# The capacity sweep said parameters are not the constraint: 0.59M -> 2.406 dB, 2.0M -> 2.526,
# 4.73M at 24000 iters -> 2.386, i.e. the largest model is the worst. What every run shared was 598
# realistic training windows from 400 of 2182 eligible clips. The new set has ~5500.
#
#   r3_hs12  vs r2_hs   same 2.0M, same 12000 iters, same loss -> isolates data at the shipped size
#   r3_hs24  vs r3_hs12 same data -> asks whether more data also wants a longer schedule
#   r3_big24 vs r2_big  same 4.73M, same 24000 iters -> isolates data at the size that previously lost
#
# Run as co-tenants: measured footprint is 2.8-3.5 GB against 38 GB free on each of cards 1 and 7, which
# belong to other tenants' jobs. Every NPU on every reachable host is occupied and CPU training is
# 20x slower (120 iters did not finish in 10 minutes), so sharing is the only way this moves today.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
D=/cache/zihaohe/range_pairs_x4big

(
  export ASCEND_RT_VISIBLE_DEVICES=7
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r3_hs12 --data $D --kinds realistic \
    --iters 12000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0 --device npu:0 \
    > ../logs/r3_hs12.log 2>&1
  echo "r3_hs12 exit=$? $(date +%H:%M:%S)"
  python -u train_range2.py --tag r3_hs24 --data $D --kinds realistic \
    --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0 --device npu:0 \
    > ../logs/r3_hs24.log 2>&1
  echo "r3_hs24 exit=$? $(date +%H:%M:%S)"
) &

(
  export ASCEND_RT_VISIBLE_DEVICES=1
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r3_big24 --data $D --kinds realistic \
    --ch 128 --blocks 16 --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0 --device npu:0 \
    > ../logs/r3_big24.log 2>&1
  echo "r3_big24 exit=$? $(date +%H:%M:%S)"
) &

wait
echo R3_ALL_DONE
python3 - <<'PY'
import json
from pathlib import Path
R = Path('/cache/zihaohe/streaming_sr0/results/range_branch')
rows = [('r2_hs', 598, 12000, '2.0M'), ('r3_hs12', None, 12000, '2.0M'),
        ('r3_hs24', None, 24000, '2.0M'), ('r2_big', 598, 24000, '4.7M'),
        ('r3_big24', None, 24000, '4.7M')]
print('%-10s %6s %7s %6s %9s %9s' % ('tag', 'size', 'iters', 'wins', 'gain_dB', 'psnr'))
for t, w, it, sz in rows:
    p = R / f'{t}.json'
    if not p.exists():
        print('%-10s %6s %7s %6s %9s %9s' % (t, sz, it, w or '?', 'missing', '')); continue
    d = json.loads(p.read_text())
    v = d['final']['val']['realistic']
    print('%-10s %6s %7s %6s %9.3f %9.3f'
          % (t, sz, d['iters'], d.get('train_windows', w), v['gain_dB'], v['psnr']))
PY
