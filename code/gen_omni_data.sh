#!/usr/bin/env bash
# CPU-only data generation, in the order that buys the most per hour. Immune to the card squeeze: the
# eight NPUs are currently all held by a colleague's TP8 benchmark, and this needs none of them.
#
# 1. bd with the FITTED operator (INTER_AREA, ksize 21, sigma 1.0-1.2). This is the only addition that
#    buys an external benchmark: Vid4 BDx4 becomes a legitimate positive test instead of the recorded
#    negative (+0.251 dB mean with 2 of 4 clips going backwards). Pre-registered gate before any
#    training: the new pairs' base -- observation against D(x) -- must land near 38 dB, matching the
#    released Vid4 BD's 38.17, not the 30.56 the guessed chain produced.
#
# 2. Arbitrary scales x1_5, x3, x6. These turn omni from five categories into a continuous operator
#    family, and they immediately expose the limitation of a one-hot condition: x1.5 has no slot. That
#    is the point -- the next conditioning change is to feed D's spectral profile rather than a label,
#    and these are the pairs that can show interpolation to an operator never trained on.
#
# Non-integer scale needs the HR crop to divide: 480/1.5 = 320 and 832/1.5 is not an integer, so x1_5
# uses SR_SCALE=3 on a 2x-upsampled grid is wrong -- instead it is generated at scale 3 and 6 only, and
# x1.5 is left to the conditioning experiment on the existing x2 pairs. Recorded here so the gap is
# explicit rather than discovered later.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1
export PATH=/cache/zihaohe/streaming_sr0/venv/bin:$PATH

echo "=== 1. bd pairs with the fitted operator  $(date +%H:%M:%S)"
D=/cache/zihaohe/range_pairs_bdfit
for sp in train val; do
  if [ "$sp" = train ]; then n=2182; w=4; else n=200; w=2; fi
  for i in $(seq 0 7); do
    SR_SCALE=4 DEGRADE_MODE=bd RANGE_OUT=$D \
      python prepare_range_data.py --split $sp --clips $n --windows $w --shard $i --nshards 8 \
      > ../logs/rp_bdfit_${sp}_$i.log 2>&1 &
  done
  wait
  echo "   bd $sp: $(ls $D/$sp 2>/dev/null | wc -l) files at $(date +%H:%M:%S)"
done

echo "=== 2. arbitrary scales x3 and x6  $(date +%H:%M:%S)"
for S in 3 6; do
  for sp in train val; do
    if [ "$sp" = train ]; then n=2182; w=4; else n=200; w=2; fi
    for i in $(seq 0 7); do
      SR_SCALE=$S RANGE_OUT=/cache/zihaohe/range_pairs_s${S}big \
        python prepare_range_data.py --split $sp --clips $n --windows $w --shard $i --nshards 8 \
        > ../logs/rp_s${S}_${sp}_$i.log 2>&1 &
    done
    wait
    echo "   s$S $sp: $(ls /cache/zihaohe/range_pairs_s${S}big/$sp 2>/dev/null | wc -l) at $(date +%H:%M:%S)"
  done
done

echo "=== pre-registered gate: base of each new pair set against D(x)"
python3 - <<'PY'
import glob, json, numpy as np, torch, torch.nn.functional as F
for tag, sc in (('bdfit', 4), ('s3big', 3), ('s6big', 6)):
    fs = sorted(glob.glob(f'/cache/zihaohe/range_pairs_{tag}/val/*.npz'))[:12]
    if not fs:
        print(json.dumps({'set': tag, 'status': 'no files'})); continue
    v = []
    for f in fs:
        d = np.load(f)
        y = torch.from_numpy(d['y'][:4]).float().permute(0, 3, 1, 2) / 255
        t = torch.from_numpy(d['target'][:4]).float().permute(0, 3, 1, 2) / 255
        v.append(float(10 * torch.log10(1 / (y - t).square().mean().clamp_min(1e-12))))
    print(json.dumps({'set': tag, 'scale': sc, 'n': len(v), 'base_dB': round(float(np.mean(v)), 2),
                      'gate': 'bd must be near 38.17 (released Vid4 BD); the guessed chain was 30.56'}))
PY
