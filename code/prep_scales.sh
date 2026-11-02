#!/usr/bin/env bash
# Build range-branch training pairs at x2 and x8.
#
# Gate B asks whether R_phi is operator-agnostic or x4-specific. null_volume ordered the task family by
# unobserved volume (generation buys 0.019 LPIPS at x2 and 0.351 at x16) but that was a re-decomposition
# of x4 outputs; no model has ever been trained under a different operator. prepare_range_data is now
# scale-parameterised, so the pairs can simply be rebuilt.
#
# Written as a script because the inline for-loop lost its environment: `source env.sh` inside a
# backgrounded compound command did not reach the children, so every shard died before writing a log.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code || exit 1
mkdir -p /cache/zihaohe/streaming_sr0/logs
for S in 2 8; do
  for sp in train val; do
    if [ "$sp" = train ]; then n=400; else n=60; fi
    for i in $(seq 0 11); do
      SR_SCALE=$S RANGE_OUT=/cache/zihaohe/range_pairs_s$S \
        python prepare_range_data.py --split $sp --clips $n --windows 2 --shard $i --nshards 12 \
        > /cache/zihaohe/streaming_sr0/logs/rp_s${S}_${sp}_$i.log 2>&1 &
    done
  done
done
wait
echo ALL_PREP_DONE
for S in 2 8; do
  echo "s$S train=$(ls /cache/zihaohe/range_pairs_s$S/train 2>/dev/null | wc -l) val=$(ls /cache/zihaohe/range_pairs_s$S/val 2>/dev/null | wc -l)"
done
