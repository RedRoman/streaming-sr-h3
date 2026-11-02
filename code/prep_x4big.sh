#!/usr/bin/env bash
# Ten times the range-branch training data at x4.
#
# The capacity sweep says parameters are not the constraint: 0.59M scores 2.406 dB, 2.0M scores 2.526, and
# 4.73M trained for 24000 iters scores 2.386 -- worse than a model less than half its size. What the runs
# do share is 598 realistic training windows, drawn from 400 of the 2182 eligible clips at 2 windows each.
# So the next R_phi experiment has to move data, not width, and that needs the pairs to exist first.
#
# Writes to /cache, not /temp: the shared mount is 95% full and other tenants fill it, which shows up as
# silent truncation rather than an error. 2182 clips x 4 windows at ~2.4 MB is roughly 21 GB.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code || exit 1
# 16 av-decode workers on top of a 14-worker judge pass put the load average over 600 on this box once
# before. Wait for the judge rather than contend with it.
while pgrep -f 'deploy_eval.py --procs' > /dev/null; do sleep 60; done
OUTDIR=/cache/zihaohe/range_pairs_x4big
mkdir -p "$OUTDIR"
for sp in train val; do
  if [ "$sp" = train ]; then n=2182; w=4; else n=200; w=2; fi
  for i in $(seq 0 15); do
    SR_SCALE=4 RANGE_OUT="$OUTDIR" \
      python prepare_range_data.py --split $sp --clips $n --windows $w --shard $i --nshards 16 \
      > /cache/zihaohe/streaming_sr0/logs/rpbig_${sp}_$i.log 2>&1 &
  done
  wait
  echo "$sp done: $(ls $OUTDIR/$sp 2>/dev/null | wc -l) files at $(date +%H:%M:%S)"
done
du -sh "$OUTDIR"
