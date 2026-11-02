#!/usr/bin/env bash
# Seven-arm comparison videos, full frame plus a null-energy zoom, with the ground-truth soundtrack.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=4
source env.sh
cd code
for P in worst best; do
  python -u make_compare_video2.py --judge ../results/deploy/judge_all7.json \
    --ref plain2 --pick $P --clips 2 --zoom --audio > ../logs/vid2_$P.log 2>&1
  echo "$P exit=$? $(date +%H:%M:%S)"
  grep -aE '"clip"|"out"|ours_minus' ../logs/vid2_$P.log | tail -8
done
ls -la ../results/compare_video/
