#!/usr/bin/env bash
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=1
source env.sh
cd code
for P in worst best; do
  python -u make_compare_video.py --pick $P --clips 2 > ../logs/vid_$P.log 2>&1
  echo "$P exit=$? $(date +%H:%M:%S)"
  grep -aE '"clip"|"out"|ours_minus' ../logs/vid_$P.log | tail -6
done
ls -la ../results/compare_video/
