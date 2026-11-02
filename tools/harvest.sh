#!/bin/bash
# Scale the real-footage pool for Stage-7 training. Each worker takes a different AudioSet shard
# directory, so they do not fight over the same candidate list.
cd /cache/zihaohe/streaming_sr0/code
for s in 1 2 3 4 5 6 7 8 9 10 11 12; do
  ( SHARD_START=$s PATH=/usr/local/ffmpeg/bin:$PATH OMP_NUM_THREADS=3 \
    ../venv/bin/python -u build_real_av.py --shards 1 --per-shard 400 --keep 1000 --min-width 1280 \
      > ../logs/harvest_$s.log 2>&1 ) &
  sleep 3
done
wait
echo HARVEST_DONE >> ../logs/harvest_1.log
