#!/bin/bash
# usage: run_stage3.sh <nshards>
cd /cache/zihaohe/streaming_sr0/code
N=$1
for i in $(seq 0 $((N-1))); do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    for M in anchor anchor_lowfreq_only; do
      OMP_NUM_THREADS=8 python -u run_stream.py --mode $M --shard $i --nshards $N --out ../results/stage3_blocks 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/stage3_${M}_$i.log
    done ) &
done
wait
echo STAGE3_STREAM_DONE >> ../logs/stage3_anchor_0.log
