#!/bin/bash
# usage: run_stage4.sh <nshards>
cd /cache/zihaohe/streaming_sr0/code
N=$1
for i in $(seq 0 $((N-1))); do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    for A in causal causal_cw2; do
      OMP_NUM_THREADS=8 python -u run_stream.py --mode indep --attn $A --shard $i --nshards $N \
        --out ../results/stage4_blocks/$A 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/stage4_${A}_$i.log
    done ) &
done
wait
echo STAGE4_DONE >> ../logs/stage4_causal_0.log
