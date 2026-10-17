#!/bin/bash
# usage: run_preencode.sh <split> <nshards>
cd /cache/zihaohe/streaming_sr0/code
SPLIT=$1; N=$2
for i in $(seq 0 $((N-1))); do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh; \
    OMP_NUM_THREADS=8 python -u preencode.py --split $SPLIT --shard $i --nshards $N 2>&1 \
    | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/preencode_${SPLIT}_$i.log ) &
done
wait
echo PREENCODE_${SPLIT}_DONE >> ../logs/preencode_${SPLIT}_0.log
