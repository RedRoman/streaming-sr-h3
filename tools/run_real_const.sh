#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for i in 0 1 2 3 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const OMP_NUM_THREADS=8 ../venv/bin/python -u run_stream.py \
      --mode indep --shard $i --nshards 8 --out ../results/real_const_indep \
      > ../logs/real_const_s$i.log 2>&1 ) &
done
wait
echo REAL_CONST_DONE >> ../logs/real_const_s0.log
