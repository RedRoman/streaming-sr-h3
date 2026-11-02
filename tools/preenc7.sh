#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for sp in val train; do
  for i in 0 1 2 3 4 5 6 7; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
      OMP_NUM_THREADS=8 ../venv/bin/python -u preencode7.py --split $sp --shard $i --nshards 8 \
        >> ../logs/preenc7_${sp}_$i.log 2>&1 ) &
  done
  wait
done
echo PREENC7_DONE >> ../logs/preenc7_train_0.log
