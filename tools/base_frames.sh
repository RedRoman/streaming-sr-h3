#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for i in 0 1 2 3 4 5; do
  ( EVAL_SET=data/real_eval/const EVAL_TAG=real_const_tmp SHARD=$i OMP_NUM_THREADS=24 \
    ../venv/bin/python -u baseline_realesrgan.py --save-frames > ../logs/bf_$i.log 2>&1 ) &
  break
done
wait
echo BF_DONE >> ../logs/bf_0.log
