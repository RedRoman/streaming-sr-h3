#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for i in $(seq 0 15); do
  ( PATH=/usr/local/ffmpeg/bin:$PATH OMP_NUM_THREADS=3 ../venv/bin/python -u prepare_stage7_train.py \
      --clips 290 --windows 3 --shard $i --nshards 16 > ../logs/mk7_$i.log 2>&1 ) &
done
wait
echo STAGE7_DATA_DONE >> ../logs/mk7_0.log
