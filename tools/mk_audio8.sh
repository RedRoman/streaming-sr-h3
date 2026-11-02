#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for i in $(seq 0 11); do
  ( PATH=/usr/local/ffmpeg/bin:$PATH OMP_NUM_THREADS=2 ../venv/bin/python -u prepare_audio7.py \
      --clips 290 --shard $i --nshards 12 > ../logs/mka8_$i.log 2>&1 ) &
done
wait
echo AUDIO8_DONE >> ../logs/mka8_0.log
