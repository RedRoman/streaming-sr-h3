#!/bin/bash
cd /cache/zihaohe/streaming_sr0
PORT=29621 bash tools/run_train.sh 0,1,2,3,4,5,6,7 ../results/stage1_lora \
  --data ../data/pilot_latents --steps 2000 --rank 32 --lr 1e-4 --warmup 100 \
  --save_every 250 --val_every 250 >> logs/stage1_train.log 2>&1
echo STAGE1_EXIT=$? >> logs/stage1_train.log
