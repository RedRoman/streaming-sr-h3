#!/bin/bash
cd /cache/zihaohe/streaming_sr0
rm -rf results/train_smoke
PORT=29611 bash tools/run_train.sh 0,1 ../results/train_smoke --data ../data/pilot_latents \
  --steps 6 --val_every 6 --save_every 6 --warmup 2 > logs/train_smoke.log 2>&1
echo SMOKE_EXIT=$? >> logs/train_smoke.log
