#!/bin/bash
# Ceiling test: identical runs differing only in which guide the model sees.
cd /cache/zihaohe/streaming_sr0/code
( export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29511 \
    train_stage1.py --data ../data/stage7_latents --out ../results/stage7_oracle \
    --steps 2000 --rank 32 --guide-key guide_oracle > ../logs/s7_oracle.log 2>&1 ) &
sleep 25
( export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29512 \
    train_stage1.py --data ../data/stage7_latents --out ../results/stage7_raw \
    --steps 2000 --rank 32 --guide-key guide_raw > ../logs/s7_raw.log 2>&1 ) &
wait
echo STAGE7_TRAIN_DONE >> ../logs/s7_oracle.log
