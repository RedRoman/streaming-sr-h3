#!/bin/bash
# Perceptual-loss arm against an identical MSE-only control, 4 cards each.
cd /cache/zihaohe/streaming_sr0/code
( export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29531 \
    train_stage1.py --data ../data/stage7_latents --out ../results/s7_perc \
    --steps 2000 --rank 32 --guide-key guide_oracle --perc 1.0 > ../logs/s7_perc.log 2>&1 ) &
sleep 25
( export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29532 \
    train_stage1.py --data ../data/stage7_latents --out ../results/s7_perc_hi \
    --steps 2000 --rank 32 --guide-key guide_oracle --perc 4.0 > ../logs/s7_perc_hi.log 2>&1 ) &
wait
echo S7_PERC_DONE >> ../logs/s7_perc.log
