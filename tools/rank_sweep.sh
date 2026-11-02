#!/bin/bash
# The one quality lever not yet tried: adaptation capacity. Stage 1 and every Stage-7 arm used rank 32 for
# 2000 steps, which is a light adaptation of a 33B generator. Both arms keep the oracle guide and the
# perceptual weight that worked (1.0), so rank is the only thing that varies.
cd /cache/zihaohe/streaming_sr0/code
( export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29541 \
    train_stage1.py --data ../data/stage7_latents --out ../results/s7_r64 \
    --steps 2000 --rank 64 --guide-key guide_oracle > ../logs/s7_r64.log 2>&1 ) &
sleep 25
( export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7; source ../env.sh
  ../venv/bin/python -m torch.distributed.run --nproc_per_node=4 --master_port=29542 \
    train_stage1.py --data ../data/stage7_latents --out ../results/s7_r128 \
    --steps 2000 --rank 128 --guide-key guide_oracle > ../logs/s7_r128.log 2>&1 ) &
wait
echo RANK_SWEEP_DONE >> ../logs/s7_r64.log
