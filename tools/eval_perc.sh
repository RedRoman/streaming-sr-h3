#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
for i in 0 1 2 3; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const ARM_TAG=perc OMP_NUM_THREADS=6 ../venv/bin/python -u eval_stage7.py \
      --arm oracle --lora ../results/s7_perc/lora_step002000.safetensors --shard $i --nshards 4 \
      > ../logs/ep_perc_$i.log 2>&1 ) &
done
for i in 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const ARM_TAG=perc_hi OMP_NUM_THREADS=6 ../venv/bin/python -u eval_stage7.py \
      --arm oracle --lora ../results/s7_perc_hi/lora_step002000.safetensors --shard $((i-4)) --nshards 4 \
      > ../logs/ep_perchi_$((i-4)).log 2>&1 ) &
done
wait
echo EVAL_PERC_DONE >> ../logs/ep_perc_0.log
