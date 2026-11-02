#!/bin/bash
# raw and oracle arms share the 8 cards, 4 each; rphi waits for R_phi to converge.
cd /cache/zihaohe/streaming_sr0/code
CK=lora_step002000.safetensors
for i in 0 1 2 3; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const OMP_NUM_THREADS=6 ../venv/bin/python -u eval_stage7.py \
      --arm raw --lora ../results/stage7_raw/$CK --shard $i --nshards 4 \
      > ../logs/e7_raw_$i.log 2>&1 ) &
done
for i in 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const OMP_NUM_THREADS=6 ../venv/bin/python -u eval_stage7.py \
      --arm oracle --lora ../results/stage7_oracle/$CK --shard $((i-4)) --nshards 4 \
      > ../logs/e7_oracle_$((i-4)).log 2>&1 ) &
done
wait
echo EVAL_S7_DONE >> ../logs/e7_raw_0.log
