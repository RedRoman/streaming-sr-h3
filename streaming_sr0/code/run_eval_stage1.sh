#!/bin/bash
# usage: run_eval_stage1.sh <nshards>
cd /cache/zihaohe/streaming_sr0/code
N=$1
for i in $(seq 0 $((N-1))); do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh; \
    OMP_NUM_THREADS=8 python -u eval_stage1.py --lora ../results/stage1_lora/lora_step002000.safetensors \
      --shard $i --nshards $N 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/eval_stage1_$i.log ) &
done
wait
echo EVAL_STAGE1_DONE >> ../logs/eval_stage1_0.log
