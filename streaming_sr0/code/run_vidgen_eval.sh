#!/bin/bash
# usage: run_vidgen_eval.sh <nshards>
cd /cache/zihaohe/streaming_sr0/code
N=$1
for i in $(seq 0 $((N-1))); do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    OMP_NUM_THREADS=8 python -u eval_vidgen.py --tag frozen --shard $i --nshards $N 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/vidgen_frozen_$i.log
    OMP_NUM_THREADS=8 python -u eval_vidgen.py --tag lora --lora ../results/stage1_lora/lora_step002000.safetensors --shard $i --nshards $N 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input' >> ../logs/vidgen_lora_$i.log ) &
done
( cd /cache/zihaohe/streaming_sr0/code && OMP_NUM_THREADS=16 /cache/zihaohe/streaming_sr0/venv/bin/python -u eval_vidgen.py --tag classic 2>&1 | grep -v -i warn >> ../logs/vidgen_classic.log ) &
wait
echo VIDGEN_EVAL_DONE >> ../logs/vidgen_frozen_0.log
