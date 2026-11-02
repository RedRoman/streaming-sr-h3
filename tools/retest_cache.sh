#!/bin/bash
# Re-test the Stage-5 conclusion on real footage: does the cross-chunk KV cache still cost quality when
# the unobserved band actually carries something across a block (0.487 here against 0.103 on the
# generated clips the original was measured on)?
cd /cache/zihaohe/streaming_sr0/code
LORA=${LORA:-../results/stage7_raw/lora_step002000.safetensors}
for i in 0 1 2 3; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const OMP_NUM_THREADS=6 ../venv/bin/python -u run_stream.py \
      --mode indep --lora $LORA --shard $i --nshards 4 --out ../results/real_indep \
      > ../logs/rt_indep_$i.log 2>&1 ) &
done
for i in 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$i; source ../env.sh
    EVAL_SET=data/real_eval/const OMP_NUM_THREADS=6 ../venv/bin/python -u run_stream.py \
      --mode indep --kv --lora $LORA --shard $((i-4)) --nshards 4 --out ../results/real_cache \
      > ../logs/rt_cache_$((i-4)).log 2>&1 ) &
done
wait
echo RETEST_DONE >> ../logs/rt_indep_0.log
