#!/bin/bash
# Full 24-clip rerun of the locked configuration with R_phi on the accelerator.
# Numerically identical to t4_dcr by construction (batched K=1 call verified max abs diff 0.0), so this is
# a confirmation run, not a new arm -- but the quality table has to be produced from the config that will
# actually ship, not from one that differs by a device placement.
cd /cache/zihaohe/streaming_sr0
ORDER=$(python3 code/clip_order.py)
test -n "$ORDER" || { echo "empty ORDER" >&2; exit 1; }
LORA=../results/stage1_lora/lora_step002000.safetensors
for sh in 0 1 2 3 4 5; do
  (export ASCEND_RT_VISIBLE_DEVICES=$sh; source env.sh; cd code;
   python -u run_deploy.py --tag t4_final --turbo-steps 4 --dc rphi --merge-lora --lora "$LORA" \
     --shard $sh --nshards 6 --clips="$ORDER" > ../logs/deploy_t4final_s$sh.log 2>&1) &
  sleep 30
done
wait
echo ALLDONE > logs/t4final.done
