#!/bin/bash
# Run the existing merged t8_dcr weights with fewer sigma points. The Turbo LoRA was trained on 8
# points, but turbo_sigmas(n, 12) simply picks n points from the same family, so n=2 and n=4 are
# inference-only changes. No retraining. Cards $2-$5 for n=4, $6-7 for n=2 (2 shards each).
cd /cache/zihaohe/streaming_sr0
ORDER=$(python3 code/clip_order.py)
test -n "$ORDER" || { echo "empty ORDER" >&2; exit 1; }
LORA=${1:-../results/distill_1step/lora_step001750.safetensors}
# Use the stage1 lora, not the 1-step distilled one
LORA=../results/stage1_lora/lora_step002000.safetensors

for n in 4 2; do
  if [ $n -eq 4 ]; then c0=2; ns=4; else c0=6; ns=2; fi
  for sh in $(seq 0 $((ns-1))); do
    card=$((c0+sh))
    (export ASCEND_RT_VISIBLE_DEVICES=$card; source env.sh; cd code;
     python -u run_deploy.py --tag t${n}_dcr --turbo-steps $n --dc rphi \
       --merge-lora --lora "$LORA" \
       --shard $sh --nshards $ns --clips="$ORDER" > ../logs/deploy_t${n}dcr_s$sh.log 2>&1) &
    sleep 35
  done
done
wait
echo ALLDONE > logs/nstep.done
