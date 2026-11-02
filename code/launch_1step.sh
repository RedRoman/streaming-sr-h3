#!/bin/bash
# One-step inference with a distilled student. $1 = tag, $2 = lora checkpoint, $3 = first card.
# Same arm as t8_dcr in every respect except the step count: turbo weights merged, dc projecting onto
# R_phi(y), 24 real clips. Note the schedule has a single point, so innerDC's projection lands on it --
# effectively a post-hoc projection rather than the mid-trajectory injection the 8-step teacher used.
cd /cache/zihaohe/streaming_sr0
TAG=$1; LORA=$2; C0=$3
ORDER=$(python3 code/clip_order.py)
test -n "$ORDER" || { echo "empty ORDER" >&2; exit 1; }
for sh in 0 1 2 3; do
  card=$((C0+sh))
  (export ASCEND_RT_VISIBLE_DEVICES=$card; source env.sh; cd code;
   python -u run_deploy.py --tag $TAG --turbo-steps 1 --dc rphi --lora "$LORA" \
     --shard $sh --nshards 4 --clips="$ORDER" > ../logs/deploy_${TAG}_s$sh.log 2>&1) &
  sleep 30
done
wait
echo ALLDONE > logs/deploy_${TAG}.done
