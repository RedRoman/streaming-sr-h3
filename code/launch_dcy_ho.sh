#!/bin/bash
# dc=y on the 96-clip held-out set: the control row of the ablation.
#
# Every held-out number so far comes from one sampling run (deploy_runs/t4_r2_ho, dc=rphi) with the quota
# gate and sharpening applied as post-processing. The innerDC projection target is the one choice that
# lives inside sampling, so separating its contribution from the gate's needs a second sampling run that
# differs in that single flag. Same 4-step turbo schedule, same merged Stage-1 LoRA, same clips.
#
# Cards 3-6 are a colleague's four-card job and card 0 is the FlashVSR+ clip, so this takes 1, 2 and 7.
cd /cache/zihaohe/streaming_sr0 || exit 1
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for dev in 1 2 7; do
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_dcy_ho --turbo-steps 4 --dc y --guide y --merge-lora --lora "$LORA" \
     --shard $i --nshards 3 > ../logs/dcy_ho_s$i.log 2>&1) &
  i=$((i+1))
  sleep 30
done
wait
echo ALLDONE > logs/dcy_ho.done
