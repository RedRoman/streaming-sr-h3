#!/usr/bin/env bash
# 1. deploy r6_xl_k3 on the held-out 96, six shards -- the colleague's four-card job finished so cards
#    1,3,4,5,6,7 are free and the 20.1B DiT finally fits six ways at once.
# 2. the three big-data specialists that omni_m4's table is missing.
#
# Why these specialists and not five: omni_m4 covers {2,4,deblur,denoise} at crop 112, and a specialist
# must match it on every axis for the comparison to mean anything. x4 already has one (r5_big, K=1, 4.7M,
# 24000 iters, crop 112, +3.029 on the shared val). x2/deblur/denoise have pairs at 240x416 or 120x208 so
# crop 112 fits; x8's pairs are only 60x104, so an x8 specialist cannot be crop-112 and is left out of
# this table rather than compared across a crop difference. K=1, because OmniRange has no causal stack --
# comparing it against a K=3 specialist would charge the omni model for a feature it structurally lacks.
#
# Without these, omni's x2/deblur/denoise columns compare a 10351-window/24000-iter model against
# 598-window/6000-iter specialists, which is a confound in omni's FAVOUR and cannot be reported.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r6_xl_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors

i=0
for dev in 1 3 4 5 6 7; do
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_xl_ho --turbo-steps 4 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" --shard $i --nshards 6 \
     > ../logs/xl_ho_s$i.log 2>&1; echo "deploy shard$i exit=$? $(date +%H:%M:%S)") &
  i=$((i+1)); sleep 25
done

COMMON="--kinds realistic --ch 128 --blocks 16 --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0"
(export ASCEND_RT_VISIBLE_DEVICES=0; source env.sh; cd code
 python -u train_range2.py --tag sp_s2 --data /cache/zihaohe/range_pairs_s2big $COMMON --device npu:0 \
   > ../logs/sp_s2.log 2>&1; echo "sp_s2 exit=$? $(date +%H:%M:%S)"
 python -u train_range2.py --tag sp_deblur --data /cache/zihaohe/range_pairs_deblurbig $COMMON --device npu:0 \
   > ../logs/sp_deblur.log 2>&1; echo "sp_deblur exit=$? $(date +%H:%M:%S)") &
(export ASCEND_RT_VISIBLE_DEVICES=2; source env.sh; cd code
 python -u train_range2.py --tag sp_denoise --data /cache/zihaohe/range_pairs_denoisebig $COMMON --device npu:0 \
   > ../logs/sp_denoise.log 2>&1; echo "sp_denoise exit=$? $(date +%H:%M:%S)") &
wait
echo PUSH_XL_TRAINING_DONE
