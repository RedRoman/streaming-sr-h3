#!/usr/bin/env bash
# Does closing the guide's mid-band deficit actually improve DELIVERED quality?
#
# The sharpness complaint was localised to the guide, not the generator: pooled over 77 realistic clips
# our null band carries 0.60 of the truth's energy against FlashVSR's 1.06, and substituting the truth's
# detail into both arms still left us behind (0.0849 vs 0.0820) -- so the defect is in R_phi. The
# incumbent r5 plateaus at mid_ratio 0.8831. The two new objectives move it:
#
#   r7_energy   mid_ratio 1.0293, estimate gain +2.485 dB   (iter 16000/24000)
#   r7_deficit  mid_ratio 0.9915, estimate gain +2.379 dB   (iter 16000/24000)
#
# Both clear the pre-registered gate (mid_ratio > 0.93, estimate PSNR not down more than 0.3 dB). But
# this project has already measured that the estimate and the delivered output are NOT monotone -- a
# latent gain of 4.7 dB once arrived with a pixel loss of 6.1 dB -- so the gate is necessary and not
# sufficient. r7_energy slightly OVERSHOOTS at 1.0293, and whether 3% too much mid-band energy reads as
# sharper or as crunchy is exactly what LPIPS on the full held-out set has to answer.
#
# Matched to t4_plain2, the shipped 2-step incumbent, on the same 96 clips and the same schedule, so the
# only difference is which R_phi produced the range. Seed noise at 2 steps is 0.0009 LPIPS.
#
# These are iteration-16000 checkpoints, not final; training continues on cards 6/7 until about 14:10.
# Read as an early signal, to be re-run on the final weights if the sign is interesting.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
LORA=../results/stage1_lora/lora_step002000.safetensors
for ARM in energy deficit; do
  TAG=t2_r7${ARM}_ho
  echo "=== $TAG  4 shards on cards 2-5  $(date '+%F %T')"
  for i in 0 1 2 3; do
    CARD=$((i+2))
    (export ASCEND_RT_VISIBLE_DEVICES=$CARD
     export EVAL_SET=data/real_eval/heldout
     source env.sh
     cd code
     python -u run_deploy.py --tag "$TAG" --turbo-steps 2 --dc rphi --guide y \
       --rphi ../results/range_branch/r7_${ARM}.pt --merge-lora --lora "$LORA" \
       --shard "$i" --nshards 4 > ../logs/${TAG}_s${i}.log 2>&1
     echo "$TAG shard$i card$CARD exit=$? $(date '+%F %T')" >> ../logs/r7_deploy.status) &
  done
  wait
  echo "=== $TAG: $(ls results/deploy_runs/$TAG 2>/dev/null | wc -l)/96 at $(date '+%F %T')"
done
echo "R7 DEPLOY DONE $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/r7_deploy.status
