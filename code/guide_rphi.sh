#!/usr/bin/env bash
# Feed the DiT's keyframe condition the cleaned estimate instead of the raw observation.
#
# run_deploy has had a --guide switch from the start and every one of the 24 runs on disk used
# guide=y: the keyframe latent is encoded from the degraded LR, carrying its noise, its second-order
# blur and its JPEG or H.264 artifacts straight into the condition the DiT attends to.
#
# The prior for trying the other setting is strong and measured today. The innerDC PROJECTION target
# was moved from y to R_phi(y) under a clean single-variable comparison -- same estimator, same step
# count -- and it was worth -0.0818 LPIPS and +1.09 dB PSNR, improving on 66 and 73 of 77 clips. That
# is 5.7x our entire margin over FlashVSR+. The reason given there applies verbatim here: y contains
# everything the operator did NOT do, and forcing that into the model is not data consistency, it is
# noise injection. There is no reason the condition path should be exempt.
#
# Two arms because the two uses of R_phi(y) are separable and might not be independent:
#   g_rphi_dc_rphi   guide and projection both from the estimate
#   g_rphi_dc_y      guide from the estimate, projection from the raw observation
# The second isolates the condition path on its own against the t4_dcy_ho arm already measured.
set -u
R=/cache/zihaohe/streaming_sr0
: > $R/logs/guide_rphi.status
run () {
  local tag=$1 dc=$2 card=$3 shard=$4 nsh=$5
  ( export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source $R/env.sh
    cd $R/code
    python run_deploy.py --tag "$tag" --dc "$dc" --guide rphi \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
      --shard "$shard" --nshards "$nsh" \
      > "$R/logs/${tag}_p${shard}.log" 2>&1
    echo "$tag shard$shard exit=$? $(date '+%F %T')" >> $R/logs/guide_rphi.status ) &
}
# four cards each, both arms at once
for S in 0 1 2 3; do run g_rphi_dc_rphi rphi $S $S 4; done
for S in 0 1 2 3; do run g_rphi_dc_y    y    $((S+4)) $S 4; done
wait
for T in g_rphi_dc_rphi g_rphi_dc_y; do
  echo "$T: $(ls $R/results/deploy_runs/$T/*/stream_result.json 2>/dev/null | wc -l) clips" >> $R/logs/guide_rphi.status
done
( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/heldout
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 8 eval --name judge_guide \
    --arm shipped_guide_y=../results/deploy_runs/t4_plain2 \
    --arm guide_rphi=../results/deploy_runs/g_rphi_dc_rphi \
    --arm guide_rphi_dc_y=../results/deploy_runs/g_rphi_dc_y \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus \
    --arm bicubic=bicubic \
    --pair guide_rphi,shipped_guide_y \
    > "$R/logs/judge_guide.log" 2>&1
  echo "judge exit=$? $(date '+%F %T')" >> $R/logs/guide_rphi.status )
echo "DONE $(date '+%F %T')" >> $R/logs/guide_rphi.status
