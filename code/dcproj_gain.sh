#!/usr/bin/env bash
# dcproj restores fidelity and detail DIRECTION but loses detail ENERGY: null_cos 0.4812 -> 0.5056
# (better on 84/96) and PSNR +0.523 dB (91/96), against null energy 0.611 -> 0.474 (worse on 95/96) and
# LPIPS +0.0045 (better on only 32/96). The energy loss is the obvious cause of the LPIPS cost.
#
# Amplifying the null band has failed every time it was tried -- judge_sharp swept gain 1.3/1.6/1.9 and
# was monotonically worse. But every one of those sweeps amplified detail whose cosine was 0.48, so half
# of what got amplified was wrong. dcproj's detail has cosine 0.506, so this is the first time the knob
# acts on better-directed detail, and gain_oracle's argument (the MSE-optimal scale is below one BECAUSE
# the cosine is low) predicts the optimal scale rises with the cosine.
#
# A gain of about 1.29 restores dcproj's energy to the shipped level. Sweeping 1.15 / 1.30 / 1.45 brackets
# it. Pure post-processing of frames already on disk, no sampling, no cards.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source "$R/env.sh"
for G in 1.15 1.30 1.45; do
  T="dcp_g$(echo $G | tr -d .)"
  python -u deploy_eval.py --procs 10 quota --src ../results/deploy_runs/dcproj \
    --ckpt ../results/range_branch/r5_k3.pt --keep-range \
    --tag "$T" --s0 0.0000001 --p 1.0 --gain "$G" > "$R/logs/$T.log" 2>&1
  echo "$T gain=$G exit=$? $(date '+%F %T')" >> "$R/logs/dcproj_gain.status"
done
python -u deploy_eval.py --procs 10 eval --name judge_dcproj_gain \
  --arm shipped=../results/deploy_runs/t4_plain2 \
  --arm dcproj=../results/deploy_runs/dcproj \
  --arm dcproj_g115=../results/deploy/dcp_g115 \
  --arm dcproj_g130=../results/deploy/dcp_g130 \
  --arm dcproj_g145=../results/deploy/dcp_g145 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --pair dcproj_g115,shipped --pair dcproj_g130,shipped --pair dcproj_g145,shipped \
  --pair dcproj_g130,dcproj > "$R/logs/judge_dcproj_gain.log" 2>&1
echo "judge_dcproj_gain exit=$? $(date '+%F %T')" >> "$R/logs/dcproj_gain.status"
