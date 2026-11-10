#!/usr/bin/env bash
# The delivery set. Four sheets, each answering a different question, and each ordered by a rule so the
# clips shown are not cherry-picked.
#
#   1 synthetic x4   the headline comparison: five arms including ground truth, worst-case first by our
#                    own LPIPS against FlashVSR+, so the clips that argue against us lead
#   2 x6             an operator no arm trained on, where a one-hot cannot be evaluated at all
#   3 MVSR4x         real optics WITH a reference: the set where every generative arm falls below a
#                    near-no-op bicubic in PSNR, ours by the least
#   4 VideoLQ        real degradation with NO reference, already rendered: the frames are the only
#                    trustworthy judge there, since a NIQE-family metric cannot separate FlashVSR+ from
#                    ground truth
set -u
R=/cache/zihaohe/streaming_sr0
cd $R/code || exit 1
: > $R/logs/delivery.status
run () {
  local name=$1; shift
  echo "$(date '+%F %T') BEGIN $name" >> $R/logs/delivery.status
  if "$@" >> "$R/logs/delivery_${name}.log" 2>&1; then
    echo "$(date '+%F %T') OK    $name" >> $R/logs/delivery.status
  else
    echo "$(date '+%F %T') FAIL  $name exit=$?" >> $R/logs/delivery.status
  fi
}

d1 () {
  EVAL_SET=data/real_eval/heldout python3 make_video_real.py \
    --eval-set data/real_eval/heldout \
    --arm "LR (bicubic x4)=bicubic" \
    --arm "InfVSR=../results/baseline_infvsr" \
    --arm "FlashVSR+=../results/baseline_flashvsr_plus" \
    --arm "ours 2-step=../results/deploy_runs/t4_plain2" \
    --arm "ground truth=gt" \
    --judge results/deploy/judge_all7.json --ref plain2 \
    --clips 2 --cols 3 --crop 256 --out results/delivery
}
d2 () {
  EVAL_SET=data/real_eval/heldout_s6 python3 make_video_real.py \
    --eval-set data/real_eval/heldout_s6 \
    --arm "LR (bicubic x6)=bicubic" \
    --arm "ours, x6 never trained=../results/deploy_runs/t2_mtf_s6_ho" \
    --arm "ground truth=gt" \
    --judge results/deploy/judge_s6_true.json --ref mtf_s6 \
    --clips 2 --cols 3 --crop 256 --out results/delivery
}
d3 () {
  EVAL_SET=data/real_eval/mvsr4x python3 make_video_real.py \
    --eval-set data/real_eval/mvsr4x \
    --arm "LQ (real optics)=bicubic" \
    --arm "InfVSR=../results/baseline_infvsr_mvsr4x" \
    --arm "FlashVSR+=../results/baseline_flashvsr_plus_mvsr4x" \
    --arm "ours=../results/deploy_runs/mv_r5k3" \
    --arm "ground truth=gt" \
    --judge results/deploy/judge_mtf3way.json --ref uncond_r5k3 \
    --clips 2 --cols 3 --crop 256 --out results/delivery
}
run d1_synthetic_x4 d1
run d2_x6_holdout   d2
run d3_mvsr4x_real  d3
echo "$(date '+%F %T') DONE" >> $R/logs/delivery.status
