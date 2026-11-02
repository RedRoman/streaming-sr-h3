#!/usr/bin/env bash
# Our arm on VideoLQ: 46 clips of genuinely degraded video, no reference anywhere.
#
# Every other number in this project comes from real sources with SYNTHETIC degradation, so this is the
# only set that tests the actual deployment condition. Two things about it are honest limitations rather
# than oversights:
#   - D is assumed to be bicubic x4. R_phi conditions on the MTF of U(D(.)) and a real capture has no
#     analytic D; estimating it from one frame was measured closed here (supervised AUC 0.61-0.68). The
#     assumption is recorded in each meta.json as assumed_operator.
#   - There is no ground truth, so reference metrics are undefined. judge must not be pointed at this
#     set. The deliverable is side-by-side video plus the NR score, and this project has just measured
#     that the NR score cannot separate FlashVSR+ from ground truth -- so the video is the real output.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
CARDS=${1:-0}
NSH=${2:-1}
SH=${3:-0}
export EVAL_SET=data/real_eval/videolq
run () {
  local card=$1 sh=$2 nsh=$3
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/videolq
    source ../env.sh
    python run_deploy.py --tag vlq_plain2 --dc rphi --guide y \
      --rphi ../results/omni/omni_m5.pt --op 4 --turbo-steps 2 --merge-lora \
      --shard "$sh" --nshards "$nsh" \
      > "../logs/vlq_p${sh}.log" 2>&1
    echo "vlq shard${sh} card${card} exit=$? $(date '+%F %T')" >> ../logs/run_videolq.status
  ) &
}
cd code
IFS=, read -ra CA <<< "$CARDS"
i=0
for c in "${CA[@]}"; do
  run "$c" $((SH + i)) "$NSH"
  i=$((i+1))
done
wait
