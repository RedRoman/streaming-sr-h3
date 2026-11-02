#!/usr/bin/env bash
# Measure the seed spread on both sides of the omni-vs-specialist comparison.
#
# omni5 lands at -0.054 / +0.041 / +0.164 / -0.173 / -0.241 dB against the five specialists. Nothing in
# that line is interpretable yet: neither side has ever been trained twice, so a per-operator sign could
# be a seed. The Sol-H3 case is the precedent -- a 0.44 dB noise floor there swallowed a 1 dB arm gap and
# reversed the ordering. Three seeds each on the operator with the largest negative delta (denoise) plus
# three of the multi-operator model gives both spreads.
#
# Cards 0 and 2 are the two free ones; 1 and 7 are the judge passes and 3-6 a colleague's four-card job.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1

(
  export ASCEND_RT_VISIBLE_DEVICES=0
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  for S in 1 2; do
    python -u omni_stage1.py --tag omni5_s$S --seed $S --scales 2,4,8,deblur,denoise \
      --iters 6000 --batch 30 --crop 48 --device npu:0 --val-limit 24 \
      > ../logs/omni5_s$S.log 2>&1
    echo "omni5_s$S exit=$? $(date +%H:%M:%S)"
  done
) &

(
  export ASCEND_RT_VISIBLE_DEVICES=2
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  for S in 1 2; do
    python -u train_range2.py --tag sgl_denoise_s$S --seed $S \
      --data /cache/zihaohe/range_pairs_denoise --kinds realistic \
      --iters 6000 --batch 30 --crop 48 --device npu:0 \
      > ../logs/sgl_denoise_s$S.log 2>&1
    echo "sgl_denoise_s$S exit=$? $(date +%H:%M:%S)"
  done
) &

wait
echo ALL_SEEDS_DONE
