#!/usr/bin/env bash
# x6 hold-out at the pixel level, scored on the x6 eval set it was actually run on.
# HR frames are byte-identical across heldout and heldout_s6 (verified), but the bicubic baseline is
# not: at x6 the LR is 138x80 against 208x120, so the reference floor must come from this set.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=${1:-0}
export EVAL_SET=data/real_eval/heldout_s6
source env.sh
cd code
python -u deploy_eval.py --procs 10 eval --name judge_s6_true \
  --arm mtf_s6=../results/deploy_runs/t2_mtf_s6_ho \
  --arm bicubic=bicubic \
  > ../logs/judge_s6_true.log 2>&1
echo "EXIT=$? $(date '+%F %T')" >> ../logs/judge_s6_true.status
