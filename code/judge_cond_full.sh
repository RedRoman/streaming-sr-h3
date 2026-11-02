#!/usr/bin/env bash
# The conditioning comparison on every realistic held-out clip, not a quarter of them.
# Earlier tables were built from 24 of the 77 realistic clips because two fill passes ran
# data/real_eval/const by omitting $EVAL_SET. All three arms are now complete at 77/77.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=${1:-0}
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 12 eval --name judge_cond_full \
  --arm mtf_x4=../results/deploy_runs/t2_mtf_x4 \
  --arm onehot_x4=../results/deploy_runs/t2_onehot_x4 \
  --arm mtf_s6=../results/deploy_runs/t2_mtf_s6 \
  --arm bicubic=bicubic \
  --pair mtf_x4,onehot_x4 \
  > ../logs/judge_cond_full.log 2>&1
echo "EXIT=$? $(date '+%F %T')" >> ../logs/judge_cond_full.status
