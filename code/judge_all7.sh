#!/usr/bin/env bash
# One judge pass over every arm that exists on the held-out set, so the video can label each tile with a
# number from the SAME measurement rather than stitching several judges together.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=4
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 12 eval --name judge_all7 \
  --arm infvsr=../results/baseline_infvsr \
  --arm flashvsr=../results/baseline_flashvsr \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm dcy=../results/deploy_runs/t4_dcy_ho \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --arm plain2=../results/deploy_runs/t4_plain2 \
  --arm bicubic=bicubic \
  --pair plain2,flashvsr_plus --pair plain2,dck3 --pair plain2,dcy \
  --pair plain2,infvsr --pair plain2,flashvsr \
  > ../logs/judge_all7.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
