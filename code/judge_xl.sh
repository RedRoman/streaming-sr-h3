#!/usr/bin/env bash
# Score the 13.3M K=3 estimator's sampling run against the 4.7M K=3 one it differs from in width only.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_xl \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm infvsr=../results/baseline_infvsr \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --arm dcxl=../results/deploy_runs/t4_xl_ho \
  --arm bicubic=bicubic \
  --pair dcxl,dck3 --pair dcxl,flashvsr_plus --pair dcxl,infvsr \
  > ../logs/judge_xl.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
tail -1 ../logs/judge_xl.log
