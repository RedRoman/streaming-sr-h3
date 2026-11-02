#!/usr/bin/env bash
# One judge pass over every arm, so every pair is paired-comparable.
#
# The ablation and the baseline table were scored in two separate passes, which gives means for all arms
# but paired tests only within a pass -- and the comparison that now matters most, our un-gated arm
# against FlashVSR+, falls across the boundary. Scoring them together fixes that and costs one pass.
#
# `dcr` rather than `dcr_q` is the arm to watch: the 2x2 showed the quota gate is a net regression on the
# held-out set (LPIPS +0.0085, terr_null +0.054, both with CI excluding zero, 60/77 and 67/71 clips),
# buying only +0.18 dB PSNR. The gate's gain was selected on the 24-clip const set.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_ho_full \
  --arm infvsr=../results/baseline_infvsr \
  --arm flashvsr=../results/baseline_flashvsr \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm dcy=../results/deploy_runs/t4_dcy_ho \
  --arm dcy_q=../results/deploy/ho_dcy_q \
  --arm dcr=../results/deploy_runs/t4_r2_ho \
  --arm dcr_q=../results/deploy/t4r2_ho_final \
  --arm bicubic=bicubic \
  --pair dcr,flashvsr_plus --pair dcr,flashvsr --pair dcr,infvsr \
  --pair dcr,dcr_q --pair dcr_q,flashvsr_plus --pair flashvsr,flashvsr_plus \
  --pair dcr,bicubic \
  > ../logs/judge_ho_full.log 2>&1
echo "exit=$? at $(date +%H:%M:%S)"
tail -2 ../logs/judge_ho_full.log
