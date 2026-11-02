#!/usr/bin/env bash
# Does the range substitution help a method that is not ours?
#
# The plan was to validate the decomposition on a second backbone, but the H3->Wan route is blocked at the
# frozen UltraFlash upsampler (perfect input scores 17.5 dB, 5.2 dB WORSE than bicubic), so a second
# backbone does not currently exist. This asks the same question more directly and for the cost of one CPU
# pass: replace the range band of FlashVSR+'s and InfVSR's own output with U(R_phi(y)) and keep their null
# band. If both improve, the stage is backbone-agnostic by demonstration.
#
# R_phi has never seen these clips, and it is the deployable estimate -- not the GT range -- so this is a
# configuration someone could actually ship on top of either method.
set -u
while pgrep -f 'prepare_range_data|deploy_eval.py --procs' > /dev/null; do sleep 60; done
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
for A in flashvsr_plus infvsr; do
  python -u deploy_eval.py --procs 14 rangesub --src ../results/baseline_$A --tag xb_$A \
    --ckpt ../results/range_branch/r2_hs.pt > ../logs/xb_$A.log 2>&1
  echo "$A: $(ls ../results/deploy/xb_$A | wc -l) clips, exit=$?"
done
python -u deploy_eval.py --procs 14 eval --name judge_xbackbone \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm fvp_rs=../results/deploy/xb_flashvsr_plus \
  --arm infvsr=../results/baseline_infvsr \
  --arm inf_rs=../results/deploy/xb_infvsr \
  --arm dcr=../results/deploy_runs/t4_r2_ho \
  --pair fvp_rs,flashvsr_plus --pair inf_rs,infvsr --pair dcr,fvp_rs \
  > ../logs/judge_xbackbone.log 2>&1
echo "judge exit=$? at $(date +%H:%M:%S)"
tail -2 ../logs/judge_xbackbone.log
