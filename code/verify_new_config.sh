#!/usr/bin/env bash
# Confirm the two findings on all 77 realistic clips before they become the shipped configuration.
#
# One step. An 8-clip pilot measured one step from noise at LPIPS 0.2459 against two steps at 0.2466 --
# a difference of 0.0007 inside the 0.0009 noise floor, each winning 4 of 8 -- while sampling time fell
# from 105.6 s to 59.6 s per clip. If that holds, the second step is doing nothing and end-to-end cost
# drops about a third, from 18.3x real time to roughly 11x, which is where FlashVSR+ sits. Eight clips
# have misled this project three times today, so this runs the full set.
#
# Gain 1.3. The sharpness sweep put null energy at 0.893 against our 0.597 and FlashVSR+'s 0.845, with
# directional accuracy falling only 0.4588 -> 0.4531 and staying far above FlashVSR+'s 0.3970, at a cost
# of 0.55 dB PSNR and no loss of LPIPS. That addresses the thing the viewer actually saw: our output
# reads softer than FlashVSR+ on screen, and a 0.0143 LPIPS lead does not.
#
# Both are applied here so the pair can be judged as the configuration it would ship as, with the
# single-change arms kept alongside for attribution.
set -u
R=/cache/zihaohe/streaming_sr0
: > $R/logs/verify_new.status
run () {
  local tag=$1 steps=$2 card=$3 shard=$4
  ( export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source $R/env.sh
    cd $R/code
    python run_deploy.py --tag "$tag" --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps "$steps" --merge-lora \
      --shard "$shard" --nshards 4 > "$R/logs/${tag}_p${shard}.log" 2>&1
    echo "$tag shard$shard exit=$? $(date '+%F %T')" >> $R/logs/verify_new.status ) &
}
for S in 0 1 2 3; do run step1_full 1 $S $S; done
for S in 0 1 2 3; do run step2_full 2 $((S+4)) $S; done
wait
for T in step1_full step2_full; do
  echo "$T: $(ls $R/results/deploy_runs/$T/*/stream_result.json 2>/dev/null | wc -l) clips" >> $R/logs/verify_new.status
done
# gain 1.3 on top of each, post-processing only
source $R/venv/bin/activate 2>/dev/null || true
cd $R/code
for T in step1_full step2_full; do
  EVAL_SET=data/real_eval/heldout python -u deploy_eval.py --procs 20 quota \
    --src "../results/deploy_runs/$T" --tag "${T}_g13" \
    --ckpt ../results/range_branch/r5_k3.pt --s0 0.01 --p 0.5 --gain 1.3 --keep-range \
    > "$R/logs/${T}_g13.log" 2>&1
  echo "${T}_g13 exit=$? $(date '+%F %T')" >> $R/logs/verify_new.status
done
( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/heldout
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 10 eval --name judge_new_config \
    --arm shipped_2step=../results/deploy_runs/t4_plain2 \
    --arm step2=../results/deploy_runs/step2_full \
    --arm step1=../results/deploy_runs/step1_full \
    --arm step2_g13=../results/deploy/step2_full_g13 \
    --arm step1_g13=../results/deploy/step1_full_g13 \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus \
    --arm bicubic=bicubic \
    --pair step1,step2 --pair step1_g13,step1 \
    > "$R/logs/judge_new_config.log" 2>&1
  echo "judge exit=$? $(date '+%F %T')" >> $R/logs/verify_new.status )
echo "DONE $(date '+%F %T')" >> $R/logs/verify_new.status
