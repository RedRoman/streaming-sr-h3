#!/usr/bin/env bash
# The evaluation noise floor of the SAMPLER, which has never been measured on this project.
#
# Seed variance was measured for R_phi training (0.014-0.067 dB per operator) but the sampling seed was
# hardcoded to 101, so every row of the quality table is one draw. Our LPIPS margin over FlashVSR+ is
# 0.0107 with CI [-0.0199, -0.0014]. If one arm's seed-to-seed spread is of that order the margin means
# nothing, and this project already has the precedent: in Sol-H3 a 0.44 dB noise floor swallowed a 1 dB
# arm gap and reversed the ordering, which is why `feedback_eval_noise_floor` exists.
#
# Same arm (r6_xl_k3, dc=rphi, 4-step turbo, merged Stage-1), same 96 clips, two further seeds. The full
# 96 and not a subset: the number has to be comparable to a margin computed on the full 96, and 24-clip
# subsets have already been measured to differ from it by 0.06-0.08 dB.
#
# Cards 3 and 7 are the only free ones -- 0/2 hold a colleague's 35 GB jobs and 4/5/6 a new H3 training
# run that will grow -- so one seed per card, 96 clips each, about five hours in parallel.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r6_xl_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for SEED in 202 303; do
  dev=$([ $i -eq 0 ] && echo 3 || echo 7)
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_xl_seed$SEED --turbo-steps 4 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" --seed $SEED \
     > ../logs/seed_$SEED.log 2>&1
   echo "seed $SEED exit=$? $(date +%H:%M:%S)") &
  i=$((i+1)); sleep 25
done
wait
for SEED in 202 303; do echo "seed $SEED: $(ls -d results/deploy_runs/t4_xl_seed$SEED/*/ 2>/dev/null|wc -l)/96"; done

export ASCEND_RT_VISIBLE_DEVICES=3
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_seeds \
  --arm s101=../results/deploy_runs/t4_xl_ho \
  --arm s202=../results/deploy_runs/t4_xl_seed202 \
  --arm s303=../results/deploy_runs/t4_xl_seed303 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --pair s202,s101 --pair s303,s101 --pair s303,s202 \
  > ../logs/judge_seeds.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
