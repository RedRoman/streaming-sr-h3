#!/usr/bin/env bash
# Does the better range estimate reach the output? G1's verdict step.
#
# Data scaling moved R_phi from +2.116 to +2.937 dB and the HR mid-frequency ratio from 0.8375 to 0.8708
# (1.0 is no deficit), measured on one ruler. Both are intermediate quantities. The LPIPS deficit against
# FlashVSR+ was traced to the range band twice over -- our own arm, and the post-hoc substitution applied
# to their arm -- so a sharper R_phi is the one change that should move it. This run is the only way to
# know, since latent/intermediate gains in this project have already been non-monotone with pixel quality.
#
# No quota gate: the 2x2 showed it is a net regression on the held-out set (LPIPS +0.0085, terr_null
# +0.054, both CI-excluding-zero). dc=rphi is the whole method now.
#
# Cards 1 and 7 came free at 21:17; 0 and 2 hold a colleague's 35 GB jobs and 3-6 a four-card job, and the
# 20.1B DiT needs 61-65 GB so it cannot share. Two shards is what is available.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r3_big24.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for dev in 1 7; do
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout;
   source env.sh; cd code;
   python -u run_deploy.py --tag t4_r3_ho --turbo-steps 4 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" --shard $i --nshards 2 \
     > ../logs/r3_ho_s$i.log 2>&1) &
  i=$((i+1))
  sleep 30
done
wait
echo "sampling done $(ls results/deploy_runs/t4_r3_ho | wc -l) dirs at $(date +%H:%M:%S)"

export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_r3 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm infvsr=../results/baseline_infvsr \
  --arm dcr=../results/deploy_runs/t4_r2_ho \
  --arm dcr3=../results/deploy_runs/t4_r3_ho \
  --arm bicubic=bicubic \
  --pair dcr3,dcr --pair dcr3,flashvsr_plus --pair dcr3,infvsr \
  > ../logs/judge_r3.log 2>&1
echo "judge exit=$? at $(date +%H:%M:%S)"
