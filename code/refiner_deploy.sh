#!/usr/bin/env bash
# The Open-H3 2K refiner LoRA on the held-out 96, composed with the arm that currently wins.
#
# Single variable against dcxl: same R_phi (r6_xl_k3), same 4-step turbo schedule, same merged Stage-1
# LoRA, same innerDC projection onto R_phi(y). The refiner is added on top and nothing else changes --
# in particular we do NOT adopt its own 3-step Euler schedule or its (1-s)*upscaled + s*noise init,
# because that would move several variables at once and the question here is only whether its detail
# prior transfers.
#
# Why it is worth a run at all: R_phi's own axes are drying up at the output. The width step just bought
# +0.175 dB on the estimate but only -0.0020 LPIPS, a transfer efficiency of 0.011 per dB against 0.030
# for the K step. The refiner acts on the null band instead, which is a different pool.
#
# The risk is domain: it was trained to regenerate H3's own 768P drafts, not degraded camera footage.
# That is exactly what this measures.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
test -s assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors || { echo 'refiner missing' >&2; exit 1; }
grep -aq 'refiner fires: True' logs/refiner_smoke.log || { echo 'smoke did not confirm the adapter fires; refusing to spend 6 cards' >&2; exit 1; }

RPHI=../results/range_branch/r6_xl_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for dev in 1 3 4 5 6 7; do
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_ref_ho --turbo-steps 4 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" \
     --refiner ../assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors \
     --shard $i --nshards 6 > ../logs/ref_ho_s$i.log 2>&1
   echo "shard$i exit=$? $(date +%H:%M:%S)") &
  i=$((i+1)); sleep 25
done
wait
n=$(ls results/deploy_runs/t4_ref_ho | wc -l)
echo "refiner sampling: $n dirs at $(date +%H:%M:%S)"
test "$n" -ge 96 || { echo 'incomplete, not scoring' >&2; exit 1; }

export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_ref \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm dcxl=../results/deploy_runs/t4_xl_ho \
  --arm dcref=../results/deploy_runs/t4_ref_ho \
  --arm bicubic=bicubic \
  --pair dcref,dcxl --pair dcref,flashvsr_plus \
  > ../logs/judge_ref.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
