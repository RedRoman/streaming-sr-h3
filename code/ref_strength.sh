#!/usr/bin/env bash
# Is the refiner regression a scale error or a domain mismatch? Six clips, three strengths.
#
# dcref came in at LPIPS 0.2407 against dcxl's 0.1964 -- worse on all six metrics, significantly, with
# only 6 of 77 clips improving. What makes that worth one more probe is the size: the folded delta is
# only 0.83% of the base weight norm, and a 0.8% perturbation producing a 22% relative LPIPS loss is a
# lot. Two explanations with different consequences:
#   scale wrong   -- Open-H3 reports the refiner at rank 384 / alpha 384, so scale 1.0, but the file
#                    itself does not say so. Then there is a minimum at some s < 1 and the adapter is
#                    usable at reduced strength.
#   domain wrong  -- it was trained to refine H3's own 768P drafts, not degraded camera footage, so the
#                    direction is simply wrong here and quality falls monotonically from s = 0.
#
# Six clips is a diagnostic, not a result: the strengths are compared against each other on the SAME six,
# and anything that looks promising gets the full 96 before it is believed.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
CLIPS=$(python3 -c "import json;print(','.join(json.load(open('data/real_eval/heldout/manifest.json'))['clips'][:6]))")
RPHI=../results/range_branch/r6_xl_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
i=0
for S in 0.25 0.50; do
  dev=$([ $i -eq 0 ] && echo 1 || echo 3)
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code;
   python -u run_deploy.py --tag t4_refs$S --turbo-steps 4 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" \
     --refiner ../assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors \
     --refiner-strength $S --clips "$CLIPS" > ../logs/refs_$S.log 2>&1
   echo "s=$S exit=$? $(date +%H:%M:%S)") &
  i=$((i+1)); sleep 25
done
wait
export ASCEND_RT_VISIBLE_DEVICES=1
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 12 eval --name judge_refs \
  --arm dcxl=../results/deploy_runs/t4_xl_ho \
  --arm s025=../results/deploy_runs/t4_refs0.25 \
  --arm s050=../results/deploy_runs/t4_refs0.50 \
  --arm s100=../results/deploy_runs/t4_ref_ho \
  --pair s025,dcxl --pair s050,dcxl --pair s100,dcxl \
  > ../logs/judge_refs.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
