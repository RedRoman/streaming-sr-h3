#!/usr/bin/env bash
# Does the 2-step adapter hold quality? The only latency lever that matters.
#
# The deployed path is 1.2156 s/frame and the DiT is 81.0% of it, so halving the step count is worth
# about 40% end to end -- more than everything else on the list combined (pipelining 18%, guide reuse
# 2.5%, light VAE 3%, the last two dropped). Everything cheaper has been measured and spent.
#
# pdmd_2step is a per-layer-rank LoRA (2 to 128, avg 38) over the same 208 modules as the base; preflight
# confirms 0 absent and 0 shape mismatches. It is folded in like the Stage-1 adapter, for the same
# reasons -- no live parameters for ComfyUI's device move to miss, and no float32 branch per step.
#
# Two arms, one variable each against the current best (r5_k3, 4 steps, no step adapter):
#   step2      pdmd_2step adapter, 2 sampling steps
#   step2_only 4 steps WITH the adapter, to separate "the adapter changes the model" from "two steps is
#              fewer steps" -- otherwise a regression cannot be attributed.
#
# Read-out: the sampling-seed noise floor is sd 0.0018 LPIPS, so a delta under 0.0036 is not a result.
# A 2-step arm within that band would be a 40% latency cut for free.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r5_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors
STEPLORA=../assets/speed/minimax_h3_pdmd_2step_lora_avg_rank_38_bf16.safetensors

python code/preflight.py --cards 4,5,6,7 --need-gb 45 \
  --eval-set data/real_eval/heldout --ckpt "${STEPLORA#../}" --ckpt results/range_branch/r5_k3.pt || {
  echo 'preflight failed, not launching' >&2; exit 1; }

i=0
for ARM in step2:2 step2_only:4; do
  TAG=${ARM%%:*}; ST=${ARM##*:}
  for SH in 0 1; do
    dev=$(echo "4 5 6 7" | cut -d' ' -f$((i+1)))
    (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code
     python -u run_deploy.py --tag t4_$TAG --turbo-steps $ST --dc rphi --guide y \
       --rphi "$RPHI" --merge-lora --lora "$LORA" --refiner "$STEPLORA" \
       --shard $SH --nshards 2 > ../logs/${TAG}_s$SH.log 2>&1
     echo "$TAG shard$SH exit=$? $(date +%H:%M:%S)") &
    i=$((i+1)); sleep 25
  done
done
wait
for TAG in step2 step2_only; do
  echo "t4_$TAG: $(ls -d results/deploy_runs/t4_$TAG/*/ 2>/dev/null|wc -l)/96"
done
export ASCEND_RT_VISIBLE_DEVICES=4
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_step2 \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --arm step2=../results/deploy_runs/t4_step2 \
  --arm step2_only=../results/deploy_runs/t4_step2_only \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --pair step2,dck3 --pair step2_only,dck3 --pair step2,step2_only \
  > ../logs/judge_step2.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
