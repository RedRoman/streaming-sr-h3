#!/usr/bin/env bash
# Deploy the three Stage-1 LoRA arms once training finishes, then judge all three against the shipped
# config. Each arm is the shipped t4_plain2 configuration with only the adapter swapped, so the arms are
# single-variable against each other and against shipping:
#   a_r32_p0   control, retrained on the rebuilt pilot latents
#   b_r32_p4   + LPIPS on decoded pixels below sigma 0.6   (the pixel-supervision hypothesis)
#   c_r128_p0  + rank 128                                   (the capacity hypothesis)
#
# Cards 1-3 take shard 0 of 2 as soon as training ends; cards 4,5,7 take shard 1 once the dcproj run
# frees them. 48 clips per shard at a measured 115 s per clip is about 92 minutes per shard.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
STEP=${STEP:-002000}

while pgrep -f "train_stage1.py" >/dev/null; do sleep 60; done
echo "training finished $(date '+%F %T')" >> "$R/logs/queue.status"

arm () {                      # arm <tag> <card> <shard> <nshards>
  local tag=$1 card=$2 sh=$3 nsh=$4
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export EVAL_SET=data/real_eval/heldout
    source "$R/env.sh"
    python run_deploy.py --tag "dep_$tag" --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
      --lora "$R/results/lora_$tag/lora_step$STEP.safetensors" \
      --lora-rank "$(python -c "import json;print(json.load(open('$R/results/lora_$tag/config.json'))['rank'])")" \
      --shard "$sh" --nshards "$nsh" > "$R/logs/dep_${tag}_s${sh}.log" 2>&1
    echo "dep_$tag shard$sh card$card exit=$? $(date '+%F %T')" >> "$R/logs/lora_deploy.status"
  ) &
}

arm a_r32_p0  1 0 2
arm b_r32_p4  2 0 2
arm c_r128_p0 3 0 2

while pgrep -f "run_deploy.py --tag dcproj" >/dev/null; do sleep 60; done
echo "dcproj finished, taking cards 4,5,7 $(date '+%F %T')" >> "$R/logs/queue.status"
arm a_r32_p0  4 1 2
arm b_r32_p4  5 1 2
arm c_r128_p0 7 1 2
wait
echo "lora deploys done $(date '+%F %T')" >> "$R/logs/lora_deploy.status"

export ASCEND_RT_VISIBLE_DEVICES=0
source "$R/env.sh"
EVAL_SET=data/real_eval/heldout python -u deploy_eval.py --procs 10 eval --name judge_lora_arms \
  --arm shipped=../results/deploy_runs/t4_plain2 \
  --arm a_r32_p0=../results/deploy_runs/dep_a_r32_p0 \
  --arm b_r32_p4=../results/deploy_runs/dep_b_r32_p4 \
  --arm c_r128_p0=../results/deploy_runs/dep_c_r128_p0 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --pair b_r32_p4,a_r32_p0 --pair c_r128_p0,a_r32_p0 --pair a_r32_p0,shipped \
  > "$R/logs/judge_lora_arms.log" 2>&1
echo "judge_lora_arms exit=$? $(date '+%F %T')" >> "$R/logs/lora_deploy.status"
