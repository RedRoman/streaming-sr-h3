#!/usr/bin/env bash
# Priority queue for the first free card.
#
#   1. LMS Sharp smoke, one clip, strengths 1.0 and 0.5. Sharp is the one component of
#      Open-H3-Regenerate-2K that has never been on this model. The refiner alone was measured and lost
#      (LPIPS +0.0211 with 10/96 better, null cosine -0.0414 worse on 91/96) though it DID supply detail
#      (null energy 0.579 -> 0.776). Sharp is a same-resolution adapter with no draft or scale
#      assumption, and our deficit is exactly detail energy: 0.611 against FlashVSR+'s 0.850.
#      The released sharp_curve_branches.safetensors cannot supply it -- every curve_B in that file is
#      exactly zero -- so the weights come from rank slice 512:576 of the merged adapter, converted by
#      convert_openh3_to_comfy.py, whose recipe reproduces Kijai's pre-converted refiner with max
#      relative error 0.0 over all 416 tensors.
#   2. dcproj on the two real-degradation sets, which still gates dcproj + gain 1.15 from shipping.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
SHARP="$R/assets/openh3_split/openh3_sharp64_comfy.safetensors"
RPHI=../results/range_branch/r5_k3.pt
CLIP=1jyM3quZ0h0

# `npu-smi info -i <card>` prints nothing matching N/ 65536 on this host, so the obvious per-card query
# silently yields an empty string, the -n test always fails, and the waiter sleeps forever even once a
# card frees. That is exactly what happened for the first 3h12m of this queue. The full table does work
# and lists the eight cards in index order, so read it once and index into it.
free_card () {
  local mem i=0
  for mem in $(npu-smi info 2>/dev/null | grep -oE '[0-9]+/ 65536' | cut -d/ -f1 | tr -d ' '); do
    [ "$mem" -lt 16000 ] && { echo "$i"; return 0; }
    i=$((i+1))
  done
  return 1
}
CARD=""
while [ -z "$CARD" ]; do CARD=$(free_card) || sleep 240; done
echo "queue_sharp got card $CARD $(date '+%F %T')" >> "$R/logs/queue.status"

run () { # run <tag> <extra args...>
  local tag=$1; shift
  ( export ASCEND_RT_VISIBLE_DEVICES=$CARD
    export EVAL_SET=data/real_eval/heldout
    source "$R/env.sh"
    python run_deploy.py --tag "$tag" --dc rphi --guide y --rphi "$RPHI" \
      --turbo-steps 2 --merge-lora "$@" > "$R/logs/$tag.log" 2>&1
    echo "$tag exit=$? $(date '+%F %T')" >> "$R/logs/sharp_lora.status" )
}

run sharp_s10_smoke --refiner "$SHARP" --refiner-strength 1.0 --clips "$CLIP"
run sharp_s05_smoke --refiner "$SHARP" --refiner-strength 0.5 --clips "$CLIP"

( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/heldout
  source "$R/env.sh"
  python -u deploy_eval.py --procs 6 eval --name judge_sharp_smoke \
    --arm shipped=../results/deploy_runs/t4_plain2 \
    --arm sharp_s10=../results/deploy_runs/sharp_s10_smoke \
    --arm sharp_s05=../results/deploy_runs/sharp_s05_smoke \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus \
    > "$R/logs/judge_sharp_smoke.log" 2>&1
  echo "judge_sharp_smoke exit=$? $(date '+%F %T')" >> "$R/logs/sharp_lora.status" )

for SET in realvsr mvsr4x; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$CARD
    export EVAL_SET="data/real_eval/$SET"
    source "$R/env.sh"
    python run_deploy.py --tag "dcproj_$SET" --dc rphi --guide y --rphi "$RPHI" \
      --turbo-steps 2 --merge-lora --dc-project > "$R/logs/dcproj_$SET.log" 2>&1
    echo "dcproj_$SET exit=$? $(date '+%F %T')" >> "$R/logs/sharp_lora.status" )
done
echo "queue_sharp done $(date '+%F %T')" >> "$R/logs/sharp_lora.status"
