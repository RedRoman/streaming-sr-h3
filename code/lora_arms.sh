#!/usr/bin/env bash
# Three Stage-1 LoRA arms, single-variable from a common control, on the rebuilt pilot latents
# (2874 train / 152 val). The shipped lora_step002000 was trained on an earlier data build, so it cannot
# serve as the control -- arm A is that control, retrained on the same data as B and C.
#
#   A  rank 32,  perc 0   control
#   B  rank 32,  perc 4   pixel-supervised. This is the arm today's mapper experiment argues for: a
#                         least-squares map on latent targets improved every latent statistic
#                         (detail cosine 0.862 -> 0.884, r2 0.73) and cost the picture 1.17 dB and
#                         0.056 LPIPS. The LoRA itself is trained on velocity MSE, which is a
#                         latent-domain target, so the same objection applies to it. --perc adds an
#                         LPIPS term on decoded pixels below sigma 0.6 and costs nothing at inference.
#   C  rank 128, perc 0   capacity. OmniVR's adapter is rank 384 / 3803M against our rank 32 / 149M.
#                         The adapter is merged at deploy, so rank is also free at inference.
#
# Validation always reports pure velocity MSE (see train_stage1.py), so B is comparable to A and C on it,
# but velocity MSE is exactly the proxy under suspicion -- the arms get judged by a deploy run and
# deploy_eval, not by the training curve.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
STEPS=${STEPS:-2000}
run () {
  local card=$1 tag=$2 rank=$3 perc=$4
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    source "$R/env.sh"
    # venv/bin has no torchrun, and PATH then falls through to the system anaconda whose torch cannot
    # import comfy_aimdo -- that is what killed the first launch. Call the module from the venv python.
    "$R/venv/bin/python" -m torch.distributed.run --nproc_per_node=1 \
      --master_port=$((29600 + card)) train_stage1.py \
      --out "$R/results/lora_$tag" --steps "$STEPS" --rank "$rank" --perc "$perc" \
      > "$R/logs/lora_$tag.log" 2>&1
    echo "lora_$tag card$card exit=$? $(date '+%F %T')" >> "$R/logs/lora_arms.status"
  ) &
}
run 1 a_r32_p0   32  0
run 2 b_r32_p4   32  4
run 3 c_r128_p0 128  0
wait
echo "lora arms all done $(date '+%F %T')" >> "$R/logs/lora_arms.status"
