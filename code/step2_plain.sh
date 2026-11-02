#!/usr/bin/env bash
# Two sampling steps with NO third-party adapter: just turbo_sigmas(2, 12.0) off the same 8-step delta.
#
# Why this and not the pdmd adapter: pdmd at its own 2 steps cost +0.0518 LPIPS (28.8 noise floors) and
# at 4 steps +0.0933, with the damage landing in lpips_gtrange (0.1368 -> 0.2043 / 0.2482), i.e. in the
# generated detail. The Open-H3 refiner failed the same way (+0.0443 LPIPS, +0.0496 gtrange). Two
# independent third-party H3 adapters, same failure band -- both were trained on H3's own generation
# distribution, not on a restoration pipeline with a guide latent and an innerDC projection.
#
# The turbo delta is a weight edit, not a schedule, so evaluating it on 2 points of the shift-12 family
# instead of 4 needs no adapter at all. An older 19-clip measurement on the const set had 2 steps at
# LPIPS 0.2225 against the 8-step teacher's 0.2261 -- better, not worse -- but that was the r2-era R_phi
# with the quota gate still on, so it cannot be carried over. This re-measures it on the current
# configuration (r5_k3, no gate) over the full 96 held-out clips.
#
# Read-out fixed before running: sampling-seed noise floor is sd 0.0018 LPIPS, so a delta under 0.0036
# is not a result. Within that band, 2 steps is a 38.7% cut of the deployed path for free.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
RPHI=../results/range_branch/r5_k3.pt
LORA=../results/stage1_lora/lora_step002000.safetensors

python code/preflight.py --cards 4,5,6,7 --need-gb 45 \
  --eval-set data/real_eval/heldout --ckpt results/range_branch/r5_k3.pt --tag t4_plain2 || {
  echo 'preflight failed, not launching' >&2; exit 1; }

for SH in 0 1 2 3; do
  dev=$((SH + 4))
  (export ASCEND_RT_VISIBLE_DEVICES=$dev; export EVAL_SET=data/real_eval/heldout; source env.sh; cd code
   python -u run_deploy.py --tag t4_plain2 --turbo-steps 2 --dc rphi --guide y \
     --rphi "$RPHI" --merge-lora --lora "$LORA" --shard $SH --nshards 4 \
     > ../logs/plain2_s$SH.log 2>&1
   echo "shard$SH exit=$? $(date +%H:%M:%S)") &
  sleep 20
done
wait
echo "t4_plain2: $(ls -d results/deploy_runs/t4_plain2/*/ 2>/dev/null|wc -l)/96 at $(date +%H:%M:%S)"

export ASCEND_RT_VISIBLE_DEVICES=4
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code
python -u deploy_eval.py --procs 14 eval --name judge_plain2 \
  --arm dck3=../results/deploy_runs/t4_k3_ho \
  --arm plain2=../results/deploy_runs/t4_plain2 \
  --arm step2=../results/deploy_runs/t4_step2 \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --pair plain2,dck3 --pair plain2,step2 --pair plain2,flashvsr_plus \
  > ../logs/judge_plain2.log 2>&1
echo "judge exit=$? $(date +%H:%M:%S)"
