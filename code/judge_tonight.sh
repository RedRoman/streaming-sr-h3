#!/usr/bin/env bash
# Two judges, run back to back so they do not contend for CPU.
#
# judge_dcproj   the latent-manifold projection arm (encode(project(decode(z0))) handed back instead of
#                z0 + delta) against shipping, plus the three matched temporal-smoothing controls that
#                decide whether the VAE round trip's temporal gain is a manifold effect or a low-pass.
#                m0_rt is the round trip itself; it only covers 76 clips, so pairs against it report 76.
# judge_suppress the suppression side of the null-band gate at strengths that actually bite. gain_oracle
#                says the optimal rescaling is 0.613 globally and a per-site median of 0.224; every
#                previous sweep used gain >= 1.0, and the only completed gate arms were near-identity.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source "$R/env.sh"

python -u deploy_eval.py --procs 12 eval --name judge_dcproj \
  --arm shipped=../results/deploy_runs/t4_plain2 \
  --arm dcproj=../results/deploy_runs/dcproj \
  --arm rt=../results/deploy/m0_rt \
  --arm tsmooth050=../results/deploy/tsmooth050 \
  --arm tsmooth100=../results/deploy/tsmooth100 \
  --arm tsmooth150=../results/deploy/tsmooth150 \
  --pair dcproj,shipped --pair rt,shipped --pair tsmooth050,shipped \
  --pair tsmooth100,shipped --pair tsmooth150,shipped \
  > "$R/logs/judge_dcproj.log" 2>&1
echo "judge_dcproj exit=$? $(date '+%F %T')" >> "$R/logs/judge_tonight.status"

python -u deploy_eval.py --procs 12 eval --name judge_suppress \
  --arm shipped=../results/deploy_runs/t4_plain2 \
  --arm g090=../results/deploy/sup_g090 \
  --arm g075=../results/deploy/sup_g075 \
  --arm g060=../results/deploy/sup_g060 \
  --arm s02_g10=../results/deploy/sup_s02_g10 \
  --arm s05_g10=../results/deploy/sup_s05_g10 \
  --arm s02_g075=../results/deploy/sup_s02_g075 \
  --arm bicubic=bicubic \
  --pair g090,shipped --pair g075,shipped --pair g060,shipped \
  --pair s02_g10,shipped --pair s05_g10,shipped --pair s02_g075,shipped \
  > "$R/logs/judge_suppress.log" 2>&1
echo "judge_suppress exit=$? $(date '+%F %T')" >> "$R/logs/judge_tonight.status"
