#!/usr/bin/env bash
# The gate's parameter space, on a basis that is finally the identity when it should be.
#
# out' = range(sampled) + null(sampled) * clip((s/s0)^p, 0, 1) * gain,  s = local std of U(R_phi(y))
#
# Only one point of this space was ever tried, s0=0.02 p=1.0, and the earlier gain sweep ran on top of
# a gate that also replaced the range component with U(R_phi(y)) -- which loses energy (0.9878 against
# 1.0003 measured, worst case 0.9449 against 0.9872) and cannot be recovered by any gain on the null
# term. With --keep-range the w=1 case is now bit-identical to the shipped arm (verified, max abs diff
# 0-1 from png rounding), so what this sweep measures is the gate and the gain, not the substitution.
#
# Why a gate at all rather than a global gain: null_cos is 0.4876, so only about half the generated
# detail points the right way. A global multiplier amplifies both halves. s is the only signal
# available at deployment that correlates with where generation is reliable -- it reads U(R_phi(y)),
# never the truth.
#
# Judged on LPIPS AND null_cos together. Pushing null energy toward the truth's level is progress only
# if the energy stays in the right place, and this project has measured that no-reference metrics
# cannot tell those apart.
set -u
R=/cache/zihaohe/streaming_sr0
cd $R/code || exit 1
: > $R/logs/quota3.status
for S0 in 0.005 0.02 0.08; do
  for P in 0.5 1.0 2.0; do
    for G in 1.0 1.3; do
      T="q$(echo ${S0}_${P}_${G} | tr -d '.')"
      EVAL_SET=data/real_eval/heldout python -u deploy_eval.py --procs 24 quota \
        --src ../results/deploy_runs/t4_plain2 --tag "$T" \
        --ckpt ../results/range_branch/r5_k3.pt \
        --s0 "$S0" --p "$P" --gain "$G" --keep-range \
        > "$R/logs/$T.log" 2>&1
      echo "s0=$S0 p=$P gain=$G -> $T exit=$? $(date '+%F %T')" >> $R/logs/quota3.status
    done
  done
done
# one judge over the whole grid plus the untouched arm
ARGS=""
for S0 in 0.005 0.02 0.08; do for P in 0.5 1.0 2.0; do for G in 1.0 1.3; do
  T="q$(echo ${S0}_${P}_${G} | tr -d '.')"
  test -d "$R/results/deploy/$T" && ARGS="$ARGS --arm $T=../results/deploy/$T"
done; done; done
( export ASCEND_RT_VISIBLE_DEVICES=2
  export EVAL_SET=data/real_eval/heldout
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 8 eval --name judge_quota3 \
    $ARGS --arm plain2=../results/deploy_runs/t4_plain2 --arm bicubic=bicubic \
    > "$R/logs/judge_quota3.log" 2>&1
  echo "judge exit=$? $(date '+%F %T')" >> $R/logs/quota3.status )
echo "DONE $(date '+%F %T')" >> $R/logs/quota3.status
