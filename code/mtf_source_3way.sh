#!/usr/bin/env bash
# Can MTF conditioning use a descriptor MEASURED on real degradation, rather than one computed from an
# assumed operator? This is the question the earlier MVSR4x run could not answer, because it used
# r5_k3 -- cfg desc:false, a single-operator estimator that takes no descriptor at all. The
# "descriptor shape is off the training manifold" prediction was never exercised.
#
# A 2x2 that separates the two things that differ between an assumed and a measured descriptor:
#
#            descriptor                              convention        operator
#   A        operator_mtf(4), white-noise probe      probe             x4 bicubic
#   C        synthetic_x4_control_mtf                natural image     x4 bicubic
#   B        measured_radial_mtf                     natural image     the real lens
#
#   A vs C  is the convention effect alone. It is not small: the same x4 operator measured both ways
#           sits L2 1.02 apart, because operator_mtf probes with white noise while a natural image has
#           a 1/f spectrum that makes the high-band ratios noisy.
#   C vs B  is the operator effect at fixed convention -- the number the prediction is actually about.
#
# Without C, a drop in B could be either, and reporting it as "the operator assumption is wrong" would
# repeat exactly the mistake made earlier today with psnr_gtrange: reading a quantity outside the
# conditions under which it means anything.
set -u
R=/cache/zihaohe/streaming_sr0
cd $R || exit 1
: > $R/logs/mtf3way.status
for SRC in op control measured; do
  TAG="mv_mtf_$SRC"
  for S in 0 1 2 3 4 5 6 7; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$S
      export EVAL_SET=data/real_eval/mvsr4x
      source $R/env.sh
      cd $R/code
      python run_deploy.py --tag "$TAG" --dc rphi --guide y \
        --rphi ../results/omni/omni_mtf_k3.pt --op 4 --mtf-source "$SRC" \
        --turbo-steps 2 --merge-lora --shard $S --nshards 8 \
        > "$R/logs/mtf3way_${SRC}_p${S}.log" 2>&1 ) &
  done
  wait
  n=$(ls $R/results/deploy_runs/$TAG/*/stream_result.json 2>/dev/null | wc -l)
  echo "$SRC: $n clips $(date '+%F %T')" >> $R/logs/mtf3way.status
done
# one judge over all three plus the unconditioned arm already measured, and the baselines
( export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/mvsr4x
  source $R/env.sh
  cd $R/code
  python -u deploy_eval.py --procs 10 eval --name judge_mtf3way \
    --arm mtf_op=../results/deploy_runs/mv_mtf_op \
    --arm mtf_control=../results/deploy_runs/mv_mtf_control \
    --arm mtf_measured=../results/deploy_runs/mv_mtf_measured \
    --arm uncond_r5k3=../results/deploy_runs/mv_r5k3 \
    --arm flashvsr_plus=../results/baseline_flashvsr_plus_mvsr4x \
    --arm infvsr=../results/baseline_infvsr_mvsr4x \
    --arm bicubic=bicubic \
    --pair mtf_measured,mtf_control --pair mtf_control,mtf_op \
    > "$R/logs/judge_mtf3way.log" 2>&1
  echo "judge exit=$? $(date '+%F %T')" >> $R/logs/mtf3way.status )
echo "DONE $(date '+%F %T')" >> $R/logs/mtf3way.status
