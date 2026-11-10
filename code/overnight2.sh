#!/usr/bin/env bash
# GPU chain, rewritten. The first version lost three stages to a shell bug of mine: a stage function
# did `cd code` and, because stage() calls functions in the current shell, every later stage inherited
# that directory and could not even create its own log. Stages now run in a subshell from a fixed
# absolute root, and logs and status use absolute paths.
#
# Stage order changed too. The MVSR4x result that looked like a framework failure was measured against
# bicubic alone, and on this dataset bicubic is nearly a no-op: the package ships LQ and GT both at
# 1024x1024, the LR here is LQ downsampled by four, so upsampling it back reproduces LQ. Comparing a
# generative method to "do nothing" on real paired capture is close to rigged, because the LQ-GT pair
# is only 21.5-28.1 dB apart to begin with -- parallax and registration error that punishes any method
# that synthesises detail and leaves an untouched one alone. InfVSR and FlashVSR+ are the controls that
# separate "our framework fails here" from "this dataset punishes everything generative", and InfVSR
# has published numbers on MVSR4x, so it is the right anchor.
set -u
R=/cache/zihaohe/streaming_sr0
cd $R || exit 1
ST=$R/logs/overnight2.status
: > $ST
say () { echo "$(date '+%F %T') $*" >> $ST; }

stage () {           # stage <name> <fn>
  local name=$1; shift
  say "BEGIN $name"
  if ( cd $R && "$@" ) >> "$R/logs/on2_${name}.log" 2>&1; then
    say "OK    $name"
  else
    say "FAIL  $name exit=$?  (continuing)"
  fi
}

# --- 1. the controls that decide whether the MVSR4x reading was about us or about the dataset
mv_flashvsr_plus () {
  test -f code/baseline_flashvsr_plus.py || return 1
  for S in 0 1 2 3; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$S
      export EVAL_SET=data/real_eval/mvsr4x
      export FVP_OUT_SUFFIX=_mvsr4x
      export SHARD=$S NSHARDS=4
      source $R/env.sh
      cd $R/code
      python baseline_flashvsr_plus.py ) &
  done
  wait
  n=$(ls -d results/baseline_flashvsr_plus_mvsr4x/*/ 2>/dev/null | wc -l); echo "fvp_mvsr4x: $n"; test "$n" -ge 15
}

mv_infvsr () {
  test -f code/baseline_infvsr.py || return 1
  for S in 0 1 2 3; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$((S+4))
      export EVAL_SET=data/real_eval/mvsr4x
      export INFVSR_OUT_SUFFIX=_mvsr4x
      export SHARD=$S NSHARDS=4
      source $R/env.sh
      cd $R/code
      python baseline_infvsr.py ) &
  done
  wait
  n=$(ls -d results/baseline_infvsr_mvsr4x/*/ 2>/dev/null | wc -l); echo "infvsr_mvsr4x: $n"; test "$n" -ge 15
}

# --- 2. rejudge MVSR4x with every arm present, and WITHOUT psnr_gtrange as a headline:
#        GTrange is the x4-bicubic range of the truth, which is not what the observation determines
#        when the real operator is a lens. It stays in the json, it does not stay in the argument.
mv_judge_all () {
  export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/mvsr4x
  source $R/env.sh
  cd $R/code
  A="--arm ours=../results/deploy_runs/mv_r5k3 --arm bicubic=bicubic"
  test -d ../results/baseline_flashvsr_plus_mvsr4x && A="$A --arm flashvsr_plus=../results/baseline_flashvsr_plus_mvsr4x"
  test -d ../results/baseline_infvsr_mvsr4x && A="$A --arm infvsr=../results/baseline_infvsr_mvsr4x"
  python -u deploy_eval.py --procs 10 eval --name judge_mvsr4x_all $A \
    --pair ours,bicubic
}

# --- 3. VideoLQ with the shipped estimator, so the real-degradation number is produced by the same
#        R_phi as the baseline table (today's run used omni_m5, the multi-operator arm)
videolq_r5k3 () {
  for S in 0 1 2 3 4 5 6 7; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$S
      export EVAL_SET=data/real_eval/videolq
      source $R/env.sh
      cd $R/code
      python run_deploy.py --tag vlq_r5k3 --dc rphi --guide y \
        --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
        --shard $S --nshards 8 ) &
  done
  wait
  n=$(ls results/deploy_runs/vlq_r5k3/*/stream_result.json 2>/dev/null | wc -l)
  echo "vlq_r5k3: $n"; test "$n" -ge 46
}

# --- 4. the 17.5x data point, which closes the starvation question on the conditioning comparison
sweep_full () {
  for i in 0 1 2 3; do
    case $i in
      0) C=mtf;    SD=0;; 1) C=mtf;    SD=1;;
      2) C=onehot; SD=0;; 3) C=onehot; SD=1;;
    esac
    ( export ASCEND_RT_VISIBLE_DEVICES=$((i+1))
      export OMNI_DATA_SUFFIX=big
      source $R/env.sh
      cd $R/code
      python -u omni_stage1.py --tag "omni_${C}_dfull_s${SD}" \
        --scales 2,4,8,deblur,denoise --cond "$C" --k 3 \
        --iters 12000 --log-every 3000 --batch 30 --crop 48 --ch 96 --blocks 12 \
        --val-limit 24 --seed "$SD" --device npu:0 ) &
  done
  wait
  test -f results/omni/omni_mtf_dfull_s0.json
}

stage mv_flashvsr_plus mv_flashvsr_plus
stage mv_infvsr        mv_infvsr
stage mv_judge_all     mv_judge_all
stage videolq_r5k3     videolq_r5k3
stage sweep_full       sweep_full
say "GPU CHAIN DONE"
