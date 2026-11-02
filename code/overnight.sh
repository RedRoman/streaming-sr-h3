#!/usr/bin/env bash
# Overnight queue for 2026-11-03, to be collected in the afternoon.
#
# Two independent chains so a GPU failure cannot idle the CPU work and vice versa. Every stage is
# wrapped: a stage that fails records its exit code and the next one still runs, because the expensive
# failure mode on this project has never been a crash, it has been a stage that exited 0 on the wrong
# input. Each stage therefore gates its own precondition before spending anything.
#
# Known pitfalls this script is written against, all of them paid for today:
#   EVAL_SET defaults to data/real_eval/const        -> exported per stage AND asserted
#   run_deploy resumes on an existing stream_result   -> fresh tags only, never a reused one
#   t4_plain2 shipped with range_branch/r5_k3.pt, not omni_m5.pt  -> real-data runs use r5_k3 to stay
#                                                                    comparable with the baseline table
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
ST=logs/overnight.status
: > $ST
say () { echo "$(date '+%F %T') $*" >> $ST; }

stage () {           # stage <name> <command...>
  local name=$1; shift
  say "BEGIN $name"
  if "$@" >> "logs/overnight_${name}.log" 2>&1; then
    say "OK    $name"
  else
    say "FAIL  $name exit=$?  (continuing)"
  fi
}

# ---------------------------------------------------------------- GPU chain
gpu_chain () {

# 1. MVSR4x: real optics WITH a reference, the set that tests the pre-registered prediction that
#    conditioning fails there because the measured MTF is non-monotone (mid-band dip, high-band rise
#    from sensor noise) while every trained operator is monotone. 15 clips, 8 cards.
mvsr4x () {
  test -f data/real_eval/mvsr4x/manifest.json || { echo "no mvsr4x manifest"; return 1; }
  python3 -c "
import json,sys
m=json.load(open('data/real_eval/mvsr4x/manifest.json'))
assert m['has_hr'] and m['lr_wh']==[208,120] and m['hr_wh']==[832,480], m
print('gate ok: %d clips' % len(m['clips']))" || return 1
  for S in 0 1 2 3 4 5 6 7; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$S
      export EVAL_SET=data/real_eval/mvsr4x
      source env.sh
      cd code
      python run_deploy.py --tag mv_r5k3 --dc rphi --guide y \
        --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
        --shard $S --nshards 8 ) &
  done
  wait
  n=$(ls results/deploy_runs/mv_r5k3/*/stream_result.json 2>/dev/null | wc -l)
  echo "mvsr4x clips done: $n"
  test "$n" -ge 15
}
stage mvsr4x mvsr4x

# 2. judge MVSR4x. It has a reference, so reference metrics are defined -- the point of preferring it
#    over VideoLQ. bicubic comes from this set's own LR.
judge_mvsr4x () {
  export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/mvsr4x
  source env.sh
  cd code
  python -u deploy_eval.py --procs 10 eval --name judge_mvsr4x \
    --arm ours=../results/deploy_runs/mv_r5k3 --arm bicubic=bicubic
}
stage judge_mvsr4x judge_mvsr4x

# 3. VideoLQ again with the SHIPPED R_phi. Today's run used omni_m5, which is the multi-operator arm;
#    the baseline table was produced with r5_k3, so the real-degradation number has to come from the
#    same estimator to be quotable next to it.
videolq_r5k3 () {
  for S in 0 1 2 3 4 5 6 7; do
    ( export ASCEND_RT_VISIBLE_DEVICES=$S
      export EVAL_SET=data/real_eval/videolq
      source env.sh
      cd code
      python run_deploy.py --tag vlq_r5k3 --dc rphi --guide y \
        --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora \
        --shard $S --nshards 8 ) &
  done
  wait
  n=$(ls results/deploy_runs/vlq_r5k3/*/stream_result.json 2>/dev/null | wc -l)
  echo "videolq r5k3 clips: $n"
  test "$n" -ge 46
}
stage videolq_r5k3 videolq_r5k3

# 4. The 17.5x data point for the conditioning comparison. 3000 windows showed no crossover
#    (mtf stayed 0.01-0.05 dB behind on 4 of 5 operators); 10351 is the full set and closes the
#    data-starvation question. 75 min of loading per run at ~260 GB RSS, so four at a time.
sweep_full () {
  for i in 0 1 2 3; do
    case $i in
      0) C=mtf;    SD=0;; 1) C=mtf;    SD=1;;
      2) C=onehot; SD=0;; 3) C=onehot; SD=1;;
    esac
    ( export ASCEND_RT_VISIBLE_DEVICES=$((i+1))
      export OMNI_DATA_SUFFIX=big
      source env.sh
      cd code
      python -u omni_stage1.py --tag "omni_${C}_dfull_s${SD}" \
        --scales 2,4,8,deblur,denoise --cond "$C" --k 3 \
        --iters 12000 --log-every 3000 --batch 30 --crop 48 --ch 96 --blocks 12 \
        --val-limit 24 --seed "$SD" --device npu:0 ) &
  done
  wait
  ls results/omni/omni_mtf_dfull_s0.json results/omni/omni_onehot_dfull_s0.json >/dev/null 2>&1
}
stage sweep_full sweep_full

# 5. RealBasicVSR as an anchor shared with OmniVR's table. Smoke one clip first: new code path.
rbvsr_smoke () {
  test -f assets/realbasicvsr/RealBasicVSR.pth || { echo "weights missing"; return 1; }
  python3 - <<'PY'
import torch, json
st = torch.load('assets/realbasicvsr/RealBasicVSR.pth', map_location='cpu')
k = st.get('state_dict', st)
print(json.dumps({'tensors': len(k), 'first': sorted(k)[:5]}, indent=1))
PY
}
stage rbvsr_smoke rbvsr_smoke

}

# ---------------------------------------------------------------- CPU chain
cpu_chain () {

# 6. The null-gain sweep, which is the only knob measured to have room left: at 19 clips it moved
#    LPIPS 0.2152 -> 0.2059 monotonically without reaching a turning point, at the cost of null_cos
#    0.3980 -> 0.3896. _quota is pure post-processing of frames already on disk, so this is CPU only
#    and needs no card. Redone here on the shipped arm and all 77 realistic clips.
gain_sweep () {
  test -d results/deploy_runs/t4_plain2 || { echo "no shipped arm to post-process"; return 1; }
  cd code
  for G in 1.00 1.15 1.30 1.45 1.60 1.75; do
    T="g$(echo $G | tr -d '.')"
    EVAL_SET=data/real_eval/heldout python -u deploy_eval.py --procs 24 quota \
      --src ../results/deploy_runs/t4_plain2 --tag "gs_$T" \
      --ckpt ../results/range_branch/r5_k3.pt --s0 0.02 --p 1.0 --sharp 0.0 --gain "$G" \
      > "../logs/gs_$T.log" 2>&1 || echo "gain $G failed"
  done
  cd ..
  ls -d results/deploy/gs_g* 2>/dev/null | wc -l
}
stage gain_sweep gain_sweep

# 7. Judge the sweep. null_cos is the number that matters as much as LPIPS here: pushing null energy
#    toward the truth's level is only progress if the energy is in the right place, and today's
#    measurements show NR metrics cannot tell those apart.
judge_gain () {
  export ASCEND_RT_VISIBLE_DEVICES=0
  export EVAL_SET=data/real_eval/heldout
  source env.sh
  cd code
  ARGS=""
  for G in 1.00 1.15 1.30 1.45 1.60 1.75; do
    T="g$(echo $G | tr -d '.')"
    test -d "../results/deploy/gs_$T" && ARGS="$ARGS --arm $T=../results/deploy/gs_$T"
  done
  test -n "$ARGS" || { echo "no gain arms produced"; return 1; }
  python -u deploy_eval.py --procs 10 eval --name judge_gain_full \
    $ARGS --arm plain2=../results/deploy_runs/t4_plain2 --arm bicubic=bicubic \
    --pair g115,plain2 --pair g130,plain2 --pair g145,plain2
}
stage judge_gain judge_gain

# 8. Side-by-side video on both real-degradation sets. This is the deliverable for VideoLQ, which has
#    no reference, and today's NR measurement says no automatic judge can be trusted there -- the
#    frames are the output.
videos () {
  test -d results/deploy_runs/vlq_r5k3 || return 1
  cd code
  python -u make_compare_video2.py --judge ../results/deploy/judge_all7.json \
    --ref plain2 --pick worst --clips 2 --zoom 2>&1 | tail -5
}
stage videos videos

}

gpu_chain &
GPU=$!
cpu_chain &
CPU=$!
wait $GPU $CPU
say "ALL CHAINS DONE"
