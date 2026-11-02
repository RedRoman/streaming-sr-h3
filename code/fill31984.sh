#!/usr/bin/env bash
# Six cards on the host that holds every pair set, now that its reservation pool is stopped.
#
#   card 2  the x6 holdout -- the experiment that separates the two conditioning signals
#   card 3  mtf    seeds 1 then 2
#   card 4  onehot seeds 1 then 2
#   card 5  mtf and onehot at seed 3
#   card 6  r7_deficit  (restarted; the first attempt was killed at iteration 2000 by a reservation
#                        holder that took the card during the dataset load, which patches/claim_card_early
#                        now prevents)
#   card 7  r7_energy
#
# The seeds are not padding. The mtf-minus-onehot deltas on trained operators are -0.014 to -0.144 dB and
# this project has never measured a seed spread for either arm, so the SIGN of each per-operator delta is
# currently uninterpretable -- exactly the objection the x4 line raised against its own width ablation
# before the sampling-seed noise floor was measured and invalidated it. Three seeds per arm is the minimum
# that gives a spread to compare those deltas against.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1

run () {  # run <card> <logname> <command...>
  local card=$1 name=$2; shift 2
  ( cd /cache/zihaohe/streaming_sr0
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export OMNI_DATA_SUFFIX=
    source env.sh
    cd code
    "$@" > "../logs/$name.log" 2>&1
    echo "$name card$card exit=$? $(date '+%F %T')" >> ../logs/fill31984.status ) &
}

# card 2: holdout, on the big x6 set (6836/178 windows, never trained on by any arm)
( cd /cache/zihaohe/streaming_sr0
  export ASCEND_RT_VISIBLE_DEVICES=2
  export OMNI_DATA_SUFFIX=big
  source env.sh
  cd code
  python -u omni_holdout.py --ckpt omni_mtf_k3 --ckpt omni_onehot_k3 --ops 6,3 \
    --val-limit 24 --out omni_holdout_s0.json > ../logs/omni_holdout_s0.log 2>&1
  echo "holdout_s0 card2 exit=$? $(date '+%F %T')" >> ../logs/fill31984.status ) &

run 3 omni_seed_mtf    bash -c './omni_cond.sh 3 mtf    omni_mtf_k3_s1    1; ./omni_cond.sh 3 mtf    omni_mtf_k3_s2    2'
run 4 omni_seed_onehot bash -c './omni_cond.sh 4 onehot omni_onehot_k3_s1 1; ./omni_cond.sh 4 onehot omni_onehot_k3_s2 2'
run 5 omni_seed3       bash -c './omni_cond.sh 5 mtf    omni_mtf_k3_s3    3; ./omni_cond.sh 5 onehot omni_onehot_k3_s3 3'
run 6 r7_deficit_run   ./r7_sharp.sh 6 deficit
run 7 r7_energy_run    ./r7_sharp.sh 7 energy
wait
echo "FILL DONE $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/fill31984.status
