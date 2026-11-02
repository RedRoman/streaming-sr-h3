#!/usr/bin/env bash
# Is the mtf-vs-onehot null result a product of training-set size?
#
# The published arms read ~590 windows per operator. This file's own DATA comment records that data, not
# capacity, binds this family: 8.7x the data moved the x4 estimate +0.457 dB and deployed LPIPS -0.0178,
# "and the omni deficits on deblur (-0.19 dB) and denoise (-0.26 dB) were therefore measured under the
# same starvation". A 17-dim conditioning vector is more data-hungry than a 6-dim one, so a tie at 590
# windows cannot be read as a tie in general. The big sets hold 10351-13672 per operator, 17x more, and
# were never used for THIS comparison.
#
# PRE-REGISTERED:
#   1. If mtf-minus-onehot improves monotonically with data, the 590-window result is starvation and the
#      in-vocabulary claim must be restated at the larger size.
#   2. If it is flat, the tie is a property of the signals and not of the sample count -- which is the
#      stronger version of the negative result, and the one worth reporting.
#   3. deblur is watched separately: it is the only operator with a consistent sign (4/4 seeds, -0.109),
#      so if any operator moves with data it should be that one.
# Seeds, not a single run: the effect under test is ~0.01-0.1 dB against a seed sd of 0.01-0.04.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
LIMIT=${1:-3000}
run () {
  local card=$1 cond=$2 seed=$3
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    export OMNI_DATA_SUFFIX=big
    source env.sh
    cd code
    python -u omni_stage1.py --tag "omni_${cond}_d${LIMIT}_s${seed}" \
      --scales 2,4,8,deblur,denoise --cond "$cond" --k 3 --train-limit "$LIMIT" \
      --iters 12000 --log-every 2000 --batch 30 --crop 48 --ch 96 --blocks 12 \
      --val-limit 24 --seed "$seed" --device npu:0 \
      > "../logs/sweep_${cond}_d${LIMIT}_s${seed}.log" 2>&1
    echo "${cond}_d${LIMIT}_s${seed} card${card} exit=$? $(date '+%F %T')" >> ../logs/sweep_data.status
  ) &
}
# card 0 is held by the x6 judge
run 1 mtf    0
run 2 mtf    1
run 3 mtf    2
run 4 mtf    3
run 5 onehot 0
run 6 onehot 1
run 7 onehot 2
wait
# the fourth onehot seed, once a card frees
run 1 onehot 3
wait
echo "ALL DONE limit=$LIMIT $(date '+%F %T')" >> ../logs/sweep_data.status
