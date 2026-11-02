#!/usr/bin/env bash
# Three single-variable follow-ups to r3_big24, plus the next data doubling.
#
# r3_big24 (4.7M, 24000 iters, 5184 windows, K=1) is the new reference. Data scaling moved the estimate
# +0.457 dB, a longer schedule +0.200, and capacity +0.164 -- and capacity only started paying once the
# data was there: at 598 windows the 4.7M model was 0.14 dB WORSE than the 2.0M one. Both the capacity
# ceiling and the K>1 question were therefore decided under data starvation and have to be re-asked.
#
#   r4_xl   ch192/blk20 (~10M), everything else identical  -> is the capacity ramp still climbing
#   r4_k3   K=3 causal stack, everything else identical    -> mech6 measured range content 0.614-correlated
#                                                             across a block stride, so temporal context
#                                                             should help; the old K sweep said it did not,
#                                                             but that was at 598 windows
#   windows 4 -> 8                                         -> the clip list is exhausted at 2182, so the
#                                                             only remaining data axis is windows per clip
#
# Both trainings start before the new windows land, so they see the same 5184 windows r3_big24 saw;
# train_windows is recorded in each checkpoint's json so the attribution stays checkable.
#
# Cards 1 and 7 are the deploy verification (20.1B DiT, cannot share). Cards 0 and 2 hold a colleague's
# 35 GB jobs with 29 GB free each, and R_phi training measured 2.8-3.5 GB, so these go there as guests.
# Data generation takes 8 shards rather than 16: the deploy run is already loading PNGs and the load
# average is 71.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
D=/cache/zihaohe/range_pairs_x4big
COMMON="--data $D --kinds realistic --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0"

(
  export ASCEND_RT_VISIBLE_DEVICES=0
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r4_xl --ch 192 --blocks 20 $COMMON --device npu:0 \
    > ../logs/r4_xl.log 2>&1
  echo "r4_xl exit=$? $(date +%H:%M:%S)"
) &

(
  export ASCEND_RT_VISIBLE_DEVICES=2
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r4_k3 --k 3 --ch 128 --blocks 16 $COMMON --device npu:0 \
    > ../logs/r4_k3.log 2>&1
  echo "r4_k3 exit=$? $(date +%H:%M:%S)"
) &

(
  export ASCEND_RT_VISIBLE_DEVICES=0
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  for i in $(seq 0 7); do
    SR_SCALE=4 RANGE_OUT=$D python prepare_range_data.py --split train --clips 2182 --windows 8 \
      --shard $i --nshards 8 > ../logs/rpbig8_$i.log 2>&1 &
  done
  wait
  echo "windows=8 done: $(ls $D/train | wc -l) files at $(date +%H:%M:%S)"
) &

wait
echo R4_ALL_DONE
