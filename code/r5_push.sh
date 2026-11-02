#!/usr/bin/env bash
# Data doubled again (5184 -> 10351 realistic windows), and the K>1 question re-asked on top of it.
#
#   r5_big  vs r3_big24  same 4.7M / 24000 iters / K=1 -> isolates the second data doubling
#   r5_k3   vs r5_big    same data / size / schedule, K=3 -> isolates the causal stack
#
# K>1 lost the first time it was tried, but that sweep ran on 598 windows, and the capacity sweep that
# also lost there has since reversed (4.7M was 0.14 dB worse than 2.0M at 598 windows and is 0.16 dB
# better at 5184). mech6 measured range content 0.614-correlated across a block stride against 0.103 for
# the null band, so the information is there to be used. The K=3 path was broken until today -- it built a
# 9-channel stem and was fed one frame -- and is verified by a 24-iter run before this launches.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
D=/cache/zihaohe/range_pairs_x4big
COMMON="--data $D --kinds realistic --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0 --ch 128 --blocks 16"
(
  export ASCEND_RT_VISIBLE_DEVICES=0
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r5_big $COMMON --device npu:0 > ../logs/r5_big.log 2>&1
  echo "r5_big exit=$? $(date +%H:%M:%S)"
) &
(
  export ASCEND_RT_VISIBLE_DEVICES=2
  source /cache/zihaohe/streaming_sr0/env.sh
  cd code
  python -u train_range2.py --tag r5_k3 --k 3 $COMMON --device npu:0 > ../logs/r5_k3.log 2>&1
  echo "r5_k3 exit=$? $(date +%H:%M:%S)"
) &
wait
echo R5_DONE
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code
python -u eval_range.py ../results/range_branch/r3_big24.pt ../results/range_branch/r5_big.pt \
  ../results/range_branch/r5_k3.pt --data $D --device npu:0 2>&1 | grep -viE 'owner|warnings.warn|x = torch'
