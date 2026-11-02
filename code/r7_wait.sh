#!/usr/bin/env bash
# The two one-sided-loss variants, started only once cards are actually free.
#
# They were launched at 19:04 with preflight passing (cards 4/5/6 had 60.6 GB free) and died silently
# about two hours later when an eight-card TP benchmark claimed every NPU at 61.5 GB. No traceback: the
# allocator simply failed. Nothing was saved either, because train_range2 writes its first checkpoint at
# iters/12 = 2000 and they had not reached it.
#
# So: poll preflight instead of trusting a single pre-launch check, and ask for the first checkpoint much
# earlier so a kill costs minutes rather than hours.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
for try in $(seq 1 240); do
  if python code/preflight.py --cards 4,5 --need-gb 20 --eval-set data/real_eval/heldout > /tmp/pf.txt 2>&1; then
    echo "cards free at $(date +%H:%M:%S) after $try checks"; break
  fi
  sleep 300
done
grep -E 'FAIL|PASS  card' /tmp/pf.txt | head -4
python code/preflight.py --cards 4,5 --need-gb 20 --eval-set data/real_eval/heldout > /dev/null 2>&1 || {
  echo 'still no cards after 20h, giving up' >&2; exit 1; }

D=/cache/zihaohe/range_pairs_x4big
COMMON="--data $D --kinds realistic --ch 128 --blocks 16 --iters 24000 --batch 32 --crop 112 --k 3"
(export ASCEND_RT_VISIBLE_DEVICES=4; source env.sh; cd code
 python -u train_range2.py --tag r7_deficit $COMMON --hf-w 1.0 --std-w 2.0 \
   --std-mode deficit --hf-mode l1 --device npu:0 > ../logs/r7_deficit.log 2>&1
 echo "r7_deficit exit=$? $(date +%H:%M:%S)") &
(export ASCEND_RT_VISIBLE_DEVICES=5; source env.sh; cd code
 python -u train_range2.py --tag r7_energy $COMMON --hf-w 2.0 --std-w 2.0 \
   --std-mode deficit --hf-mode energy --device npu:0 > ../logs/r7_energy.log 2>&1
 echo "r7_energy exit=$? $(date +%H:%M:%S)") &
wait
export ASCEND_RT_VISIBLE_DEVICES=4
source env.sh
cd code
python -u eval_range.py ../results/range_branch/r5_k3.pt ../results/range_branch/r7_deficit.pt \
  ../results/range_branch/r7_energy.pt --data $D --device npu:0 2>&1 \
  | grep -aviE 'owner|warnings.warn|x = torch'
