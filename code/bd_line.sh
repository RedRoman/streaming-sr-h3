#!/usr/bin/env bash
# Make Vid4 a benchmark we can legitimately report a gain on.
#
# Vid4 ships two degradations and both are currently outside R_phi's training family: BI is y = D(x)
# exactly, so the identity is optimal and our realistic-trained estimator loses 12.84 dB; BD is
# blur-then-direct-subsample with no noise and no compression, where we measured +0.251 dB mean with
# 2 of 4 clips going backwards -- noise, not a result. The cause is specific and measurable, not vague
# domain drift: the estimator learned to remove noise and compression and finds neither.
#
# So BD becomes its own operator. Three steps, each a single variable against something that exists:
#   1. pairs      prepare_range_data with DEGRADE_MODE=bd, same clips/windows as the x4 big set
#   2. sp_bd      a specialist on them, matched to r5_big (4.7M, K=1, 24000 iters, crop 112)
#   3. Vid4       eval sp_bd on Vid4 BDx4, against the r6_xl_k3 number already on record
#
# If sp_bd lands a clear positive gain on Vid4 BDx4 where r6_xl_k3 got +0.251 with 2/4 clips negative,
# the public-benchmark hole closes and the limitation becomes "the operator must be in the training
# family", which is a statement we can also support with the omni result.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=1
source env.sh
cd code
D=/cache/zihaohe/range_pairs_bd
for sp in train val; do
  if [ "$sp" = train ]; then n=2182; w=4; else n=200; w=2; fi
  for i in $(seq 0 7); do
    SR_SCALE=4 DEGRADE_MODE=bd RANGE_OUT=$D \
      python prepare_range_data.py --split $sp --clips $n --windows $w --shard $i --nshards 8 \
      > ../logs/rp_bd_${sp}_$i.log 2>&1 &
  done
  wait
  echo "bd $sp: $(ls $D/$sp 2>/dev/null | wc -l) files at $(date +%H:%M:%S)"
done

python -u train_range2.py --tag sp_bd --data $D --kinds realistic \
  --ch 128 --blocks 16 --iters 24000 --batch 32 --crop 112 --hf-w 1.0 --std-w 1.0 --device npu:0 \
  > ../logs/sp_bd.log 2>&1
echo "sp_bd exit=$? $(date +%H:%M:%S)"

for CK in results/range_branch/sp_bd.pt results/range_branch/r6_xl_k3.pt; do
  echo "##### Vid4 BDx4 with $CK"
  python -u bench_vid4.py --deg BDx4 --rphi "$CK" --out "results/vid4_BDx4_$(basename $CK .pt).json" 2>&1 \
    | grep -aviE 'owner|warnings.warn|UserWarning|lr = lr' | tail -12
done
