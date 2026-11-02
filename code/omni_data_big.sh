#!/usr/bin/env bash
# Regenerate every operator's pairs at the scale the x4 line has shown to matter.
#
# x2/x8 come from the video manifest (2182 clips, 8 windows each, against the old 400 x 2); deblur and
# denoise are synthesised from the x4 pairs, so they inherit the big set by pointing MAKE_OPS_SRC at it.
# Without this, retraining omni on "more data" would only move the x4 slice and the deblur/denoise
# deficits -- the two largest -- would stay in the starved regime that produced them.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code
for S in 2 8; do
  for sp in train val; do
    if [ "$sp" = train ]; then n=2182; w=8; else n=200; w=2; fi
    for i in $(seq 0 7); do
      SR_SCALE=$S RANGE_OUT=/cache/zihaohe/range_pairs_s${S}big \
        python prepare_range_data.py --split $sp --clips $n --windows $w --shard $i --nshards 8 \
        > ../logs/rp_${S}big_${sp}_$i.log 2>&1 &
    done
    wait
    echo "s$S $sp: $(ls /cache/zihaohe/range_pairs_s${S}big/$sp 2>/dev/null | wc -l) at $(date +%H:%M:%S)"
  done
done
for OP in deblur denoise; do
  MAKE_OPS_SRC=/cache/zihaohe/range_pairs_x4big \
    python -u make_ops.py --op $OP --out /cache/zihaohe/range_pairs_${OP}big --procs 24 \
    > ../logs/mkops_${OP}big.log 2>&1
  echo "$OP: train=$(ls /cache/zihaohe/range_pairs_${OP}big/train 2>/dev/null|wc -l) at $(date +%H:%M:%S)"
done
echo OMNI_DATA_BIG_DONE
