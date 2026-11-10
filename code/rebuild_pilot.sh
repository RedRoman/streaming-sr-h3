#!/usr/bin/env bash
# Rebuild the Stage-1 training latents. data/pilot_latents was gone; the source dataset survives at
# /temp/h3_unified_dataset_20260918 with its manifests (2182 train / 114 val clips).
#
# Why this is being rebuilt at all: OmniVR's public adapter is rank 384 and 3803M parameters against
# our rank 32 and 149M -- 26x the adaptation capacity -- and it was trained on a 100K-shot corpus while
# ours saw a pilot. On screen our output reads softer than FlashVSR+ and the viewer called it, so the
# question is whether our ceiling is the method or the adapter. That is testable without OmniVR running:
# retrain at a higher rank on the same data and see whether sharpness follows.
#
# Measured cost: 7.7 s per sample, 3.7 MB per sample. 2182 clips x 2 windows is about 70 minutes on
# eight cards and 16 GB on disk.
set -u
R=/cache/zihaohe/streaming_sr0
: > $R/logs/pilot_rebuild.status
for S in 0 1 2 3 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$S
    source $R/env.sh
    cd $R/code
    python preencode.py --split train --windows 2 --shard $S --nshards 8 \
      --out $R/data/pilot_latents > "$R/logs/preenc_train_$S.log" 2>&1
    echo "train shard$S exit=$? $(date '+%F %T')" >> $R/logs/pilot_rebuild.status ) &
done
wait
( export ASCEND_RT_VISIBLE_DEVICES=0
  source $R/env.sh
  cd $R/code
  python preencode.py --split val --windows 2 --out $R/data/pilot_latents \
    > "$R/logs/preenc_val.log" 2>&1
  echo "val exit=$? $(date '+%F %T')" >> $R/logs/pilot_rebuild.status )
echo "train files: $(ls $R/data/pilot_latents/train/*.safetensors 2>/dev/null | wc -l)" >> $R/logs/pilot_rebuild.status
echo "val files:   $(ls $R/data/pilot_latents/val/*.safetensors 2>/dev/null | wc -l)" >> $R/logs/pilot_rebuild.status
echo "DONE $(date '+%F %T')" >> $R/logs/pilot_rebuild.status
