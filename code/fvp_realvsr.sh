#!/usr/bin/env bash
# FlashVSR+ on RealVSR, so the second real-degradation set carries the same baseline as the first.
# Same Ascend fixes as fvp_heldout.sh (dense mask via expand+reshape, channels_last off), published
# code path otherwise. 50 clips of 50 frames, sharded over six cards.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
for i in 0 1 2 3 4 5; do
  (
    export ASCEND_RT_VISIBLE_DEVICES=$i
    export EVAL_SET=data/real_eval/realvsr
    export FVP_OUT_SUFFIX=_realvsr
    export FVP_MASK_VERIFY=1 FVP_MASK_IMPL=fast FVP_MASK_CACHE=4
    export SHARD=$i NSHARDS=6
    source /cache/zihaohe/streaming_sr0/env.sh
    cd code || exit 1
    python -u /tmp/fvp_entry.py > "/cache/zihaohe/streaming_sr0/logs/fvp_rvsr$i.log" 2>&1
    echo "fvp_rvsr shard$i card$i exit=$? $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/fvp_realvsr.status
  ) &
done
wait
echo "fvp_rvsr all done $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/fvp_realvsr.status
