#!/usr/bin/env bash
# One FlashVSR+ clip with the mask cache in place, no wall-clock cap.
#
# Before the cache a single clip had produced nothing after four hours, with 99.96% of profiled self time
# inside the dense attention-mask expansion. FVP_MASK_VERIFY=2 makes the first two cache hits rebuild the
# mask and assert bitwise equality, so the log proves the cache both fired and is exact.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export FVP_MASK_VERIFY=2
export FVP_MASK_IMPL=${FVP_MASK_IMPL:-fast}
export FVP_MASK_CACHE=${FVP_MASK_CACHE:-4}
source /cache/zihaohe/streaming_sr0/env.sh
mkdir -p /cache/zihaohe/streaming_sr0/logs
cd code || exit 1
CLIP=$(python3 -c "import json;print(json.load(open('../data/real_eval/const/manifest.json'))['clips'][0])")
echo "clip=$CLIP start=$(date +%H:%M:%S)"
CLIPS="$CLIP" python -u /tmp/fvp_entry.py > /cache/zihaohe/streaming_sr0/logs/fvp_one2.log 2>&1
echo "EXIT=$? end=$(date +%H:%M:%S)"
grep -E 'MASK_CACHE|\{"clip' /cache/zihaohe/streaming_sr0/logs/fvp_one2.log | tail -5
