#!/usr/bin/env bash
# Finish one FlashVSR+ clip without a wall-clock cap.
#
# The arm is >20 min/clip at 512x896 through their SDPA backend, against 46 s for FlashVSR, so the
# earlier 1700 s cap killed it mid-clip. Correctness has to be confirmed on one clip before committing
# roughly ten card-hours to all 120.
set -u
while pgrep -f 'timeout 1700 python /tmp/fvp_entry.py' > /dev/null; do sleep 20; done
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code || exit 1
CLIP=$(python3 -c "import json;print(json.load(open('../data/real_eval/const/manifest.json'))['clips'][0])")
CLIPS="$CLIP" python /tmp/fvp_entry.py > /cache/zihaohe/streaming_sr0/logs/fvp_one.log 2>&1
echo "EXIT=$?"
grep -oE '\{"clip.*' /cache/zihaohe/streaming_sr0/logs/fvp_one.log | tail -2
