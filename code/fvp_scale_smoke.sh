#!/usr/bin/env bash
# One clip at x2 and x8 before committing cards to 96 each.
#
# FlashVSR+ is a x4 method. Our eval sets all share the same 480x832 output and differ only on the LR
# side, so running it at x2/x8 means a different input resolution for the same output -- but whether
# run_stream derives the ratio from --height/--width or assumes 4x internally is unknown, and the x4
# invocation happened to be exactly 4x so it never had to be distinguished. If it silently does 4x the
# output geometry will be wrong and the assert in the driver will catch it; if it errors we learn that
# instead. Either way it costs two clips, not 192.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
for SC in 2 8; do
  SET=$([ "$SC" = 4 ] && echo heldout || echo heldout_s$SC)
  CLIP=$(python3 -c "import json;print(json.load(open('data/real_eval/$SET/manifest.json'))['clips'][0])")
  echo "########## x$SC  clip=$CLIP"
  (export ASCEND_RT_VISIBLE_DEVICES=1
   export EVAL_SET=data/real_eval/$SET
   export FVP_SCALE=$SC
   export FVP_MASK_VERIFY=1 FVP_MASK_IMPL=fast FVP_MASK_CACHE=2 FVP_MASK_MAX_ENTRIES=4e9
   export CLIPS="$CLIP"
   export FVP_OUT_SUFFIX=_s$SC
   source env.sh; cd code
   python -u /tmp/fvp_entry.py 2>&1 | grep -aviE 'owner|warnings.warn|UserWarning|ImportWarning|^\s*\*|^\s*The |^\s*torch\.' | tail -6)
done
