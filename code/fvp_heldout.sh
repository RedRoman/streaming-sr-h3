#!/usr/bin/env bash
# FlashVSR+ over the 96-clip held-out set, so the comparison table has a second modern baseline.
#
# This arm was unusable until two Ascend-specific costs were removed, both measured rather than assumed:
#   * the dense attention mask was rebuilt on every attention call through two chained repeat_interleave
#     calls -- 78.55 s per build for the [1,12,7168,7168] mask that the window geometry produces, against
#     0.088 s for the equivalent expand+reshape (888x, bitwise equal, asserted at runtime). 240 builds per
#     clip is the 5 hours the clip spent producing nothing.
#   * the TinyDecoder's NHWC convolution storage, which Ascend rejects outright; their own CLI exposes
#     --no-tiny-decoder-channels-last, so the arm stays the published code path.
# One clip now takes 58.5 s for 124 frames and scores +2.545 dB over bicubic, i.e. the output is aligned
# and sane, not merely present.
#
# Cards 1, 2 and 7 are the dc=y sampling run and 3-6 are a colleague's job, so this takes card 0 alone:
# roughly 96 x 80 s including per-clip model load, about two and a quarter hours.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
export FVP_MASK_VERIFY=1
export FVP_MASK_IMPL=fast
export FVP_MASK_CACHE=4
source /cache/zihaohe/streaming_sr0/env.sh
cd code || exit 1
python -u /tmp/fvp_entry.py > /cache/zihaohe/streaming_sr0/logs/fvp_ho_plus.log 2>&1
echo "EXIT=$? at $(date +%H:%M:%S)"
echo "clips with output: $(ls -d /cache/zihaohe/streaming_sr0/results/baseline_flashvsr_plus/*/ 2>/dev/null | wc -l)"
grep -c failed /cache/zihaohe/streaming_sr0/logs/fvp_ho_plus.log || true
