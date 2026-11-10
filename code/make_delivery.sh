#!/usr/bin/env bash
# The delivery path, now including the block-handover fusion as a standard step.
#
# Our video is produced 22 frames at a time with a 5-frame overlap, and the overlap used to be cut
# rather than blended: the later block's copy of those frames was written to disk and thrown away.
# Adjacent emitted frames therefore came from independent diffusion samples and the picture jumped at
# every handover -- frame-to-frame change at a handover measured 3.13x what it is inside a block, while
# FlashVSR+ sits at 0.67x, so the jump was ours and not a cost of streaming.
#
# Cross-fading the overlap on a cosine profile removes most of it: the handover jump drops 63%
# (0.9936 -> 0.3701) and improves on 76 of 77 clips, with image quality unchanged (LPIPS -0.0001,
# PSNR +0.03). It is post-processing over frames already on disk, so it costs no sampling.
#
# This is the standard overlap-blend from the literature, not a contribution. A variant that fused only
# the generated component and took the observed component from a single block -- which is what the
# range/null decomposition would permit and what Morphe's "naive blending blurs" result suggests ought
# to matter -- measured identically (LPIPS +0.0000, handover jump +0.0003), so that hypothesis is dead
# and the simple version ships.
set -u
R=/cache/zihaohe/streaming_sr0
cd $R/code || exit 1
SRC=${1:-results/deploy_runs/t4_plain2}
TAG=${2:-delivery_fused}
source $R/venv/bin/activate 2>/dev/null || true
python3 seam_fuse.py --src "$SRC" --tag "$TAG" --profile cosine --range-from current --procs 16
echo "fused frames in results/deploy/$TAG"
