#!/usr/bin/env bash
# The actual delivery, four sets, with the block-handover cross-fade applied as it is in shipping.
# Clip order is worst-first by our own LPIPS where a judge exists, so the cases that argue against us
# lead rather than trail.
set -u
R=/cache/zihaohe/streaming_sr0
cd "$R/code" || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source "$R/env.sh"
V="$R/results/videos_20261105"
mkdir -p "$V"

fuse () { # fuse <src tag> <out tag> <eval set>
  python -u seam_fuse.py --src "$R/results/deploy_runs/$1" --tag "$2" --eval-set "$3" --procs 6 \
    > "$R/logs/fuse_$2.log" 2>&1
  echo "fuse $2 exit=$? $(date '+%F %T')" >> "$R/logs/videos.status"
}
fuse t4_plain2   fused_heldout data/real_eval/heldout
fuse rvsr_plain2 fused_realvsr data/real_eval/realvsr
fuse mv_r5k3     fused_mvsr4x  data/real_eval/mvsr4x
fuse vlq_plain2  fused_videolq data/real_eval/videolq

vid () { python -u make_video_real.py "$@" >> "$R/logs/videos.log" 2>&1; echo "vid $* exit=$?" >> "$R/logs/videos.status"; }

# RealVSR: real phone degradation, reference exists, we beat FlashVSR+ on it
vid --eval-set data/real_eval/realvsr --clips 4 --cols 4 --crop 256 \
    --arm "LR (bicubic)=bicubic" \
    --arm "FlashVSR+=$R/results/baseline_flashvsr_plus_realvsr" \
    --arm "ours=$R/results/deploy/fused_realvsr" \
    --arm "GT=gt" \
    --judge "$R/results/deploy/judge_realvsr.json" --ref ours \
    --out "$V/1_realvsr"

# MVSR4x: real optics, reference exists but registration-dominated
vid --eval-set data/real_eval/mvsr4x --clips 4 --cols 4 --crop 256 \
    --arm "LR (bicubic)=bicubic" \
    --arm "FlashVSR+=$R/results/baseline_flashvsr_plus_mvsr4x" \
    --arm "ours=$R/results/deploy/fused_mvsr4x" \
    --arm "GT=gt" \
    --out "$V/2_mvsr4x"

# VideoLQ: real degradation, no reference anywhere
vid --eval-set data/real_eval/videolq --clips 4 --cols 2 --crop 256 \
    --arm "LR (bicubic)=bicubic" \
    --arm "ours=$R/results/deploy/fused_videolq" \
    --out "$V/3_videolq"

# held-out: synthetic degradation, our main benchmark
vid --eval-set data/real_eval/heldout --clips 4 --cols 4 --crop 256 \
    --arm "LR (bicubic)=bicubic" \
    --arm "FlashVSR+=$R/results/baseline_flashvsr_plus" \
    --arm "ours=$R/results/deploy/fused_heldout" \
    --arm "GT=gt" \
    --judge "$R/results/deploy/judge_new_config.json" --ref shipped_2step \
    --out "$V/4_heldout"
echo "videos done $(date '+%F %T')" >> "$R/logs/videos.status"
ls -la "$V"
