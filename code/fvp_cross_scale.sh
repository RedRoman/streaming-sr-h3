#!/usr/bin/env bash
# FlashVSR+ at x2 and x8: a fixed-scale method used off its design point, which is the honest
# alternative to one estimator covering the scale family.
#
# Verified on one clip per scale before committing 192: output geometry is correct (480x832, so
# run_stream does derive the ratio from --height/--width rather than assuming 4x), and the result is
# sane -- x2 gives +2.909 dB over bicubic with LPIPS 0.1229 vs 0.7113, while x8 gives -0.135 dB, i.e.
# it falls BELOW bicubic on PSNR off its design point while still winning LPIPS. That is the effect
# worth measuring at scale.
#
# FVP_MASK_MAX_ENTRIES=4e9 because their guard is a 2.0 GB memory ceiling and our geometry lands at
# 2,013,265,920 entries, 0.66% over, on 64 GB cards. Their default is untouched.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
i=0
for SC in 2 8; do
  for SH in 0 1 2; do
    dev=$(echo "1 3 4 5 6 7" | cut -d' ' -f$((i+1)))
    (export ASCEND_RT_VISIBLE_DEVICES=$dev
     export EVAL_SET=data/real_eval/heldout_s$SC
     export FVP_SCALE=$SC FVP_OUT_SUFFIX=_s$SC
     export FVP_MASK_IMPL=fast FVP_MASK_CACHE=2 FVP_MASK_MAX_ENTRIES=4e9 FVP_MASK_VERIFY=0
     export SHARD=$SH NSHARDS=3
     source env.sh; cd code
     python -u /tmp/fvp_entry.py > ../logs/fvpx_s${SC}_$SH.log 2>&1
     echo "x$SC shard$SH exit=$? $(date +%H:%M:%S)") &
    i=$((i+1)); sleep 20
  done
done
wait
for SC in 2 8; do
  echo "x$SC: $(ls -d results/baseline_flashvsr_plus_s$SC/*/ 2>/dev/null | wc -l) clips, failures $(ls results/baseline_flashvsr_plus_s$SC/FAILED_* 2>/dev/null | wc -l)"
done
