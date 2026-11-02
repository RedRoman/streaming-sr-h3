#!/usr/bin/env bash
# The sharpness line: stop measuring the low band with a distance and measure it with the statistic the
# pipeline actually reads.
#
# The complaint is specific and was reproduced by band_energy.py: pooled over 77 realistic clips our null
# band carries 0.60 of the truth's energy where FlashVSR carries 1.06, and R_phi's own mid_ratio sits at
# 0.8831 against the truth's 1.0. cross_band.py localised it -- substituting the truth's detail into both
# arms still leaves us behind (0.0849 vs 0.0820) -- so the deficit is in the GUIDE, not in the generator.
#
# An L1 between two local-std maps is still a distance, and a distance is minimised partly by being
# conservative: the first pair of runs with std_w=1 moved the gain to +2.1 dB and left mid_ratio at 0.81,
# unchanged. Both arms here remove that fixed point:
#
#   deficit  penalise only std(pred) < std(target). One-sided, so under-contrast is never traded against
#            over-contrast, and there is no conservative optimum.
#   energy   match the local RMS of the high pass rather than the high pass itself, so the model is asked
#            for the right amount of detail where it cannot know the phase.
#
# Pre-registered: mid_ratio must move above 0.93 for either arm to be worth deploying, and the estimate's
# PSNR against D(x) must not drop more than 0.3 dB -- a sharper guide that is wrong is the failure mode
# the quota gate would then amplify, since the gate scales generated detail by the guide's own local std.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
CARD=$1; ARM=$2
export ASCEND_RT_VISIBLE_DEVICES=$CARD
source env.sh
cd code
case "$ARM" in
  deficit) EXTRA="--std-mode deficit --std-w 2.0 --hf-w 1.0" ;;
  energy)  EXTRA="--hf-mode energy  --hf-w 2.0 --std-w 1.0" ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
python -u train_range2.py --tag r7_$ARM --data /cache/zihaohe/range_pairs_x4big --kinds realistic \
  --ch 128 --blocks 16 --k 3 --iters 24000 --batch 32 --crop 112 $EXTRA --device npu:0
echo "EXIT=$? $(date '+%F %T')"
