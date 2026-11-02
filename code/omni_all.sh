#!/usr/bin/env bash
# One range branch over all five operators, and the matched comparison against the five specialists.
#
# Two families have been shown separately: scales 2/4/8 (one operator at three strengths) and
# 4/deblur/denoise (three operators that differ in kind, including one whose null space is empty). Neither
# run covers both at once, so "one model over the operator family" is still two half-claims. This trains
# the union and scores it on identical data against every specialist that exists.
#
# Waits for the FlashVSR+ held-out sweep to release card 0 rather than contending with it.
set -u
while pgrep -f 'fvp_entry.py' > /dev/null; do sleep 60; done
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
source /cache/zihaohe/streaming_sr0/env.sh
cd code || exit 1

for TAG in omni5 omni5sev; do
  SEV=""
  [ "$TAG" = omni5sev ] && SEV="--use-sev"
  python -u omni_stage1.py --tag "$TAG" --scales 2,4,8,deblur,denoise $SEV \
    --iters 6000 --batch 30 --crop 48 --device npu:0 --val-limit 24 \
    > ../logs/$TAG.log 2>&1
  echo "$TAG exit=$? at $(date +%H:%M:%S)"
done

# full validation split, every model on the same clips and frames
python -u omni_compare.py --ops 2,4,8,deblur,denoise \
  --single 2=r2_s2,4=sgl_sr4,8=r2_s8,deblur=sgl_deblur,denoise=sgl_denoise \
  --omni omni5,omni5sev --out compare_all5.json --device cpu \
  > ../logs/compare_all5.log 2>&1
echo "compare exit=$? at $(date +%H:%M:%S)"
tail -12 ../logs/compare_all5.log
