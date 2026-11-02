#!/usr/bin/env bash
# omni_m4 against the three big-data specialists it is now matched to, on identical validation data.
#
# Until today this comparison could not be made: omni_m4 saw 10351 windows for 24000 iters while the
# only specialists that existed for x2/deblur/denoise were trained on 598 windows for 6000 iters -- a
# confound in omni's favour, so those columns were not reportable. sp_s2 / sp_deblur / sp_denoise are
# now matched on data, schedule, width (4.7M), crop (112) and K (1, because OmniRange has no causal
# stack and charging omni for a feature it lacks would not be a fair test). x4's control is r5_big.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export OMNI_DATA_SUFFIX=big
source env.sh
cd code
python -u omni_compare.py --ops 2,4,deblur,denoise \
  --single 2=sp_s2,4=r5_big,deblur=sp_deblur,denoise=sp_denoise \
  --omni omni_m4 --out cmp_m4_matched.json --device npu:0 \
  > ../logs/cmp_m4_matched.log 2>&1
echo "exit=$? $(date +%H:%M:%S)"
grep -aviE 'owner|warnings.warn|UserWarning' ../logs/cmp_m4_matched.log | tail -12
