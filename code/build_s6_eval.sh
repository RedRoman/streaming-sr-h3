#!/usr/bin/env bash
# A x6 held-out evaluation set on the same 96 clips as heldout / heldout_s2 / heldout_s8.
#
# This is the missing half of the strongest form of the conditioning claim. Today's x6 result is at the
# ESTIMATE level only: an MTF-conditioned branch trained on {2,4,8,deblur,denoise} gains +1.794 dB on x6
# pairs it never saw, while a one-hot cannot be evaluated there at all. The project has already measured
# that estimate and delivered quality are not monotone, so that result is not yet a claim about output.
# Running it end to end needs an x6 eval set, and none existed.
#
# Reproduced exactly as heldout was: candidates_big.json, --exclude the 24 const tuning clips, --keep 96,
# seed 20261019, variant const. Verified before launching that this selection reproduces heldout's clip
# set. CPU only -- both hosts' NPUs are currently running colleagues' jobs, and this needs none.
#
# Two geometry facts recorded rather than discovered later:
#   832/6 = 138.667, so the LR is 138x80 and the operator is slightly anisotropic (6.029 horizontal
#   against 6.000 vertical). range_pairs_s6big was generated the same way, so the eval set and the pair
#   set share one operator, which is what matters for the comparison.
#   operator_desc.operator_mtf(6) probes at round(832/6)=139 wide, one pixel off from the data's 138.
#   The knee sits at 1/6 either way; noted because it is a real inconsistency, not because it is large.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export PATH=/cache/zihaohe/streaming_sr0/venv/bin:$PATH
for i in $(seq 0 7); do
  SR_SCALE=6 python -u code/prepare_real_eval.py \
    --cand candidates_big.json --exclude data/real_av/exclude_const24.txt \
    --keep 96 --out-name heldout_s6 --variant const --shard "$i" --nshards 8 \
    > logs/s6eval_$i.log 2>&1 &
done
wait
n=$(ls -d data/real_eval/heldout_s6/*/ 2>/dev/null | wc -l)
echo "heldout_s6: $n clip dirs at $(date '+%F %T')"
python - <<'PY'
import json
from pathlib import Path
R=Path('/cache/zihaohe/streaming_sr0')
a=set(json.loads((R/'data/real_eval/heldout/manifest.json').read_text())['clips'])
m=R/'data/real_eval/heldout_s6/manifest.json'
if m.exists():
    b=set(json.loads(m.read_text())['clips'])
    print(json.dumps({'s6_clips':len(b),'heldout_clips':len(a),'same_clip_set':a==b,
                      'missing':sorted(a-b)[:5]}))
    one=sorted(b)[0]
    meta=json.loads((R/'data/real_eval/heldout_s6'/one/'meta.json').read_text())
    print(json.dumps({'clip':one,'scale':meta['scale'],'hr_wh':meta['hr_wh'],
                      'degradation_kind':meta['degradation'].get('kind')}))
else: print('{"error":"no manifest written"}')
PY
echo "S6 EVAL DONE $(date '+%F %T')"
