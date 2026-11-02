#!/usr/bin/env bash
# Four-way held-out comparison: add FlashVSR+ to the existing InfVSR / FlashVSR / ours table.
#
# Waits for the FlashVSR+ sweep and refuses to score a partial arm -- a baseline evaluated on a subset of
# the clips the others were evaluated on is not a comparison, and the 24-clip subsets already differed from
# the full set by 0.06-0.08 dB, the same size as the effects being measured.
#
# FlashVSR and FlashVSR+ are both marked approximate in the table: neither CUDA kernel exists on Ascend.
# FlashVSR's masked-SDPA substitute drops 4.08% of the mask; FlashVSR+'s is their own exact expansion
# (verified bitwise against repeat_interleave), so only FlashVSR is numerically approximate -- but both
# are off their intended kernel and neither latency figure is usable.
set -u
while pgrep -f 'fvp_entry.py' > /dev/null; do sleep 60; done
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=0
export EVAL_SET=data/real_eval/heldout
source env.sh
cd code

need=$(python3 -c "import json;print(len(json.load(open('../data/real_eval/heldout/manifest.json'))['clips']))")
have=$(python3 - <<'PY'
import json
from pathlib import Path
R = Path('/cache/zihaohe/streaming_sr0')
clips = json.loads((R / 'data/real_eval/heldout/manifest.json').read_text())['clips']
print(sum((R / 'results/baseline_flashvsr_plus' / c / 'result.json').exists() for c in clips))
PY
)
echo "FlashVSR+ held-out clips with result.json: $have / $need"
test "$have" -eq "$need" || { echo "incomplete, not scoring" >&2; exit 1; }

python -u deploy_eval.py --procs 12 eval --name judge_ho_4way \
  --arm infvsr=../results/baseline_infvsr \
  --arm flashvsr=../results/baseline_flashvsr \
  --arm flashvsr_plus=../results/baseline_flashvsr_plus \
  --arm ours=../results/deploy/t4r2_ho_final \
  --arm bicubic=bicubic \
  --pair ours,flashvsr_plus --pair ours,flashvsr --pair ours,infvsr \
  --pair flashvsr,flashvsr_plus \
  > ../logs/judge_ho_4way.log 2>&1
echo "judge exit=$? at $(date +%H:%M:%S)"
tail -2 ../logs/judge_ho_4way.log
