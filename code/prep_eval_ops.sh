#!/usr/bin/env bash
# Pixel-level held-out evaluation sets for the operators other than x4.
#
# This is the actual blocker for an omni deliverable, and neither of us had named it: every pixel-level
# number in this project -- the main table, the ablation, the baselines, every deploy run -- is on a
# single operator, x4 realistic. omni's whole claim is "one model over the operator family", and that
# claim has never been evaluated at the output, only on R_phi's intermediate estimate. There is no
# evaluation set to do it on.
#
# The same 96 clips as data/real_eval/heldout, reproduced exactly: prepare_real_eval picks
# candidates_big.selected minus exclude_eval24 and takes the first 96, and seeds the degradation with
# SEED + i, so passing the same candidate file, exclusion list and --keep gives the same clips in the
# same order with the same degradation draws -- only the scale changes. The HR frames are byte-identical
# by construction, which is a free check that the clip set really is the same.
#
# CPU only (video decode plus PNG writes); cards 3 and 7 are the seed-noise run and the rest belong to
# other tenants.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
# env.sh ends with ${ASCEND_RT_VISIBLE_DEVICES:?...}, and an unset :? parameter expansion makes the
# sourcing shell exit outright -- `|| fallback` cannot catch that, which is why the first attempt died
# before launching a single worker and left no logs at all. This work is CPU-only, so the value is
# arbitrary; it just has to be set.
export ASCEND_RT_VISIBLE_DEVICES=0
source env.sh
cd code
for S in 2 8; do
  for i in $(seq 0 7); do
    SR_SCALE=$S python prepare_real_eval.py --cand candidates_big.json \
      --exclude ../data/real_eval/exclude_eval24.txt --keep 96 --out-name heldout_s$S \
      --shard $i --nshards 8 > ../logs/prep_ho_s${S}_$i.log 2>&1 &
  done
  wait
  echo "heldout_s$S: $(ls -d ../data/real_eval/heldout_s$S/*/ 2>/dev/null | wc -l) clips at $(date +%H:%M:%S)"
done
python3 - <<'PY'
import json
from pathlib import Path
R = Path('/cache/zihaohe/streaming_sr0/data/real_eval')
base = json.loads((R / 'heldout/manifest.json').read_text())['clips']
for s in (2, 8):
    d = R / f'heldout_s{s}'
    man = d / 'manifest.json'
    clips = sorted(p.name for p in d.iterdir() if p.is_dir())
    same = set(clips) == set(base)
    # the HR frames must be byte-identical to the x4 set, or the clip windows differ and nothing is comparable
    hr_ok = None
    if clips:
        a = json.loads((R / 'heldout' / clips[0] / 'meta.json').read_text())['frame_records']['0']['hr_sha256']
        b = json.loads((d / clips[0] / 'meta.json').read_text())['frame_records']['0']['hr_sha256']
        hr_ok = (a == b)
    print(json.dumps({'set': f'heldout_s{s}', 'clips': len(clips), 'same_clip_set': same,
                      'hr_byte_identical': hr_ok, 'manifest_present': man.exists()}))
PY
