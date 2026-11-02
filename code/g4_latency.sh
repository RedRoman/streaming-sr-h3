#!/usr/bin/env bash
# G4: the end-to-end latency table, in the shipped configuration, warm.
#
# Needed to settle the framing. The quality table now leads or ties every baseline, but our sampling run
# logs 186.9 s for 124 frames (1.51 s/frame) against FlashVSR+'s measured 58.2 s (0.470 s/frame) on the
# same card and the same clips -- and FlashVSR+ is 1.4B against our 20.1B. A paper framed as streaming
# cannot ship 1.51 s/frame; one framed as a backbone-agnostic range/null stage can, provided the number
# is reported honestly and the stage's own share is separated from the backbone's.
#
# Two measurement rules this project has already been burned by:
#   * the block timer starts inside the block loop, so R_phi -- computed once per clip before it -- was
#     excluded from every latency figure reported so far. bench_e2e_wall times the whole per-clip path.
#   * first call on Ascend includes operator compilation and can be 100x steady state, so every phase is
#     warmed before it is timed.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=7
export RPHI_CKPT=results/range_branch/r3_big24.pt
source env.sh
cd code
echo "=== phase breakdown (one block, warm)"
python -u bench_fixed_cost.py > ../logs/g4_fixed.log 2>&1
tail -12 ../logs/g4_fixed.log
echo
echo "=== end-to-end per clip, R_phi on CPU vs NPU"
python -u bench_e2e_wall.py --steps 4 > ../logs/g4_e2e.log 2>&1
tail -14 ../logs/g4_e2e.log
echo
echo "=== FlashVSR+ on the same clips, for reference"
python3 - <<'PY'
import json, glob, statistics
s = [json.load(open(f))['seconds'] for f in glob.glob(
     '/cache/zihaohe/streaming_sr0/results/baseline_flashvsr_plus/*/result.json')]
print(json.dumps({'clips': len(s), 'mean_s_per_clip': round(statistics.mean(s), 1),
                  's_per_frame': round(statistics.mean(s) / 124, 3),
                  'x_realtime_at_24fps': round(statistics.mean(s) / 124 * 24, 1)}))
PY
