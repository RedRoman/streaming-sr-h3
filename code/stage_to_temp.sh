#!/usr/bin/env bash
# Stage what 30778 is missing onto the shared /temp mount.
#
# 31984 has every dataset and zero free NPUs -- a colleague's TP8 benchmark has held all eight cards for
# 23 hours. 30778 has eight idle cards (3.4 GB each, driver only), the 40 GB H3 DiT, both VAEs, the turbo
# delta, the venv and the code, but only the `const` and `tvary` eval sets and none of the new pair sets.
# So the compute is on one host and the data on the other, and /temp is the only thing both can see.
#
# /temp is SFS Turbo at 97% (765 GB free) and is NFSv3: rename is forbidden, so rsync needs
# --inplace --no-perms --no-owner --no-group, and thousands of small .npz files must travel as one tar
# or the per-file round trips dominate. Both rules are from earlier damage on this mount.
#
# Ordered by value per gigabyte: the checkpoints and Vid4 unlock evaluations immediately, the held-out
# eval sets unlock every deploy run, and the pair sets unlock training.
set -u
S=/cache/zihaohe/streaming_sr0
T=/temp/zihaohe_xfer
mkdir -p "$T"
df -h /temp | tail -1

stage () {   # name, tar -C dir, paths...
  local name=$1; shift; local base=$1; shift
  if [ -f "$T/$name.done" ]; then echo "   $name already staged"; return; fi
  echo "   packing $name at $(date +%H:%M:%S)"
  tar cf "$T/$name.tar" -C "$base" "$@" 2>/dev/null
  sync; touch "$T/$name.done"
  echo "   $name $(du -h "$T/$name.tar" | cut -f1) done at $(date +%H:%M:%S)"
}

echo "=== 1. code + checkpoints + Vid4 (small, unlocks evaluation)"
stage code "$S" code
stage ckpts "$S" results/range_branch results/omni results/stage1_lora
stage vid4 /cache/zihaohe benchmarks/vid4

echo "=== 2. held-out eval sets (unlocks every deploy run)"
stage eval_heldout "$S" data/real_eval/heldout data/real_eval/heldout_s2 data/real_eval/heldout_s8

echo "=== 3. pair sets (unlocks training), largest last"
stage pairs_bdfit /cache/zihaohe range_pairs_bdfit
stage pairs_s6 /cache/zihaohe range_pairs_s6big
stage pairs_x4 /cache/zihaohe range_pairs_x4big

ls -la "$T"/*.tar 2>/dev/null
df -h /temp | tail -1
echo STAGE_DONE
