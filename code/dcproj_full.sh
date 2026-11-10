#!/usr/bin/env bash
# The dc_project arm on all 96 held-out clips: at the innerDC point, hand back
# encode(project(decode(z0))) instead of z0 + [encode(project(decode(z0))) - encode(decode(z0))].
#
# Identical to the shipped t4_plain2 in every other respect. Two things change together and both are
# intended: z0 is replaced by its VAE-manifold projection (the default keeps z0 and adds only the
# observation correction), and one of the two encodes cancels algebraically, so sampling gets cheaper.
# Smoke-measured: 8.69 s -> 6.63 s per steady-state block, -24%.
set -u
cd /cache/zihaohe/streaming_sr0/code || exit 1
CARDS=${1:-4,5,7}
IFS=, read -ra CA <<< "$CARDS"
NSH=${#CA[@]}
i=0
for c in "${CA[@]}"; do
  (
    export ASCEND_RT_VISIBLE_DEVICES=$c
    export EVAL_SET=data/real_eval/heldout
    source ../env.sh
    python run_deploy.py --tag dcproj --dc rphi --guide y \
      --rphi ../results/range_branch/r5_k3.pt --turbo-steps 2 --merge-lora --dc-project \
      --shard "$i" --nshards "$NSH" > "../logs/dcproj_p${i}.log" 2>&1
    echo "dcproj shard${i} card${c} exit=$? $(date '+%F %T')" >> ../logs/dcproj.status
  ) &
  i=$((i+1))
done
wait
echo "dcproj all done $(date '+%F %T')" >> ../logs/dcproj.status
