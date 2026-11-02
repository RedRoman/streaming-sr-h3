#!/bin/bash
# Complete the three x4/x6 conditioning arms from shard 1 of 4 to all 96 clips.
# The MTF-vs-one-hot comparison is only valid at matched data and capacity, and the x6 holdout is the
# arm that separates the two signals at all, so both need the full realistic set, not a quarter of it.
cd "$(dirname "$0")"
run () {
  local card=$1 tag=$2 rphi=$3 op=$4 shard=$5
  (
    export ASCEND_RT_VISIBLE_DEVICES=$card
    source ../env.sh
    python run_deploy.py --tag "$tag" --dc rphi --guide y --rphi "../results/omni/$rphi" \
      --op "$op" --turbo-steps 2 --merge-lora --shard "$shard" --nshards 4 \
      > "../logs/fill_${tag}_s${shard}.log" 2>&1
    echo "${tag} shard${shard} card${card} exit=$? $(date '+%F %T')" >> ../logs/fill_x6_full.status
  ) &
}
# round 1: seven cards
run 0 t2_mtf_s6    omni_mtf_k3.pt    6 0
run 1 t2_mtf_s6    omni_mtf_k3.pt    6 2
run 2 t2_mtf_s6    omni_mtf_k3.pt    6 3
run 3 t2_mtf_x4    omni_mtf_k3.pt    4 0
run 4 t2_mtf_x4    omni_mtf_k3.pt    4 2
run 5 t2_onehot_x4 omni_onehot_k3.pt 4 0
run 6 t2_onehot_x4 omni_onehot_k3.pt 4 2
wait
# round 2: the remaining two
run 0 t2_mtf_x4    omni_mtf_k3.pt    4 3
run 1 t2_onehot_x4 omni_onehot_k3.pt 4 3
wait
echo "ALL DONE $(date '+%F %T')" >> ../logs/fill_x6_full.status
