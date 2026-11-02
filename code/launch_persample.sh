#!/usr/bin/env bash
# Wait for the deploy to release cards, then train the per-sample-MTF arm on four seeds.
# Four, not one: the deblur delta this is trying to move is -0.144 dB, and the seed spread on these arms
# is the only thing that says whether that is a result or noise. A 12000-iter run measured 12.6 min.
set -u
cd "$(dirname "$0")"
free_cards () {
  npu-smi info 2>/dev/null | awk '
    /^\| [0-7] +910B3/ {card=$2}
    /HBM-Usage|\/ 65536/ {if (match($0, /([0-9]+) \/ 65536/, m) && m[1] < 8000 && card != "") {print card; card=""}}'
}
for i in $(seq 1 180); do
  n=$(free_cards | wc -l)
  if [ "$n" -ge 4 ]; then
    echo "$(date '+%F %T') $n cards free, launching" >> ../logs/persample.status
    break
  fi
  sleep 60
done
CARDS=($(free_cards | head -4))
SEEDS=(0 1 2 3)
for i in 0 1 2 3; do
  c=${CARDS[$i]:-$i}
  ( ./deblur_persample.sh "$c" "omni_mtf_ps_s${SEEDS[$i]}" "${SEEDS[$i]}" \
      > "../logs/persample_s${SEEDS[$i]}.log" 2>&1
    echo "ps_s${SEEDS[$i]} card$c exit=$? $(date '+%F %T')" >> ../logs/persample.status ) &
done
wait
echo "ALL DONE $(date '+%F %T')" >> ../logs/persample.status
