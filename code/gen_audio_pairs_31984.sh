#!/usr/bin/env bash
# Audio latent pairs on this host, so the one experiment that makes the joint audio-video claim true can
# run where the cards are. The pairs on the other host cannot be copied (78 MB of npz, no host-to-host
# route, /temp refuses writes) but they can be REGENERATED: the source pool is on /temp, which is
# readable here, the audio VAE was found at /temp/zihaohe_omni and copied in with a matching sha256
# (8e505d95...), and the per-window seed is derived from the window name, so the pairs are reproducible
# rather than merely similar.
# Four shards, not eight: cards 4-7 are running the mtf x4 deploy and have about 13 GB free each. The
# audio VAE is 600 MB and the claim allocation is 1 GB, so four fit and eight would not.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
for sp in val train; do
  if [ "$sp" = train ]; then n=2182; w=2; else n=220; w=1; fi
  for i in $(seq 0 3); do
    CARD=$((4 + i))
    (export ASCEND_RT_VISIBLE_DEVICES=$CARD
     source env.sh
     cd code
     python -u prepare_audio_pairs.py --split "$sp" --clips "$n" --windows "$w" \
       --shard "$i" --nshards 4 > "../logs/ap84_${sp}_$i.log" 2>&1) &
  done
  wait
  echo "audio pairs $sp: $(ls /cache/zihaohe/audio_pairs/$sp 2>/dev/null | wc -l) at $(date '+%F %T')"
done
echo "AUDIO PAIRS DONE $(date '+%F %T')"
