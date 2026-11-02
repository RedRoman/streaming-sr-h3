#!/usr/bin/env bash
# The joint audio-video result on every eligible held-out clip, not the 12 that established it.
#
# The 12-clip run showed audio mel 29.262 -> 8.344 dB with video PSNR bit-identical on 12/12 clips.
# Both halves need the full set: the audio gain to be a number worth reporting, and the video
# invariance because "the DiT was not disturbed" is the whole joint claim and 12 clips is a weak basis
# for an exact-equality statement.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
for S in 0 1 2 3 4 5 6 7; do
  ( export ASCEND_RT_VISIBLE_DEVICES=$S
    source env.sh
    cd code
    python audio_joint.py --arms A,C --clips 0 --shard $S --nshards 8 \
      --out "results/audio_joint_full_s${S}.json" > "../logs/audio_full_s${S}.log" 2>&1
    echo "audio_full s${S} card${S} exit=$? $(date '+%F %T')" >> ../logs/audio_joint_full.status ) &
done
wait
echo "ALL DONE $(date '+%F %T')" >> ../logs/audio_joint_full.status
