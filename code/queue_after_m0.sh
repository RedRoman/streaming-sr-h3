#!/usr/bin/env bash
# Wait for the m0 mapper-feasibility apply pass to free the cards, then run RealVSR.
set -u
while pgrep -f "m0_latent_map.py apply" >/dev/null; do sleep 60; done
echo "m0 apply finished $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/queue.status
bash /cache/zihaohe/streaming_sr0/code/run_realvsr.sh 0,1,2,3,4,5
echo "realvsr finished $(date '+%F %T')" >> /cache/zihaohe/streaming_sr0/logs/queue.status
