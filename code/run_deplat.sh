#!/usr/bin/env bash
# Deployed-path latency on one clip. Script, not an inline command: a grep pattern with a '|' in it got
# re-split by the background session wrapper and ran "warnings.warn" as a command.
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=7
source env.sh
cd code
python -u bench_deploy_latency.py --steps 4 > ../logs/deploy_latency.log 2>&1
echo "EXIT=$? $(date +%H:%M:%S)"
tail -14 ../logs/deploy_latency.log
