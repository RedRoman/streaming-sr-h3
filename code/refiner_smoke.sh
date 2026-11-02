#!/usr/bin/env bash
set -u
cd /cache/zihaohe/streaming_sr0 || exit 1
export ASCEND_RT_VISIBLE_DEVICES=4
source env.sh
cd code
python -u refiner_smoke.py --steps 4 > ../logs/refiner_smoke.log 2>&1
echo "smoke exit=$? $(date +%H:%M:%S)"
grep -aE 'refiner fires|block seconds|SystemExit|Error' ../logs/refiner_smoke.log | tail -5
