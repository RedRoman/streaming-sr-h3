#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=${1:-3}
source ../env.sh
python npu_parity.py --out ../results/npu_parity_bear256 2>&1 | grep -v -i -E 'warn|clip_x' | tee ../logs/parity_full.log
