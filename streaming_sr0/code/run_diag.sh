#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=${1:-0}
source ../env.sh
OMP_NUM_THREADS=16 python -u diag_ref_path.py 2>&1 | grep -v -i -E 'warn|clip_x' > ../logs/step1_diag.log
echo DIAG_DONE >> ../logs/step1_diag.log
