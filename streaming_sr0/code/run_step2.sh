#!/bin/bash
# usage: run_step2.sh <npu_id> <shard> <nshards> [extra args for run_memory.py]
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=$1
S=$2; N=$3; shift 3
source ../env.sh
OMP_NUM_THREADS=16 python -u run_memory.py --shard $S --nshards $N "$@" 2>&1 | grep -v -i -E 'warn|clip_x' >> ../logs/step2_shard$S.log
echo SHARD_FINISHED >> ../logs/step2_shard$S.log
