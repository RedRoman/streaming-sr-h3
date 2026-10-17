#!/bin/bash
# usage: run_step1.sh <npu_id> <shard> <nshards> [extra args for run_refs.py]
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=$1
S=$2; N=$3; shift 3
source ../env.sh
OMP_NUM_THREADS=16 python -u run_refs.py --shard $S --nshards $N "$@" 2>&1 | grep -v -i -E 'warn|clip_x' >> ../logs/step1_shard$S.log
echo SHARD_FINISHED >> ../logs/step1_shard$S.log
