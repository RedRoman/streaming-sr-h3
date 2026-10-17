#!/bin/bash
# usage: run_step0.sh <npu_id> <shard> <nshards>
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=$1
source ../env.sh
OMP_NUM_THREADS=16 python -u run_blocks.py --shard $2 --nshards $3 2>&1 | grep -v -i -E 'warn|clip_x' >> ../logs/step0_shard$2.log
echo SHARD_FINISHED >> ../logs/step0_shard$2.log
