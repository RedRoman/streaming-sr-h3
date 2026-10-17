#!/bin/bash
# usage: run_encode.sh <npu_id> [--only seq ...]
cd /cache/zihaohe/streaming_sr0/code
export ASCEND_RT_VISIBLE_DEVICES=$1; shift
source ../env.sh
OMP_NUM_THREADS=16 python -u encode_contexts.py "$@" 2>&1 | grep -v -i -E 'warn' >> ../logs/encode_contexts.log
echo ENCODE_EXIT >> ../logs/encode_contexts.log
