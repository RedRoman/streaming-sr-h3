#!/bin/bash
# usage: run_train.sh <cards e.g. 0,1,2,3> <out_dir> [extra args]
# Uses our venv's python explicitly: a bare `torchrun` resolves to the system anaconda install.
cd /cache/zihaohe/streaming_sr0/code
CARDS=$1; OUT=$2; shift 2
export ASCEND_RT_VISIBLE_DEVICES=$CARDS
source ../env.sh
N=$(echo $CARDS | tr ',' '\n' | wc -l)
export HCCL_CONNECT_TIMEOUT=1800
OMP_NUM_THREADS=8 /cache/zihaohe/streaming_sr0/venv/bin/python -m torch.distributed.run \
  --nproc-per-node=$N --master-port=${PORT:-29571} \
  train_stage1.py --out $OUT "$@" 2>&1 | grep -v -i -E 'warn|clip_x|cast_to_input'
