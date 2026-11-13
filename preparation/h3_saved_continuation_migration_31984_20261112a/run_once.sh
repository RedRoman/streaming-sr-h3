#!/bin/sh
set -eu
exec timeout --signal=KILL 600s env TORCH_DEVICE_BACKEND_AUTOLOAD=0 ASCEND_RT_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -u /cache/zihaohe/omni_openh3_20261111/preparation/h3_saved_continuation_migration_31984_20261112a/pack_missing.py --config /cache/zihaohe/omni_openh3_20261111/preparation/h3_saved_continuation_migration_31984_20261112a/config.json --config-sha256 ac561f1ab13ede68ab97e885b4b2e8897e1c58411e2bd429e24e614963d19cb4
