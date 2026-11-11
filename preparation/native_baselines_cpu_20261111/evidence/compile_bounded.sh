#!/bin/bash
set -u
cd /cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_cpu_20261111_v1 || exit 1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 MMCV_WITH_OPS=1 FORCE_CUDA=0 FORCE_NPU=0 FORCE_MLU=0 FORCE_MPS=0 MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$PWD/pip_cache" TMPDIR="$PWD/tmp" TORCH_EXTENSIONS_DIR="$PWD/torch_extensions" PATH="$PWD/venv/bin:$PATH"
date -Is > compile.started
timeout --signal=TERM --kill-after=10s 280s python -m pip install --disable-pip-version-check --no-build-isolation --no-deps -v ./mmcv-3a5fb4b96ab1b241e7379adf7c326ef6edef6515 > compile.log 2>&1
result=$?
printf '%s\n' "$result" > compile.exitcode
date -Is > compile.finished
exit "$result"
