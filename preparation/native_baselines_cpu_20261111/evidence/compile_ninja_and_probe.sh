#!/bin/bash
set -u
cd /cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_cpu_20261111_v1 || exit 1
for check in $(seq 1 75); do
  test -f compile.exitcode && break
  sleep 1
done
test -f compile.exitcode || { printf 'previous_compile_still_running\n' > resume.status; exit 1; }
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 MMCV_WITH_OPS=1 FORCE_CUDA=0 FORCE_NPU=0 FORCE_MLU=0 FORCE_MPS=0 MAX_JOBS=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$PWD/pip_cache" TMPDIR="$PWD/tmp" TORCH_EXTENSIONS_DIR="$PWD/torch_extensions" PATH="$PWD/venv/bin:$PATH"
printf 'compiling\n' > resume.status
date -Is > compile_ninja.started
timeout --signal=TERM --kill-after=10s 600s python -m pip install --disable-pip-version-check --no-build-isolation --no-deps -v ./mmcv-3a5fb4b96ab1b241e7379adf7c326ef6edef6515 > compile_ninja.log 2>&1
result=$?
printf '%s\n' "$result" > compile_ninja.exitcode
date -Is > compile_ninja.finished
if [ "$result" -ne 0 ]; then printf 'compile_failed_or_timed_out\n' > resume.status; exit "$result"; fi
printf 'probing\n' > resume.status
for baseline in faver aver; do
 timeout --signal=TERM --kill-after=10s 120s python verify_official_cpu.py --assets /cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_20261111/third_party/native_video_baselines_20261111 --baseline "$baseline" --output "$PWD/$baseline" > "$baseline.log" 2>&1
 result=$?
 printf '%s\n' "$result" > "$baseline.exitcode"
done
printf 'probes_finished_read_individual_receipts\n' > resume.status
