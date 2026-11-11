#!/bin/bash
cd /cache/zihaohe/omni_openh3_20261111/preparation/native_baselines_cpu_20261111_v1 || exit 1
date -Is > build.started
timeout --signal=TERM --kill-after=10s 280s bash build_inner.sh > build.log 2>&1
result=$?
printf '%s\n' "$result" > build.exitcode
date -Is > build.finished
exit "$result"
