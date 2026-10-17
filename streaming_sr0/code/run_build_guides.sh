#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
OMP_NUM_THREADS=8 /cache/zihaohe/streaming_sr0/venv/bin/python -u build_memory_guides.py "$@" 2>&1 | grep -v -i warn >> ../logs/step2_build.log
echo BUILD_EXIT >> ../logs/step2_build.log
