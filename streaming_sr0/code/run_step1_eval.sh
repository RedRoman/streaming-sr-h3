#!/bin/bash
cd /cache/zihaohe/streaming_sr0/code
export OMP_NUM_THREADS=32 TORCH_HOME=/cache/zihaohe/streaming_sr0/torch_home
/cache/zihaohe/streaming_sr0/venv/bin/python -u ref_metrics.py 2>&1 | grep -v -i warn > ../logs/step1_metrics.log
/cache/zihaohe/streaming_sr0/venv/bin/python -u ref_panels.py 2>&1 | grep -v -i warn > ../logs/step1_panels.log
echo EVAL_DONE >> ../logs/step1_metrics.log
