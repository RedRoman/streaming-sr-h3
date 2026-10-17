#!/bin/bash
cd /cache/zihaohe/streaming_sr0 && venv/bin/python tools/refetch_check.py 2>&1 | grep -v -i warn > logs/refetch_check.log
