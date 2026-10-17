#!/bin/bash
cd /cache/zihaohe/streaming_sr0 && venv/bin/python tools/rebuild_te.py 2>&1 | grep -v -i warn > logs/rebuild_te.log; echo REBUILD_EXIT >> logs/rebuild_te.log
