#!/bin/bash
set -e
cd /cache/zihaohe/streaming_sr0
[ -x venv/bin/python ] || /cache/yunfeng/envs/minimax-h3-npu/bin/python -m venv --system-site-packages venv
venv/bin/python -c "import torch, torch_npu; print('base ok', torch.__version__, torch_npu.__version__)"
venv/bin/python -m pip install --no-cache-dir -i http://repo.myhuaweicloud.com/repository/pypi/simple \
  --trusted-host repo.myhuaweicloud.com \
  "comfy-kitchen==0.2.37" "comfy-aimdo==0.5.6" simpleeval blake3 alembic "SQLAlchemy>=2.0" filelock \
  sentencepiece spandrel "kornia>=0.7.1" "pydantic>=2,<3" "pydantic-settings>=2,<3" requests
venv/bin/python -m pip freeze > logs/venv_freeze.txt
echo VENV_DONE
