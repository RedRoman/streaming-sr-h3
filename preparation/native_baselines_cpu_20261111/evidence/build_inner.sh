#!/bin/bash
set -euo pipefail
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 MMCV_WITH_OPS=1 FORCE_CUDA=0 FORCE_NPU=0 FORCE_MLU=0 FORCE_MPS=0 MAX_JOBS=2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PIP_CACHE_DIR="$PWD/pip_cache" TMPDIR="$PWD/tmp" TORCH_EXTENSIONS_DIR="$PWD/torch_extensions"
/cache/zihaohe/omni_openh3_20261111/venv/bin/python -m venv --system-site-packages "$PWD/venv"
export PATH="$PWD/venv/bin:$PATH"
python -c 'import sys,torch; print("ISOLATED_PREFIX",sys.prefix,"TORCH",torch.__version__,torch.__file__,"NPU_IMPORTED", "torch_npu" in sys.modules,flush=True)'
python -m pip install --disable-pip-version-check mmengine==0.10.7 addict==2.4.0 yapf==0.43.0
python - <<'INNER'
import urllib.request,tarfile,hashlib,pathlib
url='https://github.com/open-mmlab/mmcv/archive/3a5fb4b96ab1b241e7379adf7c326ef6edef6515.tar.gz'
p=pathlib.Path('mmcv-official.tar.gz'); urllib.request.urlretrieve(url,p)
print('MMCV_ARCHIVE_SHA256',hashlib.sha256(p.read_bytes()).hexdigest(),flush=True)
with tarfile.open(p) as t: t.extractall('.',filter='data')
INNER
python -m pip install --disable-pip-version-check --no-build-isolation --no-deps -v ./mmcv-3a5fb4b96ab1b241e7379adf7c326ef6edef6515
python -c 'import mmcv, mmcv.ops; print("MMCV_OPS_READY",mmcv.__version__,flush=True)'
