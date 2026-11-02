"""Launch FlashVSR+'s own run_stream under torch_npu.

run_stream.py calls torch.cuda.is_available() and aborts with "CUDA is not available" before touching a
model. transfer_to_npu has to be imported in the SAME interpreter for the mapping to take effect, and the
baseline driver starts run_stream as a subprocess, so the shim never reached it. This wrapper imports the
shim first and then hands control to their module unchanged, which keeps the arm the published code.

The two explicit overrides are for the probes transfer_to_npu does not redirect: an availability check and
a device index. Neither changes any computation.
"""
import runpy
import sys
from pathlib import Path

# the package is not installed; make the repo importable the way `pip install -e .` would
REPO = Path('/cache/zihaohe/baselines/FlashVSR_Plus')
sys.path.insert(0, str(REPO))

import torch
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

torch.cuda.is_available = lambda: True
torch.cuda.current_device = lambda: 0
if not hasattr(torch.cuda, 'get_device_capability'):
    torch.cuda.get_device_capability = lambda *a, **k: (8, 0)


# Report whether the mask cache actually fired. A patch that is installed but never reached looks exactly
# like one that works, so the counts are printed unconditionally at exit, including on a crash.
import atexit as _atexit
import json as _json
import time as _time

_T0 = _time.perf_counter()


def _report():
    try:
        from models.attention_backends.kernels import mask_cache_stats
        st = mask_cache_stats()
    except Exception as exc:                                  # noqa: BLE001
        st = {'unavailable': repr(exc)}
    st['wall_s'] = round(_time.perf_counter() - _T0, 1)
    print('MASK_CACHE ' + _json.dumps(st, default=float), flush=True)


_atexit.register(_report)

if __name__ == '__main__':
    sys.argv[0] = 'run_stream'
    runpy.run_module('inference.run_stream', run_name='__main__')
