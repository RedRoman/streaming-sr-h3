"""Run the InfVSR baseline under torch_npu.

The baseline script is the CUDA-written official pipeline; the only adaptation is importing torch_npu's
transfer_to_npu shim before the repo is imported, which is what the original 30794 invocation did. Kept
as a separate entry point so baseline_infvsr.py stays a faithful copy of the file that produced the
published 24-clip numbers -- the held-out set has to be scored by the same code.

Selection is by environment, as the baseline expects: EVAL_SET, CLIPS, SHARD, NSHARDS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/cache/zihaohe/streaming_sr0/code')))

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import baseline_infvsr

if __name__ == '__main__':
    baseline_infvsr.main()
