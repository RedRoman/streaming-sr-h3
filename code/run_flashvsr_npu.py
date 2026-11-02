"""Run the FlashVSR baseline under torch_npu.

Same arrangement as run_infvsr_npu: the repo is CUDA-written, so torch_npu's transfer_to_npu shim is
imported before the repo is, and baseline_flashvsr.py stays a byte-for-byte copy of the file that was
used on 30794. That matters here more than for InfVSR, because the FlashVSR numbers quoted in this
project so far came from a single clip and were never reproduced; the point of this run is to replace
them with a full-set measurement under the same protocol as everything else.

Its CUDA block-sparse kernel is replaced by code/block_sparse_attn.py, a masked-SDPA shim whose self-test
asserts equivalence to dense attention under an all-true mask. Quality is therefore comparable; latency is
NOT, and must not be reported from this run.

Selection is by environment, as the baseline expects: EVAL_SET, CLIPS, SHARD, NSHARDS, FLASHVSR_WEIGHTS.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path('/cache/zihaohe/streaming_sr0/code')))

import torch  # noqa: F401
import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401

import baseline_flashvsr

if __name__ == '__main__':
    baseline_flashvsr.main()
