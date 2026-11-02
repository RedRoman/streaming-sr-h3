"""A text embedding for the borrowed SR-DiT, since a deployed stream has no caption.

Their handoff is explicit that an empty prompt is not a substitute for the clip's own PE prompt, and the
first run used a zero embedding, so the quality it produced is a lower bound of unknown tightness. This
encodes one neutral caption with the official UMT5-XXL and saves it, which is the honest deployment
stand-in: a fixed generic embedding available without captioning the stream. Comparing it against zeros on
the same clips measures how much the text conditioning is actually worth here.
"""
import argparse, sys
from pathlib import Path

PKG = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/ascend_eval')
WEIGHTS = Path('/temp/liusonghua/h3_wan_full_pipeline_20261022/weights')
sys.path.insert(0, str(PKG))

import torch
import torch_npu  # noqa: F401


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', default='a high quality, detailed, sharp real-world video')
    ap.add_argument('--out', default='/cache/zihaohe/streaming_sr0/results/wan_route/prompt_generic.pt')
    a = ap.parse_args()
    import common  # noqa: F401  (installs the NPU runtime patches)
    common.MODELS = WEIGHTS
    dev = torch.device('npu:0'); torch.npu.set_device(dev)
    # t5.py evaluates torch.cuda.current_device() as a default argument, i.e. at import time, and this
    # build of torch has no CUDA. The call only ever supplies a default that we override anyway.
    torch.cuda.current_device = lambda: 0
    from wan.modules.t5 import T5EncoderModel
    enc = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=dev,
                         checkpoint_path=str(WEIGHTS / 'Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth'),
                         tokenizer_path=str(WEIGHTS / 'Wan2.1-T2V-1.3B/google/umt5-xxl'))
    with torch.no_grad():
        out = enc([a.text], dev)
    e = out[0] if isinstance(out, (list, tuple)) else out
    e = e if e.dim() == 3 else e[None]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(e.float().cpu(), a.out)
    print({'text': a.text, 'shape': list(e.shape), 'out': a.out})


if __name__ == '__main__':
    main()
