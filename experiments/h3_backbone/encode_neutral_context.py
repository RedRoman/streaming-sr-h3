"""Cache real native H3 text-only conditioning in an isolated process."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import numpy as np
from PIL import Image

PROMPT = "Restore the observed video faithfully. Preserve the visible scene geometry, motion, colors, and appearance. Recover consistent fine details supported by the observed frames."


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--encoder", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--observed-lr-image", type=Path, help="Optional already observed LR-upsample PNG, never HR semantic context")
    args = p.parse_args()
    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram = 7.0
    import comfy.sd
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.55)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    clip = comfy.sd.load_clip([str(args.encoder)], clip_type=comfy.sd.CLIPType.MINIMAX,
                             model_options={"dtype": torch.bfloat16})
    images=[]
    if args.observed_lr_image is not None:
        images=[torch.from_numpy(np.array(Image.open(args.observed_lr_image).convert('RGB'))).float()[None]/255]
    with torch.no_grad():
        cond = clip.encode_from_tokens_scheduled(clip.tokenize(PROMPT,images=images))
    context = cond[0][0][0].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
    tags = cond[0][1]["minimax_token_tags"].detach().long().cpu().contiguous()
    assert context.ndim == 2 and context.shape[-1] == 5120
    assert tags.shape == context.shape[:1] and bool(((tags == 1)|(tags == 0)).all())
    if images:assert bool((tags==0).any())
    else:assert bool((tags==1).all())
    assert torch.isfinite(context).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"prompt": PROMPT, "prompt_embeds": context, "text_token_tags": tags}, args.output)
    provenance = {"status": "complete", "prompt": PROMPT,
                  "protocol": "Native Comfy MiniMaxH3Tokenizer raw text, no images/chat/special tokens; unnormalized Qwen3VL layer50, TEXT_TAG=1",
                  "text_encoder_precision": "existing Qwen3VL32B NVFP4-AWQ storage, BF16 compute; independently label DiT precision",
                  "encoder_path": str(args.encoder), "encoder_bytes": args.encoder.stat().st_size,
                  "context_shape": list(context.shape), "tags_shape": list(tags.shape),
                  "visual_context": bool(images), "vision_tag_count": int((tags==0).sum()),
                  "observed_lr_image": str(args.observed_lr_image) if images else None,
                  "observed_lr_image_sha256": hashlib.sha256(args.observed_lr_image.read_bytes()).hexdigest() if images else None,
                  "context_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  "uses_future_gt_semantics": False, "memory_fraction": .55,
                  "seconds": time.perf_counter() - started,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    if images:
        provenance['protocol']='Actual MiniMaxH3Tokenizer raw neutral prompt + observedLR firstimage vision block; unnormalizedQwen3VL layer50, visiontag0/texttag1; no HR vision'
    args.output.with_suffix(".json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance), flush=True)


if __name__ == "__main__":
    main()
