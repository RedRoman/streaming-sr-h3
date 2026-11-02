"""Fold the Stage-1 LoRA into the base weights for inference.

Measured on this device at 10920 tokens: the four linears of one block take 35.7 ms dense, 48.0 ms with the
LoRA branch as `lora.py` computes it, i.e. 0.61 s per 50-block step, 26% of the step. Rank 32 adds 0.75% of
the parameters, so the cost is not arithmetic -- it is that the branch runs in float32 (`x.to(torch.float32)`)
while the rest of the model is bfloat16, and float32 matmul has no tensor-core path here.

Nothing about the trained adapter requires it to stay separate at inference time:

    y = W x + b + s * (B (A x))  =  (W + s * B A) x + b

so folding it is algebraically exact. What is not exact is the rounding: the fold happens in float32 and the
result is cast back to the base dtype, so the merged weight differs from the pair by one rounding step. That
is checked here rather than assumed -- `verify` compares merged against unmerged outputs on random input and
reports the relative error, and `main` prints it for one layer of each shape before any model is touched.
"""
import json

import torch
from torch import nn

import lora as L


@torch.no_grad()
def merge_(model, strength=1.0, verify_samples=0):
    """Replace every LoRALinear with its base nn.Linear, weights updated in place. Returns a record."""
    merged, rel_errs, skipped = 0, [], 0
    # same two trees inject() walks: the 50 DiT blocks and, when present, the 2 token_refiner blocks.
    # Merging only model.blocks would leave the token_refiner adapters live, which is how the device
    # mismatch appeared in the first place.
    blocks = list(model.blocks)
    tr = getattr(model, 'token_refiner', None)
    if tr is not None and hasattr(tr, 'blocks'):
        blocks += list(tr.blocks)
    for i, block in enumerate(blocks):
        for t in L.TARGETS:
            parent_name, attr = t.split('.')
            parent = getattr(block, parent_name)
            mod = getattr(parent, attr)
            if not isinstance(mod, L.LoRALinear):
                skipped += 1
                continue
            base = mod.base
            delta = (mod.lora_b.float() @ mod.lora_a.float()) * (mod.scale * strength)
            assert delta.shape == base.weight.shape, (delta.shape, base.weight.shape)
            if verify_samples and len(rel_errs) < verify_samples:
                x = torch.randn(64, base.in_features, device=base.weight.device, dtype=base.weight.dtype)
                before = mod(x).float()
            base.weight.add_(delta.to(base.weight.dtype))
            setattr(parent, attr, base)
            if verify_samples and len(rel_errs) < verify_samples:
                after = base(x).float()
                rel_errs.append(float((after - before).abs().mean() / before.abs().mean().clamp_min(1e-8)))
            merged += 1
    return {'merged_modules': merged, 'skipped_not_lora': skipped, 'strength': strength,
            'verify_rel_err': rel_errs}


if __name__ == '__main__':
    import torch_npu  # noqa: F401
    dev = torch.device('npu', 0)
    torch.npu.set_compile_mode(jit_compile=False)
    out = {}
    for name, K, N in [('qkv_proj', 5376, 21504), ('out_proj', 7168, 5376),
                       ('fc1', 5376, 28672), ('fc2', 14336, 5376)]:
        base = nn.Linear(K, N).to(dev, torch.bfloat16).eval()
        mod = L.LoRALinear(base, 32).to(dev)
        with torch.no_grad():
            mod.lora_b.normal_(0, 0.02)          # a zero B would make the test vacuous
            x = torch.randn(256, K, device=dev, dtype=torch.bfloat16)
            before = mod(x).float()
            delta = (mod.lora_b.float() @ mod.lora_a.float()) * mod.scale
            base.weight.add_(delta.to(base.weight.dtype))
            after = base(x).float()
            out[name] = {'rel_err': float((after - before).abs().mean() / before.abs().mean()),
                         'max_abs': float((after - before).abs().max())}
        print(json.dumps({name: {k: round(v, 6) for k, v in out[name].items()}}), flush=True)
        del base, mod, x
        torch.npu.empty_cache()
