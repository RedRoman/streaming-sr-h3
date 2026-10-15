"""Exact pretrained native MiniMax-H3 visual VAE, independently loaded."""
import hashlib,sys
from contextlib import contextmanager
from pathlib import Path
import torch

CANONICAL_SHA='7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522'


def load_vae(weights,comfy,device='cuda',dtype=torch.float16,verify_hash=True):
    weights=Path(weights)
    if not weights.is_file():raise FileNotFoundError('Complete verified H3 VAE required: '+str(weights))
    if verify_hash:
        with weights.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=CANONICAL_SHA:raise ValueError('H3 VAE canonical SHA256 mismatch')
    sys.path.insert(0,str(Path(comfy).resolve()))
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    from safetensors.torch import load_file
    vae=MiniMaxH3VideoVAE(num_layers=36,operations=comfy.ops.disable_weight_init)
    vae.load_state_dict(load_file(str(weights),device='cpu'),strict=True,assign=True)
    return vae.to(device=device,dtype=dtype).eval().requires_grad_(False)


def encode_repeated(vae,rgb,context=22):
    # Temporal copies condition the native VAE only. They are not additional
    # physical evidence packets and never enter model packet accounting.
    x=(rgb*2-1).unsqueeze(2).repeat(1,1,context,1,1)
    return vae.encode(x.to(next(vae.parameters()).dtype).clone()).float()


def decode_last(vae,z):
    y=vae.decode(z.to(next(vae.decoder.parameters()).dtype))
    return y[:,:,-1].to(device=z.device,dtype=torch.float32)


@contextmanager
def differentiable_decoder_math():
    # Same audited scoped upstream eager route as peer h3_core: official
    # inference fused RoPE currently has no autograd registration. No file or
    # weight modification, and ordinary RGB evaluation uses native math again.
    import comfy.model_management
    import comfy.quant_ops
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    old_training=comfy.model_management.in_training;old_rope=comfy.quant_ops.ck.rms_rope_split_half
    try:
        comfy.model_management.in_training=True
        comfy.quant_ops.ck.rms_rope_split_half=rms_rope_split_half
        yield
    finally:
        comfy.model_management.in_training=old_training
        comfy.quant_ops.ck.rms_rope_split_half=old_rope
