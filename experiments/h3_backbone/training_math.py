"""Scoped native H3 training math; official source files remain untouched."""
from contextlib import contextmanager
import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self,base,rank=4):
        super().__init__();self.base=base;self.rank=rank
        self.a=nn.Parameter(torch.randn(rank,base.in_features,device=base.weight.device,dtype=torch.float32)*.01)
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))

    def forward(self,x):
        out=self.base(x)
        delta=torch.nn.functional.linear(torch.nn.functional.linear(x.float(),self.a),self.b)
        return out+delta.to(out.dtype)


def inject_attention_lora(model,rank=4,block_indices=None):
    blocks=list(model.blocks) if hasattr(model,'blocks') else [model]
    names=[]
    for i,block in enumerate(blocks):
        if block_indices is not None and i not in block_indices:continue
        for name in ['qkv_proj','out_proj']:
            setattr(block.attn,name,LoRALinear(getattr(block.attn,name),rank))
            names.extend([f'blocks.{i}.attn.{name}.a',f'blocks.{i}.attn.{name}.b'])
    return names


@contextmanager
def native_training_math(functional_residuals=True):
    import comfy.model_management as mm
    import comfy.quant_ops
    import comfy.ldm.minimax.model as implementation
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    saved=(mm.in_training,comfy.quant_ops.ck.rms_rope_split_half,implementation._mod_scale_shift,implementation._mod_gate)
    def scale_shift(h,shift,scale,segments):
        return torch.cat([h[a:b]*(1.0+implementation._mod_row(scale,row,h.dtype))+implementation._mod_row(shift,row,h.dtype) for a,b,row in segments],dim=0)
    def gate(x,gates,other,segments):
        return torch.cat([torch.addcmul(x[a:b],other[a:b],implementation._mod_row(gates,row,x.dtype)) for a,b,row in segments],dim=0)
    try:
        mm.in_training=True
        comfy.quant_ops.ck.rms_rope_split_half=rms_rope_split_half
        if functional_residuals:
            implementation._mod_scale_shift=scale_shift
            implementation._mod_gate=gate
        yield
    finally:
        mm.in_training,comfy.quant_ops.ck.rms_rope_split_half,implementation._mod_scale_shift,implementation._mod_gate=saved
