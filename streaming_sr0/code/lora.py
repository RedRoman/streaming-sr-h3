"""LoRA for the ComfyUI MiniMax-H3 DiT. Wrappers expose .weight/.bias so comfy.ops.linear_input_act (which reads
linear.weight before calling the module in training mode) keeps working. B is zero-initialized: zero adapter == base."""
import math

import torch
from torch import nn
from torch.nn import functional as F

TARGETS = ('attn.qkv_proj', 'attn.out_proj', 'mlp.fc1', 'mlp.fc2')


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha=None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = (alpha if alpha is not None else rank) / rank
        dev = base.weight.device
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=torch.float32))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        out = self.base(x)
        delta = F.linear(F.linear(x.to(torch.float32), self.lora_a), self.lora_b) * self.scale
        return out + delta.to(out.dtype)


def inject(model, rank, alpha=None, targets=TARGETS):
    names = []
    for i, block in enumerate(model.blocks):
        for t in targets:
            parent_name, attr = t.split('.')
            parent = getattr(block, parent_name)
            base = getattr(parent, attr)
            if isinstance(base, LoRALinear):
                continue
            setattr(parent, attr, LoRALinear(base, rank, alpha))
            names.append(f'blocks.{i}.{t}')
    return names


def lora_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if n.endswith('.lora_a') or n.endswith('.lora_b')]


def state_dict(model, extra=None):
    sd = {n: p.detach().float().cpu().contiguous() for n, p in lora_parameters(model)}
    for k, m in (extra or {}).items():
        sd.update({f'{k}.{n}': v.detach().float().cpu().contiguous() for n, v in m.state_dict().items()})
    return sd


def load(model, sd, strict=True):
    params = dict(lora_parameters(model))
    missing = [n for n in params if n not in sd]
    if strict:
        assert not missing, missing[:5]
    with torch.no_grad():
        for n, p in params.items():
            if n in sd:
                p.copy_(sd[n].to(p.device, p.dtype))
    return len(params) - len(missing)
