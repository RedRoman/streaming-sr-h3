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


def inject(model, rank, alpha=None, targets=TARGETS, token_refiner=False):
    """rank may be an int, or a dict keyed by target name for adapters whose rank differs per module.

    The Open-H3 2K refiner needs the dict form: it is rank 384 on attn.out_proj / mlp.fc1 / mlp.fc2 and
    rank 1152 on attn.qkv_proj, because qkv is a fused projection and carries three factors of 384.
    Injecting a single rank and zero-padding would work but would trip the strict load and waste memory.
    """
    names = []
    # token_refiner is opt-in. The Stage-1 adapter was trained before this tree was reachable, so its
    # checkpoint has no token_refiner tensors; injecting them by default made the strict load of
    # Stage-1 fail on 16 parameters that no file was ever going to fill. Only the Open-H3 refiner,
    # which does adapt those 2 blocks, asks for them.
    trees = [('', model.blocks)]
    tr = getattr(model, 'token_refiner', None)
    if token_refiner and tr is not None and hasattr(tr, 'blocks'):
        trees.append(('token_refiner.', tr.blocks))
    for prefix, blocks in trees:
      for i, block in enumerate(blocks):
        for t in targets:
            parent_name, attr = t.split('.')
            parent = getattr(block, parent_name)
            base = getattr(parent, attr)
            if isinstance(base, LoRALinear):
                continue
            full = f'{prefix}blocks.{i}.{t}'
            # Keys may be the full module name or just the target type. The Open-H3 refiner is uniform
            # per type (384, 1152 on the fused qkv) so the type key was enough; pdmd_2step is per LAYER
            # -- out_proj ranks run 2..72 and qkv_proj 2..128 across the 52 blocks -- and a type-keyed
            # lookup silently injected one block's rank everywhere, which surfaced as
            # "size of tensor a (69) must match the size of tensor b (91)" at load.
            r = (rank.get(full, rank.get(t)) if isinstance(rank, dict) else rank)
            a = (alpha.get(full, alpha.get(t)) if isinstance(alpha, dict) else alpha)
            if r is None:
                raise KeyError('no rank for %s' % full)
            setattr(parent, attr, LoRALinear(base, r, a))
            names.append(f'{prefix}blocks.{i}.{t}')
    return names


def convert_openh3_refiner(sd):
    """Open-H3 refiner LoRA keys -> the names lora.load() expects.

    Theirs: diffusion_model.blocks.{i}.{target}.lora_{A,B}.weight   (A: [r, in], B: [out, r])
    Ours:   blocks.{i}.{target}.lora_{a,b}
    Verified against the base checkpoint before any download: all 208 adapted modules resolve to an
    existing weight in assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors with B@A exactly the
    base shape, and the adapter never touches adaln_proj -- the one module whose layout differs between
    the pruned and unpruned H3 trees. So this is a rename, not a conversion.
    """
    out, ranks, alphas = {}, {}, {}
    for k, v in sd.items():
        kk = k
        for pre in ('model.diffusion_model.', 'diffusion_model.'):
            if kk.startswith(pre):
                kk = kk[len(pre):]
        for suf, slot in (('.lora_A.weight', 'lora_a'), ('.lora_B.weight', 'lora_b'),
                          ('.lora_down.weight', 'lora_a'), ('.lora_up.weight', 'lora_b')):
            if kk.endswith(suf):
                mod = kk[:-len(suf)]
                out[f'{mod}.{slot}'] = v
                ranks[mod] = min(v.shape)
                break
        else:
            if kk.endswith('.alpha'):
                # pdmd_2step ships one alpha scalar per module, so the scale is alpha/rank per module
                # rather than the 1.0 the Open-H3 refiner's uniform alpha == rank implied. Pulling them
                # out here keeps strict_unused meaningful: nothing is left in sd that no parameter wants.
                alphas[kk[:-len('.alpha')]] = float(v.reshape(-1)[0]) if v.numel() else None
    if not out:
        raise SystemExit('no lora_A/lora_B tensors found; is this an adapter file?')
    missing_alpha = [m for m in ranks if m not in alphas]
    if alphas and missing_alpha:
        raise SystemExit('%d modules have no alpha while %d do; refusing to guess the scale'
                         % (len(missing_alpha), len(alphas)))
    return out, ranks, (alphas or None)


def lora_parameters(model):
    return [(n, p) for n, p in model.named_parameters() if n.endswith('.lora_a') or n.endswith('.lora_b')]


def state_dict(model, extra=None):
    sd = {n: p.detach().float().cpu().contiguous() for n, p in lora_parameters(model)}
    for k, m in (extra or {}).items():
        sd.update({f'{k}.{n}': v.detach().float().cpu().contiguous() for n, v in m.state_dict().items()})
    return sd


def load(model, sd, strict=True, strict_unused=False):
    """strict guards model params with no tensor; strict_unused guards tensors no param consumed.

    Only the first direction was checked before, and it hid a real partial application: the Open-H3
    refiner adapts 208 modules -- 50 DiT blocks x 4 plus 2 token_refiner blocks x 4 -- while inject()
    only walked model.blocks, so 200 were applied, every model parameter found a tensor, strict passed,
    and 16 tensors were silently dropped. That is the same failure class as an adapter that loads and
    does nothing, just quieter.
    """
    params = dict(lora_parameters(model))
    missing = [n for n in params if n not in sd]
    if strict:
        assert not missing, missing[:5]
    if strict_unused:
        unused = [n for n in sd if n not in params]
        assert not unused, ('%d tensors in the file matched no parameter' % len(unused), unused[:5])
    with torch.no_grad():
        for n, p in params.items():
            if n in sd:
                p.copy_(sd[n].to(p.device, p.dtype))
    return len(params) - len(missing)


@torch.no_grad()
def refiner_weight_ratio(model):
    """mean ||s*B@A|| / ||W|| over the injected modules.

    The adapter's scale is taken to be 1.0 because Open-H3 reports the refiner at rank 384 / alpha 384
    and says the per-adapter strengths are folded into B. That is a claim about a file, not something we
    can see, so the magnitude of the delta it implies is reported instead of trusted: a ratio of order
    1e-2..1e-1 is a fine-tune, a ratio above 1 would mean the scale is wrong.
    """
    rs = []
    for block in model.blocks:
        for t in TARGETS:
            parent, attr = t.split('.')
            mod = getattr(getattr(block, parent), attr)
            if not isinstance(mod, LoRALinear):
                continue
            d = (mod.lora_b.float() @ mod.lora_a.float()) * mod.scale
            rs.append(float(d.norm() / mod.base.weight.float().norm().clamp_min(1e-8)))
    return {'n': len(rs), 'mean': float(sum(rs) / max(len(rs), 1)),
            'min': min(rs) if rs else None, 'max': max(rs) if rs else None}
