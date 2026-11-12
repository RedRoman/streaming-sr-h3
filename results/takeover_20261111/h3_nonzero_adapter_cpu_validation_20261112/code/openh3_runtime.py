"""Frozen Open-H3 component merge plus a separate, zero-initialized adaptation LoRA.

Only converted ComfyUI adapters are accepted. No legacy Stage-1 checkpoint or
legacy Turbo is loaded. Native scheduler/audio/backend parity remains a separate
gate; this is an explicit ComfyUI port, not the upstream Diffusers pipeline.
"""
import contextlib
import json

import torch
from safetensors import safe_open

import lora as L
from openh3_assets import read_manifest


def _module_keys(handle):
    modules = {}
    for key in handle.keys():
        name = key
        for prefix in ('model.diffusion_model.', 'diffusion_model.'):
            if name.startswith(prefix):
                name = name[len(prefix):]
        for suffix, slot in (('.lora_A.weight', 'A'), ('.lora_B.weight', 'B'), ('.alpha', 'alpha')):
            if name.endswith(suffix):
                modules.setdefault(name[:-len(suffix)], {})[slot] = key
                break
        else:
            raise ValueError('unexpected pretrained tensor: ' + key)
    if any(not {'A', 'B'} <= set(v) for v in modules.values()):
        raise ValueError('unpaired pretrained LoRA tensors')
    return modules


@torch.no_grad()
def merge_pretrained(model, manifest_path, verified_manifest=None):
    m = verified_manifest or read_manifest(manifest_path, verify=True)
    if not m['_files_sha256_verified']:
        raise ValueError('header-only validation cannot authorize a pretrained load')
    if getattr(model, '_openh3_pretrained_record', None) or L.lora_parameters(model):
        raise ValueError('pretrained merge requires a pristine model; refusing adapter stacking')
    expected = {f'blocks.{i}.{t}' for i in range(len(model.blocks)) for t in L.TARGETS}
    tr = getattr(model, 'token_refiner', None)
    if tr is not None:
        expected |= {f'token_refiner.blocks.{i}.{t}' for i in range(len(tr.blocks)) for t in L.TARGETS}
    if len(expected) != 208:
        raise ValueError('expected 50 DiT + 2 token-refiner blocks (208 linears)')
    with contextlib.ExitStack() as stack:
        opened = []
        for c in m['components']:
            h = stack.enter_context(safe_open(m['_paths'][c['role']], framework='pt', device='cpu'))
            keys = _module_keys(h)
            if set(keys) != expected:
                raise ValueError('%s module coverage mismatch: missing=%s extra=%s' %
                                 (c['role'], sorted(expected - set(keys))[:4], sorted(set(keys) - expected)[:4]))
            opened.append((c['role'], h, keys))
        # Fail before mutating weights on shape or coverage errors.
        for role, h, keys in opened:
            for name in sorted(expected):
                mod = model.get_submodule(name)
                a = h.get_slice(keys[name]['A']).get_shape()
                b = h.get_slice(keys[name]['B']).get_shape()
                if a[0] != b[1] or (b[0], a[1]) != tuple(mod.weight.shape):
                    raise ValueError('adapter shape mismatch: %s/%s' % (role, name))
        curve_count = 0
        if 'sharp' in m['_paths']:
            with safe_open(m['_paths']['sharp_curve'], framework='pt', device='cpu') as h:
                bs = [k for k in h.keys() if 'curve_B' in k]
                if not bs:
                    raise ValueError('Sharp curve file contains no curve_B tensors')
                for k in bs:
                    if torch.count_nonzero(h.get_tensor(k)).item():
                        raise ValueError('nonzero Sharp curves need a native curve runtime; refusing partial load')
                curve_count = len(bs)
        for name in sorted(expected):
            mod = model.get_submodule(name)
            merged = mod.weight.float().clone()
            for _, h, keys in opened:
                slots = keys[name]
                a = h.get_tensor(slots['A']).to(merged.device, torch.float32)
                b = h.get_tensor(slots['B']).to(merged.device, torch.float32)
                scale = float(h.get_tensor(slots['alpha'])) / a.shape[0] if 'alpha' in slots else 1.0
                merged.addmm_(b, a, alpha=scale)
            if not torch.isfinite(merged).all():
                raise ValueError('nonfinite merged weight: ' + name)
            mod.weight.copy_(merged.to(mod.weight.dtype))
    model.eval().requires_grad_(False)
    rec = {'recipe': m['recipe'], 'fingerprint': m['_fingerprint'],
           'manifest_sha256': m['_manifest_sha256'], 'merged_modules': len(expected),
           'components': [c['role'] for c in m['components']], 'merge': 'sum_fp32_then_cast_once',
           'sharp_zero_curve_B_tensors': curve_count,
           'upstream_diffusers_parity': 'not_validated',
           'legacy_stage1_loaded': False, 'legacy_turbo_loaded': False}
    model._openh3_pretrained_record = rec
    return rec


def frozen_snapshot(model):
    return [(n, p, p._version) for n, p in model.named_parameters()
            if not n.endswith(('.lora_a', '.lora_b'))]


def assert_frozen(model, snapshot):
    for name, p, version in snapshot:
        if p.requires_grad or p.grad is not None or p._version != version:
            raise AssertionError('frozen base changed or received gradients: ' + name)
    allowed = {id(p) for _, p in L.lora_parameters(model)}
    if any(p.requires_grad and id(p) not in allowed for p in model.parameters()):
        raise AssertionError('trainable parameter outside the new adaptation LoRA')
    return {'frozen_tensors': len(snapshot), 'passed': True}


def inject_adaptation(model, rank, seed):
    if not getattr(model, '_openh3_pretrained_record', None):
        raise ValueError('load and freeze verified Open-H3 pretrained weights first')
    if L.lora_parameters(model):
        raise ValueError('adaptation already injected')
    torch.manual_seed(seed)
    names = L.inject(model, rank)
    for name, p in L.lora_parameters(model):
        p.requires_grad_(True)
        if name.endswith('.lora_b') and torch.count_nonzero(p).item():
            raise AssertionError('new LoRA B is not zero: ' + name)
    return names


@torch.no_grad()
def merge_adaptation_for_inference(model):
    """Fold only the new adapter, avoiding unmanaged LoRA tensors during offload.

    Compute W + scale * B@A in FP32, then cast once to W's storage dtype.
    BF16 inference is consequently not bit-identical to the separate FP32 branch.
    Training and checkpoint serialization retain the independent adapter.
    """
    if not getattr(model, '_openh3_pretrained_record', None):
        raise ValueError('inference adaptation requires the verified pretrained base')
    modules = [(name, mod) for name, mod in model.named_modules() if isinstance(mod, L.LoRALinear)]
    if not modules:
        raise ValueError('no independent adaptation found to merge')
    dtypes = set()
    for name, mod in modules:
        base = mod.base
        merged = base.weight.float().clone()
        merged.addmm_(mod.lora_b.to(merged.device, torch.float32),
                      mod.lora_a.to(merged.device, torch.float32), alpha=mod.scale)
        if not torch.isfinite(merged).all():
            raise ValueError('nonfinite adaptation merge: ' + name)
        dtypes.add(str(base.weight.dtype))
        base.weight.copy_(merged.to(base.weight.dtype))
        parent, attr = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), attr, base)
    if L.lora_parameters(model):
        raise AssertionError('unmanaged adaptation parameters remain after inference merge')
    model.eval().requires_grad_(False)
    return {'merged_modules': len(modules), 'method': 'fp32_W_plus_scaled_BA_then_single_storage_cast',
            'storage_dtypes': sorted(dtypes), 'live_adapter_parameters': 0,
            'rounding': 'BF16 storage rounds the merged weight once; separate-branch bit parity is not claimed'}


@contextlib.contextmanager
def adapters_disabled(model):
    mods = [(m, m.scale) for m in model.modules() if isinstance(m, L.LoRALinear)]
    try:
        for m, _ in mods:
            m.scale = 0.0
        yield
    finally:
        for m, scale in mods:
            m.scale = scale


@torch.no_grad()
def zero_output_parity(model, forward):
    """Run the actual model on identical input/noise; forward must reset its RNG."""
    if any(torch.count_nonzero(p).item() for n, p in L.lora_parameters(model) if n.endswith('.lora_b')):
        raise AssertionError('zero parity must run before loading/training an adaptation')
    a = forward().detach().float()
    with adapters_disabled(model):
        b = forward().detach().float()
    if not torch.isfinite(a).all() or not torch.equal(a, b):
        raise AssertionError('zero adaptation changed prediction: max_abs=%s' % float((a-b).abs().max()))
    return {'max_abs': 0.0, 'prediction_elements': a.numel(), 'passed': True}


def adapter_metadata(pretrained, training, rank):
    return {'openh3_provenance': json.dumps({'pretrained_fingerprint': pretrained['fingerprint'],
                                           'rank': rank, 'training': training}, sort_keys=True)}


def check_adapter(path, pretrained, rank, training=None):
    with safe_open(str(path), framework='pt', device='cpu') as f:
        meta = json.loads((f.metadata() or {}).get('openh3_provenance', '{}'))
    if meta.get('pretrained_fingerprint') != pretrained['fingerprint'] or meta.get('rank') != rank:
        raise ValueError('checkpoint belongs to another base/recipe/rank (or has no provenance)')
    if training is not None and meta.get('training') != training:
        raise ValueError('resume changes the data manifest or training objective')
    return meta
