"""Merge a lightx2v MiniMax-H3 turbo LoRA into the base DiT weights, and build its sigma schedule.

Merge rather than stack: the Stage-1 restoration LoRA already wraps the same four modules per block with
LoRALinear, and adding a second wrapper would change the arithmetic those wrappers were trained under.
Folding the turbo delta into `base.weight` *before* the Stage-1 wrappers are injected makes the turbo'd
model their base, which is exactly the composition we want to measure.

Variant choice matters. lightx2v ships 544p (shift 12) and 768p (shift 6) families; our clips are 832x480
and every schedule in this project is shift 12, which is also what the Stage-1 LoRA was trained under, so
the 544p family is the only one that composes without changing two things at once.
"""
import json
import torch
from safetensors.torch import load_file

PREFIX = 'diffusion_model.'


def turbo_sigmas(n_steps, shift=12.0):
    """lightx2v's schedule: evaluation points q_i = (N-i)/N on the unshifted grid, then the shift map.

    Same closed form as the project's 49-step SIGMAS, only with fewer points, so the Stage-1 LoRA stays on
    the sigma family it was trained on."""
    g = torch.linspace(1, 0, n_steps + 1)
    return shift * g / (1 + (shift - 1) * g)


def _check_schedule():
    # the published worked example: 4 steps, video shift 12 -> [1, 0.9730, 0.9231, 0.8000] -> 0
    got = turbo_sigmas(4, 12.0)
    want = torch.tensor([1.0, 0.9730, 0.9231, 0.8000, 0.0])
    assert torch.allclose(got, want, atol=1e-4), (got, want)


_check_schedule()


def _resolve(model, path):
    m = model
    for part in path.split('.'):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


def merge(model, lora_path, strength=1.0):
    """W <- W + strength * (B @ A) * (alpha / rank), in float32, cast back to the base dtype.

    Returns a record with the per-module-type relative delta norms, so a no-op merge (wrong keys, wrong
    scale) is visible in the log rather than silently producing an unchanged model."""
    sd = load_file(str(lora_path))
    mods = sorted({k[len(PREFIX):].rsplit('.lora_', 1)[0].rsplit('.alpha', 1)[0]
                   for k in sd if k.startswith(PREFIX)})
    stats, used = {}, 0
    for path in mods:
        a = sd[f'{PREFIX}{path}.lora_A.weight']
        b = sd[f'{PREFIX}{path}.lora_B.weight']
        alpha = float(sd[f'{PREFIX}{path}.alpha'])
        used += 3
        lin = _resolve(model, path)
        w = lin.weight
        scale = strength * alpha / a.shape[0]
        delta = (b.to(torch.float32) @ a.to(torch.float32)).mul_(scale)
        assert delta.shape == w.shape, (path, delta.shape, tuple(w.shape))
        rel = float(delta.norm() / w.float().norm())
        with torch.no_grad():
            w.add_(delta.to(device=w.device, dtype=w.dtype))
        del delta
        kind = path.split('.')[-1] if 'token_refiner' not in path else 'token_refiner.' + path.split('.')[-1]
        stats.setdefault(kind, []).append(rel)
    assert used == len([k for k in sd if k.startswith(PREFIX)]), 'some LoRA tensors were not consumed'
    rec = {'lora': str(lora_path), 'strength': strength, 'modules_merged': len(mods),
           'tensors_consumed': used,
           'relative_delta_norm': {k: {'mean': float(torch.tensor(v).mean()), 'max': float(torch.tensor(v).max())}
                                   for k, v in sorted(stats.items())}}
    assert min(min(v) for v in stats.values()) > 1e-6, 'merge was a no-op'
    return rec


if __name__ == '__main__':
    import sys
    print(json.dumps({'sigmas_4step_shift12': [round(x, 4) for x in turbo_sigmas(4, 12).tolist()],
                      'sigmas_8step_shift12': [round(x, 4) for x in turbo_sigmas(8, 12).tolist()],
                      'sigmas_49step_shift12_head': [round(x, 4) for x in turbo_sigmas(49, 12).tolist()[:4]]},
                     indent=1))
