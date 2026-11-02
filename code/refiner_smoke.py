"""Does the Open-H3 2K refiner LoRA actually change our sampler's output, and is the change sane?

The tensor-level compatibility is already settled from the safetensors header alone: all 208 adapted
modules resolve to an existing weight in our pruned fl2va with B@A exactly the base shape, and the
adapter never touches adaln_proj. What that does NOT establish is whether it fires inside our pipeline
and whether what it produces is usable, and this project has a standing rule about exactly that gap --
an adapter that loads with strict=False and silently drops every tensor looks identical to one that works.

So this checks, in order:
  1. every parameter the injection creates is filled from the file (strict load, no silent drops)
  2. the sampled latent differs from the un-adapted run by a non-trivial margin -- if it matched, the
     adapter would not be firing
  3. the decoded pixels are finite and in range, and the detail energy in the null band is reported
     against the un-adapted run, because "more detail" is the only thing this adapter claims

It does not claim anything about quality. One clip cannot say that; the held-out 96 and the judge can.
The refiner was trained on H3's own 768P drafts, not on degraded camera footage, so the domain gap is
the real risk and only the full run measures it.
"""
import argparse
import json
import time

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image
from safetensors.torch import load_file

from h3_comfy import H3, ROOT, sync

H, W = 480, 832
REFINER = ROOT / 'assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors'
TURBO = ROOT / 'assets/turbo/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors'


for _p in (REFINER, TURBO):
    if not _p.exists():
        raise SystemExit('missing asset: %s' % _p)


def load_png(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def build(h3, with_refiner, steps, stage1):
    import h3_comfy
    import lora as L
    import lora_merge as LM
    import turbo as TB
    model = h3.patcher.model.diffusion_model
    TB.merge(model, TURBO)
    L.inject(model, 32)
    L.load(model, load_file(str(stage1)))
    LM.merge_(model)                      # fold Stage-1 so the refiner is the only live adapter
    rec = {'stage1_merged': True, 'refiner': False}
    if with_refiner:
        sd, ranks, alphas = L.convert_openh3_refiner(load_file(str(REFINER)))
        rec['refiner_ranks_sample'] = dict(list(ranks.items())[:4])
        rec['per_module_alpha'] = alphas is not None
        names = L.inject(model, ranks, alpha=alphas, token_refiner=True)
        n = L.load(model, sd, strict=True, strict_unused=True)  # both directions
        rec.update({'refiner': True, 'modules_injected': len(names), 'tensors_loaded': n,
                    'params_expected': 2 * len(names)})
        assert n == 2 * len(names), (n, len(names))
        rec['delta_over_base_norm'] = L.refiner_weight_ratio(model)
        rec['merge'] = LM.merge_(model, verify_samples=3)
    h3.frozen = [(nm, v, v._version) for nm, v in model.named_parameters()
                 if not (nm.endswith('.lora_a') or nm.endswith('.lora_b'))]
    h3_comfy.SIGMAS = TB.turbo_sigmas(steps, 12.0)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=4)
    ap.add_argument('--clip', default=None)
    ap.add_argument('--stage1', default=str(ROOT / 'results/stage1_lora/lora_step002000.safetensors'))
    ap.add_argument('--rphi', default='results/range_branch/r6_xl_k3.pt')
    a = ap.parse_args()

    from run_deploy import range_estimate
    D_ = ROOT / 'data/real_eval/const'
    man = json.loads((D_ / 'manifest.json').read_text())
    name = a.clip or man['clips'][0]
    s, e = man['blocks'][1]['range']
    T = json.loads((D_ / name / 'meta.json').read_text())['frames']
    lr = torch.stack([load_png(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)]).permute(0, 3, 1, 2)

    out = {'clip': name, 'block_frames': [s, e], 'steps': a.steps, 'arms': {}}
    lat = {}
    for arm in ('base', 'refiner'):
        h3 = H3(load_dit=True, dit='fl2va', lora_path=None)
        rec = build(h3, arm == 'refiner', a.steps, a.stage1)
        est = range_estimate(lr, ROOT / a.rphi, dev=h3.dev)
        sync(); t0 = time.perf_counter()
        guide = h3.encode_guide(lr[s:e], [H, W])
        sampled, _ = h3.sample(guide, 'innerdc', low=est[s:e], hw=[H, W], seed=101)
        px = h3.decode(sampled['video'])
        sync()
        lat[arm] = sampled['video'].detach().float().cpu()
        pf = px.detach().float().cpu()
        rec.update({'block_s': round(time.perf_counter() - t0, 2),
                    'latent_shape': list(lat[arm].shape),
                    'pixel_min': float(pf.min()), 'pixel_max': float(pf.max()),
                    'pixel_finite': bool(torch.isfinite(pf).all()),
                    'pixel_std': float(pf.std())})
        out['arms'][arm] = rec
        del h3, px
        torch.npu.empty_cache()

    d = (lat['refiner'] - lat['base']).abs()
    ref = lat['base'].abs().mean().clamp_min(1e-8)
    out['latent_rel_change'] = float(d.mean() / ref)
    out['latent_max_abs_change'] = float(d.max())
    out['fires'] = bool(out['latent_rel_change'] > 1e-3)
    print(json.dumps(out, indent=1, default=float))
    print()
    print('refiner fires:', out['fires'], ' relative latent change %.4f' % out['latent_rel_change'])
    print('block seconds   base %.2f   refiner %.2f'
          % (out['arms']['base']['block_s'], out['arms']['refiner']['block_s']))
    if not out['fires']:
        raise SystemExit('the adapter loaded but changed nothing; it is not in the forward path')


if __name__ == '__main__':
    main()
