"""Can LMS Sharp be merged into our ComfyUI H3 as a pure weight edit? (zero inference cost if so)

Our weakest axis is detail energy: 0.611 of the truth against FlashVSR+'s 0.850, and the delivery videos
read as too soft. Every attempt to fix it by scaling our own null band failed, because gain_oracle showed
the MSE-optimal scale is BELOW one when the detail cosine is 0.48 -- amplifying detail that is half wrong
amplifies the wrong half too. A pretrained sharpness prior does not have that problem.

Open-H3-Regenerate-2K ships the 50 LMS Sharp AdaLN curve branches separately from its 312-linear refiner
LoRA, on the grounds that they "take an 8-D timestep curve instead of the layer input". In OUR checkpoint
that distinction disappears: blocks.N.adaln_proj.linear has in_features 8 and weight [96768, 8], and the
branch is curve_A [64, 8] and curve_B [96768, 64]. So curve_B @ curve_A is [96768, 8] -- exactly the shape
of the weight -- and merging is algebraically exact with no new module and no extra compute.
Our checkpoint also already carries adaln_t_table [1025, 8], the tensor their download.py fetches by byte
range from Comfy-Org, so nothing external is needed.

This checks the three things that would make that wrong:
  coverage   do the 50 branch indices match our 50 blocks exactly
  shape      does curve_B @ curve_A match adaln_proj.linear.weight for every block
  magnitude  is ||B@A|| a sane perturbation of ||W||. The README gives LMS Sharp as rank 64 with no alpha,
             which conventionally means scale 1. If the delta were the same order as the weight, the scale
             would be folded differently and merging at 1.0 would blow the model up.
No card, no model load: safetensors headers and two matrix products per block.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path('/cache/zihaohe/streaming_sr0')
BASE = ROOT / 'assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors'
SHARP = Path('/cache/zihaohe/baselines/openh3/lora/sharp_curve_branches.safetensors')


def main():
    with safe_open(str(SHARP), framework='pt') as s:
        skeys = list(s.keys())
        idx = sorted({int(k.split('transformer_blocks.')[1].split('.')[0]) for k in skeys})
        rows = []
        with safe_open(str(BASE), framework='pt') as b:
            bkeys = set(b.keys())
            ours = sorted({int(k.split('blocks.')[1].split('.')[0]) for k in bkeys
                           if k.startswith('blocks.') and 'adaln_proj.linear.weight' in k})
            for i in idx:
                A = s.get_tensor(f'transformer.transformer_blocks.{i}.adaln_proj.linear.curve_A.weight').float()
                B = s.get_tensor(f'transformer.transformer_blocks.{i}.adaln_proj.linear.curve_B.weight').float()
                W = b.get_tensor(f'blocks.{i}.adaln_proj.linear.weight').float()
                D = B @ A
                rows.append({'block': i, 'shape_match': tuple(D.shape) == tuple(W.shape),
                             'rank': A.shape[0],
                             'w_norm': float(W.norm()), 'd_norm': float(D.norm()),
                             'ratio': float(D.norm() / (W.norm() + 1e-12)),
                             'w_max': float(W.abs().max()), 'd_max': float(D.abs().max())})
    r = rows
    out = {
        'sharp_blocks': idx, 'our_blocks': ours,
        'coverage_exact': idx == ours,
        'ranks': sorted({x['rank'] for x in r}),
        'all_shapes_match': all(x['shape_match'] for x in r),
        'delta_over_weight_norm': {'min': round(min(x['ratio'] for x in r), 5),
                                   'median': round(float(torch.tensor([x['ratio'] for x in r]).median()), 5),
                                   'max': round(max(x['ratio'] for x in r), 5)},
        'per_block': r,
        'verdict': None,
    }
    ok = out['coverage_exact'] and out['all_shapes_match'] and out['delta_over_weight_norm']['max'] < 1.0
    out['verdict'] = ('mergeable as a weight edit at scale 1' if ok else
                      'NOT directly mergeable -- see which check failed')
    (ROOT / 'results/deploy/sharp_merge_probe.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'per_block'}, indent=1))
    print('first five blocks:', json.dumps(r[:5]))


main()
