"""Why the fitted latent map made things worse: how much of the truth's detail is linearly predictable.

judge_m0 says the least-squares latent map costs PSNR 1.17 dB and null cosine 0.112 against the round-trip
control. The useful question is not that it failed but which way it moved: an L2-optimal map SHRINKS the part
of the target it cannot predict. If our generated detail were merely mis-calibrated -- right direction, wrong
amplitude or spectrum -- the optimal map would amplify it. It shrank it (latent-domain detail energy down,
pixel null energy 0.575 -> 0.463), which says the detail is weakly correlated with the truth site by site.

This measures that correlation directly, so the claim is a number rather than an inference from a failure.
In latent space, with g the guide latent (what the observation already determines):
    d  = z  - g     our generated detail
    d* = z* - g     the truth's detail
and it reports cos(d, d*), the energy ratio, and the R^2 of the fitted map restricted to the detail band.
Encode only -- no decode, no DiT.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
from m0_latent_map import BLOCKS, block_px, gt_px, feat, splits, load  # same data path as the fit

D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/heldout')
OUT = ROOT / 'results/deploy'
NCLIP = int(os.environ.get('NCLIP', 8))
H, W = 480, 832


def lr_px(clip, s, e):
    d = D_ / clip / 'lr'
    return torch.stack([load(d / f'{f:06d}.png') for f in range(s, e)])


def main():
    from h3_comfy import H3
    h3 = H3(load_dit=False)
    W_ = torch.load(OUT / 'm0_W.pt', map_location='cpu', weights_only=True)['W']
    _, testc = splits()
    acc = []
    for c in testc[:NCLIP]:
        for k, (s, e) in enumerate(BLOCKS):
            z = h3.encode(block_px(c, k, s, e)).float().cpu()
            zs = h3.encode(gt_px(c, s, e)).float().cpu()
            g = h3.encode_guide(lr_px(c, s, e).permute(0, 3, 1, 2), [H, W]).float().cpu()
            d, ds = (z - g).reshape(-1).double(), (zs - g).reshape(-1).double()
            pred = (feat(z) @ W_)                                     # [T*h*w, C]
            T, h, w = z.shape[2], z.shape[3], z.shape[4]
            pred = pred.reshape(T, h, w, 24).permute(3, 0, 1, 2)[None]
            dp = (pred - g).reshape(-1).double()
            res = (zs - pred).reshape(-1).double()
            acc.append({
                'cos_detail': float(d @ ds / (d.norm() * ds.norm())),
                'energy_ratio_detail': float((d @ d) / (ds @ ds)),
                'cos_detail_mapped': float(dp @ ds / (dp.norm() * ds.norm())),
                'energy_ratio_detail_mapped': float((dp @ dp) / (ds @ ds)),
                'r2_full': float(1 - (res @ res) / ((zs.reshape(-1).double() - zs.mean().double()) ** 2).sum()),
                'r2_detail': float(1 - (res @ res) / (ds @ ds)),
            })
        print(json.dumps({'clip': c, **{kk: round(float(np.mean([a[kk] for a in acc])), 4) for kk in acc[0]}}),
              flush=True)
    agg = {kk: round(float(np.mean([a[kk] for a in acc])), 4) for kk in acc[0]}
    agg['blocks'] = len(acc)
    agg['reading'] = ('cos_detail is how aligned our generated detail is with the truth IN LATENT SPACE; '
                      'energy_ratio_detail_mapped below energy_ratio_detail is the shrinkage that makes the '
                      'map lose. r2_detail is the fraction of the truth\'s detail variance a site-local '
                      'linear map can explain from our latent.')
    (OUT / 'm0_predictability.json').write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg, indent=1))


main()
