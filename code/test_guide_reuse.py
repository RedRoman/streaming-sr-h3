"""Can the guide latent of overlapping frames be reused between blocks?

Blocks step by 17 and span 22, so consecutive blocks share 5 pixel frames. If the VAE encoded frames
independently those 5 frames' latents would be identical in both blocks and could be computed once,
cutting guide encoding by 5/22. But H3's VAE compresses time (22 pixel frames -> 7 latent frames, a 2+5k
layout) with temporal convolutions, so a latent near a block edge may depend on frames outside the shared
range -- in which case reuse is not merely an optimisation but a change of output.

This encodes the same clip as block k and block k+1 and compares the latents covering the shared frames.
The answer is binary and decides whether the optimisation is available at all.
"""
import json

import torch
import torch_npu  # noqa: F401

from h3_comfy import H3, ROOT

H, W = 480, 832


def main():
    import numpy as np
    from PIL import Image
    D = ROOT / 'data/real_eval/const'
    name = json.loads((D / 'manifest.json').read_text())['clips'][0]

    def load(t):
        return torch.from_numpy(np.asarray(Image.open(D / name / 'lr' / f'{t:06d}.png').convert('RGB')).copy()).float() / 255

    lr = torch.stack([load(t) for t in range(39)]).permute(0, 3, 1, 2)   # [39,3,h,w]
    h3 = H3(load_dit=False)

    with torch.no_grad():
        g0 = h3.encode_guide(lr[0:22], [H, W])      # block 0 covers pixel frames [0,22)
        g1 = h3.encode_guide(lr[17:39], [H, W])     # block 1 covers [17,39); shares [17,22)
    out = {'latent_shape_block0': list(g0.shape), 'latent_shape_block1': list(g1.shape)}

    # 22 pixel frames -> 7 latent frames. The shared pixels [17,22) are the last 5 of block 0 and the
    # first 5 of block 1. Compare the last latent frame of g0 against the first latent frames of g1,
    # and report the best alignment found, so a partial match is visible rather than assumed absent.
    T = g0.shape[2]
    best = None
    for off0 in range(T):
        for off1 in range(T):
            d = (g0[:, :, off0] - g1[:, :, off1]).abs()
            rel = float(d.mean() / g0[:, :, off0].abs().mean().clamp_min(1e-8))
            if best is None or rel < best['rel']:
                best = {'block0_latent_idx': off0, 'block1_latent_idx': off1, 'rel': rel,
                        'max_abs': float(d.max())}
    out['best_matching_latent_pair'] = best
    out['reusable'] = bool(best['rel'] < 0.01)

    # also report the scale of a typical difference, to tell "slightly different" from "unrelated"
    out['unrelated_baseline_rel'] = float(
        (g0[:, :, 0] - g1[:, :, T - 1]).abs().mean() / g0[:, :, 0].abs().mean().clamp_min(1e-8))
    print(json.dumps(out, indent=1))
    verdict = ('REUSABLE: overlapping frames encode identically, guide cost can drop by 5/22'
               if out['reusable'] else
               'NOT REUSABLE: the VAE temporal receptive field makes edge latents block-dependent')
    print()
    print('VERDICT:', verdict)


if __name__ == '__main__':
    main()
