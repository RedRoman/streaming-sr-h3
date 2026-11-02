"""NPU vs archived H100 parity for frozen H3 on DAVIS bear_256 (LR-only inputs; HR only for evaluation).

1) VAE encode of the official guide; 2) VAE decode of archived H100 latents vs archived PNGs;
3) 49-step base / inner-DC sampling from the archived H100 guide, compared with the H100 latents,
   PNGs and GT. Sampling is chaotic, so bitwise equality is not expected; matched quality is the bar.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from h3_comfy import H3, ROOT, sha, sync, to_u8

B = ROOT / 'data/parity_bear256'
H = B / 'results/h3_setup/sampling_dc_davis8'


def psnr_u8(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64)) ** 2).mean())
    return 10 * np.log10(255 ** 2 / max(mse, 1e-12))


def compare_u8(mine, ref):
    d = np.abs(mine.astype(np.int16) - ref.astype(np.int16))
    return {'max_abs_u8': int(d.max()), 'frac_identical': float((d == 0).mean()), 'frac_diff_gt1': float((d > 1).mean()),
            'psnr_between': float(psnr_u8(mine, ref))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--skip-dit', action='store_true')
    p.add_argument('--modes', nargs='+', default=['base', 'innerdc'])
    p.add_argument('--out', type=Path, default=ROOT / 'results/npu_parity_bear256')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    case = next(c for c in json.loads((H / 'input_manifest.json').read_text())['cases'] if c['id'] == 'bear_256')
    lr_files = [B / 'data/real_video' / q for q in case['lr_paths']]
    hr_files = [B / 'data/real_video' / q for q in case['hr_paths']]
    assert [sha(q) for q in lr_files] == case['observed_lr_sha256'] and [sha(q) for q in hr_files] == case['evaluation_hr_sha256']
    low = torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2, 0, 1) for q in lr_files]).float() / 255
    gt = np.stack([np.array(Image.open(q).convert('RGB')) for q in hr_files])
    hw = case['shape_hw']
    ref_png = lambda mode: np.stack([np.array(Image.open(H / 'readouts' / mode / 'bear_256' / f'output_{i:02d}.png').convert('RGB')) for i in range(22)])

    t = time.perf_counter()
    h3 = H3(load_dit=not a.skip_dit)
    out = {'case': 'bear_256', 'load_seconds': time.perf_counter() - t, **h3.record}
    h100_guide = load_file(str(B / case['guide']))['guide'].float()

    times = []
    for _ in range(2):
        sync(); t = time.perf_counter()
        g = h3.encode_guide(low, hw).float()
        sync(); times.append(time.perf_counter() - t)
    d = g - h100_guide
    out['encode_guide'] = {'max_abs': float(d.abs().max()), 'rms': float(d.square().mean().sqrt()),
                           'ref_rms': float(h100_guide.square().mean().sqrt()), 'seconds_cold_warm': times}

    for mode, latent in [('vae_only', h100_guide),
                         ('sampling_dc', load_file(str(H / 'readouts/sampling_dc/bear_256/sampled_av_latents.safetensors'))['video'])]:
        ts = []
        for _ in range(2):
            sync(); t = time.perf_counter()
            px = h3.decode(latent)
            sync(); ts.append(time.perf_counter() - t)
        mine = to_u8(px)
        out[f'decode_archived_{mode}'] = {**compare_u8(mine, ref_png(mode)), 'psnr_vs_gt_npu_decode': float(psnr_u8(mine, gt)),
                                          'psnr_vs_gt_h100_png': float(psnr_u8(ref_png(mode), gt)), 'seconds_cold_warm': ts}
    out['peak_allocated_GB_after_vae'] = torch.npu.max_memory_allocated() / 1e9
    (a.out / 'parity.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)
    if a.skip_dit:
        return

    for mode, ref_mode in [m for m in [('base', 'unadapted'), ('innerdc', 'sampling_dc')] if m[0] in a.modes]:
        log = lambda ev: print(json.dumps({'event': 'hook', 'mode': mode, **ev}), flush=True)
        sampled, rec = h3.sample(h100_guide, mode, low=low, hw=hw, log=log)
        save_file({k: v.contiguous() for k, v in sampled.items()}, str(a.out / f'{mode}_sampled_av_latents.safetensors'))
        ref_lat = load_file(str(H / 'readouts' / ref_mode / 'bear_256/sampled_av_latents.safetensors'))
        dv = sampled['video'].float() - ref_lat['video'].float()
        mine = to_u8(h3.decode(sampled['video']))
        for i, fr in enumerate(mine):
            Image.fromarray(fr).save(a.out / f'{mode}_output_{i:02d}.png')
        out[f'sample_{mode}'] = {**rec, 'latent_max_abs_vs_h100': float(dv.abs().max()), 'latent_rms_vs_h100': float(dv.square().mean().sqrt()),
                                 'latent_ref_rms': float(ref_lat['video'].float().square().mean().sqrt()),
                                 **compare_u8(mine, ref_png(ref_mode)),
                                 'psnr_vs_gt_npu': float(psnr_u8(mine, gt)), 'psnr_vs_gt_h100': float(psnr_u8(ref_png(ref_mode), gt)),
                                 'peak_allocated_GB': torch.npu.max_memory_allocated() / 1e9}
        (a.out / 'parity.json').write_text(json.dumps(out, indent=1))
        print(json.dumps({k: v for k, v in out[f'sample_{mode}'].items() if k != 'correction_events'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
