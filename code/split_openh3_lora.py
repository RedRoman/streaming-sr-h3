"""Split Open-H3-Regenerate-2K's merged rank-576 LoRA back into its three constituent adapters.

Their README states the merge: three adapters concatenated along the rank axis with strengths folded into
B, so B @ A equals the sum of the three scaled deltas exactly (MERGE_VERIFY.json puts the relative error
at 2.2e-8).

    Open-H3 refiner, step 5000 EMA   rank 384 / alpha 384
    LightX2V FL2VA Turbo v1.2        rank 128 / alpha 8
    LMS Sharp v1.0                   rank 64

Why bother splitting instead of mounting the whole thing: the refiner alone has already been measured on
this project's held-out 96 and it lost badly -- LPIPS +0.0211 with only 10/96 clips better, null cosine
-0.0414 worse on 91/96, every temporal metric worse (results/deploy/judge_ref.json, arm dcref against
dcxl). It did raise null energy 0.579 -> 0.776, i.e. it supplies detail, but misdirected detail. We
already have our own LightX2V turbo merged. The component that was never on the model is LMS Sharp, and
it is the one aimed at our actual deficit: detail energy 0.611 against FlashVSR+'s 0.850.

The separately released lora/sharp_curve_branches.safetensors cannot supply it: every curve_B tensor in
that file is exactly zero (0 of 6.2M entries nonzero, all 50 blocks), so B @ A = 0 and the adapter is a
no-op. The file's sha256 matches the repo's LFS hash, so that is the release and not a bad download.
Sharp's real weights are inside the merged file -- the rank slices carry mean per-rank contributions of
0.113 / 0.0028 / 0.098 on blocks.0.attn.to_q, and the tiny middle slice matches Turbo's folded
alpha/rank of 8/128 = 0.0625, which is the signature that pins the stated order.

Rather than rely on that identification, this writes all three slices as separate adapters so each can be
mounted and judged on its own. Ranks are per-module so the fused qkv projection in the ComfyUI layout
keeps working (lora.py already handles the 3x factor there).
"""
import argparse, json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = Path('/cache/zihaohe/baselines/openh3/lora/open_h3_regenerate_2k_lora.safetensors')
OUT = Path('/cache/zihaohe/streaming_sr0/assets/openh3_split')
SLICES = {'refiner384': (0, 384), 'turbo128': (384, 512), 'sharp64': (512, 576)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='', help='comma list of slice names; default all')
    a = ap.parse_args()
    want = [s for s in SLICES if not a.only or s in a.only.split(',')]
    OUT.mkdir(parents=True, exist_ok=True)
    with safe_open(str(SRC), framework='pt') as s:
        keys = [k for k in s.keys() if k.endswith('.lora_A.weight')]
        mods = sorted(k[:-len('.lora_A.weight')] for k in keys)
        print(json.dumps({'modules': len(mods)}), flush=True)
        for name in want:
            lo, hi = SLICES[name]
            out, stats = {}, []
            for m in mods:
                A = s.get_tensor(m + '.lora_A.weight')[lo:hi].contiguous()
                B = s.get_tensor(m + '.lora_B.weight')[:, lo:hi].contiguous()
                out[m + '.lora_A.weight'] = A
                out[m + '.lora_B.weight'] = B
                if len(stats) < 400:
                    stats.append(float((B.float() @ A.float()).norm()))
            p = OUT / f'openh3_{name}.safetensors'
            save_file(out, str(p), metadata={
                'source': SRC.name, 'rank_slice': f'{lo}:{hi}', 'rank': str(hi - lo),
                'modules': str(len(mods)),
                'note': 'rank-axis slice of the merged Open-H3-Regenerate-2K adapter; see '
                        'split_openh3_lora.py for why the slices are judged separately'})
            print(json.dumps({'wrote': p.name, 'rank': hi - lo, 'tensors': len(out),
                              'bytes': p.stat().st_size,
                              'delta_norm_mean': round(float(torch.tensor(stats).mean()), 4),
                              'delta_norm_max': round(float(torch.tensor(stats).max()), 4)}), flush=True)


if __name__ == '__main__':
    main()
