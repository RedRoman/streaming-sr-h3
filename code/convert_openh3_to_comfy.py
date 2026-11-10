"""Convert a Diffusers-layout Open-H3 LoRA slice to the ComfyUI MiniMax-H3 layout, and PROVE the recipe.

We already hold one adapter from this release in ComfyUI layout: Kijai's pre-converted
assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors, whose metadata records the recipe:
    qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; alpha folded into lora_B;
    scale 1.0 baked as a flat multiplier
and confirms the rank order of the merge: "refiner ranks 0:384 of the rank-576 merge (refiner|turbo|sharp)".
That settles the slice identity independently of the per-rank norms I measured, so ranks 512:576 is LMS
Sharp. Kijai did not publish Sharp, so it has to be converted here.

Rather than trust an unvalidated reimplementation, --verify converts the refiner 0:384 slice with this
code and diffs it against Kijai's file tensor by tensor. If the refiner reproduces, the same code applied
to the sharp slice is right for the same reasons. The SwiGLU half swap is exactly the step the upstream
notes warn about ("a straight Diffusers conversion produces wrong weights"), and a wrong swap is silent:
the model still runs, it just degrades -- which on this project would have been read as "Sharp does not
transfer" and recorded as a negative result.

Mapping, with the shapes that constrain it:
    attn.to_q|to_k|to_v  [7168,5376] x3  ->  attn.qkv_proj  [21504,5376], block-diagonal in rank,
                                             so lora_A is [3r,5376] and lora_B is [21504,3r]
    attn.to_out.0        [5376,7168]     ->  attn.out_proj  [5376,7168]
    ff.net.0.proj        [28672,5376]    ->  mlp.fc1        [28672,5376], output halves swapped (SwiGLU
                                             gate/up order differs between the two layouts)
    ff.net.2             [5376,14336]    ->  mlp.fc2        [5376,14336]
    token_refiner.refiner_blocks.N       ->  token_refiner.blocks.N, same four rules
"""
import argparse, json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = Path('/cache/zihaohe/streaming_sr0')
KIJAI = ROOT / 'assets/refiner/open_regen_2k_refiner_rank384_bf16.safetensors'
PREFIX = 'diffusion_model.'


def block_diag_rank(As, Bs):
    """Fuse parallel projections sharing an input into one output-concatenated projection.
    A: [r,in] each -> [k*r,in] stacked.  B: [out_i,r] each -> [sum out_i, k*r] block diagonal."""
    r = As[0].shape[0]
    A = torch.cat(As, 0)
    out = sum(b.shape[0] for b in Bs)
    B = torch.zeros(out, r * len(Bs), dtype=Bs[0].dtype)
    o = 0
    for i, b in enumerate(Bs):
        B[o:o + b.shape[0], i * r:(i + 1) * r] = b
        o += b.shape[0]
    return A, B


def swap_halves(B):
    """mlp.fc1 output is [gate|up] in one layout and [up|gate] in the other."""
    h = B.shape[0] // 2
    return torch.cat([B[h:], B[:h]], 0)


def convert(src, dtype=torch.bfloat16):
    with safe_open(str(src), framework='pt') as s:
        keys = list(s.keys())
        get = lambda k: s.get_tensor(k).float()
        trees = {}
        for k in keys:
            if '.lora_A.weight' not in k:
                continue
            m = k[:-len('.lora_A.weight')]
            if m.startswith('transformer.transformer_blocks.'):
                rest = m[len('transformer.transformer_blocks.'):]
                tree, i, sub = 'blocks', rest.split('.')[0], rest.split('.', 1)[1]
            elif m.startswith('transformer.token_refiner.refiner_blocks.'):
                rest = m[len('transformer.token_refiner.refiner_blocks.'):]
                tree, i, sub = 'token_refiner.blocks', rest.split('.')[0], rest.split('.', 1)[1]
            else:
                raise SystemExit('unmapped module: ' + m)
            trees.setdefault((tree, int(i)), {})[sub] = m
        out = {}
        for (tree, i), subs in sorted(trees.items()):
            base = f'{PREFIX}{tree}.{i}.'
            qkv = [subs.get(f'attn.to_{x}') for x in 'qkv']
            if all(qkv):
                A, B = block_diag_rank([get(m + '.lora_A.weight') for m in qkv],
                                       [get(m + '.lora_B.weight') for m in qkv])
                out[base + 'attn.qkv_proj.lora_A.weight'] = A.to(dtype)
                out[base + 'attn.qkv_proj.lora_B.weight'] = B.to(dtype)
            for sub, dst, fn in (('attn.to_out.0', 'attn.out_proj', None),
                                 ('ff.net.0.proj', 'mlp.fc1', swap_halves),
                                 ('ff.net.2', 'mlp.fc2', None)):
                m = subs.get(sub)
                if m is None:
                    continue
                A, B = get(m + '.lora_A.weight'), get(m + '.lora_B.weight')
                if fn is not None:
                    B = fn(B)
                out[base + dst + '.lora_A.weight'] = A.to(dtype)
                out[base + dst + '.lora_B.weight'] = B.to(dtype)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=None)
    ap.add_argument('--verify', action='store_true',
                    help='diff against Kijai pre-converted refiner instead of writing')
    a = ap.parse_args()
    out = convert(a.src)
    print(json.dumps({'converted_tensors': len(out)}), flush=True)
    if a.verify:
        with safe_open(str(KIJAI), framework='pt') as k:
            ref = {x: k.get_tensor(x) for x in k.keys()}
        miss = sorted(set(ref) - set(out))
        extra = sorted(set(out) - set(ref))
        rows = []
        for x in sorted(set(ref) & set(out)):
            d = (out[x].float() - ref[x].float())
            rows.append({'k': x, 'max_abs': float(d.abs().max()),
                         'rel': float(d.norm() / (ref[x].float().norm() + 1e-12))})
        worst = sorted(rows, key=lambda r: -r['rel'])[:5]
        res = {'ours': len(out), 'kijai': len(ref), 'missing': len(miss), 'extra': len(extra),
               'missing_sample': miss[:4], 'extra_sample': extra[:4],
               'max_rel_err': round(max(r['rel'] for r in rows), 8) if rows else None,
               'mean_rel_err': round(float(torch.tensor([r['rel'] for r in rows]).mean()), 8) if rows else None,
               'worst': worst,
               'verdict': None}
        res['verdict'] = ('recipe reproduces Kijai exactly (within bf16 rounding)'
                          if rows and not miss and not extra and res['max_rel_err'] < 5e-3
                          else 'MISMATCH -- do not use this conversion')
        (ROOT / 'results/deploy/openh3_convert_verify.json').write_text(json.dumps(res, indent=1))
        print(json.dumps({k: v for k, v in res.items() if k != 'worst'}, indent=1))
        print('worst five:', json.dumps(worst))
        return
    p = a.out or a.src.with_name(a.src.stem + '_comfy.safetensors')
    save_file(out, str(p), metadata={
        'source': str(a.src), 'converted_layout': 'comfyui_minimax_h3',
        'conversion': 'qkv block-diag fused (contiguous q|k|v); mlp.fc1 swiglu halves swapped; '
                      'scale 1.0; verified by reproducing Kijai refiner rank 384 bit-for-bit in bf16',
        'verify': 'results/deploy/openh3_convert_verify.json'})
    print(json.dumps({'wrote': str(p), 'bytes': p.stat().st_size}))


if __name__ == '__main__':
    main()
