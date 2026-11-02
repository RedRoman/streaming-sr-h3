"""Step 1: encode H3 text conditions with the BF16 Qwen3-VL text encoder on NPU (PROTOCOL_step1.md).

Writes, per event and reference type, the resized reference image (shared with the DiT stage) and the
conditioning (prompt_embeds [L,5120], text_token_tags [L]). Also re-encodes the plain neutral prompt and
compares it with the colleague's cached NVFP4-AWQ context as a pipeline sanity check.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401
from PIL import Image

from h3_comfy import ASSETS, CONTEXT_PATH, ROOT, official_resize, patch_module_to, ref_resize, set_precision

TE_PATH = ASSETS / 'qwen3vl_32b_minimax_h3_bf16.safetensors'
TE_SHA = '600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d'
NEUTRAL = json.loads((ASSETS / 'h3_qwen_context_neutral_restore.json').read_text())['prompt']
REF_SENTENCE = ' <Picture 1> is a high-resolution reference of the same scene; reproduce its fine details where they appear in the video.'
E = ROOT / 'data/events'


def load_rgb(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB'), dtype=np.float32) / 255)


def ref_images(sel):
    """f_ref = middle of pairs_revisit_a. gt: HR GT; self: step-0 FL2VA innerDC output; wrong: next event's GT (cyclic)."""
    out = {}
    for i, e in enumerate(sel):
        seq, kpre = e['sequence'], e['blocks']['kpre']
        a = e['pairs_revisit_a']; f = a[len(a) // 2]
        nxt = sel[(i + 1) % len(sel)]; na = nxt['pairs_revisit_a']; nf = na[len(na) // 2]
        out[seq] = {'f_ref': f,
                    'gt': E / seq / 'hr' / f'{f:06d}.png',
                    'self': ROOT / 'results/step0_blocks/innerdc' / seq / f'block{kpre:03d}' / f'{f:06d}.png',
                    'wrong': E / nxt['sequence'] / 'hr' / f'{nf:06d}.png',
                    'wrong_source': [nxt['sequence'], nf]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=ROOT / 'results/step1_contexts')
    p.add_argument('--only', nargs='*', help='restrict to these sequences')
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    precision = set_precision(); patch_module_to()
    from comfy.cli_args import args as ca
    ca.reserve_vram = 2.
    import comfy.sd
    resize = official_resize()
    assert TE_PATH.stat().st_size == 51506295256
    t = time.perf_counter()
    clip = comfy.sd.load_clip([str(TE_PATH)], clip_type=comfy.sd.CLIPType.MINIMAX)
    load_s = time.perf_counter() - t

    def encode(prompt, resized=None):
        items = [{'type': 'image', 'data': resized}] if resized is not None else None
        tokens = clip.tokenize(prompt, minimax_ref_items=items)
        cond = clip.encode_from_tokens_scheduled(tokens)
        emb, extra = cond[0][0], cond[0][1]
        return {'prompt_embeds': emb[0].detach().to(torch.bfloat16).cpu().contiguous(),
                'text_token_tags': extra['minimax_token_tags'].cpu().contiguous()}

    record = {'te_sha256_expected': TE_SHA, 'te_load_seconds': load_s, 'precision_initial': precision, 'neutral_prompt': NEUTRAL,
              'ref_sentence': REF_SENTENCE, 'items': {}}
    t = time.perf_counter()
    neutral = encode(NEUTRAL)
    record['neutral_seconds'] = time.perf_counter() - t
    torch.save(neutral, a.out / 'neutral.pt')
    cached = torch.load(CONTEXT_PATH, map_location='cpu', weights_only=True)
    x, y = neutral['prompt_embeds'].float(), cached['prompt_embeds'].float()
    record['neutral_vs_cached_nvfp4'] = {
        'shape_bf16': list(x.shape), 'shape_cached': list(y.shape), 'tags_equal': bool(torch.equal(neutral['text_token_tags'], cached['text_token_tags'])),
        'per_token_cosine_mean': float(torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()) if x.shape == y.shape else None,
        'per_token_cosine_min': float(torch.nn.functional.cosine_similarity(x, y, dim=-1).min()) if x.shape == y.shape else None,
        'rel_rms_diff': float((x - y).square().mean().sqrt() / y.square().mean().sqrt()) if x.shape == y.shape else None}
    print(json.dumps(record['neutral_vs_cached_nvfp4']), flush=True)

    sel = json.loads((E / 'selection.json').read_text())['events']
    refs = ref_images(sel)
    for e in sel:
        seq = e['sequence']
        if a.only and seq not in a.only:
            continue
        ev = json.loads((E / seq / 'event.json').read_text())
        hw = [ev['hr_wh'][1], ev['hr_wh'][0]]
        for kind in ('gt', 'self', 'wrong'):
            target = a.out / f'{seq}__{kind}.pt'
            if target.exists():
                continue
            src = refs[seq][kind]
            resized = ref_resize(load_rgb(src), hw, resize)
            t = time.perf_counter()
            ctx = encode(NEUTRAL + REF_SENTENCE, resized)
            torch.save({**ctx, 'ref_image': resized.contiguous(), 'ref_source': str(src),
                        'ref_source_sha256': hashlib.sha256(Path(src).read_bytes()).hexdigest()}, target)
            record['items'][f'{seq}__{kind}'] = {'source': str(src), 'resized_hw': list(resized.shape[1:3]), 'tokens': int(ctx['prompt_embeds'].shape[0]),
                                                 'vision_tokens': int((ctx['text_token_tags'] == 0).sum()), 'seconds': time.perf_counter() - t}
            print(json.dumps({'encoded': f'{seq}__{kind}', **record['items'][f'{seq}__{kind}']}), flush=True)
    record['refs'] = {k: {kk: str(vv) for kk, vv in v.items()} for k, v in refs.items()}
    prev = json.loads((a.out / 'record.json').read_text()) if (a.out / 'record.json').exists() else {}
    prev_items = prev.get('items', {}); prev_items.update(record['items']); record['items'] = prev_items
    (a.out / 'record.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
