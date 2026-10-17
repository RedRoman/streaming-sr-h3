"""Step 1: frozen H3 Ref2VA + LR guide keyframe + inner-DC on the post-return block kpost, with/without a reference
image (PROTOCOL_step1.md). LR-only video input; the reference comes from before the absence. Resumable, sharded."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

from h3_comfy import H3, ROOT, sync, to_u8

CONDS = ['none', 'gt', 'self', 'wrong']


def write(p, d):
    q = Path(str(p) + '.tmp'); q.write_text(json.dumps(d, indent=1)); q.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', type=Path, default=ROOT / 'data/events')
    p.add_argument('--contexts', type=Path, default=ROOT / 'results/step1_contexts')
    p.add_argument('--out', type=Path, default=ROOT / 'results/step1_blocks')
    p.add_argument('--conds', nargs='+', default=CONDS)
    p.add_argument('--only', nargs='*')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((a.events / 'selection.json').read_text())['events']
    jobs = [(e['sequence'], c) for e in sel if not a.only or e['sequence'] in a.only for c in a.conds]
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    h3 = H3(load_dit=True, dit='ref2va')
    write(a.out / f'runtime_shard{a.shard}.json', {**h3.record, 'jobs': len(jobs), 'conds': a.conds})
    neutral = torch.load(a.contexts / 'neutral.pt', map_location='cpu', weights_only=True)
    for seq, cond in jobs:
        ev = json.loads((a.events / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        folder = a.out / f'r2v_{cond}' / seq / f'block{k:03d}'
        if (folder / 'result.json').exists():
            continue
        ctx_path = a.contexts / ('neutral.pt' if cond == 'none' else f'{seq}__{cond}.pt')
        if not ctx_path.exists():
            print(json.dumps({'skip_missing_context': str(ctx_path)}), flush=True); continue
        frames = list(range(17 * k, 17 * k + 22))
        lr_paths = [a.events / seq / 'lr' / f'{f:06d}.png' for f in frames]
        assert [hashlib.sha256(q.read_bytes()).hexdigest() for q in lr_paths] == [ev['frame_records'][str(f)]['lr_sha256'] for f in frames]
        low = torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2, 0, 1) for q in lr_paths]).float() / 255
        hw = [ev['hr_wh'][1], ev['hr_wh'][0]]
        ctx = neutral if cond == 'none' else torch.load(ctx_path, map_location='cpu', weights_only=True)
        folder.mkdir(parents=True, exist_ok=True)
        torch.npu.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        guide = h3.encode_guide(low, hw)
        refs = None if cond == 'none' else [h3.ref_block(ctx['ref_image'])]
        sampled, rec = h3.sample(guide, 'innerdc', low=low, hw=hw, seed=101,
                                 context={'prompt_embeds': ctx['prompt_embeds'], 'text_token_tags': ctx['text_token_tags']}, refs=refs)
        save_file({kk: v.contiguous() for kk, v in sampled.items()}, str(folder / 'sampled_av_latents.safetensors'))
        px = h3.decode(sampled['video'])
        sync()
        assert px.shape == (22, hw[0], hw[1], 3) and torch.isfinite(px).all()
        files = {}
        for f, fr in zip(frames, to_u8(px)):
            q = folder / f'{f:06d}.png'; Image.fromarray(fr).save(q); files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
        write(folder / 'result.json', {'status': 'complete', 'sequence': seq, 'block': k, 'cond': cond, 'dit': 'ref2va',
                                       'context_file': ctx_path.name, 'context_tokens': int(ctx['prompt_embeds'].shape[0]),
                                       'ref_source': ctx.get('ref_source'), 'ref_hw': list(ctx['ref_image'].shape[1:3]) if cond != 'none' else None,
                                       'total_seconds': time.perf_counter() - t0, 'peak_allocated_GB': torch.npu.max_memory_allocated() / 1e9,
                                       'outputs_sha256': files, 'HR_input_video': False,
                                       **{kk: v for kk, v in rec.items() if kk != 'mode'}, 'sampler_mode': rec.get('mode')})
        print(json.dumps({'done': seq, 'cond': cond, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps({'shard': a.shard, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
