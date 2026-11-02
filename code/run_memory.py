"""Step 2 (PROTOCOL_step2.md): FL2VA + inner-DC on kpost with memory-augmented guides. LR is the only video
observation used by inner-DC; the guide may carry registered memory detail. Resumable, sharded."""
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

MODES = ['none_rerun', 'mem_gt_oflow_nogate', 'mem_wrong_oflow_nogate', 'mem_gt_oflow', 'mem_wrong_oflow', 'mem_gt_lrflow', 'mem_self_lrflow']


def write(p, d):
    q = Path(str(p) + '.tmp'); q.write_text(json.dumps(d, indent=1)); q.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', type=Path, default=ROOT / 'data/events')
    p.add_argument('--guides', type=Path, default=ROOT / 'data/step2_guides')
    p.add_argument('--out', type=Path, default=ROOT / 'results/step2_blocks')
    p.add_argument('--modes', nargs='+', default=MODES)
    p.add_argument('--only', nargs='*')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((a.events / 'selection.json').read_text())['events']
    jobs = [(e['sequence'], m) for m in a.modes for e in sel if not a.only or e['sequence'] in a.only]  # mode-priority order
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    h3 = H3(load_dit=True, dit='fl2va')
    write(a.out / f'runtime_shard{a.shard}.json', {**h3.record, 'jobs': len(jobs), 'modes': a.modes})
    for seq, mode in jobs:
        ev = json.loads((a.events / seq / 'event.json').read_text())
        k = ev['blocks']['kpost']
        folder = a.out / mode / seq / f'block{k:03d}'
        if (folder / 'result.json').exists():
            continue
        gpath = a.guides / seq / f"guide_{mode[len('mem_'):]}.npy"
        if mode != 'none_rerun' and not gpath.exists():
            print(json.dumps({'skip_missing_guide': str(gpath)}), flush=True); continue
        frames = list(range(17 * k, 17 * k + 22))
        lr_paths = [a.events / seq / 'lr' / f'{f:06d}.png' for f in frames]
        assert [hashlib.sha256(q.read_bytes()).hexdigest() for q in lr_paths] == [ev['frame_records'][str(f)]['lr_sha256'] for f in frames]
        low = torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2, 0, 1) for q in lr_paths]).float() / 255
        hw = [ev['hr_wh'][1], ev['hr_wh'][0]]
        folder.mkdir(parents=True, exist_ok=True)
        torch.npu.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        if mode == 'none_rerun':
            guide = h3.encode_guide(low, hw)
        else:
            pixels = torch.from_numpy(np.load(gpath))
            assert pixels.shape == (22, hw[0], hw[1], 3)
            guide = h3.encode_guide_pixels(pixels, hw)
        sampled, rec = h3.sample(guide, 'innerdc', low=low, hw=hw, seed=101)
        save_file({kk: v.contiguous() for kk, v in sampled.items()}, str(folder / 'sampled_av_latents.safetensors'))
        save_file({'guide': guide}, str(folder / 'guide_latent.safetensors'))
        px = h3.decode(sampled['video'])
        sync()
        assert px.shape == (22, hw[0], hw[1], 3) and torch.isfinite(px).all()
        files = {}
        for f, fr in zip(frames, to_u8(px)):
            q = folder / f'{f:06d}.png'; Image.fromarray(fr).save(q); files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
        write(folder / 'result.json', {'status': 'complete', 'sequence': seq, 'block': k, 'mode': mode, 'dit': 'fl2va',
                                       'guide_source': 'LR' if mode == 'none_rerun' else str(gpath),
                                       'guide_sha256': None if mode == 'none_rerun' else hashlib.sha256(gpath.read_bytes()).hexdigest(),
                                       'total_seconds': time.perf_counter() - t0, 'peak_allocated_GB': torch.npu.max_memory_allocated() / 1e9,
                                       'outputs_sha256': files, **{kk: v for kk, v in rec.items() if kk != 'mode'}, 'sampler_mode': rec.get('mode')})
        print(json.dumps({'done': seq, 'mode': mode, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)
    print(json.dumps({'shard': a.shard, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
