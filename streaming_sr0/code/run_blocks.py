"""Run frozen H3 (vae_only / innerdc) on the stream blocks of each selected revisit event. LR-only inputs; resumable."""
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


def write(p, d):
    q = Path(str(p) + '.tmp'); q.write_text(json.dumps(d, indent=1)); q.replace(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--events', type=Path, default=ROOT / 'data/events')
    p.add_argument('--out', type=Path, default=ROOT / 'results/step0_blocks')
    p.add_argument('--modes', nargs='+', default=['vae_only', 'innerdc', 'innerdc_s202'])
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    p.add_argument('--lora', type=Path, default=None); p.add_argument('--lora-rank', type=int, default=32)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((a.events / 'selection.json').read_text())['events']
    jobs = [(e['sequence'], name, k) for e in sel for name, k in sorted(e['blocks'].items())]
    jobs = [j for i, j in enumerate(jobs) if i % a.nshards == a.shard]
    h3 = H3(load_dit=any(m != 'vae_only' for m in a.modes), lora_path=a.lora, lora_rank=a.lora_rank)
    write(a.out / f'runtime_shard{a.shard}.json', {**h3.record, 'jobs': len(jobs), 'modes': a.modes})
    pending, waited = list(jobs), 0
    while pending and waited < 3600:
        ready = [j for j in pending if (a.events / j[0] / 'event.json').exists()]
        if not ready:
            time.sleep(30); waited += 30; continue
        pending = [j for j in pending if j not in ready]
        for job in ready:
            run_job(a, h3, *job)
    print(json.dumps({'shard': a.shard, 'finished': True, 'unprocessed': pending}), flush=True)


def run_job(a, h3, seq, name, k):
    ev = json.loads((a.events / seq / 'event.json').read_text())
    frames = list(range(17 * k, 17 * k + 22))
    lr_paths = [a.events / seq / 'lr' / f'{f:06d}.png' for f in frames]
    assert [hashlib.sha256(q.read_bytes()).hexdigest() for q in lr_paths] == [ev['frame_records'][str(f)]['lr_sha256'] for f in frames]
    low = torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2, 0, 1) for q in lr_paths]).float() / 255
    hw = [ev['hr_wh'][1], ev['hr_wh'][0]]
    for mode in a.modes:
        if mode == 'innerdc_s202' and name != 'kpre':
            continue  # seed-sensitivity diagnostic runs on kpre only
        folder = a.out / mode / seq / f'block{k:03d}'
        if (folder / 'result.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        torch.npu.reset_peak_memory_stats()
        sync(); t0 = time.perf_counter()
        guide = h3.encode_guide(low, hw)
        sync(); t_guide = time.perf_counter() - t0
        rec = {}
        if mode == 'vae_only':
            latent = guide
        else:
            sampled, rec = h3.sample(guide, 'base' if mode == 'base' else 'innerdc', low=low, hw=hw, seed=202 if mode == 'innerdc_s202' else 101)
            latent = sampled['video']
            save_file({kk: v.contiguous() for kk, v in sampled.items()}, str(folder / 'sampled_av_latents.safetensors'))
        sync(); t1 = time.perf_counter()
        px = h3.decode(latent)
        sync(); t_dec = time.perf_counter() - t1
        assert px.shape == (22, hw[0], hw[1], 3) and torch.isfinite(px).all()
        u8 = to_u8(px)
        files = {}
        for f, fr in zip(frames, u8):
            q = folder / f'{f:06d}.png'; Image.fromarray(fr).save(q); files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
        write(folder / 'result.json', {'status': 'complete', 'sequence': seq, 'block_name': name, 'block': k, 'frames': [frames[0], frames[-1]],
                                       'mode': mode, 'hr_wh': ev['hr_wh'], 'guide_seconds': t_guide, 'decode_seconds': t_dec,
                                       'total_seconds': time.perf_counter() - t0, 'peak_allocated_GB': torch.npu.max_memory_allocated() / 1e9,
                                       'outputs_sha256': files, 'HR_input': False, **{kk: v for kk, v in rec.items() if kk != 'mode'}, 'sampler_mode': rec.get('mode')})
        print(json.dumps({'done': seq, 'block': k, 'mode': mode, 'seconds': round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == '__main__':
    main()
