"""Stage 2 (PROTOCOL_stage2.md): run the Stage-1 LoRA over a whole 124-frame clip, block by block.

mode indep : every block is sampled independently (Stage-1 behaviour, current baseline)
mode carry : for k>0 the 5 overlap frames of the guide carry the previous block's decoded output instead of U(LR)

Per-block outputs are kept separately so block-boundary disagreement can be measured; the assembled stream
(each frame taken from the block that newly emits it) is written to stream/.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import causal_mask as CM
from h3_comfy import H3, ROOT, sync, to_u8

D = ROOT / 'data/stage2_eval'


def unobserved(x):
    """N(x) = x - U(D(x)): the component a x4 antialiased-bicubic observation does not determine. x: [T,H,W,3]."""
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // 4, w // 4), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def write(p, d):
    q = Path(str(p) + '.tmp'); q.write_text(json.dumps(d, indent=1)); q.replace(p)


def up(lr, hw):
    t = lr.unsqueeze(0) if lr.ndim == 3 else lr
    return F.interpolate(t, size=hw, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--lora', type=Path, default=ROOT / 'results/stage1_lora/lora_step002000.safetensors')
    p.add_argument('--lora-rank', type=int, default=32)
    p.add_argument('--mode', choices=['indep', 'carry', 'anchor', 'anchor_lowfreq_only'], required=True)
    p.add_argument('--tau', type=float, default=1.0)
    p.add_argument('--ref-dir', type=Path, default=ROOT / 'results/stage2_blocks/indep')
    p.add_argument('--attn', default='bidir', help='bidir | causal | causal_cw2')
    p.add_argument('--out', type=Path, default=ROOT / 'results/stage2_blocks')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    clips = [c for i, c in enumerate(man['clips']) if i % a.nshards == a.shard]
    blocks = man['blocks']
    h3 = H3(load_dit=True, dit='fl2va', lora_path=a.lora, lora_rank=a.lora_rank)
    chunk = int(a.attn.split('cw')[1]) if 'cw' in a.attn else 1
    handle = CM.install(h3.patcher.model.diffusion_model, 'causal' if a.attn.startswith('causal') else 'bidir', chunk)
    write(a.out / f'runtime_{a.mode}_shard{a.shard}.json', {**h3.record, 'mode': a.mode, 'attn': a.attn,
                                                            'attn_chunk': chunk, 'clips': len(clips)})
    for name in clips:
        m = json.loads((D / name / 'meta.json').read_text())
        hw = [m['hr_wh'][1], m['hr_wh'][0]]
        base = a.out / a.mode / name
        if (base / 'stream_result.json').exists():
            continue
        lr_all = torch.stack([torch.from_numpy(np.array(Image.open(D / name / 'lr' / f'{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1)
                              for f in range(m['frames'])]).float() / 255
        prev_tail = None          # decoded HR pixels of the previous block's last 5 emitted frames
        stream = {}               # frame -> pixels, each from the block that newly emits it
        recs = []
        for b in blocks:
            k, (s, e) = b['k'], b['range']
            fo = base / f'block{k:03d}'
            fo.mkdir(parents=True, exist_ok=True)
            low = lr_all[s:e]
            ref_tail = None
            if a.mode == 'anchor' and prev_tail is not None:
                # the energy reference is this block sampled without any carry, which the indep run already
                # produced with the same guide and seed -- read it instead of re-sampling
                rd = a.ref_dir / name / f'block{k:03d}'
                ref_tail = torch.stack([torch.from_numpy(np.array(Image.open(rd / f'{f:06d}.png').convert('RGB')).copy())
                                        for f in range(s, s + 5)]).float() / 255
            upx = up(low, hw).permute(0, 2, 3, 1).contiguous()    # [22,H,W,3]
            anchor_rec = {}
            if prev_tail is not None and a.mode != 'indep':
                if a.mode == 'carry':
                    upx[:5] = prev_tail                            # raw carry (Stage-2 baseline)
                else:
                    # observation-anchored carry: low frequency always from the current observation,
                    # and (unless disabled) the carried high-frequency energy is capped at the
                    # energy this block produces without any carry.
                    base5 = upx[:5].clone()
                    hf = unobserved(prev_tail)
                    if a.mode == 'anchor':
                        if ref_tail is None:
                            raise RuntimeError('anchor mode needs the no-carry reference for this block')
                        e_ref = float(unobserved(ref_tail).square().sum().sqrt())
                        e_hf = float(hf.square().sum().sqrt())
                        r = e_hf / max(e_ref, 1e-12)
                        if r > a.tau:
                            hf = hf * (a.tau / r)
                        anchor_rec = {'energy_ratio_carry_over_ref': r, 'scaled': bool(r > a.tau), 'tau': a.tau}
                    upx[:5] = (base5 + hf).clamp(0, 1)
            torch.npu.reset_peak_memory_stats(); sync(); t0 = time.perf_counter()
            guide = h3.encode_guide_pixels(upx, hw)
            t_guide = time.perf_counter() - t0
            sampled, rec = h3.sample(guide, 'innerdc', low=low, hw=hw, seed=101)
            t1 = time.perf_counter()
            px = h3.decode(sampled['video']); sync()
            t_dec = time.perf_counter() - t1
            assert px.shape == (22, hw[0], hw[1], 3) and torch.isfinite(px).all()
            u8 = to_u8(px)
            files = {}
            for i, f in enumerate(range(s, e)):
                q = fo / f'{f:06d}.png'; Image.fromarray(u8[i]).save(q)
                files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
            new_lo = b['new'][0]
            for i, f in enumerate(range(s, e)):
                if f >= new_lo:
                    stream[f] = u8[i]
            prev_tail = px[-5:].clone()
            r = {'block': k, 'range': [s, e], 'new': b['new'], 'guide_seconds': t_guide, **anchor_rec,
                 'sample_seconds': rec['seconds'], 'decode_seconds': t_dec,
                 'total_seconds': time.perf_counter() - t0, 'peak_GB': torch.npu.max_memory_allocated() / 1e9,
                 'outputs_sha256': files}
            recs.append(r)
            write(fo / 'result.json', {'status': 'complete', 'clip': name, 'mode': a.mode, **r})
        sd = base / 'stream'; sd.mkdir(exist_ok=True)
        for f, im in sorted(stream.items()):
            Image.fromarray(im).save(sd / f'{f:06d}.png')
        write(base / 'stream_result.json', {'status': 'complete', 'clip': name, 'mode': a.mode, 'attn': a.attn,
                                            'frames_emitted': sorted(stream), 'blocks': recs,
                                            'lora_sha256': h3.record.get('lora', {}).get('sha256'),
                                            'wait_frames_per_block': 22 - 5, 'HR_input': a.mode == 'carry' and False})
        print(json.dumps({'done': name, 'mode': a.mode, 'blocks': len(recs),
                          'sec_total': round(sum(r['total_seconds'] for r in recs), 1)}), flush=True)
    handle.remove()
    print(json.dumps({'shard': a.shard, 'mode': a.mode, 'attn': a.attn, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
