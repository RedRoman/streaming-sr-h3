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
import kv_cache as KV
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
    p.add_argument('--kv', action='store_true', help='cross-chunk KV cache (Stage 5)')
    p.add_argument('--kv-window', type=int, default=2)
    p.add_argument('--kv-tail', type=int, default=2)
    p.add_argument('--kv-quota', action='store_true', help='observation energy quota on cached K/V')
    p.add_argument('--abs-noise', action='store_true',
                   help="draw the sampling noise once per clip and slice it by clip-absolute latent frame. "
                        'Blocks overlap by 5 pixel frames, but those frames sit at the END of block k and the '
                        'START of block k+1, so per-block noise gives the two blocks different noise for the '
                        'same instants and they disagree there. Indexing by absolute frame makes them solve '
                        'nearly the same problem, which is block consistency without any cross-block memory.')
    p.add_argument('--kv-commit', choices=['output', 'guide'], default='output',
                   help="what the cache stores: the chunk's own output, or its pure observation "
                        '(control: the guide carries no generated content, so anything that still '
                        'drifts is an artifact of attending to cached history, not feedback)')
    p.add_argument('--out', type=Path, default=ROOT / 'results/stage2_blocks')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    man = json.loads((D / 'manifest.json').read_text())
    clips = [c for i, c in enumerate(man['clips']) if i % a.nshards == a.shard]
    blocks = man['blocks']
    h3 = H3(load_dit=True, dit='fl2va', lora_path=a.lora, lora_rank=a.lora_rank)
    chunk = int(a.attn.split('cw')[1]) if 'cw' in a.attn else 1
    cache = KV.H3KVCache(window=a.kv_window, tail_frames=a.kv_tail) if a.kv else None
    if a.kv:
        handle = KV.install(h3.patcher.model.diffusion_model, cache)
    else:
        handle = CM.install(h3.patcher.model.diffusion_model, 'causal' if a.attn.startswith('causal') else 'bidir', chunk)
    write(a.out / f'runtime_{a.mode}_shard{a.shard}.json', {**h3.record, 'mode': a.mode, 'attn': a.attn, 'kv': bool(a.kv),
                                                            'attn_chunk': chunk, 'clips': len(clips)})
    for name in clips:
        m = json.loads((D / name / 'meta.json').read_text())
        hw = [m['hr_wh'][1], m['hr_wh'][0]]
        base = a.out / a.mode / name
        if (base / 'stream_result.json').exists():
            continue
        lr_all = torch.stack([torch.from_numpy(np.array(Image.open(D / name / 'lr' / f'{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1)
                              for f in range(m['frames'])]).float() / 255
        if cache is not None:
            cache.clear()
        clip_noise = None
        if a.abs_noise:
            import comfy.nested_tensor
            lt = (m['frames'] - 5) // 17 * 5 + 2                  # 124 pixel frames -> 37 latent frames
            at = round(m['frames'] / 24 * 40)
            g = torch.Generator().manual_seed(101)
            clip_noise = (torch.randn(1, 24, lt, hw[0] // 16, hw[1] // 16, generator=g),
                          torch.randn(1, 32, 2, at, generator=g))
        prev_tail = None          # decoded HR pixels of the previous block's last 5 emitted frames
        n_block0 = None           # block 0 runs with an empty cache, so its residual norm is un-accumulated
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
            if cache is not None:
                # block k covers pixel frames [17k, 17k+22), i.e. clip latent frames [5k, 5k+7)
                cache.latent_start = 5 * k
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
            nz = None
            if clip_noise is not None:
                import comfy.nested_tensor
                a0 = round(s / 24 * 40)
                nz = comfy.nested_tensor.NestedTensor(
                    (clip_noise[0][:, :, 5 * k:5 * k + 7].clone(),
                     clip_noise[1][:, :, :, a0:a0 + round((e - s) / 24 * 40)].clone()))
            sampled, rec = h3.sample(guide, 'innerdc', low=low, hw=hw, seed=101, noise=nz)
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
            kv_rec = {}
            if cache is not None:
                sync(); tk = time.perf_counter()
                lat = sampled['video']
                g = guide.to(device=lat.device, dtype=lat.dtype)
                # residual of the finished chunk against this block's own observation: what the model generated
                # beyond U(LR). This is the quantity that accumulates once the cache feeds it to the next chunk.
                rc = lat - g
                n_c = float(rc.float().square().sum().sqrt())
                if k == 0:
                    n_block0 = n_c
                anchor = {'resid_norm': n_c}
                if a.kv_commit == 'guide':
                    lat = g            # control arm: cache the observation, never the generated content
                if a.kv_quota and k > 0:
                    # reference = this block sampled without the cache, which the Stage-2 indep run already
                    # produced from the same guide and seed; read it instead of sampling the block twice.
                    rd = a.ref_dir / name / f'block{k:03d}'
                    refpx = torch.stack([torch.from_numpy(np.array(Image.open(rd / f'{f:06d}.png').convert('RGB')).copy())
                                         for f in range(s, e)]).float() / 255
                    z_ref = h3.encode(refpx).to(device=lat.device, dtype=lat.dtype)
                    n_r = float((z_ref - g).float().square().sum().sqrt())
                    rho = n_c / max(n_r, 1e-12)
                    if rho > a.tau:
                        lat = g + rc * (a.tau / rho)
                    anchor |= {'rho': rho, 'scaled': bool(rho > a.tau), 'tau': a.tau,
                               'ref_resid_norm': n_r,
                               # what a cache-free reference would have cost: block 0 needs no second pass
                               'rho_vs_block0': n_c / max(n_block0, 1e-12)}
                kv_rec = {'kv': KV.commit_pass(h3, lat, guide, hw, cache), 'anchor': anchor,
                          'kv_commit_seconds': time.perf_counter() - tk}
                cache.advance()
            r = {'block': k, 'range': [s, e], 'new': b['new'], 'guide_seconds': t_guide, **anchor_rec, **kv_rec,
                 'sample_seconds': rec['seconds'], 'decode_seconds': t_dec,
                 'total_seconds': time.perf_counter() - t0, 'peak_GB': torch.npu.max_memory_allocated() / 1e9,
                 'outputs_sha256': files}
            recs.append(r)
            write(fo / 'result.json', {'status': 'complete', 'clip': name, 'mode': a.mode, **r})
        sd = base / 'stream'; sd.mkdir(exist_ok=True)
        for f, im in sorted(stream.items()):
            Image.fromarray(im).save(sd / f'{f:06d}.png')
        write(base / 'stream_result.json', {'status': 'complete', 'clip': name, 'mode': a.mode, 'attn': a.attn, 'kv': bool(a.kv),
                                            'frames_emitted': sorted(stream), 'blocks': recs,
                                            'lora_sha256': h3.record.get('lora', {}).get('sha256'),
                                            'wait_frames_per_block': 22 - 5, 'HR_input': a.mode == 'carry' and False})
        print(json.dumps({'done': name, 'mode': a.mode, 'blocks': len(recs),
                          'sec_total': round(sum(r['total_seconds'] for r in recs), 1)}), flush=True)
    handle.remove()
    print(json.dumps({'shard': a.shard, 'mode': a.mode, 'attn': a.attn, 'kv': bool(a.kv), 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
