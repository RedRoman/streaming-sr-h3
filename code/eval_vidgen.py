"""Stage-1 readout on the VIDGEN eval set (native 1344x768 source, so the ground truth keeps full detail).
Conditions: frozen H3 + inner-DC, LoRA + inner-DC, and the classical Dykstra projection baseline."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, torch
from PIL import Image
from safetensors.torch import save_file
from h3_comfy import H3, ROOT, sync, to_u8
from pixel_consistency import BicubicConsistency
import torch.nn.functional as F
D = ROOT / 'data/vidgen_eval'
def write(p, d):
    q = Path(str(p)+'.tmp'); q.write_text(json.dumps(d, indent=1)); q.replace(p)
p = argparse.ArgumentParser()
p.add_argument('--lora', type=Path, default=None); p.add_argument('--lora-rank', type=int, default=32)
p.add_argument('--mode', default='innerdc'); p.add_argument('--tag', required=True)
p.add_argument('--out', type=Path, default=ROOT/'results/vidgen_blocks')
p.add_argument('--shard', type=int, default=0); p.add_argument('--nshards', type=int, default=1)
a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
clips = json.loads((D/'manifest.json').read_text())['clips']
clips = [c for i, c in enumerate(clips) if i % a.nshards == a.shard]
if a.tag == 'classic':
    for name in clips:
        m = json.loads((D/name/'meta.json').read_text()); fo = a.out/'classic'/name
        if (fo/'result.json').exists(): continue
        fo.mkdir(parents=True, exist_ok=True)
        fr = m['frames']; hw = [m['hr_wh'][1], m['hr_wh'][0]]
        lr = torch.stack([torch.from_numpy(np.array(Image.open(D/name/'lr'/f'{f:06d}.png').convert('RGB')).copy()).permute(2,0,1) for f in fr]).float()/255
        up = F.interpolate(lr.double(), size=hw, mode='bicubic', align_corners=False, antialias=True).clamp(0,1)
        op = BicubicConsistency(tuple(hw), (hw[0]//4, hw[1]//4))
        proj, _ = op.project(up, lr.double(), iterations=25)
        files = {}
        for f, im in zip(fr, proj.permute(0,2,3,1).numpy()):
            q = fo/f'{f:06d}.png'; Image.fromarray(np.uint8(np.clip(im,0,1)*255+.5)).save(q)
            files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
        write(fo/'result.json', {'status':'complete','clip':name,'mode':'classic','outputs_sha256':files})
        print(json.dumps({'done':name,'mode':'classic'}), flush=True)
    raise SystemExit(0)
h3 = H3(load_dit=True, dit='fl2va', lora_path=a.lora, lora_rank=a.lora_rank)
write(a.out/f'runtime_{a.tag}_shard{a.shard}.json', {**h3.record, 'clips': len(clips), 'tag': a.tag})
for name in clips:
    m = json.loads((D/name/'meta.json').read_text()); fo = a.out/a.tag/name
    if (fo/'result.json').exists(): continue
    fr = m['frames']; hw = [m['hr_wh'][1], m['hr_wh'][0]]
    paths = [D/name/'lr'/f'{f:06d}.png' for f in fr]
    assert [hashlib.sha256(q.read_bytes()).hexdigest() for q in paths] == [m['frame_records'][str(f)]['lr_sha256'] for f in fr]
    low = torch.stack([torch.from_numpy(np.array(Image.open(q).convert('RGB')).copy()).permute(2,0,1) for q in paths]).float()/255
    fo.mkdir(parents=True, exist_ok=True)
    torch.npu.reset_peak_memory_stats(); sync(); t0 = time.perf_counter()
    guide = h3.encode_guide(low, hw)
    sampled, rec = h3.sample(guide, a.mode, low=low, hw=hw, seed=101)
    save_file({k: v.contiguous() for k, v in sampled.items()}, str(fo/'sampled_av_latents.safetensors'))
    px = h3.decode(sampled['video']); sync()
    assert px.shape == (22, hw[0], hw[1], 3) and torch.isfinite(px).all()
    files = {}
    for f, im in zip(fr, to_u8(px)):
        q = fo/f'{f:06d}.png'; Image.fromarray(im).save(q); files[q.name] = hashlib.sha256(q.read_bytes()).hexdigest()
    write(fo/'result.json', {'status':'complete','clip':name,'tag':a.tag,'mode':a.mode,
        'lora': str(a.lora) if a.lora else None, 'lora_sha256': h3.record.get('lora',{}).get('sha256'),
        'total_seconds': time.perf_counter()-t0, 'peak_allocated_GB': torch.npu.max_memory_allocated()/1e9,
        'outputs_sha256': files, 'HR_input': False, **{k: v for k, v in rec.items() if k != 'mode'}})
    print(json.dumps({'done':name,'tag':a.tag,'seconds':round(time.perf_counter()-t0,1)}), flush=True)
print(json.dumps({'shard':a.shard,'tag':a.tag,'finished':True}), flush=True)
