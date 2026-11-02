"""Consistency checks for the cross-chunk KV cache, with memory probes at each stage."""
import json
import numpy as np, torch
from PIL import Image
import kv_cache as KV
from h3_comfy import H3, ROOT
D = ROOT/'data/stage2_eval'
name = json.loads((D/'manifest.json').read_text())['clips'][0]
m = json.loads((D/name/'meta.json').read_text()); hw = [m['hr_wh'][1], m['hr_wh'][0]]
def low_of(s, e):
    return torch.stack([torch.from_numpy(np.array(Image.open(D/name/'lr'/f'{f:06d}.png').convert('RGB')).copy()).permute(2,0,1)
                        for f in range(s, e)]).float()/255
def mem(tag):
    print(json.dumps({'mem': tag, 'alloc_GB': round(torch.npu.memory_allocated()/1e9, 2),
                      'reserved_GB': round(torch.npu.memory_reserved()/1e9, 2)}), flush=True)
h3 = H3(load_dit=True, dit='fl2va', lora_path=ROOT/'results/stage1_lora/lora_step002000.safetensors')
mem('after_load')
low0 = low_of(0, 22); g0 = h3.encode_guide(low0, hw)
cache = KV.H3KVCache(window=2, tail_frames=2)
h = KV.install(h3.patcher.model.diffusion_model, cache)
try:
    out0, _ = h3.sample(g0, 'base', low=low0, hw=hw, seed=101)
    mem('after_sample0')
    torch.npu.empty_cache(); mem('after_empty_cache')
    st = KV.commit_pass(h3, out0['video'], g0, hw, cache)
    mem('after_commit')
    print(json.dumps({'check': 'commit_chunk0', 'cache': st}), flush=True)
    cache.advance()
    low1 = low_of(17, 39); g1 = h3.encode_guide(low1, hw)
    withhist, _ = h3.sample(g1, 'base', low=low1, hw=hw, seed=101)
    mem('after_sample1_with_history')
    print(json.dumps({'check': 'chunk1_with_history', 'cache': cache.stats()}), flush=True)
finally:
    h.remove()
torch.npu.empty_cache()
nohist, _ = h3.sample(g1, 'base', low=low1, hw=hw, seed=101)
d = (withhist['video'].float() - nohist['video'].float())
print(json.dumps({'check': 'history_changes_output', 'max_abs': float(d.abs().max()),
                  'rel_rms': float(d.square().mean().sqrt()/nohist['video'].float().square().mean().sqrt())}), flush=True)
