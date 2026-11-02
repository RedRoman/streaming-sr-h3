"""Parity + effect check for the causal mask before any evaluation run."""
import json, numpy as np, torch
from PIL import Image
import causal_mask as CM
from h3_comfy import H3, ROOT, sync
D = ROOT/'data/stage2_eval'
name = json.loads((D/'manifest.json').read_text())['clips'][0]
m = json.loads((D/name/'meta.json').read_text()); hw = [m['hr_wh'][1], m['hr_wh'][0]]
low = torch.stack([torch.from_numpy(np.array(Image.open(D/name/'lr'/f'{f:06d}.png').convert('RGB')).copy()).permute(2,0,1) for f in range(0,22)]).float()/255
h3 = H3(load_dit=True, dit='fl2va', lora_path=ROOT/'results/stage1_lora/lora_step002000.safetensors')
g = h3.encode_guide(low, hw)
outs = {}
for mode, chunk in [('bidir',1), ('allvisible',1), ('causal',1), ('causal_cw2',2)]:
    if mode == 'allvisible':
        # same code path as causal but with an all-True mask: must match bidir bitwise
        orig = CM.build_mask
        CM.build_mask = lambda layout, c, dev: torch.ones(layout.seq_len, layout.seq_len, dtype=torch.bool, device=dev)
        h = CM.install(h3.patcher.model.diffusion_model, 'causal', 1)
    else:
        h = CM.install(h3.patcher.model.diffusion_model, mode, chunk)
    try:
        s, _ = h3.sample(g, 'base', low=low, hw=hw, seed=101)
        outs[mode] = s['video'].float()
    finally:
        h.remove()
        if mode == 'allvisible': CM.build_mask = orig
    print(mode, 'rms', round(float(outs[mode].square().mean().sqrt()), 4), flush=True)
b = outs['bidir']
for k, v in outs.items():
    d = (v-b)
    print(json.dumps({'mode': k, 'max_abs_vs_bidir': float(d.abs().max()), 'rel_rms_vs_bidir': float(d.square().mean().sqrt()/b.square().mean().sqrt())}), flush=True)
