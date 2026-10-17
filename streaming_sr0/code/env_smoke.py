"""Relocated-env smoke test: VAE-only on bottle-14 block 315 must reproduce 30794's PNG hashes."""
import hashlib, io, json
import numpy as np, torch
from PIL import Image
from h3_comfy import H3, ROOT, to_u8
E = ROOT / 'data/events/bottle-14'
ev = json.loads((E / 'event.json').read_text())
k = 315; frames = list(range(17 * k, 17 * k + 22))
low = torch.stack([torch.from_numpy(np.array(Image.open(E / 'lr' / f'{f:06d}.png').convert('RGB')).copy()).permute(2, 0, 1) for f in frames]).float() / 255
h3 = H3(load_dit=False)
px = h3.decode(h3.encode_guide(low, [ev['hr_wh'][1], ev['hr_wh'][0]]))
out = {}
for f, fr in list(zip(frames, to_u8(px)))[:3]:
    b = io.BytesIO(); Image.fromarray(fr).save(b, format='PNG'); out[f'{f:06d}.png'] = hashlib.sha256(b.getvalue()).hexdigest()
print(json.dumps(out))
