"""Apply the FL2VA-derived qkv row permutation to the official Ref2VA qkv weights inside the rebuilt file, then verify SHA."""
import glob, hashlib, json, struct, sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT = Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3')
NAME = 'minimax_h3_ref2va_pruned_bf16.safetensors'
REB, SHA = ROOT / (NAME + '.rebuild'), '37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe'
perms = json.load(open(ROOT / 'qkv_perms_fl2va.json'))
ref = next(iter(perms.values()))
assert all(p == ref for p in perms.values()), 'qkv permutation is not global'
perm = torch.tensor(ref)
off = {}
for fp in sorted(glob.glob('/cache/songhua/models/MiniMax-H3/Ref2VA/transformer/*.safetensors')):
    with safe_open(fp, 'pt') as f:
        for k in f.keys():
            off[k] = fp
with open(REB, 'r+b') as w:
    n = struct.unpack('<Q', w.read(8))[0]; hdr = json.loads(w.read(n)); hdr.pop('__metadata__', None)
    done = 0
    for k in perms:
        with safe_open(off[k], 'pt') as f:
            t = f.get_tensor(k)[perm].contiguous()
        lo, hi = hdr[k]['data_offsets']; b = t.view(-1).view(torch.uint8).numpy().tobytes(); assert len(b) == hi - lo
        w.seek(8 + n + lo); w.write(b); done += 1
h = hashlib.sha256()
with open(REB, 'rb') as r:
    while blk := r.read(64 << 20):
        h.update(blk)
ok = h.hexdigest() == SHA
print(json.dumps({'qkv_rewritten': done, 'global_perm': True, 'sha256': h.hexdigest(), 'matches_published': ok}), flush=True)
if ok:
    REB.replace(ROOT / NAME)
sys.exit(0 if ok else 2)
