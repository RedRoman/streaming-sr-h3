"""Recover the row permutation Comfy applies to qkv_proj weights, from FL2VA (both sides SHA-verified/official)."""
import glob, json, hashlib
import numpy as np
import torch
from safetensors import safe_open
C = '/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors'
O = '/cache/songhua/models/MiniMax-H3/FL2VA/transformer'
off = {}
for fp in sorted(glob.glob(f'{O}/*.safetensors')):
    with safe_open(fp, 'pt') as f:
        for k in f.keys():
            off[k] = fp
perms = {}
with safe_open(C, 'pt') as c:
    keys = [k for k in c.keys() if k.endswith('qkv_proj.weight') or k.endswith('qkv_proj.bias')]
    for k in sorted(keys):
        a = c.get_tensor(k)
        with safe_open(off[k], 'pt') as f:
            b = f.get_tensor(k)
        rows_b = {hashlib.md5(r.numpy().tobytes()).hexdigest(): i for i, r in enumerate(b.view(b.shape[0], -1).view(torch.int16))}
        perm = [rows_b.get(hashlib.md5(r.numpy().tobytes()).hexdigest(), -1) for r in a.view(a.shape[0], -1).view(torch.int16)]
        perms[k] = perm
        ok = -1 not in perm and torch.equal(a, b[perm])
        print(k, tuple(a.shape), 'perm found' if -1 not in perm else 'UNMATCHED', 'exact' if ok else 'NOT exact', 'identity' if perm == list(range(len(perm))) else '')
groups = {}
for k, p in perms.items():
    groups.setdefault(json.dumps(p[:64]) + str(len(p)), []).append(k)
print('distinct permutations (by prefix):', len(groups), {g[:40]: len(v) for g, v in groups.items()})
p0 = np.array(next(iter(perms.values())))
print('perm head', p0[:20].tolist(), 'len', len(p0))
json.dump(perms, open('/cache/zihaohe/streaming_sr0/assets/comfy_h3/qkv_perms_fl2va.json', 'w'))
