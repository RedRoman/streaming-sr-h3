import json, struct, sys, glob
from pathlib import Path
def header(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]; return json.loads(f.read(n))
comfy = header('/cache/zihaohe/streaming_sr0/assets/comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors')
meta = comfy.pop('__metadata__', None)
ck = {k: (v['dtype'], tuple(v['shape'])) for k, v in comfy.items()}
off = {}
for p in sorted(glob.glob('/cache/songhua/models/MiniMax-H3/FL2VA/transformer/*.safetensors')):
    h = header(p); h.pop('__metadata__', None)
    off.update({k: (v['dtype'], tuple(v['shape'])) for k, v in h.items()})
print('comfy keys', len(ck), 'official keys', len(off))
print('comfy sample', sorted(ck)[:6]); print('official sample', sorted(off)[:6])
common = set(ck) & set(off)
print('common', len(common), 'only comfy', len(set(ck) - set(off)), 'only official', len(set(off) - set(ck)))
print('only comfy ex', sorted(set(ck) - set(off))[:8]); print('only official ex', sorted(set(off) - set(ck))[:8])
same = sum(ck[k] == off[k] for k in common); print('same dtype/shape among common', same)
print('dtypes comfy', {d for d, _ in ck.values()}, 'official', {d for d, _ in off.values()})
print('comfy metadata keys', list(meta)[:10] if meta else None)
diff = sorted(k for k in common if ck[k] != off[k])
import collections
pat = collections.Counter('.'.join(p for p in k.split('.') if not p.isdigit()) for k in diff)
print(pat.most_common(10))
for k in diff[:6]: print(k, 'comfy', ck[k], 'official', off[k])
print('adaln_t_table', ck['adaln_t_table'])
