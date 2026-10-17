import json, struct, glob, collections, re
p0 = '/cache/zihaohe/streaming_sr0/assets/comfy_h3/qwen3vl_32b_minimax_h3_bf16.safetensors.parts/00000'
with open(p0, 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; comfy = json.loads(f.read(n))
comfy.pop('__metadata__', None)
ck = {k: (v['dtype'], tuple(v['shape'])) for k, v in comfy.items()}
off = {}
for p in sorted(glob.glob('/cache/songhua/models/MiniMax-H3/Ref2VA/text_encoder/*.safetensors')):
    with open(p, 'rb') as f:
        m = struct.unpack('<Q', f.read(8))[0]; h = json.loads(f.read(m))
    h.pop('__metadata__', None)
    off.update({k: (v['dtype'], tuple(v['shape'])) for k, v in h.items()})
norm = lambda k: re.sub(r'\.\d+\.', '.N.', k)
print('comfy', len(ck), 'official', len(off))
print('comfy patterns', collections.Counter(norm(k).split('.N.')[0] for k in ck).most_common(8))
print('official patterns', collections.Counter(norm(k).split('.N.')[0] for k in off).most_common(8))
cl = sorted({int(m.group(1)) for k in ck for m in [re.search(r'layers\.(\d+)\.', k)] if m and 'visual' not in k})
ol = sorted({int(m.group(1)) for k in off for m in [re.search(r'layers\.(\d+)\.', k)] if m and 'visual' not in k})
print('comfy LM layers', cl[:2], '...', cl[-1:], len(cl), '| official LM layers', len(ol))
print('comfy examples', sorted(ck)[:4], [k for k in sorted(ck) if 'visual' in k][:3])
print('official examples', sorted(off)[:4], [k for k in sorted(off) if 'visual' in k][:3])
print('dtypes', {d for d, _ in ck.values()}, {d for d, _ in off.values()})
