"""Rebuild qwen3vl_32b_minimax_h3_bf16.safetensors byte-for-byte from the official HF shards on this host:
copy the Comfy-Org header verbatim (first 8+n bytes, from the downloaded first part) and place every renamed
tensor at the header's data_offsets. If the published SHA256 matches, the weights are identical."""
import glob, hashlib, json, re, struct, sys
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3')
PART0 = ROOT / 'qwen3vl_32b_minimax_h3_bf16.safetensors.parts/00000'
OUT = ROOT / 'qwen3vl_32b_minimax_h3_bf16.safetensors'
SHA, SIZE = '600d567f6a9629c8574e8e7041b199bdd9c59a986afa7906910a81919610607d', 51506295256
OFFICIAL = '/cache/songhua/models/MiniMax-H3/Ref2VA/text_encoder'


def rename(k):
    m = re.match(r'model\.language_model\.layers\.(\d+)\.(.*)', k)
    if m:
        return f'model.layers.{m.group(1)}.{m.group(2)}' if int(m.group(1)) < 50 else None
    if k == 'model.language_model.embed_tokens.weight':
        return 'model.embed_tokens.weight'
    return k[len('model.'):] if k.startswith('model.visual.') else None


with open(PART0, 'rb') as f:
    raw8 = f.read(8); n = struct.unpack('<Q', raw8)[0]; hraw = f.read(n)
hdr = json.loads(hraw); hdr.pop('__metadata__', None)
end = max(v['data_offsets'][1] for v in hdr.values())
assert 8 + n + end == SIZE, (8 + n + end, SIZE)
tmp = OUT.with_name(OUT.name + '.rebuild')
with open(tmp, 'wb') as w:
    w.write(raw8); w.write(hraw); w.truncate(SIZE)
    done = 0
    for fp in sorted(glob.glob(f'{OFFICIAL}/*.safetensors')):
        with safe_open(fp, 'pt') as f:
            for k in f.keys():
                ck = rename(k)
                if not ck:
                    continue
                info = hdr[ck]; t = f.get_tensor(k).contiguous()
                assert list(t.shape) == info['shape'] and t.dtype == torch.bfloat16 and info['dtype'] == 'BF16', ck
                lo, hi = info['data_offsets']; b = t.view(-1).view(torch.uint8).numpy().tobytes(); assert len(b) == hi - lo
                w.seek(8 + n + lo); w.write(b); done += 1
assert done == len(hdr), (done, len(hdr))
h = hashlib.sha256()
with open(tmp, 'rb') as r:
    while blk := r.read(64 << 20):
        h.update(blk)
ok = h.hexdigest() == SHA
print(json.dumps({'tensors_written': done, 'sha256': h.hexdigest(), 'matches_published': ok}), flush=True)
if ok:
    tmp.replace(OUT)
sys.exit(0 if ok else 2)
