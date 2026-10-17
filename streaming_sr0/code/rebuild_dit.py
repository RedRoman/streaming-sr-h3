"""Rebuild minimax_h3_ref2va_pruned_bf16.safetensors byte-for-byte: Comfy-Org header (from the first downloaded part),
tensors shared with the official Ref2VA shards copied locally, the ~100MB of pruned adaLN tensors fetched as exact
206 byte ranges. Accept only if the published SHA256 matches."""
import glob, hashlib, json, struct, subprocess, sys, time
from pathlib import Path
import torch
from safetensors import safe_open

ROOT = Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3')
NAME = 'minimax_h3_ref2va_pruned_bf16.safetensors'
PART0 = ROOT / (NAME + '.parts/00000')
URL = f'https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/diffusion_models/{NAME}'
SHA, SIZE = '37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe', 40225724176
OFFICIAL = '/cache/songhua/models/MiniMax-H3/Ref2VA/transformer'
DT = {'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32}


def fetch_exact(a, b):
    for attempt in range(20):
        r = subprocess.run(['curl', '-sS', '-L', '--fail', '--connect-timeout', '20', '-D', '-', '-o', '/dev/stderr', '-r', f'{a}-{b}', URL],
                           capture_output=True)
        heads, body = r.stdout.decode(errors='replace').lower(), r.stderr
        if r.returncode == 0 and f'content-range: bytes {a}-{b}/{SIZE}' in heads and len(body) >= b - a + 1:
            return body[-(b - a + 1):] if len(body) > b - a + 1 else body
        time.sleep(3)
    raise RuntimeError(f'range {a}-{b} failed')


with open(PART0, 'rb') as f:
    raw8 = f.read(8); n = struct.unpack('<Q', raw8)[0]; hraw = f.read(n)
hdr = json.loads(hraw); hdr.pop('__metadata__', None)
assert 8 + n + max(v['data_offsets'][1] for v in hdr.values()) == SIZE
official = {}
for fp in sorted(glob.glob(f'{OFFICIAL}/*.safetensors')):
    with safe_open(fp, 'pt') as f:
        for k in f.keys():
            official[k] = fp
out = ROOT / (NAME + '.rebuild')
local = remote = 0
with open(out, 'wb') as w:
    w.write(raw8); w.write(hraw); w.truncate(SIZE)
    for k, info in sorted(hdr.items(), key=lambda kv: kv[1]['data_offsets'][0]):
        lo, hi = info['data_offsets']
        use_local = False
        if k in official:
            with safe_open(official[k], 'pt') as f:
                t = f.get_tensor(k)
            use_local = list(t.shape) == info['shape'] and t.dtype == DT[info['dtype']]
        if use_local:
            b = t.contiguous().view(-1).view(torch.uint8).numpy().tobytes(); local += 1
        else:
            b = fetch_exact(8 + n + lo, 8 + n + hi - 1); remote += 1
        assert len(b) == hi - lo, k
        w.seek(8 + n + lo); w.write(b)
print(json.dumps({'tensors': len(hdr), 'from_official_local': local, 'fetched_ranges': remote}), flush=True)
h = hashlib.sha256()
with open(out, 'rb') as r:
    while blk := r.read(64 << 20):
        h.update(blk)
ok = h.hexdigest() == SHA
print(json.dumps({'sha256': h.hexdigest(), 'matches_published': ok}), flush=True)
if ok:
    out.replace(ROOT / NAME)
sys.exit(0 if ok else 2)
