import json, struct, subprocess, glob, re
import torch
from safetensors import safe_open
PARTS = '/cache/zihaohe/streaming_sr0/assets/comfy_h3/qwen3vl_32b_minimax_h3_bf16.safetensors.parts'
URL = 'https://hf-mirror.com/Comfy-Org/MiniMax-H3/resolve/e5eb578a89295337b8ff433a035929ce0279e0b6/text_encoders/qwen3vl_32b_minimax_h3_bf16.safetensors'
with open(f'{PARTS}/00000', 'rb') as f:
    n = struct.unpack('<Q', f.read(8))[0]; hdr = json.loads(f.read(n))
def official(ck):
    m = re.match(r'model\.layers\.(\d+)\.(.*)', ck)
    k = f'model.language_model.layers.{m.group(1)}.{m.group(2)}'
    for fp in sorted(glob.glob('/cache/songhua/models/MiniMax-H3/Ref2VA/text_encoder/*.safetensors')):
        with safe_open(fp, 'pt') as f:
            if k in f.keys():
                return f.get_tensor(k)
for ck in ['model.layers.13.self_attn.k_norm.weight', 'model.layers.18.post_attention_layernorm.weight']:
    lo, hi = hdr[ck]['data_offsets']; a, b = 8 + n + lo, 8 + n + hi - 1
    raw = subprocess.run(['curl', '-sS', '-L', '--fail', '-r', f'{a}-{b}', URL], capture_output=True, check=True).stdout
    fresh = torch.frombuffer(bytearray(raw), dtype=torch.bfloat16)
    off = official(ck)
    print(ck, 'part', a // (512 << 20), 'fresh==official', bool(torch.equal(fresh, off.view(-1))), 'fresh head', fresh[:4].tolist(), 'official head', off.view(-1)[:4].tolist())
