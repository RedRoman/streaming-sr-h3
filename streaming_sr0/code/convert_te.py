"""Build the Comfy-format H3 text encoder (Qwen3-VL-32B truncated to 50 LM layers) from the official HF shards
already on this host, and verify it against the Comfy-Org file header + downloaded byte ranges."""
import argparse, glob, hashlib, json, re, struct
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file

OFFICIAL = '/cache/songhua/models/MiniMax-H3/Ref2VA/text_encoder'
PARTS = Path('/cache/zihaohe/streaming_sr0/assets/comfy_h3/qwen3vl_32b_minimax_h3_bf16.safetensors.parts')
CHUNK = 512 << 20


def rename(k):
    m = re.match(r'model\.language_model\.layers\.(\d+)\.(.*)', k)
    if m:
        return f'model.layers.{m.group(1)}.{m.group(2)}' if int(m.group(1)) < 50 else None
    if k == 'model.language_model.embed_tokens.weight':
        return 'model.embed_tokens.weight'
    if k.startswith('model.visual.'):
        return k[len('model.'):]
    return None  # lm_head, final norm


def comfy_header():
    with open(PARTS / '00000', 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        return n, json.loads(f.read(n))


def downloaded_bytes(lo, hi):
    """Return bytes [lo,hi) of the Comfy file if fully present in downloaded parts, else None."""
    out = b''
    pos = lo
    while pos < hi:
        i = pos // CHUNK; p = PARTS / f'{i:05d}'
        if not p.exists():
            return None
        off = pos - i * CHUNK; take = min(hi - pos, CHUNK - off)
        with open(p, 'rb') as f:
            f.seek(off); b = f.read(take)
        if len(b) < take:
            return None
        out += b; pos += take
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path); ap.add_argument('--verify-only', action='store_true')
    a = ap.parse_args()
    n, hdr = comfy_header(); hdr.pop('__metadata__', None)
    files = sorted(glob.glob(f'{OFFICIAL}/*.safetensors'))
    src = {}
    for fp in files:
        with safe_open(fp, 'pt') as f:
            for k in f.keys():
                ck = rename(k)
                if ck:
                    src[ck] = (fp, k)
    missing, extra = set(hdr) - set(src), set(src) - set(hdr)
    print('comfy keys', len(hdr), 'mapped', len(src), 'missing', sorted(missing)[:5], 'extra', sorted(extra)[:5], flush=True)
    assert not missing and not extra
    checked = mismatch = shape_bad = 0
    tensors = {}
    for ck, (fp, k) in sorted(src.items()):
        with safe_open(fp, 'pt') as f:
            t = f.get_tensor(k)
        info = hdr[ck]
        if list(t.shape) != info['shape'] or str(t.dtype) != 'torch.bfloat16' or info['dtype'] != 'BF16':
            shape_bad += 1; continue
        lo, hi = info['data_offsets']
        ref = downloaded_bytes(8 + n + lo, 8 + n + hi)
        if ref is not None:
            checked += 1
            if t.contiguous().view(-1).view(dtype=__import__('torch').uint8).numpy().tobytes() != ref:
                mismatch += 1
                import torch
                r = torch.frombuffer(bytearray(ref), dtype=torch.bfloat16).view(t.shape).float()
                d = (t.float() - r).abs()
                print(json.dumps({'mismatch': ck, 'shape': list(t.shape), 'max_abs': float(d.max()), 'frac_diff': float((d > 0).float().mean()),
                                  'rel_rms': float((t.float() - r).square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-12))}), flush=True)
        if not a.verify_only:
            tensors[ck] = t.contiguous()
    print(json.dumps({'tensors': len(src), 'shape_or_dtype_bad': shape_bad, 'bytes_checked_against_comfy_file': checked, 'byte_mismatch': mismatch}), flush=True)
    assert shape_bad == 0 and mismatch == 0 and checked > 0
    if not a.verify_only:
        save_file(tensors, str(a.out))
        print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
