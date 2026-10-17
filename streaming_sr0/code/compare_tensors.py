"""Compare shared tensors between a Comfy pruned file and the official shards (bytes + max abs diff)."""
import glob, json, sys
import torch
from safetensors import safe_open
comfy_path, official_dir = sys.argv[1], sys.argv[2]
off = {}
for fp in sorted(glob.glob(f'{official_dir}/*.safetensors')):
    with safe_open(fp, 'pt') as f:
        for k in f.keys():
            off[k] = fp
same = diff = skipped = 0; examples = []
with safe_open(comfy_path, 'pt') as c:
    for k in sorted(c.keys()):
        if k not in off:
            skipped += 1; continue
        a = c.get_tensor(k)
        with safe_open(off[k], 'pt') as f:
            b = f.get_tensor(k)
        if a.shape != b.shape or a.dtype != b.dtype:
            skipped += 1; continue
        if torch.equal(a, b):
            same += 1
        else:
            diff += 1
            if len(examples) < 8:
                examples.append({'k': k, 'dtype': str(a.dtype), 'max_abs': float((a.float() - b.float()).abs().max()),
                                 'rel_rms': float((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-12))})
print(json.dumps({'same': same, 'diff': diff, 'skipped': skipped, 'examples': examples}, indent=1))
