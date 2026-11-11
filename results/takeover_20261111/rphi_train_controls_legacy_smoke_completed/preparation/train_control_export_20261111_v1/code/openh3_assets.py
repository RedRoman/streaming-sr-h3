"""Pinned assets and experiment contracts for the ComfyUI Open-H3 port.

This module is deliberately standard-library-only: inspection never loads torch
or reserves an accelerator. A verified port is not a claim of Diffusers parity.
"""
import argparse
import hashlib
import json
import os
import struct
from pathlib import Path

NATIVE_SIGMAS = (0.909375, 6 / 7, 2 / 3, 0.0)
RECIPES = {
    'refiner-only-comfy': ('refiner',),
    'refiner-turbo-comfy': ('refiner', 'turbo'),
    'release-components-comfy': ('refiner', 'turbo', 'sharp'),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safetensors_header(path):
    with open(path, 'rb') as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError('truncated safetensors file: %s' % path)
        size = struct.unpack('<Q', raw)[0]
        if size > 64 << 20:
            raise ValueError('unexpected safetensors header size')
        return json.loads(f.read(size))


def asset_path(record, manifest_path):
    p = Path(record['path']).expanduser()
    return p if p.is_absolute() else Path(manifest_path).resolve().parent / p


def read_manifest(path, verify=True):
    path = Path(path).resolve()
    m = json.loads(path.read_text())
    if m.get('schema_version') != 1 or m.get('recipe') not in RECIPES:
        raise ValueError('unsupported Open-H3 asset manifest schema/recipe')
    if m.get('layout') != 'comfyui_minimax_h3':
        raise ValueError('this loader requires already converted ComfyUI weights')
    roles = [c['role'] for c in m['components']]
    if roles != list(RECIPES[m['recipe']]):
        raise ValueError('component order/roles disagree with recipe: %r' % roles)
    records = {k: m[k] for k in ('base', 'vae', 'context')}
    records.update({c['role']: c for c in m['components']})
    if 'sharp' in roles:
        if m.get('sharp_curve_policy') != 'require_zero_B':
            raise ValueError('Sharp requires an explicit require_zero_B curve policy')
        records['sharp_curve'] = m['sharp_curve']
    resolved = {}
    for name, r in records.items():
        p = asset_path(r, path)
        digest = r.get('sha256', '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('missing or invalid SHA256 for %s' % name)
        if not p.is_file() or p.stat().st_size != r['bytes']:
            raise ValueError('missing/size-mismatched asset %s: %s' % (name, p))
        if verify and sha256(p) != digest:
            raise ValueError('SHA256 mismatch for %s: %s' % (name, p))
        resolved[name] = str(p.resolve())
    identity = {'recipe': m['recipe'], 'layout': m['layout'],
                'assets': {k: r['sha256'] for k, r in records.items()},
                'merge': 'sum_fp32_then_cast_once', 'sigma': list(NATIVE_SIGMAS)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return {**m, '_paths': resolved, '_fingerprint': fingerprint,
            '_manifest_sha256': sha256(path), '_files_sha256_verified': verify}


def select_samples(path, data_root, split):
    """JSON {samples:[{source_id, latent_path|path|id}]} or a legacy list.

    The Open-H3 trainer requires source_id on every row and separate manifests.
    A relative latent_path is relative to the manifest; an id maps under data/split.
    """
    p = Path(path).resolve()
    d = json.loads(p.read_text())
    rows = d['samples'] if isinstance(d, dict) else d
    if not isinstance(rows, list) or not rows:
        raise ValueError('sample manifest must contain a nonempty samples list')
    result = []
    for row in rows:
        row = {'path': row} if isinstance(row, str) else row
        val = row.get('latent_path', row.get('path'))
        if val is None:
            q = Path(data_root) / split / (str(row['id']) + '.safetensors')
        else:
            q = Path(val).expanduser()
            q = q if q.is_absolute() else p.parent / q
        q = q.resolve()
        if not q.is_file() or q.suffix != '.safetensors':
            raise ValueError('missing latent safetensors: %s' % q)
        result.append({'path': q, 'source_id': row.get('source_id')})
    if len({r['path'] for r in result}) != len(result):
        raise ValueError('duplicate sample path in %s' % p)
    return result


def validate_split_sources(train, val):
    if any(not r['source_id'] for r in train + val):
        raise ValueError('Open-H3 manifests require source_id for every sample')
    overlap = {r['source_id'] for r in train} & {r['source_id'] for r in val}
    if overlap:
        raise ValueError('train/val source overlap: %s' % sorted(overlap))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('manifest', type=Path)
    ap.add_argument('--headers-only', action='store_true',
                    help='check sizes and headers; explicitly does NOT verify file SHA256')
    a = ap.parse_args()
    m = read_manifest(a.manifest, verify=not a.headers_only)
    out = {'recipe': m['recipe'], 'fingerprint': m['_fingerprint'],
           'files_sha256_verified': m['_files_sha256_verified'], 'components': {}}
    for c in m['components']:
        h = safetensors_header(m['_paths'][c['role']])
        keys = [k for k in h if k != '__metadata__']
        out['components'][c['role']] = {'tensors': len(keys),
                                        'metadata': h.get('__metadata__', {})}
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
