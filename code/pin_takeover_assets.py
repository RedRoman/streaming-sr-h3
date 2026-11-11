"""Hash the existing shared release assets without changing any source file."""
import argparse
import hashlib
import json
from pathlib import Path


def record(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(16 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError('asset changed during hashing: ' + str(path))
    result = {'path': str(path.resolve()), 'bytes': after.st_size, 'sha256': digest.hexdigest()}
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    names = ['refiner_only.json', 'refiner_turbo.json']
    if any((a.out / name).exists() for name in names):
        p.error('refusing to overwrite pinned manifests')
    common = {
        'schema_version': 1, 'layout': 'comfyui_minimax_h3',
        'upstream': 'https://nvlabs.github.io/Sana/Open-H3-Regenerate-2K/',
        'port_scope': 'ComfyUI FL2VA port, cached neutral text, noise audio; native Diffusers parity unverified',
        'base': record(a.assets / 'comfy_h3/minimax_h3_fl2va_pruned_bf16.safetensors'),
        'vae': record(a.assets / 'comfy_h3/minimax_h3_video_vae_fp16.safetensors'),
        'context': record(a.assets / 'comfy_h3/h3_qwen_context_neutral_restore.pt'),
    }
    refiner = {'role': 'refiner', **record(a.assets / 'refiner/open_regen_2k_refiner_rank384_bf16.safetensors')}
    turbo = {'role': 'turbo', **record(a.assets / 'openh3_split/openh3_turbo128_comfy.safetensors')}
    for name, recipe, components in [
        (names[0], 'refiner-only-comfy', [refiner]),
        (names[1], 'refiner-turbo-comfy', [refiner, turbo]),
    ]:
        dest = a.out / name
        dest.write_text(json.dumps({**common, 'recipe': recipe, 'components': components}, indent=2) + '\n')
    print('Pinned manifests written.', flush=True)


if __name__ == '__main__':
    main()
