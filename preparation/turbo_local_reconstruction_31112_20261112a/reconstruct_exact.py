"""Recreate the exact existing Turbo file from a local compatible representation.

CPU NumPy only. Every output tensor and the complete output file must match
the independently hashed original31984 file. No approximate asset substitution.
Original inputs and all failed transfers stay untouched; publication is separate.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

ROOT = Path('/cache/zihaohe/omni_openh3_20261111')
OWN = ROOT / 'preparation/turbo_local_reconstruction_31112_20261112a'
INPUT = Path('/cache/zihaohe/streaming_sr0/assets/turbo/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors')
EXPECTED = '0b5e411b519ea5041734d44fc102b0dfd6b6a29737048ad1a80a522fb633fa99'
SIZE = 1956171936
HEADER_SHA = '9d397dd3fcaf0de036421d0377240d63e464773a9a3316354803ee0b5a5210ac'
MANIFEST_SHA = '35097c986eef94ea8e7fb668236abc1add562b83ee0ed7dee2ad90ac6752a61c'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def stat_identity(path):
    s = Path(path).stat()
    return {k: getattr(s, k) for k in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns')}


def save(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    with Path(path).open('x') as f:
        f.write(data)


def transform_b(data):
    # BF16 -> FP32 is exact; the documented alpha/rank is a power of two.
    # Explicit BF16 round-to-nearest-even also handles any subnormal boundary.
    a = (np.frombuffer(data, dtype='<u2').astype('<u4') << 16).view('<f4')
    require(bool(np.isfinite(a).all()), 'nonfinite source B tensor')
    scaled = a * np.float32(0.0625)
    bits = scaled.view('<u4')
    rounded = bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))
    return (rounded >> 16).astype('<u2').tobytes()


def main():
    require(Path(__file__).resolve().parent == OWN, 'unexpected reconstruction namespace')
    for name in ('result.json', 'failure.json', 'output.partial', 'output.ready'):
        require(not (OWN/name).exists() and not (OWN/name).is_symlink(), 'existing output; no overwrite or retry')
    require(not any(k.split('.')[0] in {'torch', 'torch_npu', 'comfy'} for k in sys.modules), 'model/backend unexpectedly imported')
    started = time.monotonic()
    record = {'schema': 'openh3.exact_local_turbo_reconstruction.v1', 'complete': False,
              'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'script_sha256': sha_file(__file__), 'input': str(INPUT),
              'method': 'A unchanged; B alpha/rank=0.0625 baked into BF16, original header bytes/order restored',
              'models': 0, 'NPU': 0, 'publication_performed': False, 'tensor_results': {}}
    try:
        require(sha_file(OWN/'source_tensors.json') == MANIFEST_SHA, 'original tensor manifest changed')
        require(sha_file(OWN/'source_header.bin') == HEADER_SHA, 'original exact header changed')
        original = json.loads((OWN/'source_tensors.json').read_text())
        header = (OWN/'source_header.bin').read_bytes()
        require(len(header) == 8 + struct.unpack('<Q', header[:8])[0], 'invalid original header length')
        target_header = json.loads(header[8:])
        rows = original['tensors']
        require(original['source_sha256'] == EXPECTED and original['bytes'] == SIZE and len(rows) == 416, 'wrong original file identity')
        require(set(rows) == set(target_header) - {'__metadata__'}, 'original tensor/header keys differ')
        record['input_stat_before'] = stat_identity(INPUT)
        record['input_full_sha256'] = sha_file(INPUT)
        with INPUT.open('rb') as f, (OWN/'output.partial').open('xb') as out:
            n = struct.unpack('<Q', f.read(8))[0]
            require(0 < n < 1 << 20, 'unbounded input header')
            source = json.loads(f.read(n))
            alpha_keys = {k.removesuffix('.lora_A.weight') + '.alpha' for k in rows if k.endswith('.lora_A.weight')}
            require(len(alpha_keys) == 208 and set(source)-{'__metadata__'} == set(rows)|alpha_keys, 'local component layout differs')
            position = 0
            full = hashlib.sha256(header)
            out.write(header)
            for key, row in sorted(rows.items(), key=lambda x: x[1]['data_offsets'][0]):
                entry = source[key]
                a, b = row['data_offsets']
                require(row['dtype'] == entry['dtype'] == 'BF16' and row['shape'] == entry['shape'], 'shape/dtype changed: '+key)
                require(row['data_offsets'] == target_header[key]['data_offsets'] and a == position, 'original output offset gap/overlap')
                require(b-a == entry['data_offsets'][1]-entry['data_offsets'][0], 'tensor byte length differs')
                if key.endswith('.lora_B.weight'):
                    base = key.removesuffix('.lora_B.weight')
                    ar = source[base+'.alpha']
                    require(ar['dtype'] == 'F32' and ar['shape'] == [] and ar['data_offsets'][1]-ar['data_offsets'][0] == 4, 'invalid alpha scalar')
                    f.seek(8+n+ar['data_offsets'][0]); alpha = struct.unpack('<f', f.read(4))[0]
                    rank = source[base+'.lora_A.weight']['shape'][0]
                    require(alpha/rank == 0.0625, 'unexpected effective alpha/rank')
                else:
                    require(key.endswith('.lora_A.weight'), 'unexpected tensor role')
                f.seek(8+n+entry['data_offsets'][0])
                left = b-a; digest = hashlib.sha256()
                while left:
                    data = f.read(min(1 << 20, left)); require(bool(data) and len(data) % 2 == 0, 'short BF16 read')
                    left -= len(data)
                    if key.endswith('.lora_B.weight'):
                        data = transform_b(data)
                    digest.update(data); full.update(data); out.write(data)
                actual = digest.hexdigest()
                require(actual == row['tensor_sha256'], 'exact original tensor mismatch: '+key)
                record['tensor_results'][key] = actual
                position = b
            out.flush(); os.fsync(out.fileno())
        require(8+struct.unpack('<Q',header[:8])[0]+position == SIZE, 'output length differs')
        require(full.hexdigest() == EXPECTED, 'whole reconstructed stream SHA differs')
        require((OWN/'output.partial').stat().st_size == SIZE and sha_file(OWN/'output.partial') == EXPECTED, 'independent disk size/SHA differs')
        record['input_stat_after'] = stat_identity(INPUT)
        require(record['input_stat_after'] == record['input_stat_before'], 'input changed during reconstruction')
        os.link(OWN/'output.partial', OWN/'output.ready')
        (OWN/'output.partial').unlink()
        record.update(complete=True, output=str(OWN/'output.ready'), output_bytes=SIZE,
                      output_sha256=EXPECTED, exact_tensors_verified=len(rows), independently_rehashed_disk=True)
    except BaseException as error:
        record['failure'] = repr(error)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic()-started
        record['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save(OWN/('result.json' if record['complete'] else 'failure.json'),record)
    print(json.dumps({k:record[k] for k in ('complete','exact_tensors_verified','output_bytes','output_sha256','elapsed_seconds')}))


if __name__ == '__main__':
    main()
