"""Reserve a deterministic, project-new DIV2K cohort without image inspection.

Only archive metadata, PNG headers, CRC and SHA are read. No image decoding,
model inference, metric, crop choice or visual/source-content selection occurs.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import time
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/confirmation_div2k'
URL = 'https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip'
SALT = 'h3-confirmation-20261015-v1'
COUNT = 32


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(URL, method='HEAD'), timeout=30) as r:
        size = int(r.headers['Content-Length']); etag = r.headers.get('ETag')
        modified = r.headers.get('Last-Modified')
    identity = {'url': URL, 'bytes': size, 'etag': etag, 'last_modified': modified}
    existing = OUT / 'archive_identity.json'
    if existing.exists():
        assert json.loads(existing.read_text()) == identity, 'Archive changed; do not silently mix versions'
    else:
        existing.write_text(json.dumps(identity, indent=2) + '\n')

    def fetch(lo, hi):
        assert 0 <= lo <= hi < size and hi - lo < 16 * 1024**2
        headers = {'Range': f'bytes={lo}-{hi}', 'Accept-Encoding': 'identity'}
        if etag:
            headers['If-Match'] = etag
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=45) as r:
                    assert r.status == 206 and r.headers.get('Content-Range') == f'bytes {lo}-{hi}/{size}'
                    assert not etag or r.headers.get('ETag') == etag
                    raw = r.read(hi - lo + 2)
                    assert len(raw) == hi - lo + 1
                    return raw
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
                time.sleep(1 + attempt)

    tail = fetch(max(0, size - 65557), size - 1)
    pos = tail.rfind(b'PK\x05\x06'); assert pos >= 0
    record = struct.unpack_from('<4s4H2IH', tail, pos)
    count, cdsize, offset = record[4:7]
    assert count < 65535 and offset < 0xffffffff
    central = fetch(offset, offset + cdsize - 1)
    entries = []; pos = 0
    while pos < len(central):
        h = struct.unpack_from('<4s6H3I5H2I', central, pos); assert h[0] == b'PK\x01\x02'
        name = central[pos+46:pos+46+h[10]].decode('utf-8')
        entries.append({'name': name, 'method': h[4], 'crc32': h[7], 'compressed_bytes': h[8],
                        'uncompressed_bytes': h[9], 'local_offset': h[16]})
        pos += 46 + h[10] + h[11] + h[12]
    assert len(entries) == count
    images = [e for e in entries if e['name'].endswith('.png')]
    assert len(images) == 100
    ranked = sorted(images, key=lambda e: sha((SALT + '/' + Path(e['name']).name).encode()))
    chosen = ranked[:COUNT]
    plan = {'status': 'sealed_for_confirmation', 'selection': 'First32 SHA256(salt/basename), no pixel/content/metric selection',
            'salt': SALT, 'archive': identity, 'central_sha256': sha(central), 'entries': chosen,
            'official_split': 'DIV2K validation', 'project_role': 'New-source confirmation after candidate/protocol freeze',
            'restriction': 'No training, tuning, image viewing or model evaluation until method/checkpoints/metrics are fixed',
            'scope': 'Project-new sources; no claim about foundation-model pretraining exposure'}
    (OUT / 'sealed_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (OUT / 'archive_index.json').write_text(json.dumps({'entries': entries, **identity}, indent=2) + '\n')
    folder = OUT / 'sealed_HR'; folder.mkdir(exist_ok=True)

    def download(item):
        path = folder / Path(item['name']).name
        if path.exists():
            raw = path.read_bytes()
        else:
            lo = item['local_offset']
            header = fetch(lo, lo + 29)
            h = struct.unpack('<4s5H3I2H', header); assert h[0] == b'PK\x03\x04'
            meta = fetch(lo + 30, lo + 30 + h[9] + h[10] - 1)
            assert meta[:h[9]].decode('utf-8') == item['name']
            start = lo + 30 + h[9] + h[10]; end = start + item['compressed_bytes']
            chunks = [fetch(a, min(end, a + 8*1024**2) - 1) for a in range(start, end, 8*1024**2)]
            packed = b''.join(chunks)
            assert item['method'] in (0, 8)
            raw = zlib.decompress(packed, -15) if item['method'] == 8 else packed
        assert len(raw) == item['uncompressed_bytes'] and zlib.crc32(raw) & 0xffffffff == item['crc32']
        assert raw[:8] == b'\x89PNG\r\n\x1a\n' and raw[12:16] == b'IHDR'
        width, height = struct.unpack('>II', raw[16:24])
        if not path.exists():
            path.write_bytes(raw)
        return {'name': path.name, 'path': str(path.relative_to(ROOT)), 'bytes': len(raw), 'sha256': sha(raw),
                'zip_crc32': f"{item['crc32']:08x}", 'width': width, 'height': height,
                'pixels_decoded_or_viewed': False}

    records = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(download, e) for e in chosen]
        for future in as_completed(futures):
            records.append(future.result())
            (OUT / 'LIVE_STATE.json').write_text(json.dumps({'phase': 'download', 'completed': len(records), 'target': COUNT}, indent=2))
            print(json.dumps({'verified': len(records), 'target': COUNT, 'file': records[-1]['name']}), flush=True)
    manifest = {**plan, 'files': sorted(records, key=lambda x: x['name']), 'download_complete': True,
                'png_count': len(records), 'source_hash_equality_audit': 'New dataset; PNG bytes CRC/SHA checked. Cross-dataset perceptual duplicates not yet inspected.'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'LIVE_STATE.json').write_text(json.dumps({'phase': 'complete_sealed', 'verified': len(records), 'no_model_evaluation': True}, indent=2))
    print('Complete: byte-verified sealed cohort; no pixel inspection/evaluation.', flush=True)


if __name__ == '__main__':
    main()
