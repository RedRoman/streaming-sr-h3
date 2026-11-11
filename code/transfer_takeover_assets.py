"""Copy only listed project assets, hashing each input and verifying existing files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(16 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    ap.add_argument('--source-root', type=Path)
    ap.add_argument('--object-store-staging', action='store_true',
                    help='No rename is available; publish verified manifest last as the completion marker.')
    a = ap.parse_args()
    records = []
    for i, row in enumerate(json.loads(a.manifest.read_text())['files']):
        rel = Path(row['relative'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('unsafe relative path')
        source_rel = Path(row.get('staged_relative', row['relative']))
        if source_rel.is_absolute() or '..' in source_rel.parts:
            raise ValueError('unsafe staged relative path')
        source = a.source_root / source_rel if a.source_root else Path(row['source'])
        dest = a.destination / rel
        expected = row.get('sha256') or digest(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            temp = dest.with_suffix(dest.suffix + '.partial')
            if a.object_store_staging:
                if temp.exists() and digest(temp) == expected:
                    # Reuse an already verified staging object without unsupported rename.
                    records.append({**row, 'sha256': expected, 'bytes': temp.stat().st_size,
                                    'staged_relative': str(temp.relative_to(a.destination))})
                    print(json.dumps({'done': i + 1, 'relative': str(rel), 'reused_staged_object': True}), flush=True)
                    continue
                shutil.copyfile(source, dest)
                if digest(dest) != expected:
                    raise ValueError('copy checksum mismatch: ' + str(dest))
            else:
                if temp.exists():
                    raise ValueError('partial transfer exists: ' + str(temp))
                shutil.copyfile(source, temp)
                if digest(temp) != expected:
                    raise ValueError('copy checksum mismatch: ' + str(temp))
                temp.rename(dest)
        elif digest(dest) != expected:
            raise ValueError('refusing to replace different existing file: ' + str(dest))
        records.append({**row, 'sha256': expected, 'bytes': dest.stat().st_size})
        print(json.dumps({'done': i + 1, 'relative': str(rel), 'bytes': dest.stat().st_size}), flush=True)
    (a.destination / 'transfer_manifest_verified.json').write_text(json.dumps({'schema': 'openh3.transfer.v1', 'files': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
