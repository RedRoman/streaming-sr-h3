"""Verify and unpack one immutable project backup into a new namespace."""
from pathlib import Path, PurePosixPath
import argparse
import datetime
import hashlib
import json
import tarfile

p = argparse.ArgumentParser()
p.add_argument('--archive', type=Path, required=True)
p.add_argument('--sha256', required=True)
a = p.parse_args()
assert hashlib.sha256(a.archive.read_bytes()).hexdigest() == a.sha256
dest = a.archive.parent / 'verified_payload'
assert not dest.exists(), 'Never overwrite an existing verified backup'
with tarfile.open(a.archive, 'r:gz') as tar:
    members = tar.getmembers()
    names = [m.name for m in members]
    assert len(names) == len(set(names))
    for m in members:
        q = PurePosixPath(m.name)
        assert m.isfile() and not q.is_absolute() and '..' not in q.parts
    raw_manifest = tar.extractfile('MANIFEST.json').read()
    manifest = json.loads(raw_manifest)
    assert set(names) == set(manifest['files']) | {'MANIFEST.json'}
    for rel, expected in manifest['files'].items():
        data = tar.extractfile(rel).read()
        assert len(data) == expected['bytes']
        assert hashlib.sha256(data).hexdigest() == expected['sha256'], rel
    dest.mkdir(exist_ok=False)
    for m in members:
        target = dest / m.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as f:
            f.write(tar.extractfile(m).read())
for rel, expected in manifest['files'].items():
    assert hashlib.sha256((dest / rel).read_bytes()).hexdigest() == expected['sha256']
receipt = dict(schema='openh3.followup_backup_verification.v1',
    complete=True, verified_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    archive=str(a.archive), archive_sha256=a.sha256, archive_bytes=a.archive.stat().st_size,
    files=len(manifest['files']), manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
    payload=str(dest), archive_members_verified=True, extracted_files_verified=True,
    overwritten_files=0)
with (a.archive.parent / 'verification.json').open('x') as f:
    f.write(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
