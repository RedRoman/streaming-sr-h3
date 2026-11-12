"""Back up changed small evidence after the accepted 14:05 point-in-time snapshot."""
from pathlib import Path
import datetime
import hashlib
import io
import json
import subprocess
import tarfile

root = Path.cwd()
dest = root / "preparation/followup_20261112_full22_terminal"
prior = root / "preparation/followup_20261112_1405/manifest.json"
baseline = json.loads(prior.read_text())["files"]
names = set(subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], text=True).split("\0"))
names.update(subprocess.check_output(["git", "diff", "--name-only", "-z", "HEAD"], text=True).split("\0"))
manifest = {
    "schema": "openh3.followup_small_backup.v1",
    "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "scope": "Delta small source/evidence after full22 completed; large arrays and the separately mirrored terminal archive are excluded",
    "baseline_manifest_sha256": hashlib.sha256(prior.read_bytes()).hexdigest(),
    "files": {},
}
payload = {}
for name in sorted(names):
    if not name or name.startswith("preparation/followup_20261112_full22_terminal/"):
        continue
    path = root / name
    if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
        continue
    if name.endswith(".tar.gz") or path.suffix in (".png", ".npy", ".npz", ".bin"):
        continue
    assert path.suffix in (".py", ".json", ".jsonl", ".log", ".md"), name
    assert path.stat().st_size <= 8 << 20, name
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if baseline.get(name, {}).get("sha256") == digest:
        continue
    manifest["files"][name] = {"bytes": len(data), "sha256": digest}
    payload[name] = data
archive = dest / "backup.tar.gz"
blob = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
with tarfile.open(archive, "x:gz") as tar:
    for name, data in {**payload, "MANIFEST.json": blob}.items():
        item = tarfile.TarInfo(name)
        item.size = len(data)
        tar.addfile(item, io.BytesIO(data))
for name, expected in manifest["files"].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected["sha256"], name
(dest / "manifest.json").write_bytes(blob)
receipt = {
    "files": len(payload), "archive_bytes": archive.stat().st_size,
    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
    "manifest_sha256": hashlib.sha256(blob).hexdigest(),
    "destination_host": "31984",
    "destination": "/cache/zihaohe/omni_openh3_20261111/preparation/followup_20261112_full22_terminal",
}
(dest / "pack_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(receipt))
