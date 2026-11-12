"""Bounded immutable small evidence; no gradient/adapter/base weight arrays."""
import hashlib
import io
import json
from pathlib import Path
import tarfile


ARRAY_ROLES = set() | {m+'_'+s for m in
    ('reference','zero','updated','reloaded','merged') for s in ('video','audio','decoder_input')}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def package(root, prep, run, own, out, terminal, audit=None, evidence=None):
    root, prep, run, own, out = map(lambda p: Path(p).resolve(), (root, prep, run, own, out))
    evidence=Path(evidence or run/'eval').resolve()
    if evidence.parent!=run:raise ValueError('producer evidence path escaped run')
    candidates = [out / "finalize_receipt.json"]
    candidates += [out / name for name in ("terminal_audit.json", "auditor.log") if (out / name).is_file()]
    for directory in (prep, own):
        candidates += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".log")]
    candidates += list((prep / "code").glob("*.py")) + list((prep / "code/tests").glob("*.py"))
    admitted_arrays = {}
    if terminal:
        candidates += [run / "queue_status.json"]
        for names in (('report.json','failure.json','partial.json'),('frozen_initial.json',),('frozen_final.json',)):
            chosen=next((evidence/n for n in names if (evidence/n).is_file()),None)
            if chosen is not None:candidates.append(chosen)
        candidates += list((run / "logs").glob("*.log"))
        for role, record in (audit or {}).get("archive_arrays", {}).items():
            if role not in ARRAY_ROLES:
                raise ValueError("unapproved small-array role: " + role)
            p = Path(record["path"]).resolve()
            if not p.is_relative_to(evidence/'arrays') or p.suffix != ".npy" or p.stat().st_size > 1 << 20:
                raise ValueError("small array outside admitted run or size bound")
            admitted_arrays[p] = record["file_sha256"]
            candidates.append(p)
    files = sorted(set(p.resolve() for p in candidates))
    if len(files) > 128:
        raise ValueError("small evidence exceeds 128 files")
    blobs, rows, total = {}, {}, 0
    for path in files:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("archive path outside workspace")
        if path.suffix == ".npy":
            if path not in admitted_arrays:
                raise ValueError("unadmitted array")
        elif path.suffix not in (".py", ".json", ".md", ".log"):
            raise ValueError("artifact outside small allowlist")
        if path.stat().st_size > (64 << 20) - total:
            raise ValueError("uncompressed evidence exceeds 64MiB")
        data = path.read_bytes(); total += len(data)
        if total > 64 << 20:
            raise ValueError("evidence changed/exceeded size bound")
        digest = sha(data)
        if path in admitted_arrays and digest != admitted_arrays[path]:
            raise ValueError("admitted output array changed after CPU audit")
        name = str(path.relative_to(root)); blobs[name] = data
        rows[name] = {"sha256": digest, "bytes": len(data)}
    manifest = {"schema": "openh3.real_adamw_small_evidence.v1", "files": rows,
                "gradient_arrays_included": False, "model_weights_included": False,
                "diagnostic_adapter_checkpoint_included":False,"parameter_arrays_included":False,
                "small_arrays_roles": sorted((audit or {}).get("archive_arrays", {})) if terminal else []}
    blobs["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if total + len(blobs["MANIFEST.json"]) > 64 << 20:
        raise ValueError("manifest exceeds total evidence bound")
    target = out / "evidence.tar.gz"
    with target.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as packed:
            for name, data in sorted(blobs.items()):
                member = tarfile.TarInfo(name); member.size = len(data); member.mode = 0o644; member.mtime = 0
                packed.addfile(member, io.BytesIO(data))
    return {"path": str(target), "sha256": sha(target.read_bytes()), "bytes": target.stat().st_size,
            "members": len(blobs), "uncompressed_source_bytes": total,
            "manifest_sha256": sha(blobs["MANIFEST.json"])}
