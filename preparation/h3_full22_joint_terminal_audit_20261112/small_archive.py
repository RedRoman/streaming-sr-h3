"""Bounded immutable small evidence; excludes every gradient and model weight."""
import hashlib
import io
import json
from pathlib import Path
import tarfile


ARRAY_ROLES = {"reference_video", "candidate_video", "reference_audio", "candidate_audio",
               "reference_decoder_cast", "candidate_decoder_cast", "reference_stage_decoder_cast", "candidate_stage_decoder_cast", "guide", "raw_warm_start", "packed_initial", "packed_noise"}


def sha(value):
    return hashlib.sha256(value).hexdigest()


def package(root, prep, run, own, out, terminal, audit=None):
    root, prep, run, own, out = map(lambda p: Path(p).resolve(), (root, prep, run, own, out))
    candidates = [out / "finalize_receipt.json"]
    candidates += [out / name for name in ("terminal_audit.json", "auditor.log") if (out / name).is_file()]
    for directory in (prep, own):
        candidates += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".log")]
    candidates += list((prep / "code").glob("*.py")) + list((prep / "code/tests").glob("*.py"))
    admitted_arrays = {}
    if terminal:
        candidates += [run / "queue_status.json"]
        candidates += list((run / "eval").glob("*.json")) + list((run / "logs").glob("*.log"))
        for role, record in (audit or {}).get("archive_arrays", {}).items():
            if role not in ARRAY_ROLES:
                raise ValueError("unapproved small-array role: " + role)
            p = Path(record["path"]).resolve()
            if not p.is_relative_to(run / "eval") or p.suffix != ".npy" or p.stat().st_size > 1 << 20:
                raise ValueError("small array outside admitted run or size bound")
            admitted_arrays[p] = record["file_sha256"]
            candidates.append(p)
    files = sorted(set(p.resolve() for p in candidates))
    if len(files) > 100:
        raise ValueError("small evidence exceeds 100 files")
    blobs, rows, total = {}, {}, 0
    for path in files:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("archive path outside workspace")
        if path.suffix == ".npy":
            if path not in admitted_arrays:
                raise ValueError("unadmitted array")
        elif path.suffix not in (".py", ".json", ".md", ".log"):
            raise ValueError("artifact outside small allowlist")
        if path.stat().st_size > (32 << 20) - total:
            raise ValueError("uncompressed evidence exceeds 32MiB")
        data = path.read_bytes(); total += len(data)
        if total > 32 << 20:
            raise ValueError("evidence changed/exceeded size bound")
        digest = sha(data)
        if path in admitted_arrays and digest != admitted_arrays[path]:
            raise ValueError("admitted output array changed after CPU audit")
        name = str(path.relative_to(root)); blobs[name] = data
        rows[name] = {"sha256": digest, "bytes": len(data)}
    manifest = {"schema": "openh3.full22_joint_rgb_small_evidence.v1", "files": rows,
                "gradient_arrays_included": False, "model_weights_included": False,
                "small_arrays_roles": sorted((audit or {}).get("archive_arrays", {})) if terminal else []}
    blobs["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if total + len(blobs["MANIFEST.json"]) > 32 << 20:
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
