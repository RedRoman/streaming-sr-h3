"""One immutable small cache receipt archive; large RGB stays at its source."""
import hashlib
import io
import json
from pathlib import Path
import tarfile


def package(root, prep, run, own, out, audit):
    root, prep, run, own, out = [Path(p).resolve() for p in (root, prep, run, own, out)]
    candidates = [out / "terminal_audit.json"]
    candidates += [p for p in out.iterdir() if p.is_file() and p.name in ("audit.log", "invocation_receipt.json")]
    for directory in (prep, own):
        candidates += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".py", ".json", ".md", ".log")]
    candidates += list((prep / "code").glob("*.py")) + list((prep / "code/tests").glob("*.py"))
    candidates += [p for p in run.glob("*.json") if p.is_file()]
    evidence = Path(audit.get("producer_output", str(run / "eval"))).resolve()
    if evidence.parent != run:
        raise ValueError("producer evidence is not a direct child of the admitted run")
    candidates += list(evidence.glob("*.json")) + list((run / "logs").glob("*.log"))
    admitted = {}
    for role, row in audit.get("archive_arrays", {}).items():
        path = Path(row["path"]).resolve()
        if role not in ("Z_raw", "Z_R") or not path.is_relative_to(run) or path.suffix != ".npy":
            raise ValueError("only admitted Z_raw and Z_R NPY caches may be archived")
        if path.stat().st_size > 2 << 20:
            raise ValueError("latent exceeds individual small-array budget")
        admitted[path] = row["file_sha256"]
        candidates.append(path)
    paths = sorted(set(p.resolve() for p in candidates))
    if len(paths) > 100:
        raise ValueError("small evidence exceeds 100 payload files")
    blobs, rows, total = {}, {}, 0
    for path in paths:
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("small evidence path escaped workspace")
        if path.suffix == ".npy":
            if path not in admitted:
                raise ValueError("unadmitted cache array")
        elif path.suffix not in (".py", ".json", ".md", ".log"):
            raise ValueError("disallowed evidence extension")
        if path.stat().st_size > (16 << 20) - total:
            raise ValueError("small evidence exceeds 16 MiB")
        data = path.read_bytes(); total += len(data)
        if total > 16 << 20:
            raise ValueError("small evidence changed during packaging")
        digest = hashlib.sha256(data).hexdigest()
        if path in admitted and digest != admitted[path]:
            raise ValueError("latent changed after independent CPU audit")
        name = str(path.relative_to(root)); blobs[name] = data
        rows[name] = {"sha256": digest, "bytes": len(data)}
    manifest = {"schema": "openh3.train_vae_cache_portable_small_evidence.v1", "files": rows,
                "large_RGB_included": False, "weights_included": False,
                "input_guide_NPZ_included": False,
                "small_arrays_roles": sorted(audit.get("archive_arrays", {}))}
    payload = (json.dumps(manifest, indent=2) + "\n").encode()
    if total + len(payload) > 16 << 20:
        raise ValueError("manifest exceeds small evidence budget")
    blobs["MANIFEST.json"] = payload
    archive = out / "evidence.tar.gz"
    with archive.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as tar:
            for name, data in sorted(blobs.items()):
                member = tarfile.TarInfo(name); member.size = len(data)
                member.mode = 0o644; member.mtime = 0
                tar.addfile(member, io.BytesIO(data))
    return {"path": str(archive), "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "bytes": archive.stat().st_size, "members": len(blobs),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "uncompressed_payload_bytes": total, "large_RGB_stays_remote": True}
