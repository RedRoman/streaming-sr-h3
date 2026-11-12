"""Explicit once-only CPU assembly after all 15 transferred parts are present."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import time

MANIFEST_SHA = "46a3825c7f22d7510986e7941caf389eeeaec7a90e9f2587f9446fd5851171f3"
TOTAL_BYTES = 1956171936
TOTAL_SHA = "0b5e411b519ea5041734d44fc102b0dfd6b6a29737048ad1a80a522fb633fa99"
ROOT = Path("/cache/zihaohe/omni_openh3_20261111")
TARGET = ROOT / "assets/openh3_split/openh3_turbo128_comfy.safetensors"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def new_json(path, value):
    with path.open("x") as f: f.write(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-sha256", required=True)
    args = parser.parse_args(); own = Path(__file__).resolve().parent
    if own != ROOT / "preparation/turbo_transfer_31112_20261112" or sha(Path(__file__)) != args.self_sha256:
        raise ValueError("wrong owned workspace or assembly source SHA")
    if sha(own / "manifest.json") != MANIFEST_SHA: raise ValueError("part manifest SHA differs")
    manifest = json.loads((own / "manifest.json").read_text()); rows = manifest["parts"]
    if (manifest["bytes"] != TOTAL_BYTES or manifest["sha256"] != TOTAL_SHA
            or [r["name"] for r in rows] != ["part_%03d" % i for i in range(15)]
            or sum(r["bytes"] for r in rows) != TOTAL_BYTES):
        raise ValueError("exact 15-part identity differs")
    for row in rows:
        p = own / "parts" / row["name"]
        if p.is_symlink() or not p.is_file() or p.stat().st_size != row["bytes"]:
            raise ValueError("not all complete regular part files are present: " + row["name"])
    partial, ready = own / "openh3_turbo128_comfy.safetensors.partial", own / "openh3_turbo128_comfy.safetensors.ready"
    if any(p.exists() or p.is_symlink() for p in (partial, ready)):
        raise FileExistsError("existing assembly refused; inspect prior evidence")
    result = {"schema": "openh3.exact_turbo_asset_assembly.v1", "complete": False,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "manifest_sha256": MANIFEST_SHA, "source_sha256": args.self_sha256,
              "target": str(TARGET), "parts_verified": [], "NPU_calls": 0, "model_calls": 0,
              "existing_assets_overwritten": False, "manifest_modified": False}
    new_json(own / "assembly_claim.json", result); started = time.monotonic()
    def timeout(signum, frame): raise TimeoutError("900-second CPU assembly limit")
    signal.signal(signal.SIGALRM, timeout); signal.alarm(900)
    try:
        if TARGET.exists() or TARGET.is_symlink():
            if not TARGET.is_file() or TARGET.stat().st_size != TOTAL_BYTES or sha(TARGET) != TOTAL_SHA:
                raise FileExistsError("pre-existing manifest target differs; refused overwrite")
            result.update(complete=True, state="already_matching_target", actual_sha256=TOTAL_SHA)
        else:
            whole = hashlib.sha256(); written = 0
            with partial.open("xb") as dst:
                for row in rows:
                    p = own / "parts" / row["name"]; before = p.stat(); ph = hashlib.sha256(); count = 0
                    with p.open("rb") as src:
                        for block in iter(lambda: src.read(8 << 20), b""):
                            if count + len(block) > row["bytes"]: raise ValueError("part grew beyond bound")
                            dst.write(block); ph.update(block); whole.update(block); count += len(block); written += len(block)
                    after = p.stat()
                    if ((before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino)
                            or count != row["bytes"] or ph.hexdigest() != row["sha256"]):
                        raise ValueError("part changed or actual SHA differs: " + row["name"])
                    result["parts_verified"].append(row)
                dst.flush(); os.fsync(dst.fileno())
            if written != TOTAL_BYTES or whole.hexdigest() != TOTAL_SHA or sha(partial) != TOTAL_SHA:
                raise ValueError("assembled full-file bytes/SHA differs")
            os.link(partial, ready); partial.unlink()
            TARGET.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(str(ready), TARGET)
            result.update(complete=True, state="published_new_verified_symlink", actual_bytes=written,
                          actual_sha256=TOTAL_SHA, ready_path=str(ready), symlink_target=os.readlink(TARGET))
    except BaseException as error:
        result.update(state="failed", complete=False, error=repr(error), partial_preserved=partial.exists())
    finally:
        signal.alarm(0); result.update(elapsed_seconds=time.monotonic()-started,
            finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        new_json(own / "assembly_receipt.json", result)
    print(json.dumps(result)); return 0 if result["complete"] else 2


if __name__ == "__main__": raise SystemExit(main())
