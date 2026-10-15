"""Resumable small-range download; final names require canonical full SHA256."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

import requests

from download_assets import ASSETS, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=list(ASSETS), required=True)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    asset = ASSETS[args.asset]
    args.directory.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    target = args.directory / Path(asset["path"]).name
    partial = target.with_suffix(target.suffix + ".mirror.partial")
    progress_path = target.with_suffix(target.suffix + ".mirror.ranges.json")
    chunk_bytes = 64 * 1024**2
    count = math.ceil(asset["bytes"] / chunk_bytes)
    completed = set()
    started = time.monotonic()
    status = {"asset": args.asset, "artifact": asset, "revision": REVISION,
              "endpoint": args.endpoint, "chunk_bytes": chunk_bytes, "workers": args.workers,
              "status": "running", "pid": os.getpid()}

    def persist():
        status.update(completed_chunks=len(completed), total_chunks=count,
                      complete_bytes=sum(min(chunk_bytes, asset["bytes"] - i * chunk_bytes) for i in completed),
                      elapsed_seconds=time.monotonic() - started)
        temporary = args.status.with_suffix(args.status.suffix + ".tmp")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(args.status)

    def digest(path):
        result = hashlib.sha256()
        with path.open("rb") as handle:
            while block := handle.read(32 * 1024**2):
                result.update(block)
        return result.hexdigest()

    if target.exists():
        if target.stat().st_size != asset["bytes"] or digest(target) != asset["sha256"]:
            raise ValueError("Existing final artifact fails canonical validation")
        status.update(status="complete", reused=True, verified_sha256=asset["sha256"], path=str(target))
        persist()
        print(json.dumps(status), flush=True)
        return
    if partial.exists() and progress_path.exists():
        prior = json.loads(progress_path.read_text())
        if prior["sha256"] != asset["sha256"] or prior["chunk_bytes"] != chunk_bytes:
            raise ValueError("Resume manifest differs from requested artifact")
        completed = set(prior["completed_chunks"])
    descriptor = os.open(partial, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, asset["bytes"])
    lock = threading.Lock()
    url = f"{args.endpoint}/Comfy-Org/MiniMax-H3/resolve/{REVISION}/{asset['path']}"
    proxies = {"http": args.proxy, "https": args.proxy}

    def download(index):
        low, high = index * chunk_bytes, min((index + 1) * chunk_bytes, asset["bytes"]) - 1
        for attempt in range(5):
            try:
                with requests.get(url, headers={"Range": f"bytes={low}-{high}"}, stream=True,
                                  timeout=(20, 120), proxies=proxies) as response:
                    response.raise_for_status()
                    expected = f"bytes {low}-{high}/{asset['bytes']}"
                    if response.status_code != 206 or response.headers.get("content-range") != expected:
                        raise ValueError("Server did not return the requested exact byte range")
                    position = low
                    for block in response.iter_content(4 * 1024**2):
                        if position + len(block) > high + 1:
                            raise ValueError("Range response exceeded requested byte count")
                        if os.pwrite(descriptor, block, position) != len(block):
                            raise OSError("Short positional write")
                        position += len(block)
                    if position != high + 1:
                        raise ValueError("Truncated range response")
                with lock:
                    completed.add(index)
                    update = progress_path.with_suffix(progress_path.suffix + ".tmp")
                    update.write_text(json.dumps({"sha256": asset["sha256"], "chunk_bytes": chunk_bytes,
                                                  "completed_chunks": sorted(completed)}))
                    update.replace(progress_path)
                    persist()
                    print(json.dumps({"event": "range_complete", "asset": args.asset,
                                      "complete_bytes": status["complete_bytes"],
                                      "elapsed_seconds": status["elapsed_seconds"]}), flush=True)
                return
            except (requests.RequestException, OSError, ValueError) as exc:
                print(json.dumps({"event": "range_retry", "asset": args.asset, "index": index,
                                  "attempt": attempt + 1, "error": str(exc)}), flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2**attempt, 10))
    persist()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(download, i) for i in range(count) if i not in completed]):
                future.result()
        os.fsync(descriptor)
        status["status"] = "verifying_sha256"
        persist()
        actual = digest(partial)
        if actual != asset["sha256"]:
            raise ValueError(f"Full canonical SHA256 mismatch: {actual}")
        partial.replace(target)
        status.update(status="complete", verified_sha256=actual, path=str(target))
    except Exception as exc:
        status.update(status="failed", exception=repr(exc))
        raise
    finally:
        os.close(descriptor)
        persist()
        print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
