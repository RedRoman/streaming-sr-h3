"""Foreground, fixed-revision HF ranges with canonical SHA256 verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import subprocess
import time


REVISION = "e5eb578a89295337b8ff433a035929ce0279e0b6"
ASSETS = {
    "vae": {"path": "vae/minimax_h3_video_vae_fp16.safetensors", "bytes": 5207808496,
            "sha256": "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522", "parts": 2},
    "dit": {"path": "diffusion_models/minimax_h3_fl2va_pruned_bf16.safetensors", "bytes": 40225724176,
            "sha256": "a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4", "parts": 8},
}


def emit(value):
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--assets", nargs="+", choices=list(ASSETS), default=["vae", "dit"])
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    manifest = {"revision": REVISION, "status": "running", "assets": {}}

    def piece(asset, index):
        item = ASSETS[asset]
        width = (item["bytes"] + item["parts"] - 1) // item["parts"]
        low, high = index * width, min((index + 1) * width, item["bytes"]) - 1
        path = args.directory / (Path(item["path"]).name + f".hf-range-{index:02d}")
        if path.exists() and path.stat().st_size == high - low + 1:
            return asset, index, path
        url = f"https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/{REVISION}/{item['path']}"
        emit({"event": "range_start", "asset": asset, "index": index, "low": low, "high": high})
        subprocess.run(["curl", "--proxy", args.proxy, "--silent", "--show-error", "--fail",
                        "--location", "--range", f"{low}-{high}", "--retry", "3",
                        "--max-time", "7200", "--output", str(path), url], check=True)
        actual = path.stat().st_size
        if actual != high - low + 1:
            raise ValueError(f"Invalid range size {asset}/{index}: {actual} != {high-low+1}")
        emit({"event": "range_complete", "asset": asset, "index": index, "bytes": actual})
        return asset, index, path

    def assemble(asset, paths):
        item = ASSETS[asset]
        final = args.directory / Path(item["path"]).name
        temporary = args.directory / (final.name + ".verified.part")
        digest = hashlib.sha256()
        count = 0
        with temporary.open("wb") as target:
            for index in sorted(paths):
                with paths[index].open("rb") as source:
                    while chunk := source.read(16 * 1024 * 1024):
                        target.write(chunk)
                        digest.update(chunk)
                        count += len(chunk)
        actual_hash = digest.hexdigest()
        if count != item["bytes"] or actual_hash != item["sha256"]:
            raise ValueError(f"Integrity mismatch {asset}: size={count}, sha256={actual_hash}")
        temporary.replace(final)
        for path in paths.values():
            path.unlink()
        record = {**item, "local_path": str(final), "verified_sha256": actual_hash,
                  "elapsed_seconds": time.monotonic() - started}
        manifest["assets"][asset] = record
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
        emit({"event": "asset_verified", "asset": asset, **record})

    finished = {asset: {} for asset in args.assets}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = {pool.submit(piece, asset, index) for asset in args.assets
                       for index in range(ASSETS[asset]["parts"])}
            while pending:
                completed, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
                for future in completed:
                    asset, index, path = future.result()
                    finished[asset][index] = path
                    if len(finished[asset]) == ASSETS[asset]["parts"]:
                        assemble(asset, finished[asset])
                emit({"event": "progress", "elapsed_seconds": time.monotonic() - started,
                      "pending_ranges": len(pending),
                      "partial_bytes": {asset: sum(p.stat().st_size for p in args.directory.glob(
                          Path(ASSETS[asset]["path"]).name + ".hf-range-*")) for asset in args.assets}})
        manifest["status"] = "complete"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["exception"] = repr(exc)
        raise
    finally:
        manifest["total_elapsed_seconds"] = time.monotonic() - started
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
