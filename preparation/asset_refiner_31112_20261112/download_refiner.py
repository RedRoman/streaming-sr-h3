"""One fixed public Refiner object, HTTPS, no retry/resume, no model use."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

URL = "https://hf-mirror.com/Kijai/MiniMax-H3-experimental/resolve/c2e7b89b1800123c49a0f620a1932c01d8952256/loras/minimax_h3_open_regenerate_2k_refiner_rank_384_bf16.safetensors"
REVISION = "c2e7b89b1800123c49a0f620a1932c01d8952256"
EXPECTED_BYTES = 5868409512
EXPECTED_SHA = "4bf30291343eef5e51ee867ba37dd31bce3133d1da1ace1a245a7130b88818c2"
PARTIAL = "open_regen_2k_refiner_rank384_bf16.safetensors.partial"
READY = "open_regen_2k_refiner_rank384_bf16.safetensors.ready"
WALL_SECONDS = 7200


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def exact_copy(response, destination, expected_bytes, progress):
    digest, count = hashlib.sha256(), 0
    while count < expected_bytes:
        block = response.read(min(1 << 20, expected_bytes - count))
        if not block:
            raise ValueError("early EOF: %d of %d bytes" % (count, expected_bytes))
        destination.write(block); digest.update(block); count += len(block)
        progress(count)
    if response.read(1):
        raise ValueError("body exceeds fixed maximum size")
    return count, digest.hexdigest()


def publish_verified(partial, ready, expected_bytes, expected_sha):
    """Atomic no-overwrite publication, then remove the partial name.

    A hard link fails if READY exists. Unlike os.rename on POSIX, it cannot
    silently overwrite a preexisting target. Both names are in the same dir.
    """
    if partial.stat().st_size != expected_bytes or file_sha(partial) != expected_sha:
        raise ValueError("on-disk size/full SHA does not match fixed object")
    os.link(partial, ready)
    partial.unlink()


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file, code, message, headers, newurl):
        return None


def open_fixed_object(record):
    opener = urllib.request.build_opener(NoRedirect(), urllib.request.HTTPSHandler(context=ssl.create_default_context()))
    current = URL
    for hop in range(2):
        parsed = urllib.parse.urlsplit(current)
        expected_host = "hf-mirror.com" if hop == 0 else "cas-bridge.xethub.hf.co"
        if (parsed.scheme != "https" or parsed.hostname != expected_host
                or parsed.port not in (None, 443) or parsed.username or parsed.password):
            raise ValueError("unexpected HTTPS redirect destination")
        request = urllib.request.Request(current, headers={"Accept-Encoding": "identity", "User-Agent": "OpenH3-fixed-asset-preparation/1"})
        try:
            response = opener.open(request, timeout=30)
        except urllib.error.HTTPError as error:
            response = error
        headers = response.headers; status = response.code
        record["HTTP_hops"].append({"hop": hop, "host": parsed.hostname, "status": status,
            "content_length": headers.get("Content-Length")})
        if hop == 0:
            try:
                if (status != 302 or headers.get("X-Linked-Etag", "").strip('"') != EXPECTED_SHA
                        or headers.get("X-Linked-Size") != str(EXPECTED_BYTES)
                        or headers.get("X-Repo-Commit") != REVISION):
                    raise ValueError("fixed revision redirect size/SHA/commit metadata differs")
                current = urllib.parse.urljoin(current, headers["Location"])
            finally:
                response.close()
        else:
            if (status != 200 or headers.get("Content-Length") != str(EXPECTED_BYTES)
                    or headers.get("Content-Encoding", "identity") != "identity"):
                response.close()
                raise ValueError("final status, size or content encoding differs")
            record["final_CDN_ETag_not_file_SHA"] = headers.get("ETag")
            return response
    raise AssertionError("unreachable")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-sha256", required=True)
    args = parser.parse_args()
    own = Path(__file__).resolve().parent
    if file_sha(__file__) != args.self_sha256:
        raise ValueError("downloader source differs from frozen launch")
    partial, ready = own / PARTIAL, own / READY
    record = {"schema": "openh3.refiner_asset_download.v1", "state": "starting", "complete": False,
              "pid": os.getpid(), "host": "huawei-modelarts-31112", "hostname": os.uname().nodename,
              "started_utc": utc(), "source_sha256": args.self_sha256, "public_pinned_URL": URL,
              "revision": REVISION, "expected_bytes": EXPECTED_BYTES, "expected_sha256": EXPECTED_SHA,
              "partial_path": str(partial), "ready_path": str(ready), "max_wall_seconds": WALL_SECONDS,
              "socket_timeout_seconds": 30, "TLS_certificate_verified": True, "HTTP_hops": [],
              "download_attempts": 1, "automatic_retries": 0, "resume": False,
              "model_calls": 0, "NPU_calls": 0, "manifest_mutations": 0, "shared_asset_mutations": 0,
              "downloaded_bytes": 0, "publication": "exclusive hardlink then unlink partial, no overwrite"}
    write_new_json(own / "invocation.json", record)
    started, last_save = time.monotonic(), [0.0]
    def save():
        record.update(updated_utc=utc(), elapsed_seconds=time.monotonic() - started)
        temp = own / "status.json.tmp"
        temp.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        os.replace(temp, own / "status.json")
    def interrupted(signum, frame):
        raise TimeoutError("bounded download interrupted by signal %d" % signum)
    signal.signal(signal.SIGALRM, interrupted); signal.signal(signal.SIGTERM, interrupted)
    signal.alarm(WALL_SECONDS); save()
    try:
        if partial.exists() or partial.is_symlink() or ready.exists() or ready.is_symlink():
            raise FileExistsError("existing partial/ready refused; no overwrite or resume")
        if shutil.disk_usage(own).free < EXPECTED_BYTES + (256 << 20):
            raise OSError("insufficient free disk for bounded object")
        record["state"] = "connecting"; save()
        with open_fixed_object(record) as response, partial.open("xb") as destination:
            record["state"] = "downloading"; save()
            def progress(count):
                record["downloaded_bytes"] = count
                now = time.monotonic()
                if now - last_save[0] >= 15:
                    last_save[0] = now; save()
                    print(json.dumps({"state": record["state"], "bytes": count, "elapsed_seconds": record["elapsed_seconds"]}), flush=True)
            count, stream_sha = exact_copy(response, destination, EXPECTED_BYTES, progress)
            destination.flush(); os.fsync(destination.fileno())
        record["stream_sha256"] = stream_sha
        if count != EXPECTED_BYTES or stream_sha != EXPECTED_SHA:
            raise ValueError("downloaded size/stream SHA mismatch")
        record["state"] = "verifying_on_disk"; save()
        publish_verified(partial, ready, EXPECTED_BYTES, EXPECTED_SHA)
        record.update(state="completed", complete=True, actual_bytes=ready.stat().st_size,
                      actual_file_sha256=EXPECTED_SHA, ready_published=True, full_file_rehashed=True)
    except BaseException as error:
        record.update(state="failed", complete=False, error_type=type(error).__name__, error=str(error),
                      partial_preserved=partial.exists(), partial_bytes=partial.stat().st_size if partial.exists() else 0)
    finally:
        signal.alarm(0); record["finished_utc"] = utc(); save()
        write_new_json(own / "receipt.json", record)
    print(json.dumps({key: record.get(key) for key in ("state", "complete", "pid", "downloaded_bytes", "actual_file_sha256", "error")}), flush=True)
    return 0 if record["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
