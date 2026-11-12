"""Launch only this owned, bounded public-file preparation process once."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--launcher-sha256", required=True)
    args = parser.parse_args(); own = Path(__file__).resolve().parent
    script = own / "download_refiner.py"
    if digest(script) != args.source_sha256 or digest(__file__) != args.launcher_sha256:
        raise ValueError("source differs from frozen launch")
    timeout = shutil.which("timeout")
    if timeout is None: raise RuntimeError("required external wall-clock limiter unavailable")
    claim = {"schema": "openh3.refiner_download_launch.v1", "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
             "source_sha256": args.source_sha256, "launcher_sha256": args.launcher_sha256,
             "max_wall_seconds": 7200, "cleanup_grace_seconds": 10, "automatic_retries": 0,
             "model_calls": 0, "NPU_calls": 0, "dispatch_count": 1}
    with (own / "launch_claim.json").open("x") as stream:
        stream.write(json.dumps(claim, indent=2) + "\n")
    command = [timeout, "--signal=TERM", "--kill-after=10s", "7200s", sys.executable, "-u", str(script),
               "--self-sha256", args.source_sha256]
    with (own / "download.log").open("x") as stream:
        process = subprocess.Popen(command, cwd=own, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"})
    claim.update(state="dispatched", supervisor_pid=process.pid, process_group=process.pid,
                 command=command, status_path=str(own / "status.json"), receipt_path=str(own / "receipt.json"),
                 log_path=str(own / "download.log"))
    with (own / "launch_receipt.json").open("x") as stream:
        stream.write(json.dumps(claim, indent=2) + "\n")
    print(json.dumps(claim))


if __name__ == "__main__": main()
