"""Bounded CPU audit launcher; a fresh audit.log is mandatory."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

base = Path(__file__).resolve().parent
script = base / "audit_input_only.py"
expected = "b0fd9700922ada5a73be056b2353f82ead56b477da256d4f4fc7a2ec3f21e32b"
assert hashlib.sha256(script.read_bytes()).hexdigest() == expected
environment = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
               "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
with (base / "audit.log").open("x") as stream:
    result = subprocess.run([
        "timeout", "--signal=TERM", "--kill-after=15s", "285s",
        "/cache/zihaohe/omni_openh3_20261111/venv/bin/python", str(script)],
        cwd=base, env=environment, stdout=stream, stderr=subprocess.STDOUT)
receipt = {"returncode": result.returncode,
           "finished_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
           "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "audit_log_sha256": hashlib.sha256((base / "audit.log").read_bytes()).hexdigest(),
           "model_forwards": 0, "NPU_calls": 0}
with (base / "audit_execution_receipt.json").open("x") as stream:
    json.dump(receipt, stream, indent=2)
    stream.write("\n")
print(json.dumps(receipt), flush=True)
raise SystemExit(result.returncode)
