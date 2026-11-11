"""Single authorized bounded CPU attempt. No resume or automatic retry."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

base = Path(__file__).resolve().parent
snapshot = base / "snapshot"
admission = base / "preflight.json"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(admission) != "1a3715a1abe4ea850658f119badd73a2c95ad1015da887736c3e2e57d0645561":
    raise ValueError("preflight receipt mismatch")
preflight = json.loads(admission.read_text())
for relative, expected in preflight["snapshot_files_sha256"].items():
    if sha(snapshot / relative) != expected:
        raise ValueError("frozen source mismatch: " + relative)
receipt_path = base / "launch_receipt.json"
output = Path(preflight["output"])
if receipt_path.exists() or output.exists():
    raise FileExistsError("one attempt only; preserve any existing run")
command = ["timeout", "--signal=TERM", "--kill-after=5s", "630s", sys.executable,
    "code/materialize_mixed_temporal_pilot.py",
    "--plan", "configs/openh3_20261111/mixed_temporal_data_plan_20261111.json",
    "--execution-contract", "preparation/mixed_temporal_materializer_20261111/execution_contract.json",
    "--train-manifest", "results/takeover_20261111/materialized/train_materialized.jsonl",
    "--dev-manifest", "results/takeover_20261111/materialized/dev_materialized.jsonl",
    "--source", preflight["source_path"], "--source-id", "2BdYaGuVAKo",
    "--mode", "materialize", "--output", str(output), "--max-seconds", "600"]
receipt = {"schema": "openh3.mixed_temporal_smoke_launch.v1", "status": "running",
           "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "preflight_sha256": sha(admission), "launcher_sha256": sha(__file__),
           "snapshot_files_sha256": preflight["snapshot_files_sha256"],
           "command": command, "cwd": str(snapshot), "output": str(output),
           "automatic_retries": 0, "npu_calls": 0}
receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
with (base / "smoke.log").open("xb") as log:
    result = subprocess.run(command, cwd=snapshot, stdout=log, stderr=subprocess.STDOUT)
receipt.update(exit_code=result.returncode,
               status="process_finished_inspect_receipt" if result.returncode == 0 else "failed_preserved_no_retry",
               finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               log_sha256=sha(base / "smoke.log"))
temporary = receipt_path.with_suffix(".partial")
temporary.write_text(json.dumps(receipt, indent=2) + "\n")
temporary.replace(receipt_path)
print(json.dumps(receipt))
raise SystemExit(result.returncode)
