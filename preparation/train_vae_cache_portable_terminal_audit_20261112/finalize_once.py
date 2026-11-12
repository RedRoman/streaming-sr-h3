"""One bounded CPU-only wait and one terminal audit; producer is read-only.

The four auditor sources are pinned by admission.json. This wrapper's source
SHA is separately required on its launch command, avoiding any config cycle.
Its fresh status directory is distinct from the fresh auditor output directory.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from audit_terminal import admission
from audit_arrays import file_sha, require


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def wait_terminal(path, notify, *, max_wait=900, poll=30, clock=time.monotonic, sleep=time.sleep):
    began, polls = clock(), 0
    while True:
        polls += 1
        try:
            payload = Path(path).read_bytes(); record = json.loads(payload)
            observation = {"record": record, "sha256": hashlib.sha256(payload).hexdigest()}
        except (FileNotFoundError, json.JSONDecodeError) as error:
            observation = {"record": None, "read_error": repr(error)}
        elapsed = clock() - began; notify(polls, elapsed, observation)
        if observation["record"] is not None and observation["record"].get("state") in ("completed", "failed", "not_launched"):
            return {"terminal": True, **observation}
        if elapsed >= max_wait:
            return {"terminal": False, **observation}
        sleep(min(poll, max_wait - elapsed))


def final_state(queue_state, returncode, audited_status):
    complete = (queue_state == "completed" and returncode == 0 and audited_status.get("complete") is True
                and audited_status.get("state") == "completed")
    return {"complete": complete, "state": "completed" if complete else "failed"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    parser.add_argument("--self-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True, help="new separate runs/<name> status directory")
    args = parser.parse_args(); root = args.root.resolve(); own = Path(__file__).resolve().parent
    require(file_sha(__file__) == args.self_sha256, "finalizer source differs from reviewed launch SHA")
    config, paths = admission(root, own, args.config, args.config_sha256)
    out = (root / args.out).resolve()
    require(out.parent == root / "runs" and out not in (paths["run"], paths["out"]), "finalizer status path overlap/escape")
    out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    status = {"schema": "openh3.train_vae_cache_portable_cpu_finalizer.v1", "state": "starting", "complete": False,
              "pid": os.getpid(), "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "admission_sha256": args.config_sha256, "admission": config,
              "finalizer_source_sha256": args.self_sha256,
              "wait_limit_seconds": 900, "poll_seconds": 30, "audit_timeout_seconds": 180,
              "audit_invocations": 0, "NPU_calls": 0, "model_job_interventions": 0,
              "audit_output": str(paths["out"]), "quality_verdict": "not_assessed"}
    def save():
        status["elapsed_seconds"] = time.monotonic() - began
        status["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(out / "finalize_status.json", status)
    save()
    try:
        require(not paths["out"].exists(), "auditor output already exists; no duplicate invocation")
        status["state"] = "waiting_for_terminal"
        def notify(polls, elapsed, observation):
            status.update(polls=polls, wait_seconds=elapsed, last_queue_observation=observation); save()
        observed = wait_terminal(paths["run"] / "queue_status.json", notify)
        status["terminal_observation"] = observed
        if not observed["terminal"]:
            raise TimeoutError("900-second bounded wait expired; auditor not invoked; producer untouched")
        require(file_sha(__file__) == args.self_sha256, "finalizer source changed while waiting")
        admission(root, own, args.config, args.config_sha256)
        require(not paths["out"].exists(), "auditor output appeared while waiting; no duplicate invocation")
        status.update(state="auditing_terminal", audit_invocations=1); save()
        environment = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OPENBLAS_NUM_THREADS": "1",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with (out / "auditor.log").open("x") as stream:
            result = subprocess.run([sys.executable, str(own / "audit_terminal.py"), "--root", str(root),
                "--config", str(args.config.resolve()), "--config-sha256", args.config_sha256],
                cwd=root, env=environment, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        status["audit_returncode"] = result.returncode
        result_path = paths["out"] / "audit_status.json"
        audited = json.loads(result_path.read_text())
        status["audit_status_sha256"] = file_sha(result_path)
        status["audit_sha256"] = audited.get("audit_sha256")
        status["archive"] = audited.get("archive")
        status.update(final_state(observed["record"]["state"], result.returncode, audited))
        if not status["complete"]:
            status["error"] = "Producer failure or independent CPU audit failure preserved; no retry"
    except Exception as error:
        status.update(state="failed", complete=False, error=repr(error))
    save()
    receipt = out / "finalize_receipt.json"
    with receipt.open("x") as stream:
        stream.write(json.dumps(status, indent=2, allow_nan=False) + "\n")
    print(json.dumps(status)); return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
