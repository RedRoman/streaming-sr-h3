"""Configured bounded CPU-only nonzero terminal finalizer; no model actions."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from small_archive import package


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path); temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def admission(root, own, path, expected_sha):
    root, own, path = map(lambda p: Path(p).resolve(), (root, own, path))
    if not own.is_relative_to(root) or path.parent != own or sha_file(path) != expected_sha:
        raise ValueError("unreviewed admission path/SHA")
    config = json.loads(path.read_text())
    fields = {"schema", "run", "prep", "out", "producer_sha256", "trajectory_sha256",
              "execution_contract_sha256", "queue_sha256", "target_sha256", "source_sha256",
              "source_contract_sha256", "producer_output_subdir"}
    if set(config) != fields or config["schema"] != "openh3.nonzero_adapter_cpu_finalizer_admission.v1":
        raise ValueError("unknown admission schema/fields")
    paths = {}
    for key, parent in (("run", "runs"), ("prep", "preparation"), ("out", "runs")):
        p = Path(config[key])
        if p.is_absolute() or len(p.parts) != 2 or p.parts[0] != parent or ".." in p.parts:
            raise ValueError("unbounded workspace path")
        paths[key] = (root / p).resolve()
        if not paths[key].is_relative_to(root):
            raise ValueError("workspace escape")
    if paths["run"] == paths["out"] or paths["prep"] == own:
        raise ValueError("producer/finalizer path overlap")
    component = config['producer_output_subdir']
    if not isinstance(component,str) or not component or Path(component).is_absolute() or '/' in component or '\\' in component or Path(component).parts != (component,) or component in ('.','..'):
        raise ValueError('producer output must be one safe directory component')
    paths['evidence'] = (paths['run'] / component).resolve()
    if paths['evidence'].parent != paths['run']:
        raise ValueError('producer output escaped run')
    expected_sources = {"audit_terminal.py", "audit_arrays.py", "audit_checkpoint.py", "finalize_once.py", "small_archive.py"}
    if set(config["source_sha256"]) != expected_sources:
        raise ValueError("incomplete finalizer source lock")
    digests = [config[k] for k in fields if k.endswith("sha256") and k != "source_sha256"] + list(config["source_sha256"].values())
    if any(not isinstance(x,str) or len(x) != 64 or any(c not in "0123456789abcdef" for c in x) for x in digests):
        raise ValueError("invalid SHA in admission")
    return config, paths


def verify(config, paths, own):
    pins = {own / name: digest for name, digest in config["source_sha256"].items()}
    pins[paths["prep"] / "execution_contract.json"] = config["execution_contract_sha256"]
    pins[paths["prep"] / "queue.json"] = config["queue_sha256"]
    pins[paths['prep'] / 'code/check_h3_nonzero_adapter.py'] = config['producer_sha256']
    pins[paths['prep'] / 'code/openh3_train_trajectory.py'] = config['trajectory_sha256']
    for path, expected in pins.items():
        if sha_file(path) != expected:
            raise ValueError("frozen finalizer/producer pin mismatch: " + str(path))


def wait_terminal(path, notify, *, max_wait=4200, poll=30, clock=time.monotonic, sleep=time.sleep):
    start, polls = clock(), 0
    while True:
        polls += 1
        try:
            raw = Path(path).read_bytes(); record = json.loads(raw)
            observation = {"record": record, "sha256": hashlib.sha256(raw).hexdigest()}
        except (FileNotFoundError, json.JSONDecodeError) as error:
            observation = {"record": None, "read_error": repr(error)}
        elapsed = clock() - start; notify(polls, elapsed, observation)
        if observation["record"] is not None and observation["record"].get("state") in ("completed", "failed", "not_launched"):
            return {"terminal": True, **observation}
        if elapsed >= max_wait:
            return {"terminal": False, **observation}
        sleep(min(poll, max_wait - elapsed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(); root, own = args.root.resolve(), Path(__file__).resolve().parent
    config, paths = admission(root, own, args.config, args.config_sha256)
    out = paths["out"]; out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    status = {"schema": "openh3.nonzero_adapter_cpu_finalizer.v1", "state": "starting", "pid": os.getpid(),
              "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "admission": config, "admission_sha256": args.config_sha256, "complete": False,
              "wait_limit_seconds": 4200, "poll_seconds": 30, "audit_timeout_seconds": 300,
              "audit_invocations": 0, "NPU_calls": 0, "model_job_interventions": 0}
    def save():
        status["elapsed_seconds"] = time.monotonic() - began
        status["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(out / "finalize_status.json", status)
    save(); terminal, audited = False, None
    try:
        if hasattr(os,'sched_getaffinity'):
            affinity=sorted(os.sched_getaffinity(0))[:2]
            os.sched_setaffinity(0,affinity);status['CPU_affinity']=affinity
        verify(config, paths, own); status["state"] = "waiting_for_terminal"
        def notify(polls, elapsed, record):
            status.update(polls=polls, wait_seconds=elapsed, last_queue_observation=record); save()
        observation = wait_terminal(paths["run"] / "queue_status.json", notify)
        terminal = observation["terminal"]; status["terminal_observation"] = observation
        if not terminal:
            raise TimeoutError("bounded wait expired; producer untouched, auditor not invoked")
        verify(config, paths, own)
        status.update(state="auditing_terminal", audit_invocations=1); save()
        env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with (out / "auditor.log").open("x") as stream:
            result = subprocess.run([sys.executable, str(own / "audit_terminal.py"), "--root", str(root),
                "--config", str(args.config.resolve()), "--config-sha256", args.config_sha256,
                "--out", str(out / "terminal_audit.json")], cwd=root, env=env, timeout=300,
                stdout=stream, stderr=subprocess.STDOUT)
        status["audit_returncode"] = result.returncode
        audited = json.loads((out / "terminal_audit.json").read_text())
        status["audit_sha256"] = sha_file(out / "terminal_audit.json")
        status["numerical_acceptance"] = audited["numerical_acceptance"]
        status["complete"] = (result.returncode == 0 and audited.get("complete") is True and observation["record"]["state"] == "completed")
        if not status["complete"]:
            status["error"] = "Producer failure or independent acceptance failure preserved; no retry"
    except Exception as error:
        status["complete"] = False; status["error"] = repr(error)
    status["state"] = "completed" if status["complete"] else "failed"; save()
    write_json(out / "finalize_receipt.json", status)
    try:
        status["archive"] = package(root, paths["prep"], paths["run"], own, out, terminal, audited,
                                    evidence=paths['evidence'])
    except Exception as error:
        status.update(complete=False, state="failed", archive_error=repr(error))
    save(); print(json.dumps(status)); return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
