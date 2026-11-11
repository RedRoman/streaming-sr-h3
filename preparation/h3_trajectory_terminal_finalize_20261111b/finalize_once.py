"""One bounded CPU-only waiter/auditor/archiver for an already launched H3 probe.

Never imports Torch/H3, invokes npu-smi, starts/restarts/stops model jobs, reads
weights, or copies gradient arrays. An existing finalizer output refuses a new
run. No retry after an audit failure. Wait limit2400s, audit timeout180s.
"""
import argparse
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


MAX_BYTES = 16 << 20


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temp, path)


def wait_terminal(status_path, notify, *, max_wait=2400, poll=30,
                  clock=time.monotonic, sleep=time.sleep):
    """Bounded reads of this queue JSON only. Tests inject a clock; CLI is fixed."""
    start, count, last = clock(), 0, None
    while True:
        count += 1
        try:
            raw = Path(status_path).read_bytes()
            last = {"record": json.loads(raw), "sha256": sha_bytes(raw)}
        except (FileNotFoundError, json.JSONDecodeError) as error:
            last = {"record": None, "read_error": repr(error)}
        elapsed = clock() - start
        notify(count, elapsed, last)
        record = last.get("record")
        if record is not None and record.get("state") in ("completed", "failed"):
            return {"terminal": True, "polls": count, "seconds": elapsed, **last}
        if elapsed >= max_wait:
            return {"terminal": False, "polls": count, "seconds": elapsed, **last}
        sleep(min(poll, max_wait - elapsed))


def selected_files(root, run, prep, own_prep, out, terminal):
    """Bounded allowlist: no recursive run scan, weights, NPY, or model trees."""
    root = Path(root).resolve()
    candidates = [out / "finalize_receipt.json"]
    candidates += [out / name for name in ("terminal_audit.json", "auditor.log") if (out / name).is_file()]
    for directory in (prep, own_prep):
        candidates += [p for p in directory.iterdir() if p.is_file() and p.suffix in (".json", ".py", ".md", ".log")]
    if (prep / "code").is_dir():
        candidates += list((prep / "code").glob("*.py"))
        candidates += list((prep / "code/tests").glob("*.py"))
    if terminal:
        candidates += [run / "queue_status.json"]
        candidates += list((run / "logs").glob("*.log"))
        candidates += list((run / "eval").glob("*.json"))
        # Only final/reference tiny latent NPZs are relevant; never any *.npy.
        for name in ("latent_outputs.npz", "reference_latents.npz"):
            path = run / "eval" / name
            if path.is_file():
                if path.stat().st_size > 1 << 20:
                    raise ValueError("unexpectedly large final latent NPZ: " + str(path))
                candidates.append(path)
    unique = sorted({p.resolve() for p in candidates})
    if len(unique) > 100:
        raise ValueError("small-evidence file-count bound exceeded")
    for path in unique:
        if not path.is_relative_to(root) or not path.is_file() or path.suffix not in (".py", ".json", ".md", ".log", ".npz"):
            raise ValueError("artifact outside bounded allowlist: " + str(path))
        if any(part in ("adapter_gradients", "adapter_gradients_on_failure", "assets") for part in path.parts):
            raise ValueError("gradient/weight material excluded: " + str(path))
    return unique


def archive_small(root, paths, target, max_bytes=MAX_BYTES):
    """Read once into a bounded16MiB snapshot; manifest describes packed bytes."""
    root, target = Path(root).resolve(), Path(target)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    records, blobs, total = {}, {}, 0
    for path in paths:
        path = Path(path).resolve()
        name = str(path.relative_to(root))
        size = path.stat().st_size
        if size > max_bytes - total:
            raise ValueError("uncompressed small-evidence16MiB limit exceeded")
        data = path.read_bytes()
        total += len(data)
        if len(data) != size or total > max_bytes:
            raise ValueError("evidence changed size or exceeded bound")
        records[name] = {"sha256": sha_bytes(data), "bytes": len(data)}
        blobs[name] = data
    manifest = {"schema": "openh3.latent_terminal_small_evidence.v1", "files": records,
                "files_bytes": total, "gradient_arrays_included": False,
                "model_weights_included": False,
                "original_run_modified": False}
    blobs["MANIFEST.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    if total + len(blobs["MANIFEST.json"]) > max_bytes:
        raise ValueError("manifest exceeds archive byte budget")
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("xb") as stream:
        with tarfile.open(fileobj=stream, mode="w:gz") as archive:
            for name, data in sorted(blobs.items()):
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(data), 0o644, 0
                archive.addfile(item, io.BytesIO(data))
    os.replace(temporary, target)
    return {"path": str(target), "sha256": sha_file(target), "bytes": target.stat().st_size,
            "members": len(blobs), "uncompressed_source_bytes": total,
            "manifest_sha256": sha_bytes(blobs["MANIFEST.json"])}


def load_admission(root, own_prep, config_path, expected_sha):
    """Explicit new-run admission; no inferred or auto-updated producer pins."""
    root, own_prep, config_path = map(lambda p: Path(p).resolve(), (root, own_prep, config_path))
    if not own_prep.is_relative_to(root) or config_path.parent != own_prep:
        raise ValueError("config and finalizer must be in the same isolated workspace preparation")
    if sha_file(config_path) != expected_sha:
        raise ValueError("admission config SHA mismatch")
    config = json.loads(config_path.read_text())
    keys = {"schema", "run", "prep", "out", "auditor_sha256", "finalizer_sha256",
            "execution_contract_sha256", "queue_sha256", "probe_sha256", "trajectory_sha256"}
    if set(config) != keys or config["schema"] != "openh3.latent_finalizer_admission.v1":
        raise ValueError("unexpected admission schema/fields")
    for key in keys:
        if key.endswith("sha256"):
            value = config[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("invalid admitted SHA: " + key)
    paths = {}
    for key, prefix in (("run", "runs"), ("prep", "preparation"), ("out", "runs")):
        path = Path(config[key])
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != prefix or path.parts[1] in (".", ".."):
            raise ValueError("expected bounded workspace-relative path: " + key)
        paths[key] = (root / path).resolve()
        if not paths[key].is_relative_to(root):
            raise ValueError("admitted path escapes workspace: " + key)
    if paths["run"] == paths["out"] or paths["out"].is_relative_to(paths["run"]):
        raise ValueError("finalizer output overlaps producer run")
    if paths["prep"] == own_prep:
        raise ValueError("finalizer preparation must be separate from producer")
    return config, paths


def verify_runtime_pins(config, paths, own_prep):
    for path, expected in ((Path(__file__), config["finalizer_sha256"]),
                           (own_prep / "audit_terminal.py", config["auditor_sha256"]),
                           (paths["prep"] / "execution_contract.json", config["execution_contract_sha256"]),
                           (paths["prep"] / "queue.json", config["queue_sha256"])):
        if sha_file(path) != expected:
            raise ValueError("frozen finalizer/auditor/producer-admission SHA mismatch: " + str(path))
    contract = json.loads((paths["prep"] / "execution_contract.json").read_text())
    if Path(contract["root"]).resolve() != paths["prep"].parent.parent:
        raise ValueError("execution contract root differs from finalizer workspace")
    if contract["probe_sha256"] != config["probe_sha256"] or config["trajectory_sha256"] not in contract["implementation_sha256"].values():
        raise ValueError("producer/trajectory pin does not match the admitted execution contract")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    root, own_prep = args.root.resolve(), Path(__file__).resolve().parent
    config, paths = load_admission(root, own_prep, args.config, args.config_sha256)
    run, prep, out = (paths[key] for key in ("run", "prep", "out"))
    out.mkdir(parents=True, exist_ok=False)  # atomic duplicate/restart rejection
    status_path = out / "finalize_status.json"
    began = time.monotonic()
    status = {"schema": "openh3.latent_cpu_finalizer.v2", "state": "starting",
              "pid": os.getpid(), "started_at_utc": now(), "run": str(run),
              "wait_limit_seconds": 2400, "poll_seconds": 30, "audit_timeout_seconds": 180,
              "audit_invocations": 0, "NPU_calls": 0, "model_job_interventions": 0,
              "complete": False, "error": None,
              "admission": config, "admission_sha256": args.config_sha256}

    def save():
        status["updated_at_utc"] = now()
        status["elapsed_seconds"] = time.monotonic() - began
        atomic_json(status_path, status)

    save()
    terminal = False
    try:
        auditor = own_prep / "audit_terminal.py"
        verify_runtime_pins(config, paths, own_prep)
        status["state"] = "waiting_for_terminal"

        def notify(count, seconds, last):
            status.update({"polls": count, "wait_seconds": seconds, "last_queue_observation": last})
            save()

        observation = wait_terminal(run / "queue_status.json", notify)
        terminal = observation["terminal"]
        status["terminal_observation"] = observation
        if not terminal:
            raise TimeoutError("2400s wait expired; producer left untouched; audit not invoked")
        verify_runtime_pins(config, paths, own_prep)
        status["state"] = "auditing_terminal"
        status["audit_invocations"] = 1
        save()
        command = [sys.executable, str(auditor), "--run", str(run / "eval"),
                   "--execution-contract", str(prep / "execution_contract.json"),
                   "--out", str(out / "terminal_audit.json"),
                   "--contract-sha256", config["execution_contract_sha256"],
                   "--probe-sha256", config["probe_sha256"],
                   "--trajectory-sha256", config["trajectory_sha256"]]
        environment = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "1",
                       "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        with (out / "auditor.log").open("x") as stream:
            result = subprocess.run(command, cwd=root, env=environment,
                                    stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        status["audit_returncode"] = result.returncode
        if not (out / "terminal_audit.json").is_file():
            raise ValueError("auditor did not produce its evidence JSON")
        audit = json.loads((out / "terminal_audit.json").read_text())
        status["audit_sha256"] = sha_file(out / "terminal_audit.json")
        status["numerical_acceptance"] = audit["numerical_acceptance"]
        status["producer_terminal_state"] = observation["record"]["state"]
        status["complete"] = bool(audit.get("complete") is True and result.returncode == 0
                                  and status["producer_terminal_state"] == "completed")
        if not status["complete"]:
            status["error"] = "Producer failed or independent numerical/evidence acceptance failed; no promotion/retry"
    except Exception as error:
        status["error"] = repr(error)
        status["complete"] = False
    status["state"] = "completed" if status["complete"] else "failed"
    save()
    # Immutable pre-archive receipt avoids a self-referential archive hash.
    atomic_json(out / "finalize_receipt.json", status)
    try:
        selected = selected_files(root, run, prep, own_prep, out, terminal)
        status["archive"] = archive_small(root, selected, out / "evidence.tar.gz")
    except Exception as error:
        status["state"], status["complete"] = "failed", False
        status["archive_error"] = repr(error)
    save()
    print(json.dumps(status), flush=True)
    return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
