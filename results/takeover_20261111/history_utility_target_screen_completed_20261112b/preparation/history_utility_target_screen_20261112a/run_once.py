"""Single-use CPU supervisor; inspection is default, execution requires review."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value: raise ValueError(message)


def write(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n"); tmp.replace(path)


def group_memory(pgid):
    """Only read RSS for members of this newly owned process group."""
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit(): continue
        try:
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid: continue  # fields start with state,ppid,pgrp
            rss = next((int(line.split()[1]) * 1024 for line in (proc / "status").read_text().splitlines()
                        if line.startswith("VmRSS:")), 0)
            rows.append({"pid": int(proc.name), "rss_bytes": rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    return rows


def stop_owned(process, grace=30):
    try: os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError: return
    end = time.monotonic() + grace
    while time.monotonic() < end:
        process.poll()
        if not group_memory(process.pid): return
        time.sleep(.2)
    try: os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError: pass
    process.wait(timeout=5)


@contextmanager
def owned_signal_guard(process):
    """External TERM/INT follows the same owned-group cleanup as budget failure."""
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def interrupted(signum, frame):
        raise InterruptedError("supervisor received signal " + str(signum))
    for sig in previous: signal.signal(sig, interrupted)
    try:
        yield
    finally:
        # Disable repeated termination while cleaning only the newly owned PG.
        for sig in previous: signal.signal(sig, signal.SIG_IGN)
        try:
            if process.poll() is None or group_memory(process.pid):
                stop_owned(process, grace=30)
        finally:
            for sig, handler in previous.items(): signal.signal(sig, handler)


def validate_freeze(path, expected):
    require(sha(path) == expected, "freeze SHA differs")
    config = json.loads(path.read_text())
    require(config["schema"] == "openh3.history_utility_cpu_freeze.v1" and config["host"] == "31984", "wrong freeze host/schema")
    require(config["wall_seconds"] == 1800 and config["grace_seconds"] == 30 and config["memory_mib"] == 768,
            "CPU limits changed")
    for p, digest in config["pins"].items(): require(sha(p) == digest, "pin changed: " + p)
    require(config["pins"].get(str(Path(__file__).resolve())) == sha(__file__), "supervisor not pinned")
    root = Path(config["root"]).resolve()
    require(Path(config["run"]).parent == root / "runs" and Path(config["supervisor"]).parent == root / "runs",
            "run namespace outside owned root")
    require(config["run"] != config["supervisor"], "worker/supervisor directories collide")
    require(config["command"][-2:] == ["--out", config["run"]] and "--execute" not in config["command"], "unexpected command")
    require(config["command"][1] in config["pins"] and config["execution_contract"] in config["pins"], "execution not pinned")
    return config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True); p.add_argument("--freeze-sha256", required=True)
    p.add_argument("--execute", action="store_true"); p.add_argument("--admission", type=Path)
    args = p.parse_args(); config = validate_freeze(args.freeze, args.freeze_sha256)
    env = os.environ.copy()
    env.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    command = list(config["command"])
    if not args.execute:
        command[-1] = config["inspection"]
        require(not Path(command[-1]).exists(), "inspection already exists; no rerun/overwrite")
        result = subprocess.run(command, cwd=config["root"], env=env, timeout=90)
        raise SystemExit(result.returncode)
    require(args.admission is not None, "internal review admission is required")
    admission = json.loads(args.admission.read_text())
    require(admission.get("schema") == "openh3.history_utility_cpu_admission.v1" and admission.get("approved") is True
            and admission.get("freeze_sha256") == args.freeze_sha256 and admission.get("host") == "31984",
            "execution was not admitted for these exact frozen bytes")
    require(Path(config["inspection"]).is_file(), "actual default inspection missing")
    inspection = json.loads(Path(config["inspection"]).read_text())
    require(inspection["status"] == "inspected_not_materialized" and inspection["source_file_SHA_verified"] == 40,
            "inspection did not pass")
    require(not Path(config["run"]).exists(), "worker run already exists")
    supervisor = Path(config["supervisor"])
    supervisor.mkdir(parents=True, exist_ok=False)
    mem = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    require(mem >= 3 << 30 and shutil.disk_usage(supervisor).free >= 6 << 30, "insufficient free CPU memory/disk")
    affinity = sorted(os.sched_getaffinity(0)); require(bool(affinity), "no available CPU affinity")
    os.sched_setaffinity(0, {affinity[0]})
    command.append("--execute")
    status = {"schema": "openh3.history_utility_cpu_supervision.v1", "state": "starting", "complete": False,
        "freeze_sha256": args.freeze_sha256, "admission_sha256": sha(args.admission), "command": command,
        "cpu_affinity": [affinity[0]], "wall_seconds": 1800, "grace_seconds": 30, "memory_mib": 768,
        "available_memory_before_bytes": mem, "peak_owned_group_rss_bytes": 0, "owned_group_only": True}
    write(supervisor / "status.json", status)
    with (supervisor / "worker.log").open("xb") as log:
        process = subprocess.Popen(command, cwd=config["root"], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        started = time.monotonic(); status.update(state="running", worker_pid=process.pid, owned_pgid=process.pid)
        write(supervisor / "status.json", status)
        with owned_signal_guard(process):
            try:
                while process.poll() is None:
                    used = sum(r["rss_bytes"] for r in group_memory(process.pid))
                    status["peak_owned_group_rss_bytes"] = max(status["peak_owned_group_rss_bytes"], used)
                    if used > 768 << 20: raise MemoryError("owned worker+FFmpeg RSS exceeded768MiB")
                    if time.monotonic() - started >= 1800: raise TimeoutError("owned worker wall exceeded1800s")
                    time.sleep(.2)
                require(not group_memory(process.pid), "worker left live owned descendants")
                complete = Path(config["run"]) / "complete.json"
                require(process.returncode == 0 and complete.is_file(), "worker failed or completion missing")
                report = json.loads(complete.read_text()); require(report["complete"] is True, "worker report is incomplete")
                status.update(state="completed", complete=True, returncode=process.returncode, report_sha256=sha(complete))
            except BaseException as exc:
                stop_owned(process, grace=30)
                status.update(state="failed", complete=False, returncode=process.poll(), error={"type": type(exc).__name__, "message": str(exc)})
            finally:
                status["elapsed_seconds"] = time.monotonic() - started
                write(supervisor / "status.json", status)
    if not status["complete"]: raise SystemExit(1)


if __name__ == "__main__": main()
