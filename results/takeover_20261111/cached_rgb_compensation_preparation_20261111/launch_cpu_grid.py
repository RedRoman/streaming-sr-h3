"""Owned CPU worker launcher; invoke inside GNU timeout's dedicated process group."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


config_path = Path(sys.argv[1]).resolve()
config = json.loads(config_path.read_text())
root = Path(config["root"])
for name, expected in config["pins"].items():
    if sha(name) != expected:
        raise RuntimeError("launch pin mismatch: " + name)
runtime = json.loads(Path(config["runtime_manifest"]).read_text())
for name, expected in runtime["files"].items():
    if sha(Path(config["runtime_manifest"]).parent / name) != expected:
        raise RuntimeError("runtime changed after CPU validation: " + name)
inspection = json.loads(Path(config["inspection_receipt"]).read_text())
if not inspection["complete"] or not inspection["inspection_only"] or inspection["contract"]["samples"] != 12:
    raise RuntimeError("completed exact-input preflight required")
if any(row["NPU_initialized"] or row["H3_imported"] for row in inspection["cpu_observations"]):
    raise RuntimeError("preflight CPU isolation gate failed")
run = Path(config["run"])
if run.exists() or run.is_symlink():
    raise FileExistsError(run)
mem = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
allowed = sorted(os.sched_getaffinity(0))
load = os.getloadavg()
if mem["MemAvailable"] < 32 * 1024**2 or len(allowed) < 2 or load[0] >= .9 * len(allowed):
    raise RuntimeError("CPU resource admission unavailable; no worker started")


def cpu_times():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        parts = line.split()
        if parts[0].startswith("cpu") and parts[0][3:].isdigit():
            values = list(map(int, parts[1:]))
            out[int(parts[0][3:])] = (sum(values[:8]), values[3] + values[4])
    return out


before = cpu_times()
time.sleep(1)
after = cpu_times()
busy = {}
for core in allowed:
    total = after[core][0] - before[core][0]
    idle = after[core][1] - before[core][1]
    busy[core] = 1 - idle / total if total else 0.
selected = sorted(sorted(allowed, key=lambda c: (busy[c], c))[:2])
os.sched_setaffinity(0, selected)
os.nice(10)
cap = 24 * 1024**3
resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
os.environ.update({"TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "OMP_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "NUMEXPR_NUM_THREADS": "2",
    "VECLIB_MAXIMUM_THREADS": "2", "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1",
    "TORCH_HOME": str(root / "torch_home"), "TMPDIR": str(root / "tmp"), "SSR0": str(root)})
os.chdir(root)
run.mkdir(parents=True, exist_ok=False)
admission = {"schema": "openh3.cached_compensation_cpu_launch.v1",
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "worker_pid_preserved_by_exec": os.getpid(), "owned_process_group": os.getpgrp(),
    "config_sha256": sha(config_path), "launcher_sha256": sha(__file__),
    "runtime_manifest_sha256": sha(config["runtime_manifest"]),
    "inspection_receipt_sha256": sha(config["inspection_receipt"]),
    "mem_available_kib": mem["MemAvailable"], "loadavg": list(load),
    "cpu_affinity": selected, "selected_cpu_busy_fraction_prior_second": {str(c): busy[c] for c in selected},
    "affinity_is_not_an_exclusive_cpu_reservation": True, "nice": 10,
    "rlimit_as_bytes": list(resource.getrlimit(resource.RLIMIT_AS)), "max_compute_threads": 2,
    "torch_interop_threads": 1, "timeout_seconds": 7200, "kill_after_seconds": 30,
    "timeout_scope": "external GNU timeout own process group in dedicated tmux session",
    "argv": config["argv"], "reuse_partial": None, "new_run": True}
(run / "launch_admission.json").write_text(json.dumps(admission, indent=2) + "\n")
with (run / "grid.log").open("x") as log:
    log.flush()
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    os.execv(config["python"], [config["python"], *config["argv"]])
