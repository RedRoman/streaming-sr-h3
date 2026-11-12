"""One admitted full22 probe on one fixed card; no wait/retry or model imports.

The worker owns a fresh process group. RSS is the simultaneously sampled sum
of that group, not summed historical per-process peaks. Cooperative card locks
do not reserve hardware against other users; both availability checks are real.
"""
import argparse
from contextlib import contextmanager
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


RESOURCE = {"CPU_threads": 4, "wall_seconds": 3600,
    "termination_grace_seconds": 30,
    "owned_process_group_RSS_limit_bytes": 128 << 30,
    "minimum_host_available_bytes": 160 << 30,
    "minimum_disk_free_bytes": 10 << 30, "stable_seconds": 15}


def require(value, message):
    if not value: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    p = Path(path)
    value["updated_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    temp = p.with_suffix(p.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temp, p)


def resolve(root, name):
    p = Path(name)
    return p if p.is_absolute() else root / p


def group_rows(pgid):
    rows = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit(): continue
        try:
            fields = (p / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) != pgid: continue
            rss = next((int(line.split()[1]) * 1024 for line in (p / "status").read_text().splitlines()
                        if line.startswith("VmRSS:")), 0)
            rows.append({"pid": int(p.name), "state": fields[0], "rss_bytes": rss})
        except (FileNotFoundError, ProcessLookupError, PermissionError): pass
    return rows


def live_rows(pgid): return [row for row in group_rows(pgid) if row["state"] != "Z"]


def stop_owned(process, grace):
    """Signal only the PG created by this Popen(start_new_session=True)."""
    result = {"pgid": process.pid, "TERM_sent": False, "KILL_sent": False}
    try: os.killpg(process.pid, signal.SIGTERM); result["TERM_sent"] = True
    except ProcessLookupError: pass
    end = time.monotonic() + grace
    while time.monotonic() < end:
        process.poll()
        if not live_rows(process.pid): break
        time.sleep(.05)
    if live_rows(process.pid):
        try: os.killpg(process.pid, signal.SIGKILL); result["KILL_sent"] = True
        except ProcessLookupError: pass
    try: process.wait(timeout=5)
    except subprocess.TimeoutExpired: result["wait_timeout"] = True
    # Orphan zombies may await PID1 reaping; they cannot execute or hold RSS.
    until = time.monotonic() + 2
    while live_rows(process.pid) and time.monotonic() < until: time.sleep(.05)
    result["remaining_live"] = live_rows(process.pid)
    result["zombies_only"] = not result["remaining_live"] and bool(group_rows(process.pid))
    return result


class Interrupted(RuntimeError): pass


@contextmanager
def signal_guard():
    seen = {"value": False}
    def handler(signum, frame):
        if not seen["value"]:
            seen["value"] = True
            raise Interrupted("supervisor received signal %d" % signum)
    old = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGINT)}
    try: yield
    finally:
        for s, h in old.items(): signal.signal(s, h)


def supervise(command, cwd, env, log, status, status_path, *, wall, rss_limit, grace,
              sample_seconds=.25):
    """Internal low-budget fixtures use this function; production limits are fixed."""
    process, started = None, None
    with signal_guard():
        try:
            with Path(log).open("xb") as output:
                blocked = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTERM, signal.SIGINT})
                try:
                    process = subprocess.Popen(command, cwd=cwd, env=env, stdout=output,
                        stderr=subprocess.STDOUT, start_new_session=True,
                        preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, blocked))
                finally:
                    # Assignment precedes delivery of a pending TERM/INT.
                    signal.pthread_sigmask(signal.SIG_SETMASK, blocked)
            started = time.monotonic()
            status.update(state="running", producer_started=True, worker_pid=process.pid, owned_pgid=process.pid,
                command=command, peak_owned_group_RSS_bytes=0, RSS_samples=0)
            write(status_path, status)
            next_status = started
            while True:
                rows = group_rows(process.pid)
                usage = sum(row["rss_bytes"] for row in rows)
                status["peak_owned_group_RSS_bytes"] = max(status["peak_owned_group_RSS_bytes"], usage)
                status["RSS_samples"] += 1
                elapsed = time.monotonic() - started
                if usage > rss_limit: raise MemoryError("owned process-group RSS exceeded fixed limit")
                if elapsed > wall: raise TimeoutError("owned worker exceeded fixed wall budget")
                code = process.poll()
                if code is not None:
                    require(not live_rows(process.pid), "worker exited leaving live owned descendants")
                    require(code == 0, "worker nonzero return code: %s" % code)
                    status.update(state="worker_completed", returncode=code)
                    break
                if time.monotonic() >= next_status:
                    status["active_worker_seconds"] = elapsed
                    write(status_path, status); next_status = time.monotonic() + 15
                time.sleep(sample_seconds)
        except BaseException as error:
            status.update(state="failed", complete=False,
                error={"type": type(error).__name__, "message": str(error)})
        finally:
            if process is not None:
                # Includes a parent that exited while descendants remained.
                if process.poll() is None or live_rows(process.pid):
                    status["owned_cleanup"] = stop_owned(process, grace)
                status["returncode"] = process.poll()
                remaining = live_rows(process.pid)
                status["remaining_owned_live_processes"] = remaining
                if remaining:
                    status.update(state="failed", complete=False,
                        cleanup_error="owned live processes remain after TERM/KILL")
            status["active_worker_seconds"] = 0 if started is None else time.monotonic() - started
            write(status_path, status)
    return status


def verify_pins(root, pins):
    require(bool(pins), "empty runtime pins")
    return {str(resolve(root, p).resolve()): sha(resolve(root, p)) for p in pins}


def validate(plan_path, *, allow_status=False):
    plan_path = Path(plan_path).resolve(); p = read(plan_path)
    require(p["schema"] == "openh3.full22_bounded_queue.v1", "wrong queue schema")
    root = Path(p["root"])
    require(root.is_absolute() and root.is_dir(), "invalid workspace root")
    require(p["resource"] == RESOURCE, "fixed resource budget differs")
    cards = p["cards"]
    require(len(cards) == 1 and type(cards[0]) is int and 0 <= cards[0] < 8,
            "root must specify exactly one physical card")
    require(p["min_free_mb"] == 60000 and len(p["steps"]) == 1, "one step / 60000MB gate required")
    require(p["lock_dir"] == "/cache/zihaohe/.npu-reservations", "must use the shared cooperating card-lock namespace")
    expected = {str(resolve(root, k).resolve()): v for k, v in p["implementation_sha256"].items()}
    require(verify_pins(root, p["implementation_sha256"]) == expected, "runtime pin mismatch")
    for q in (Path(__file__).resolve(), root / "code/takeover_queue.py", root / "code/run_takeover_command.sh",
              root / "env.sh", resolve(root, p["execution_contract"])):
        require(expected.get(str(q.resolve())) == sha(q), "required source/contract not pinned: " + str(q))
    execution_path = resolve(root, p["execution_contract"])
    execution = read(execution_path)
    require(execution["schema"] == "openh3.full22_joint_rgb_probe_execution.v1" and execution["root"] == str(root), "wrong execution contract")
    step = p["steps"][0]; argv = step["argv"]
    require(isinstance(argv, list) and all(isinstance(x, str) for x in argv), "argv must be strings")
    require(len(argv) == 9 and argv[0] == "venv/bin/python" and argv[2] == "--pretrained-manifest"
            and argv[4] == "--execution-contract" and argv[6] == "--out" and argv[8] == "--execute", "unexpected command or extra budget options")
    require(resolve(root, argv[4+1]).resolve() == execution_path.resolve(), "argv contract differs")
    require(expected.get(str(resolve(root, argv[1]).resolve())) == execution["probe_sha256"]
            and expected.get(str(resolve(root, argv[3]).resolve())) == execution["manifest_sha256"], "entry/manifest binding differs")
    status, log, out = [resolve(root, x) for x in (p["status"], step["log"], argv[7])]
    run = status.parent
    require(run.parent == root / "runs" and status.name == "queue_status.json"
            and out == run / "eval" and log == run / "logs/probe.log", "outputs must be one fresh run/eval/log namespace")
    fresh_paths = (log, out) if allow_status else (status, log, out)
    require(not any(x.exists() or x.is_symlink() for x in fresh_paths), "prior output exists; no retry/overwrite")
    admission_path = resolve(root, p["fresh_admission"])
    a = read(admission_path)
    require(a["schema"] == "openh3.full22_launch_admission.v1" and a["approved"] is True
        and a["scope"] == "full22_single_joint_engineering_probe" and a["physical_card"] == cards[0]
        and a["queue_sha256"] == sha(plan_path) and a["execution_contract_sha256"] == sha(execution_path)
        and a["runner_sha256"] == sha(__file__), "fresh launch admission does not bind exact queue/contract/runner/card")
    # Producer admission is deliberately separate, avoiding admission SHA cycles.
    pa = execution["root_admission"]; pa_path = resolve(root, pa["path"])
    require(pa_path.resolve() != admission_path.resolve() and sha(pa_path) == pa["sha256"], "invalid producer admission binding")
    producer = read(pa_path)
    payload = {k: v for k, v in execution.items() if k != "root_admission"}
    payload_sha = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(producer["approved"] is True and producer["scope"] == a["scope"]
        and producer["probe_sha256"] == execution["probe_sha256"]
        and producer["execution_payload_sha256"] == payload_sha, "producer admission differs")
    for guard in p.get("required_json_values", []):
        require(read(resolve(root, guard["path"])).get(guard["key"]) == guard["value"], "required JSON condition changed")
    return p, expected


def resource_check(root):
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    disk = shutil.disk_usage(root).free
    require(available >= RESOURCE["minimum_host_available_bytes"], "available host memory below160GiB")
    require(disk >= RESOURCE["minimum_disk_free_bytes"], "free workspace disk below10GiB")
    return {"host_MemAvailable_bytes": available, "disk_free_bytes": disk}


def run(plan_path):
    plan, pins = validate(plan_path)
    root = Path(plan["root"]); card = plan["cards"][0]
    status_path = resolve(root, plan["status"]); status_path.parent.mkdir(parents=True, exist_ok=True)
    status = {"schema": "openh3.full22_bounded_queue_status.v1", "state": "checking", "complete": False,
        "queue_sha256": sha(plan_path), "runner_sha256": sha(__file__), "physical_card": card,
        "execution_contract_sha256": sha(resolve(root, plan["execution_contract"])), "resource": RESOURCE,
        "RSS_scope": "simultaneously sampled owned process-group RSS; not historical peak sums; not cgroup enforcement",
        "cooperative_lock_only": True, "automatic_retry": False, "producer_started": False}
    spec = importlib.util.spec_from_file_location("bound_takeover_queue", root / "code/takeover_queue.py")
    queue = importlib.util.module_from_spec(spec); spec.loader.exec_module(queue)
    with status_path.with_suffix(".lock").open("a") as slock:
        fcntl.flock(slock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not status_path.exists(), "queue already attempted")
        prior_affinity = os.sched_getaffinity(0)
        try:
            with signal_guard():
                status["first_resources"] = resource_check(root)
                first = queue.inspect(); status["first_devices"] = first
                require(card in first and queue.available(first[card], 60000), "fixed card unavailable; no waiting")
                lock_dir = Path(plan["lock_dir"]); lock_dir.mkdir(parents=True, exist_ok=True)
                with (lock_dir / ("card_%d.lock" % card)).open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    status["state"] = "stability_check"; write(status_path, status)
                    time.sleep(15)
                    # Fresh immutable pins + conditions, followed by fresh resources/device.
                    require(sha(plan_path) == status["queue_sha256"], "queue changed during stability check")
                    validate(plan_path, allow_status=True)
                    status["second_resources"] = resource_check(root)
                    second = queue.inspect(); status["second_devices"] = second
                    require(card in second and queue.available(second[card], 60000), "second fixed-card check failed; no retry")
                    affinity = sorted(prior_affinity)[:4]
                    require(len(affinity) == 4, "four CPU affinity unavailable")
                    os.sched_setaffinity(0, set(affinity)); status["cpu_affinity"] = affinity
                    env = {**os.environ, **plan.get("environment", {}), "ASCEND_RT_VISIBLE_DEVICES": str(card),
                           "SSR0": str(root), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                           "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
                    step = plan["steps"][0]; log = resolve(root, step["log"]); log.parent.mkdir(parents=True, exist_ok=True)
                    status["launch_attempted"] = True
                    supervise(["bash", str(root / "code/run_takeover_command.sh"), *step["argv"]], root, env, log,
                        status, status_path, wall=3600, rss_limit=128 << 30, grace=30)
                    if status["state"] == "worker_completed":
                        report = resolve(root, step["argv"][7]) / "report.json"
                        data = read(report)
                        require(data["schema"] == "openh3.full22_joint_rgb_probe.v1" and data["complete"] is True,
                                "producer did not produce complete full22 report")
                        status.update(state="completed", complete=True, report_sha256=sha(report),
                            acceptance="producer execution completed; independent terminal array audit pending")
        except BaseException as error:
            status.update(state="failed" if status["producer_started"] else "not_launched", complete=False,
                error={"type": type(error).__name__, "message": str(error)})
        finally:
            os.sched_setaffinity(0, prior_affinity)
            try:
                after = verify_pins(root, plan["implementation_sha256"])
                status["pins_unchanged"] = after == pins
                if not status["pins_unchanged"]: status.update(state="failed", complete=False, error={"type": "PinChanged", "message": "runtime pins changed"})
            except BaseException as error:
                status.update(state="failed", complete=False, error={"type": type(error).__name__, "message": str(error)})
            write(status_path, status)
    return status


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("plan", type=Path)
    args = p.parse_args(); status = run(args.plan)
    print(json.dumps({k: status.get(k) for k in ("state", "complete", "physical_card", "active_worker_seconds", "peak_owned_group_RSS_bytes", "error")}))
    return 0 if status["complete"] else 2


if __name__ == "__main__": raise SystemExit(main())
