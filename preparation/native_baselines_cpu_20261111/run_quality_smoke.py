"""Bounded six-forward CPU queue; immutable admission, separate inference/metrics.

The inference child reads observations only and calls the existing lossless
adapter. The metrics child alone reads target pixels. No NPU backend is loaded.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types

sys.dont_write_bytecode = True
os.environ.update(TORCH_DEVICE_BACKEND_AUTOLOAD="0", PYTHONDONTWRITEBYTECODE="1",
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def admission(args):
    assert sha(args.admission) == args.admission_sha256, "Admission changed"
    a = json.loads(Path(args.admission).read_text())
    for path, expected in a["locked_files"].items():
        assert sha(path) == expected, "Locked file changed: " + path
    return a


def metric_gate(a):
    import torch
    import cv2
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert "torch_npu" not in sys.modules
    actual = {}
    for name, expected in a["metric_versions"].items():
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            actual[name] = None
        assert actual[name] == expected, (name, actual[name], expected)
    assert cv2.__version__ == "4.11.0"
    assert sha(a["alexnet_weights"]) == a["alexnet_sha256"]
    linear = importlib.metadata.distribution("lpips").locate_file("lpips/weights/v0.1/alex.pth")
    assert sha(linear) == a["lpips_linear_sha256"]
    source = Path(a["metric_runtime"], "eval_rphi.py").read_text()
    functions = {n.name: hashlib.sha256(ast.get_source_segment(source, n).encode()).hexdigest()
                 for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    for name, expected in a["metric_function_sha256"].items():
        assert functions[name] == expected, name
    return {"passed": True, "versions": actual, "opencv_runtime": cv2.__version__,
            "torch_npu_imported": False, "alexnet_sha256": sha(a["alexnet_weights"]),
            "lpips_linear_sha256": sha(linear), "downloads": False}


def infer(a, method, task):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert "torch_npu" not in sys.modules
    row = a["records"][task]
    for name, expected in a["input_files"].items():
        # Inference only checks the selected observation, never opens clean GT.
        if name == row["observation_npz"]:
            assert sha(name) == expected
    adapter = load_module("locked_lossless_adapter", a["adapter"])
    repo = Path(a["assets"], adapter.LOCKS[method]["repo"])
    lock = json.loads(Path(a["source_lock"]).read_text())[repo.name]
    for relative, expected in lock["files"].items():
        assert sha(repo / relative) == expected
    if method == "avernet":
        for name, directory in (("basicsr", repo / "basicsr"), ("basicsr.utils", repo / "basicsr/utils"), ("basicsr.archs", repo / "basicsr/archs")):
            assert name not in sys.modules
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            package.__package__ = name
            sys.modules[name] = package
    args = types.SimpleNamespace(method=method, device="cpu", repo=repo,
        checkpoint=Path(a["assets"], "weights", adapter.LOCKS[method]["checkpoint"]),
        out=Path(a["run_paths"][method], task, "prediction"),
        input_npz=Path(row["observation_npz"]), expected_pixel_sha256=row["observation_pixel_sha256"],
        task=task, inspect_only=False)
    result = adapter.run(args)
    assert result["status"] == "inference_completed_metrics_not_run"
    assert result["input_shape"] == [22, 480, 832, 3]
    assert result["checkpoint_strict_load"] and not result["gt_read"]
    write(Path(a["run_paths"][method], task, "inference_binding.json"), {
        "method": method, "task": task, "input_only": True, "source_tree_sha256": lock["tree_sha256"],
        "aver_import_mode": "architecture_only" if method == "avernet" else None,
        "operator_fallback": False, "torch_threads": 1, "torch_npu_imported": False})


def metrics(a, method, task):
    import numpy as np
    gate = metric_gate(a)
    sys.path.insert(0, a["metric_runtime"])
    metric = load_module("locked_rgb_metrics", Path(a["metric_runtime"], "eval_rphi.py"))
    row = a["records"][task]
    arrays = {}
    for prefix in ("observation", "target"):
        path = row[prefix + "_npz"]
        assert sha(path) == a["input_files"][path]
        with np.load(path, allow_pickle=False) as z:
            u8 = np.asarray(z["frames"])
            assert u8.shape == (22, 480, 832, 3) and u8.dtype == np.uint8
            assert hashlib.sha256(u8.tobytes()).hexdigest() == row[prefix + "_pixel_sha256"]
            arrays[prefix] = u8.astype(np.float32) / 255
    run = Path(a["run_paths"][method], task)
    prediction = run / "prediction/restored_float32.npz"
    infer_receipt = json.loads((run / "prediction/run.json").read_text())
    assert sha(prediction) == infer_receipt["output_npz_sha256"]
    with np.load(prediction, allow_pickle=False) as z:
        raw, rgb = np.asarray(z["rgb_raw"]), np.asarray(z["rgb"])
    assert raw.dtype == rgb.dtype == np.float32 and rgb.shape == (22, 480, 832, 3)
    assert np.isfinite(raw).all() and np.array_equal(rgb, np.clip(raw, 0, 1))
    lpips, lpips_info = metric.load_local_lpips(a["alexnet_weights"])
    assert lpips is not None and lpips_info["available"]
    target, observed = arrays["target"], arrays["observation"]
    support, note = metric.flow_support(target, True)
    assert support and len(support) == 21
    result = {"schema": "native_quality_smoke.rgb_eval.v1", "complete": True,
        "method": method, "task": task, "record": row, "frame_indices": list(range(22)),
        "frame_stride": 1, "metric_gate": gate, "lpips": lpips_info, "flow": note,
        "model": metric.rgb_metrics(rgb, target, lpips),
        "baseline": metric.rgb_metrics(observed, target, lpips),
        "raw_model_mse": float(np.square(raw.astype(np.float64) - target).mean()),
        "clipped_component_fraction": float(((raw < 0) | (raw > 1)).mean()),
        "attribution": metric.error_attribution(rgb, observed, target),
        "temporal_model": metric.temporal_metrics(rgb, target, support),
        "temporal_baseline": metric.temporal_metrics(observed, target, support),
        "temporal_access": "offline_bidirectional_full22_max_future21",
        "quality_verdict": "requires_review_single_historical_dev_source"}
    reference = json.loads(Path(a["reference_metrics"]).read_text())
    ref = next(x for x in reference["per_sample"] if x["source_id"] == row["source_id"] and x["task"] == task)
    for key in ("source_sha256", "target_pixel_sha256", "observation_pixel_sha256"):
        assert ref[key] == row[key]
    assert ref["frame_indices"] == list(range(22)) and reference["contract"]["frame_stride"] == 1
    checks = {}
    for group, actual, expected in (("quality", result["baseline"], ref["arms"]["input"]["quality"]),
                                    ("temporal", result["temporal_baseline"], ref["arms"]["input"]["temporal"])):
        for key, value in actual.items():
            other = expected.get(key)
            if value is None or other is None:
                assert value is other, (group, key, value, other)
                continue
            tolerance = 1e-6 if key in ("lpips", "psnr", "psnr_from_mean_mse") else 1e-8
            if key in ("frames", "perfect_frames", "pairs", "valid_pairs"):
                tolerance = 0
            difference = abs(value - other)
            checks[group + "." + key] = {"absolute_difference": difference, "absolute_tolerance": tolerance}
            assert difference <= tolerance, ("Reference input mismatch", group, key, difference, tolerance)
    assert ref["flow"] == note
    result["reference_input_gate"] = {"passed": True, "checks": checks, "flow_description_matches": True,
        "prior_flow_pixel_hash_available": False,
        "support_claim": "same pixels/functions/versions; recomputed input quality, pair counts and coverage match; no prior flow tensor hash was saved"}
    result["reference"] = {"file_sha256": sha(a["reference_metrics"]), "arms": ref["arms"],
        "note": "Existing ordinary H3/R_phi stride1 metrics; no extra R_phi inference. Temporal architecture budgets differ."}
    write(run / "metrics.json", result)


def owned_rss_bytes(pid):
    total, pending, seen = 0, [pid], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        try:
            status = Path(f"/proc/{current}/status").read_text()
            total += next((int(x.split()[1]) * 1024 for x in status.splitlines() if x.startswith("VmRSS:")), 0)
            pending.extend(int(x) for x in Path(f"/proc/{current}/task/{current}/children").read_text().split())
        except (FileNotFoundError, ProcessLookupError):
            pass
    return total


def stop_group(child):
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    # The original child may exit before its children. Kill any remaining owned
    # process-group members after a bounded grace period, even if parent exited.
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            return
        child.poll()
        time.sleep(.1)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def guarded(command, logfile, seconds, rss_limit, progress_path=None):
    started, peak, reason = time.monotonic(), 0, None
    with Path(logfile).open("x") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=os.environ.copy())
        next_progress = 0
        try:
            while child.poll() is None:
                peak = max(peak, owned_rss_bytes(child.pid))
                elapsed = time.monotonic() - started
                if progress_path and elapsed >= next_progress:
                    write(progress_path, {"pid": child.pid, "elapsed_seconds": elapsed, "peak_owned_rss_bytes": peak,
                                          "wall_limit_seconds": seconds, "rss_limit_bytes": rss_limit})
                    next_progress = elapsed + 30
                if elapsed > seconds:
                    reason = "owned_process_wall_timeout"
                elif peak > rss_limit:
                    reason = "owned_process_rss_limit"
                if reason:
                    stop_group(child)
                    break
                time.sleep(1)
            code = child.wait()
        finally:
            if child.poll() is None:
                stop_group(child)
                child.wait()
    return {"returncode": code, "guard_reason": reason, "seconds": time.monotonic() - started,
            "peak_owned_rss_bytes": peak, "pid": child.pid}


def supervise(a, args):
    status_path = Path(a["workspace"], "queue_status.json")
    assert not status_path.exists(), "Queue already launched"
    state = {"status": "running", "admission_sha256": args.admission_sha256, "pid": os.getpid(),
             "started_unix": time.time(), "jobs": [], "completed_forwards": 0, "failed_models": []}
    write(status_path, state)
    first = True
    for method in ("favernet", "avernet"):
        root = Path(a["run_paths"][method])
        root.mkdir(parents=True, exist_ok=False)
        for task in ("deblur", "denoise", "compression"):
            jobdir = root / task
            jobdir.mkdir()
            for mode, python, seconds in (("infer", a["inference_python"], 1800), ("metrics", a["metrics_python"], 600)):
                state["current"] = {"method": method, "task": task, "stage": mode}
                write(status_path, state)
                cmd = [python, str(Path(__file__).resolve()), "--mode", mode, "--admission", args.admission,
                       "--admission-sha256", args.admission_sha256, "--method", method, "--task", task]
                available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
                if available < 32 * 1024**3:
                    state["status"] = "stopped_insufficient_available_memory"
                    write(status_path, state)
                    return 1
                outcome = guarded(cmd, jobdir / (mode + ".log"), seconds, 24 * 1024**3, jobdir / (mode + "_progress.json"))
                state["jobs"].append({"method": method, "task": task, "stage": mode, **outcome})
                write(jobdir / (mode + "_guard.json"), outcome)
                if outcome["returncode"] != 0 or outcome["guard_reason"]:
                    state["failed_models"].append(method)
                    if first and mode == "infer" and outcome["guard_reason"]:
                        state["status"] = "stopped_first_full_clip_over_budget"
                        write(status_path, state)
                        return 1
                    break
                if mode == "infer":
                    state["completed_forwards"] += 1
                    first = False
            if method in state["failed_models"]:
                break
    state["status"] = "complete" if not state["failed_models"] else "complete_with_model_failures"
    state["finished_unix"] = time.time()
    state.pop("current", None)
    write(status_path, state)
    return int(bool(state["failed_models"]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("supervise", "gate", "infer", "metrics"), required=True)
    p.add_argument("--admission", required=True)
    p.add_argument("--admission-sha256", required=True)
    p.add_argument("--method", choices=("favernet", "avernet"))
    p.add_argument("--task", choices=("deblur", "denoise", "compression"))
    args = p.parse_args()
    a = admission(args)
    if args.mode == "gate":
        write(Path(a["workspace"], "metric_gate.json"), metric_gate(a))
    elif args.mode == "infer":
        infer(a, args.method, args.task)
    elif args.mode == "metrics":
        metrics(a, args.method, args.task)
    else:
        def terminate(signum, _frame):
            raise SystemExit(128 + signum)
        signal.signal(signal.SIGTERM, terminate)
        return supervise(a, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
