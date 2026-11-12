"""Recover the finite P1 cohort using fresh, sequential bounded CPU workers.

The original frozen data contract, entry, numerical helper and first-source
artifacts remain unchanged. Each remaining source runs in a fresh process, so
the original single-source historical RSS guard keeps its original scope.
All workers/FFmpeg inherit the outer supervisor's owned process group.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

SCHEMA = "openh3.history_utility_materialization_recovery.v1"
REMAINING = ["4EKWTiWjPTQ", "6UZVovkR3IU", "6rJrrfRO3cA"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""): h.update(b)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def require(value, message):
    if not value: raise ValueError(message)


def write(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"); tmp.replace(path)


def validate(contract):
    require(contract["schema"] == SCHEMA and contract["remaining_sources"] == REMAINING, "wrong recovery cohort")
    require(contract["reuse_source_id"] == "2BdYaGuVAKo" and contract["scientific_continuity_approved"] is False,
            "reuse/continuity contract changed")
    require(contract["budget"] == {"wall_seconds": 1800, "grace_seconds": 30, "memory_mib": 768,
                "simultaneous_compute_workers": 1, "CPU_threads": 1, "model_calls": 0, "NPU_calls": 0}, "budget changed")
    for name, row in contract["files"].items():
        require(Path(row["path"]).is_absolute() and sha(row["path"]) == row["sha256"], "recovery pin changed: " + name)
    require(Path(contract["files"]["entry"]["path"]).resolve() == Path(__file__).resolve(), "unbound recovery entry")
    original = read(contract["files"]["original_contract"]["path"])
    require(original["files"]["entry_source"] == contract["files"]["original_entry"], "wrong original entry")
    failed = read(contract["files"]["original_failure"]["path"])
    reused = contract["files"]["reuse_receipt"]
    require(failed["complete"] is False and failed["sources_completed"] == ["2BdYaGuVAKo"]
            and failed["receipt_indices"]["train"] == [reused], "original failure/reuse binding differs")
    r = read(reused["path"])
    require(r["source"]["source_id"] == "2BdYaGuVAKo" and r["source"]["split"] == "train", "wrong reused source role")
    artifacts = [r["target"]] + [r["observations"][pid]["artifact"] for pid in original["profiles"]]
    require(contract["reuse_artifacts"] == [{"path": a["path"], "sha256": a["file_sha256"]} for a in artifacts],
            "reuse artifact inventory changed")
    for a in contract["reuse_artifacts"]: require(sha(a["path"]) == a["sha256"], "reused NPZ file changed")
    return original


def load_original(contract):
    path = contract["files"]["original_entry"]["path"]
    sys.path.insert(0, str(Path(path).parent))
    spec = importlib.util.spec_from_file_location("_frozen_p1_original_entry", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def inspect(contract):
    original = validate(contract); module = load_original(contract)
    report = module.inspect(original)
    return {**report, "recovery_schema": SCHEMA, "recovery_contract_sha256": module.D.value_sha(contract),
            "original_data_contract_file_sha256": contract["files"]["original_contract"]["sha256"],
            "reuse_file_SHA_verified": 4, "reuse_pixels_verified": False, "remaining_sources": REMAINING,
            "fresh_process_per_source": True}


def phase(contract, out, name, source_id):
    """A fresh process runs exactly one phase; no model/backend imports."""
    if name == "source": require(source_id in REMAINING, "first source cannot be rerun")
    original = validate(contract); module = load_original(contract)
    plan, profiles, registry = module.validate_contract(original)
    old = module.load_numeric(original["files"]["numeric_source"]["path"])
    budget = old.Budget(1800, 768)
    report_path = out / (name + ("_" + source_id if source_id else "") + ".json")
    require(not report_path.exists(), "phase already exists; no replay/overwrite")
    if name == "verify_reuse":
        entry = contract["files"]["reuse_receipt"]
        data = module.D.HistoryUtilityDataset([entry], plan=original["files"]["plan"]["path"],
            train_manifest=original["files"]["train_manifest"]["path"], dev_manifest=original["files"]["dev_manifest"]["path"],
            split="train", scope="engineering", verify_pixels=True, cache_bytes=128 << 20)
        require(data.source_ids == ["2BdYaGuVAKo"] and not data.research_ready, "wrong reused pixels")
        result = {"entry": entry, "dataset_sha256": data.dataset_sha256, "binding": data.binding,
                  "actual_NPZ_file_and_pixel_SHA_verified": 4, "new_target_decodes": 0, "new_codec_passes": 0}
    elif name == "source":
        require(source_id in REMAINING, "first source cannot be rerun")
        entry = module.source_receipt(original, old, plan, profiles, registry, registry[source_id],
            out / "materialized" / registry[source_id]["split"] / source_id, budget)
        result = {"entry": entry, "new_target_decodes": 1, "new_codec_passes": 6}
    elif name == "loader":
        require(source_id is None, "loader does not accept source selection")
        indices = {split: read(out / (split + "_receipt_index.json"))["receipts"] for split in ("train", "dev")}
        result = {"loader_validation": module.validate_loader(original, indices, out, budget)}
    else: raise ValueError("unknown phase")
    budget.check(); result.update(complete=True, phase=name, source_id=source_id, pid=os.getpid(),
        elapsed_seconds=time.monotonic() - budget.started, peak_parent_rss_bytes=old.peak_rss(),
        max_sampled_parent_plus_ffmpeg_rss_bytes=budget.max_combined_rss,
        conservative_nonconcurrent_parent_plus_child_peak_bytes=budget.conservative_peak_sum,
        import_guard=module.check_no_models(), model_calls=0, NPU_calls=0)
    write(report_path, result)
    return result


def run_phase(contract_path, out, name, source_id=None):
    args = [sys.executable, str(Path(__file__).resolve()), "--recovery-contract", str(contract_path),
            "--out", str(out), "--phase", name]
    if source_id: args.extend(["--source-id", source_id])
    stem = name + ("_" + source_id if source_id else "")
    with (out / (stem + ".log")).open("xb") as log:
        # The frozen outer run_once owns/limits this whole process group.
        done = subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
    require(done.returncode == 0, "fresh phase failed: " + stem)
    report = read(out / (stem + ".json")); require(report["complete"] is True, "incomplete fresh phase")
    return report


def execute(contract_path, contract, out):
    require(out.is_absolute(), "absolute new output required")
    if out.exists(): raise FileExistsError("recovery output exists; no overwrite")
    validate(contract)  # stdlib only: do not inflate coordinator RSS with pixels.
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": SCHEMA, "status": "running", "complete": False,
        "original_data_contract_file_sha256": contract["files"]["original_contract"]["sha256"],
        "receipt_indices": {"train": [], "dev": []}, "phases_completed": [], "sources_completed": [],
        "scope": "2train2dev engineering; one source reused, three newly materialized", "model_calls": 0, "NPU_calls": 0,
        "scientific_continuity_approved": False, "natural_cut_status": "unknown_engineering_only"}
    started = time.monotonic(); write(out / "partial.json", report)
    previous = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame): raise TimeoutError("owned supervisor requested termination")
    signal.signal(signal.SIGTERM, terminate)
    try:
        reuse = run_phase(contract_path, out, "verify_reuse")
        report["receipt_indices"]["train"].append(reuse["entry"]); report["sources_completed"].append("2BdYaGuVAKo")
        report["phases_completed"].append(reuse); write(out / "partial.json", report)
        for sid in REMAINING:
            result = run_phase(contract_path, out, "source", sid)
            split = "train" if sid == "4EKWTiWjPTQ" else "dev"
            report["receipt_indices"][split].append(result["entry"]); report["sources_completed"].append(sid)
            report["phases_completed"].append(result); write(out / "partial.json", report)
        for split, entries in report["receipt_indices"].items():
            write(out / (split + "_receipt_index.json"), {"schema": "openh3.history_utility_receipt_index.v1", "receipts": entries})
        validation = run_phase(contract_path, out, "loader"); report["phases_completed"].append(validation)
        report["loader_validation"] = validation["loader_validation"]
        validate(contract)
        report.update(status="completed_engineering_only", complete=True, elapsed_seconds=time.monotonic() - started,
            reused_sources=1, newly_materialized_sources=3, new_target_decodes=3, new_observation_codec_passes=18,
            inherited_completed_target_decodes=1, inherited_completed_codec_passes=6,
            prior_failed_source2_decode_attempt="Popen may have started; no completed decode evidence, not counted as a completed target",
            total_verified_target_NPZ=4, total_verified_observation_NPZ=12, independent_worker_PIDs=[r["pid"] for r in report["phases_completed"]])
        require(len(set(report["independent_worker_PIDs"])) == 5, "fresh worker identity repeated")
        write(out / "complete.json", report); (out / "partial.json").unlink()
        return report
    except BaseException as exc:
        report.update(complete=False, status="failed", elapsed_seconds=time.monotonic() - started,
                      error={"type": type(exc).__name__, "message": str(exc)})
        write(out / "failure.json", report); raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recovery-contract", type=Path, required=True); p.add_argument("--out", type=Path, required=True)
    p.add_argument("--execute", action="store_true"); p.add_argument("--phase", choices=["verify_reuse", "source", "loader"])
    p.add_argument("--source-id"); a = p.parse_args(); c = read(a.recovery_contract)
    if a.phase: phase(c, a.out, a.phase, a.source_id)
    elif a.execute: execute(a.recovery_contract, c, a.out)
    else:
        if a.out.exists(): raise FileExistsError("inspection exists")
        r = inspect(c); a.out.parent.mkdir(parents=True, exist_ok=True)
        with a.out.open("x") as f: json.dump(r, f, indent=2)


if __name__ == "__main__": main()
