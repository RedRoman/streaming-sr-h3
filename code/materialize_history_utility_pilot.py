"""Bounded P1 2-train/2-dev CPU engineering materialization; no training.

Default is read-only inspection. Frozen v1 numeric functions are imported by
verified path; its single-source main/execute/constants are never changed.
The new decoder keeps the exact v1 command/filter/timestamp math but makes its
FFmpeg subprocess inherit the owned worker process group for hard timeout.
Per-source v1 receipt fields preserve their meanings; the new cohort receipt
explicitly names the three-profile subset and unknown natural-cut status.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

import argparse
import gc
import importlib.abc
import importlib.util
import json
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time


FORBIDDEN = {"torch", "torch_npu", "comfy", "comfy_kitchen", "h3_comfy", "rphi_model"}


class NoModels(importlib.abc.MetaPathFinder):
    def __init__(self): self.attempts = []
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in FORBIDDEN:
            self.attempts.append(fullname)
            raise ImportError("model/backend import forbidden in CPU materializer: " + fullname)


IMPORT_GUARD = NoModels()
sys.meta_path.insert(0, IMPORT_GUARD)
import numpy as np
import history_utility_data as D

SCHEMA = "openh3.history_utility_materialization.v1"
CONTRACT_SCHEMA = "openh3.history_utility_materialization_execution.v1"
SELECTED = {"train": ["2BdYaGuVAKo", "4EKWTiWjPTQ"], "dev": ["6UZVovkR3IU", "6rJrrfRO3cA"]}
FROZEN_NUMERIC_SHA = "80fa35dd84e941fb009b276cc704afb07b41088ca2b80ba72604afedc40d1c37"
MAX_SECONDS = 1800
MEMORY_MIB = 768


def require(value, message):
    if not value: raise ValueError(message)


def atomic_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def check_no_models():
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in FORBIDDEN)
    require(not loaded and not IMPORT_GUARD.attempts, "forbidden model/backend import occurred")
    return {"forbidden_loaded_modules": loaded, "forbidden_import_attempts": list(IMPORT_GUARD.attempts)}


def load_numeric(path):
    require(D.file_sha(path) == FROZEN_NUMERIC_SHA, "frozen numerical source changed")
    spec = importlib.util.spec_from_file_location("_p1_frozen_materializer_numeric", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    require(module.SOURCE == "2BdYaGuVAKo" and len(module.PROFILE_IDS) == 6, "old source constants changed")
    return module


def validate_contract(contract):
    require(contract["schema"] == CONTRACT_SCHEMA and contract["scope"] == "engineering_only_no_training",
            "wrong execution contract/scope")
    require(contract["sources"] == SELECTED and contract["profiles"] == list(D.PROFILES), "cohort/profile expansion forbidden")
    require(contract["frames"] == 96 and contract["replays"] == 2 and contract["shape"] == list(D.SHAPE), "geometry/replay differs")
    require(contract["budget"] == {"wall_seconds": 1800, "termination_grace_seconds": 30, "memory_mib": 768,
                "worker_processes": 1, "threads": 1, "model_calls": 0, "NPU_calls": 0}, "fixed CPU budget differs")
    require(contract["natural_cut_status"] == "unknown_engineering_only"
            and contract["scientific_continuity_approved"] is False, "continuity must not be fabricated")
    require(contract["new_window_noise_version"] == "execution_contract_changes_window_key_and_keyed_noise",
            "new noise/window version must be explicit")
    required_files = {"numeric_source", "loader_source", "entry_source", "plan", "train_manifest", "dev_manifest", "prior_single_source_receipt"}
    require(set(contract["files"]) == required_files, "incomplete execution input bindings")
    for name, row in contract["files"].items():
        require(Path(row["path"]).is_absolute() and D.file_sha(row["path"]) == row["sha256"], "bound file changed: " + name)
    require(Path(contract["files"]["entry_source"]["path"]).resolve() == Path(__file__).resolve(), "executed entry not bound")
    require(Path(contract["files"]["loader_source"]["path"]).resolve() == Path(D.__file__).resolve(), "imported loader not bound")
    require(contract["files"]["numeric_source"]["sha256"] == FROZEN_NUMERIC_SHA, "wrong numerical source")
    plan = D._json(contract["files"]["plan"]["path"])
    profiles = [next(p for p in plan["profiles"] if p["profile_id"] == pid) for pid in D.PROFILES]
    require(contract["operator_contract_sha256"] == D.value_sha(plan["operator_contract"])
            and contract["profile_recipes_sha256"] == D.value_sha(profiles), "recipe binding changed")
    for profile in profiles:
        require(profile["task"] == "mixed_native" and profile["scale"] == 1 and not profile.get("scene_cut_frames")
                and not profile.get("codec_reset_frames"), "only native no-reset profiles allowed")
        require(profile["ordered_operators"] == ["gaussian_blur_float_rgb", "gaussian_noise_float_rgb", "quantize_rint_clip_uint8", "h264_roundtrip"], "operator order changed")
        require(len(profile["frame_parameters"]) == 96 and [r["frame"] for r in profile["frame_parameters"]] == list(range(96)), "parameter frame schedule differs")
        require({r["h264_crf"] for r in profile["frame_parameters"]} == {28}, "codec must have fixed CRF28")
    registry = D.load_source_registry(plan, contract["files"]["train_manifest"]["path"], contract["files"]["dev_manifest"]["path"])
    selected = SELECTED["train"] + SELECTED["dev"]
    require(set(contract["original_sources"]) == set(selected), "original source bindings incomplete")
    for split, ids in SELECTED.items():
        for sid in ids:
            row = contract["original_sources"][sid]
            require(registry[sid]["split"] == split and row["sha256"] == registry[sid]["source_sha256"], "source role/hash changed")
            require(Path(row["path"]).name == sid + ".mp4" and Path(row["path"]).is_absolute(), "source path does not identify selected video")
            require(Path(row["path"]).stat().st_size == row["bytes"] and D.file_sha(row["path"]) == row["sha256"], "actual original video bytes changed")
    prior = D._json(contract["files"]["prior_single_source_receipt"]["path"])
    require(prior["schema"] == D.MATERIALIZER_SCHEMA and prior["status"] == D.COMPLETE_STATUS
            and prior["source"] == registry["2BdYaGuVAKo"] and prior["target"]["shape"] == list(D.SHAPE),
            "old clean-target reference/source binding differs")
    return plan, profiles, registry


def inspect(contract):
    check_no_models()
    plan, profiles, registry = validate_contract(contract)
    old = load_numeric(contract["files"]["numeric_source"]["path"])
    probes = {sid: old.source_probe(row["path"], contract["ffprobe"]) for sid, row in contract["original_sources"].items()}
    return {"schema": SCHEMA, "status": "inspected_not_materialized", "sources": SELECTED,
            "source_file_SHA_verified": 4, "frames": 96, "profiles": list(D.PROFILES),
            "source_probes": probes, "budget": contract["budget"], "execution_contract_sha256": D.value_sha(contract),
            "new_window_noise_version": contract["new_window_noise_version"], "model_calls": 0, "NPU_calls": 0,
            "natural_cut_status": "unknown_engineering_only", "import_guard": check_no_models()}


def decode_target_owned(old, source, spec, scratch, ffmpeg, budget):
    """Same frozen command and parser, different subprocess ownership only."""
    raw, log = scratch / "target.rgb", scratch / "target_decode.log"
    command, graph = old.target_command(source, spec, raw, ffmpeg)
    with log.open("wb") as f:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=f, start_new_session=False)
        try:
            while process.poll() is None:
                budget.check(process.pid); time.sleep(.1)
            if process.returncode: raise RuntimeError("FFmpeg failed: " + str(log))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=2)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    budget.check(); old.checked_shape(raw, D.SHAPE)
    timestamps = old.parse_timestamps(log.read_text(errors="replace"))
    clean = np.memmap(raw, mode="r", dtype=np.uint8, shape=D.SHAPE)
    return clean, {"command": command, "filter_graph": graph, "accurate_input_seek": True,
        "source_pts_semantic": "original presentation timestamps retained with copyts before relative-grid setpts",
        "target_pts_semantic": "24fps window-relative output grid after explicit setpts",
        "timestamps": timestamps, "decoded_source_count": len(timestamps["source"]),
        "decoded_target_count": len(timestamps["target"]), "log_sha256": D.file_sha(log),
        "subprocess_ownership": "inherits owned worker process group; numerical command/parser unchanged"}


def source_receipt(contract, old, plan, profiles, registry, binding, out, budget):
    sid = binding["source_id"]; source = Path(contract["original_sources"][sid]["path"])
    out.mkdir(parents=True, exist_ok=False)
    probe = old.source_probe(source, contract["ffprobe"])
    spec = old.target_specification(binding, probe, contract); window = old.value_sha(spec)
    software = old.software_receipt(contract["ffmpeg"], contract["ffprobe"])
    registry_receipt = {"schema": "openh3.history_utility_engineering_registry.v1", "sources": [registry[k] for k in sorted(registry)],
        "selected_sources": SELECTED, "manifest_bindings": {s: {"sha256": contract["files"][s + "_manifest"]["sha256"]} for s in SELECTED}}
    # Shared v1 fields describe exactly the same artifacts, not a rename of an
    # incomplete receipt. Cohort/version/selected profiles are additional fields.
    receipt = {"schema": old.VERSION, "status": "started_not_complete", "producer_schema": SCHEMA,
        "producer_sha256": D.file_sha(__file__), "source": binding, "source_path": str(source),
        "source_stat": {"bytes": source.stat().st_size, "mtime_ns": source.stat().st_mtime_ns}, "source_probe": probe,
        "target_specification": spec, "target_specification_sha256": window, "window_key": window,
        "plan_file_sha256": contract["files"]["plan"]["sha256"], "execution_contract": contract,
        "registry": registry_receipt, "registry_sha256": old.value_sha(registry_receipt),
        "software": software, "software_sha256": old.value_sha(software), "materializer_sha256": FROZEN_NUMERIC_SHA,
        "source_natural_cut_status": "unknown_engineering_only", "scientific_continuity_approved": False,
        "new_split_assignment": False, "model_calls": 0, "npu_calls": 0, "observations": {},
        "selected_profiles": list(D.PROFILES), "new_window_noise_version": contract["new_window_noise_version"]}
    atomic_json(out / "receipt.partial.json", receipt)
    with tempfile.TemporaryDirectory(prefix="scratch_", dir=out) as temporary:
        scratch = Path(temporary)
        clean, receipt["decode"] = decode_target_owned(old, source, spec, scratch, contract["ffmpeg"], budget)
        shutil.copyfile(scratch / "target_decode.log", out / "target_decode.log")
        target_sha = old.pixel_sha(clean)
        if sid == "2BdYaGuVAKo":
            prior = D._json(contract["files"]["prior_single_source_receipt"]["path"])
            require(target_sha == prior["target"]["pixel_sha256"], "same-source clean target no longer reproduces old decoder pixels")
            require(window != prior["window_key"], "new contract unexpectedly reused old noise/window key")
            receipt["prior_target_parity"] = {"pixel_equal": True, "old_window_key": prior["window_key"], "new_window_key": window,
                "observation_parity_not_expected": True}
        common = {"schema": old.VERSION, **binding, "window_key": window, "target_pixel_sha256": target_sha,
            **{k: receipt[k] for k in ("plan_file_sha256", "registry_sha256", "materializer_sha256", "software_sha256")}}
        receipt["target"] = old.save_npz(out / "target.npz", clean, {**common, "role": "target", "target_specification": spec})
        for profile in profiles:
            pid = profile["profile_id"]; replays = []
            for replay in range(2):
                observed, raw, replay_receipt = old.observation(clean, profile, binding, window, scratch, replay, budget)
                replays.append(replay_receipt)
                if replay == 0:
                    artifact = old.save_npz(out / (pid + ".npz"), observed, {**common, "role": "observation", "profile": profile,
                        "parameter_schedule_sha256": old.value_sha(profile["frame_parameters"]), "noise_stream": profile["noise_stream"],
                        "blind_model_condition": "not_supplied_new_loader_required"})
                del observed; raw.unlink(); budget.check()
            require(replays[0]["observation_pixel_sha256"] == replays[1]["observation_pixel_sha256"]
                    and replays[0]["noise_schedule_sha256"] == replays[1]["noise_schedule_sha256"], "pixel/noise double replay mismatch")
            receipt["observations"][pid] = {"artifact": artifact, "task": profile["task"], "scale": 1,
                "parameter_schedule_sha256": old.value_sha(profile["frame_parameters"]), "replay_count": 2, "pixel_replay_equal": True, "replays": replays}
            require(old.pixel_sha(clean) == target_sha, "clean target was mutated")
            atomic_json(out / "receipt.partial.json", receipt)
        require(receipt["observations"][D.PROFILES[1]]["replays"][0]["noise_frames"] ==
                receipt["observations"][D.PROFILES[2]]["replays"][0]["noise_frames"], "smooth/abrupt unit noise differs")
        del clean
    require(D.file_sha(source) == binding["source_sha256"], "source changed during materialization")
    budget.check()
    receipt.update(status=D.COMPLETE_STATUS, source_unchanged=True, paired_noise_equal=True,
        paired_noise_scope=[D.PROFILES[1], D.PROFILES[2]], import_guard=check_no_models())
    atomic_json(out / "complete.json", receipt); (out / "receipt.partial.json").unlink()
    return {"path": str((out / "complete.json").resolve()), "sha256": D.file_sha(out / "complete.json")}


def validate_loader(contract, entries, out, budget):
    result = {}
    for split in ("train", "dev"):
        data = D.HistoryUtilityDataset(entries[split], plan=contract["files"]["plan"]["path"],
            train_manifest=contract["files"]["train_manifest"]["path"], dev_manifest=contract["files"]["dev_manifest"]["path"],
            split=split, scope="engineering", verify_pixels=True, cache_bytes=128 << 20)
        require(not data.research_ready and data.source_ids == sorted(SELECTED[split]), "subset falsely marked research-ready")
        if split == "train":
            schedules = {role: data.schedule(role, steps=1) for role in ("teacher", "student")}
            for role, schedule in schedules.items():
                data.validate_schedule(schedule)
                batch = data.batch(schedule["rows"])
                require(set(batch["model_inputs"]) == {"observed", "availability"}, "oracle input field leaked")
                atomic_json(out / (role + "_engineering_schedule.json"), schedule)
                budget.check(); del batch
            extra = {role + "_schedule_sha256": row["schedule_sha256"] for role, row in schedules.items()}
        else:
            rows = data.evaluation_rows()
            require(len(rows) == 2 * 5 * 96, "dev controls repeated per profile")
            # Every source/family first two causal rows, not all960 evaluations.
            chosen = [r for r in rows if r["frame_index"] in (0, 1)]
            batch = data.batch(chosen); budget.check(); del batch
            extra = {"evaluation_rows": len(rows), "evaluation_rows_sha256": D.value_sha(rows), "sampled_rows": len(chosen)}
        result[split] = {"binding": data.binding, "dataset_sha256": data.dataset_sha256, **extra}
        del data; gc.collect(); budget.check()
    return result


def execute(contract, out):
    out = Path(out)
    require(out.is_absolute(), "absolute new output required")
    if out.exists(): raise FileExistsError("output exists; no overwrite or implicit resume")
    before = inspect(contract)
    plan, profiles, registry = validate_contract(contract)
    old = load_numeric(contract["files"]["numeric_source"]["path"])
    budget = old.Budget(MAX_SECONDS, MEMORY_MIB)
    out.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "complete": False, "status": "running", "inspection": before,
        "sources_completed": [], "receipt_indices": {"train": [], "dev": []}, "model_calls": 0, "NPU_calls": 0,
        "scope": "2train2dev engineering only, not32/8 research screen", "scientific_continuity_approved": False}
    atomic_json(out / "partial.json", result)
    prior_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame): raise TimeoutError("owned supervisor requested termination")
    signal.signal(signal.SIGTERM, terminate)
    try:
        for split, ids in SELECTED.items():
            for sid in ids:
                entry = source_receipt(contract, old, plan, profiles, registry, registry[sid], out / "materialized" / split / sid, budget)
                result["receipt_indices"][split].append(entry); result["sources_completed"].append(sid)
                atomic_json(out / "partial.json", result); gc.collect(); budget.check()
        for split, entries in result["receipt_indices"].items():
            atomic_json(out / (split + "_receipt_index.json"), {"schema": "openh3.history_utility_receipt_index.v1", "receipts": entries})
        result["loader_validation"] = validate_loader(contract, result["receipt_indices"], out, budget)
        # Recheck all executable/config/source pins after producing artifacts.
        validate_contract(contract); result["import_guard"] = check_no_models()
        budget.check()
        result.update(complete=True, status="completed_engineering_only", elapsed_seconds=time.monotonic() - budget.started,
            peak_parent_rss_bytes=old.peak_rss(), max_sampled_parent_plus_ffmpeg_rss_bytes=budget.max_combined_rss,
            conservative_nonconcurrent_parent_plus_child_peak_bytes=budget.conservative_peak_sum, memory_limit_bytes=budget.bytes,
            target_decodes=4, observation_codec_passes=24, target_NPZ=4, observation_NPZ=12)
        atomic_json(out / "complete.json", result); (out / "partial.json").unlink()
        return result
    except BaseException as exc:
        result.update(complete=False, status="failed", error={"type": type(exc).__name__, "message": str(exc)},
            elapsed_seconds=time.monotonic() - budget.started, peak_parent_rss_bytes=old.peak_rss(),
            max_sampled_parent_plus_ffmpeg_rss_bytes=budget.max_combined_rss,
            conservative_nonconcurrent_parent_plus_child_peak_bytes=budget.conservative_peak_sum)
        atomic_json(out / "failure.json", result)
        raise
    finally:
        signal.signal(signal.SIGTERM, prior_handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(); contract = D._json(args.execution_contract)
    if args.execute: execute(contract, args.out)
    else:
        if args.out.exists(): raise FileExistsError("inspection output exists")
        result = inspect(contract); args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as f: json.dump(result, f, indent=2)


if __name__ == "__main__": main()
