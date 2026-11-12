"""One-shot CPU-only terminal audit of the fixed ordinary TRAIN VAE cache.

No waiting, model loading, NPU access, producer replay, repair or quality gate.
Requires a separately frozen admission; an existing output directory is refused.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from audit_arrays import array_sha, file_sha, require, verified_npy
from audit_inputs import (CHECKPOINT_SHA, GUIDE_MANIFEST_SHA, LATENT_SHAPE, RGB_SHAPE,
                          SOURCE, SOURCE_SHA, audit_guide_inputs)
from small_archive import package


ROLES = {"raw_warm_start_latent": ("Z_raw", LATENT_SHAPE),
         "rphi_guide_latent": ("Z_R", LATENT_SHAPE),
         "vae_rphi_as_returned_internal_clipped": ("VAE_R", RGB_SHAPE)}
SOURCE_FILES = {"audit_terminal.py", "audit_arrays.py", "audit_inputs.py", "small_archive.py"}


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def admission(root, own, path, expected_sha):
    root, own, path = (Path(p).resolve() for p in (root, own, path))
    require(own.is_relative_to(root) and path.parent == own and file_sha(path) == expected_sha,
            "unreviewed admission path/SHA")
    config = read_json(path)
    fields = {"schema", "run", "prep", "producer_output_subdir", "out", "producer_sha256",
              "execution_contract_sha256", "queue_sha256", "source_sha256"}
    require(set(config) == fields and config["schema"] == "openh3.train_vae_cache_cpu_audit_admission.v1",
            "unknown admission fields/schema")
    paths = {}
    for key, parent in (("run", "runs"), ("prep", "preparation"), ("out", "runs")):
        relative = Path(config[key])
        require(not relative.is_absolute() and len(relative.parts) == 2 and relative.parts[0] == parent
                and ".." not in relative.parts, "unbounded workspace path")
        paths[key] = (root / relative).resolve()
        require(paths[key].is_relative_to(root), "workspace path escape")
    subdir = Path(config["producer_output_subdir"])
    require(len(subdir.parts) == 1 and subdir.name not in (".", "..") and not subdir.is_absolute(),
            "producer output must be one child directory")
    paths["evidence"] = (paths["run"] / subdir).resolve()
    require(paths["evidence"].parent == paths["run"] and paths["run"] != paths["out"]
            and paths["prep"] != own, "producer/auditor overlap")
    require(set(config["source_sha256"]) == SOURCE_FILES, "incomplete auditor source lock")
    pins = {own / name: digest for name, digest in config["source_sha256"].items()}
    pins.update({paths["prep"] / "code/export_train_vae_cache.py": config["producer_sha256"],
                 paths["prep"] / "execution_contract.json": config["execution_contract_sha256"],
                 paths["prep"] / "queue.json": config["queue_sha256"]})
    for file, expected in pins.items():
        require(isinstance(expected, str) and len(expected) == 64 and file_sha(file) == expected,
                "frozen source/producer/contract/queue mismatch: " + str(file))
    return config, paths


def budget(report, success):
    actual = report["actual"]
    limits = {"encode_calls": 2, "decode_calls": 1, "stage_encode_calls": 2, "stage_decode_calls": 1}
    for key, limit in limits.items():
        require(type(actual[key]) is int and 0 <= actual[key] <= limit, "operation budget exceeded: " + key)
        if success:
            require(actual[key] == limit, "successful operation count incomplete: " + key)
    for key in ("Rphi_calls", "H3_sampler_calls", "NFE", "backward_calls", "optimizer_steps"):
        require(type(actual[key]) is int and actual[key] == 0, "forbidden model/optimization work: " + key)
    require(actual["forbidden_calls"] == [], "forbidden tiled/retry/model operation attempted")
    calls = actual["PIL_fromarray_calls"]
    require(set(calls) <= {"raw", "rphi"}
            and all(type(n) is int and 0 <= n <= 22 for n in calls.values()), "PIL frame budget differs")
    if success:
        require(calls == {"raw": 22, "rphi": 22}, "missing actual PIL frames")
    return actual


def loader_modes(assets):
    require(assets["VAE_and_context_actual_SHA"] is True and assets["DiT_loaded"] is False
            and assets["Rphi_model_constructed"] is False and assets["Rphi_checkpoint_deserialized_on_CPU"] is True
            and assets["actual_vae_dtype"] == "torch.float16",
            "unexpected model construction, inspection omission or deployed VAE dtype")


def input_binding(report, independent, contract):
    inspection, row = report["inspection"], independent["row"]
    require(inspection["schema"] == "openh3.train_vae_cache_input_inspection.v1" and inspection["complete"] is True,
            "missing completed input inspection")
    require(inspection["role"] == "train" and inspection["source_id"] == SOURCE
            and inspection["source_sha256"] == SOURCE_SHA and inspection["task"] == "compression"
            and inspection["sample_id"] == row["sample_id"] and inspection["export_row"] == row,
            "train/source/row identity differs")
    record = inspection["record"]
    require(record["split"] == "train" and record["source_id"] == SOURCE and record["source_sha256"] == SOURCE_SHA
            and record["task"] == "compression" and record["target_key"] == row["target_key"]
            and canonical(record) == inspection["record_sha256"] == row["record_sha256"],
            "original training record binding differs")
    require(contract["selected_sample_id"] == row["sample_id"]
            and contract["checkpoint_sha256"] == CHECKPOINT_SHA
            and contract["ordinary_guides_manifest_sha256"] == GUIDE_MANIFEST_SHA,
            "execution selected sample/legacy cache pin differs")
    root = Path(contract["root"]).resolve()
    expected_inputs = {}
    for name in ("checkpoint", "train_manifest", "dev_manifest", "ordinary_guides_manifest", "pretrained_manifest", "source_contract"):
        path = (root / contract[name]).resolve()
        key = "manifest_sha256" if name == "pretrained_manifest" else name + "_sha256"
        require(path.is_relative_to(root) and file_sha(path) == contract[key], "contracted input file SHA changed: " + name)
        expected_inputs[str(path)] = contract[key]
    require(inspection["input_files"] == expected_inputs, "producer input-file pin map differs")
    for path, digest in ((independent["guide_npz"], independent["guide_file_sha256"]),
                         (row["observation_npz"], row["observation_file_sha256"]),
                         (row["target_npz"], row["target_file_sha256"])):
        require(inspection["selected_asset_files"].get(path) == digest, "selected NPZ pin map differs")
    require(inspection["target_in_model_input"] is False and report["target_model_input"] is False
            and report["target_encode_calls"] == 0
            and inspection["model_input_arrays"] == ["raw_draft", "rphi_rgb"], "target or unexpected model input")
    require(inspection["geometry"] == RGB_SHAPE and inspection["expected_latent_shape"] == LATENT_SHAPE,
            "input/latent geometry differs")
    require(report["warm_start_role"] == "raw_warm_start_latent fixed for both future direct and compensation arms"
            and report["compensation_R"] == "existing float32 clipped rphi_rgb BEFORE deployed P uint8 preprocessing",
            "raw warm-start or compensation anchor semantics changed")
    require(inspection["model_constructed"] is False
            and all(inspection[k] == 0 for k in ("VAE_calls", "H3_calls", "Rphi_calls")),
            "CPU input inspection executed a model")
    assets = report["asset_verification"]
    loader_modes(assets)
    pretrained = read_json(root / contract["pretrained_manifest"])
    vae_record = report["pretrained"]["vae_only"]
    require(vae_record["weights"] == pretrained["vae"] and vae_record["component_binding_restored_after_load"] is True
            and vae_record["dit_loaded"] is False and vae_record["sampler_calls"] == vae_record["denoiser_evaluations"] == 0,
            "VAE-only loader identity/binding differs")
    for role in ("vae", "context"):
        asset, actual = pretrained[role], vae_record["component_binding"][role]
        manifest_file = root / contract["pretrained_manifest"]
        asset_path = (manifest_file.parent / asset["path"]).resolve()
        require(actual["path"] == str(asset_path) and actual["sha256"] == asset["sha256"]
                and actual["bytes"] == asset["bytes"], "VAE/context component binding differs")
    return {"role": "train", "source_id": SOURCE, "task": "compression", "sample_id": row["sample_id"],
            "target_not_in_model_input": "source-bound boundary/actual encode input hashes; no model replay",
            "target_pixels_loaded_by_auditor": False, "original_source_verification": independent["source_video_verification"],
            "warm_start": "Z_raw", "R": "clipped Rphi before P", "guide_manifest_sha256": GUIDE_MANIFEST_SHA}


def preprocessing(report, independent):
    require(set(report["observed_preprocessing"]) == {"raw", "rphi"}, "missing actual preprocessing observer")
    result = {}
    for label, facts in independent["preprocessing"].items():
        declared = report["inspection"]["preprocessing"][label]
        observed = report["observed_preprocessing"][label]
        for actual, expected in (("before_P_float32_sha256", "before_float32_sha256"),
                                 ("PIL_uint8_sha256", "pil_uint8_sha256"),
                                 ("after_P_float32_sha256", "after_float32_sha256")):
            require(declared[actual] == facts[expected], "independent PIL preprocessing hash differs: " + label)
        require(declared["shape"] == observed["shape"] == RGB_SHAPE
                and declared["dtype"] == observed["dtype"] == "float32"
                and declared["PIL_frames"] == observed["PIL_calls"] == 22
                and observed["matches_CPU_preprocessing"] is True
                and observed["actual_PIL_uint8_sha256"] == facts["pil_uint8_sha256"]
                and observed["actual_vae_input_sha256"] == facts["after_float32_sha256"],
                "actual model input/PIL hashes or geometry differ")
        result[label] = facts
    return {"recomputed": result, "extra_model_calls": 0,
            "scope": "same geometry official PIL truncation and FP32 restoration; no custom resize bypass"}


def frozen(report, evidence, success):
    evidence = Path(evidence).resolve()
    values = {}
    for name in ("frozen_initial", "frozen_final"):
        if name not in report:
            require(not success, "missing frozen inventory"); continue
        row = report[name]; path = (evidence / row["path"]).resolve()
        require(path.parent == evidence and file_sha(path) == row["sha256"], "frozen JSON path/SHA mismatch")
        inventory = read_json(path)
        for group in ("parameters", "registered_buffers"):
            records = inventory[group]
            require(len({r["name"] for r in records}) == len(records), "duplicate frozen tensor name")
            content = [{key: r[key] for key in ("name", "shape", "dtype", "sha256_bytes")} for r in records]
            require(canonical(content) == inventory[group + "_sha256"], "frozen canonical fingerprint differs")
            if success:
                require(all(r["requires_grad"] is False and r["grad_is_none"] is True
                            and r["device"] == "npu:0" for r in records), "frozen/resident tensor state differs")
        require(inventory["parameters"], "empty model fingerprint")
        values[name] = inventory
    if success:
        for group in ("parameters", "registered_buffers"):
            require(values["frozen_initial"][group + "_sha256"] == values["frozen_final"][group + "_sha256"],
                    "frozen parameter/buffer bytes changed")
            before, after = values["frozen_initial"][group], values["frozen_final"][group]
            require(len(before) == len(after) and all(a["name"] == b["name"] and a["version"] == b["version"]
                    and a["device"] == b["device"] for a, b in zip(before, after)), "frozen version/device records changed")
        require(report["frozen_final"]["parameter_bytes_unchanged"] is True
                and report["frozen_final"]["registered_buffer_bytes_unchanged"] is True, "producer frozen bytes failed")
        check = report["frozen_check"]
        count = sum(len(values["frozen_final"][key]) for key in ("parameters", "registered_buffers"))
        require(check["passed"] is True and check["failures"] == [] and check["tensors_checked"] == count,
                "frozen object/version/device/gradient check failed")
    return {"inventory_phases": sorted(values),
            "scope": "Independent canonical JSON/file hashes; parameter/buffer bytes and object/version facts are producer-recorded. No live model re-read or Python-cache inventory."}


def runtime(report, contract):
    before = report["math_before"]
    require(before == report["math_after"] and report["math_restored"] is True
            and report["wrappers_restored"] is True, "global math or wrapper restoration failed")
    require(before == {"in_training": False, "jit_compile_false": True,
                       "matmul_allow_hf32": False, "conv_allow_hf32": False}, "deployed inference math differs")
    expected = {str((Path(contract["root"]) / key).resolve()): value
                for key, value in contract["implementation_sha256"].items()}
    for phase in ("actual_imports_before", "actual_imports"):
        imports = report[phase]
        require(imports["passed"] is True and imports["unbound"] == [] and imports["modules"], "unbound actual import")
        for entry in imports["modules"].values():
            require(expected.get(entry["path"]) == entry["sha256"], "import source not pinned")
        require(set(imports["synthetic_namespaces"]) == {"torch.ops", "torch.classes"}, "missing Torch definition binding")
        for entry in imports["synthetic_namespaces"].values():
            require(expected.get(entry["defining_source_path"]) == entry["defining_source_sha256"], "Torch definition hash differs")
        require("comfy_kitchen.backends" in imports["namespace_packages"], "missing PEP420 evidence")
        for entry in imports["namespace_packages"].values():
            require(entry["loader_type"] == "_frozen_importlib_external.NamespaceLoader" and entry["source_files"],
                    "invalid namespace loader evidence")
            subset = {p: h for p, h in expected.items() if Path(p).suffix == ".py"
                      and any(Path(p).is_relative_to(Path(d)) for d in entry["directories"])}
            require(subset == entry["source_files"], "namespace pinned source tree differs")
    require(report["native_geometry_plan"]["policy_changed"] is False,
            "native VAE tile/chunk policy changed")
    return {"before_after_restored": True, "import_phases": 2,
            "scope": "Source-bound actual import/mode/restoration records, not model replay"}


def output_relations(report, loaded):
    raw, rphi = loaded["raw_warm_start_latent"], loaded["rphi_guide_latent"]
    decoded = report["actual_decode_input"]
    require(decoded["array_sha256"] == array_sha(rphi) and decoded["shape"] == LATENT_SHAPE
            and decoded["dtype"] == "torch.float32", "decoder did not consume exact saved Z_R")
    require(set(report["stage_inputs"]) == {"encode_raw", "encode_rphi", "decode_rphi"},
            "missing/repeated VAE whole-stage calls")
    stage = report["stage_inputs"]["decode_rphi"]
    require(stage["shape"] == LATENT_SHAPE and stage["dtype"] == "torch.float16"
            and stage["array_sha256"] == array_sha(rphi.astype(np.float16)),
            "actual FP16 decode stage input differs from saved Z_R cast")
    # The other actual stage inputs have source-recorded hashes; normalization
    # and internal native chunks are not rerun or inferred from output values.
    for label in ("encode_raw", "encode_rphi"):
        entry = report["stage_inputs"][label]
        require(entry["dtype"] == "torch.float16" and entry["shape"] == [1, 3, 22, 480, 832]
                and isinstance(entry["array_sha256"], str) and len(entry["array_sha256"]) == 64,
                "actual encode stage dtype/geometry/evidence differs")
    return {"Z_raw_sha256": array_sha(raw), "Z_R_sha256": array_sha(rphi),
            "Z_raw_equals_Z_R": bool(np.array_equal(raw, rphi)), "decode_input_Z_R_exact": True,
            "FP16_stage_decode_cast_exact": True,
            "latent_equality_is_not_a_quality_gate": True,
            "VAE_R_semantics": "as returned by H3 decode, after VAE internal clamp; no pre-clamp claim"}


def audit(root, config, paths):
    evidence = paths["evidence"]
    queue_path = paths["run"] / "queue_status.json"; queue = read_json(queue_path)
    require(queue["state"] in ("completed", "failed", "not_launched"), "producer is not terminal")
    result = {"schema": "openh3.train_vae_cache_terminal_audit.v1", "complete": False,
              "producer_terminal_state": queue["state"], "producer_output": str(evidence),
              "queue_status_sha256": file_sha(queue_path), "checks": {}, "errors": [], "archive_arrays": {},
              "actions": {"NPU_calls": 0, "model_loads": 0, "producer_writes": 0, "target_pixel_arrays_loaded": 0},
              "quality_verdict": "not_assessed", "quality_thresholds": None,
              "scope": "one already accepted TRAIN source, ordinary compression; cache engineering only"}
    success = queue["state"] == "completed"
    if queue["state"] == "not_launched":
        return {**result, "audit_complete": True, "engineering_verdict": "not_launched_not_accepted"}
    candidates = [evidence / ("manifest.json" if success else "failure.json"), evidence / "partial.json"]
    chosen = next((p for p in candidates if p.is_file()), None)
    if chosen is None:
        return {**result, "audit_complete": True, "engineering_verdict": "producer_failed_no_report" if not success else "missing_completed_report",
                "errors": ["No producer report; inspection/load may have failed before evidence creation."]}
    report = read_json(chosen)
    require(report["schema"] == "openh3.train_vae_cache.v1" and report["complete"] is success,
            "queue/report completion or schema mismatch")
    result["report_sha256"] = file_sha(chosen)
    result["producer_failure"] = report.get("failure")
    contract = read_json(paths["prep"] / "execution_contract.json")
    require(contract["schema"] == "openh3.train_vae_cache_execution.v1"
            and Path(contract["root"]).resolve() == Path(root).resolve()
            and contract["entry_sha256"] == config["producer_sha256"], "execution identity differs")
    require(report["contract"] == {**contract, "execution_contract_sha256": config["execution_contract_sha256"]},
            "embedded producer contract differs")
    def check(label, operation, available=True):
        if not available:
            result["checks"][label] = {"status": "unavailable"}; return None
        try:
            facts = operation(); result["checks"][label] = {"status": "verified", "facts": facts}; return facts
        except Exception as error:
            result["checks"][label] = {"status": "invalid", "error": repr(error)}
            result["errors"].append(label + ": " + str(error)); return None
    check("operation_budget", lambda: budget(report, success))
    independent = check("actual_guide_and_observation", lambda: audit_guide_inputs(root,
        Path(root) / contract["ordinary_guides_manifest"]), success)
    if independent is not None:
        check("train_input_contract", lambda: input_binding(report, independent, contract))
        check("actual_PIL_preprocessing", lambda: preprocessing(report, independent))
    loaded = {}
    require(set(report["arrays"]) <= set(ROLES), "unexpected cached array role")
    for role, (short, shape) in ROLES.items():
        if role not in report["arrays"]:
            if success: result["errors"].append("missing successful array: " + role)
            continue
        row = report["arrays"][role]
        try:
            path = (evidence / row["path"]).resolve()
            require(path.parent == evidence / "arrays", "array escaped producer cache")
            value, facts = verified_npy(path, row, shape)
            if short == "VAE_R":
                require(facts["min"] >= 0 and facts["max"] <= 1, "as-returned internal-clipped RGB outside range")
            loaded[role] = value
            result["checks"]["actual_array_" + short] = {"status": "verified", "facts": facts}
            if short != "VAE_R":
                result["archive_arrays"][short] = {"path": str(path), "file_sha256": row["file_sha256"]}
        except Exception as error:
            result["errors"].append(role + ": " + str(error))
    if set(loaded) == set(ROLES):
        check("actual_latent_decode_relations", lambda: output_relations(report, loaded))
    check("frozen_parameters_and_registered_buffers", lambda: frozen(report, evidence, success),
          success or "frozen_initial" in report)
    check("runtime_and_restoration", lambda: runtime(report, contract), success)
    result["audit_complete"] = True
    result["complete"] = success and not result["errors"]
    result["engineering_verdict"] = "passed_single_train_cache" if result["complete"] else (
        "producer_failed_not_accepted" if not success else "independent_audit_failed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args(); root = args.root.resolve(); own = Path(__file__).resolve().parent
    config, paths = admission(root, own, args.config, args.config_sha256)
    out = paths["out"]; out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    receipt = {"schema": "openh3.train_vae_cache_cpu_audit_invocation.v1", "pid": os.getpid(),
               "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "admission_sha256": args.config_sha256, "admission": config,
               "NPU_calls": 0, "producer_replays": 0, "large_arrays_copied": 0}
    write_json(out / "invocation_receipt.json", receipt)
    try:
        result = audit(root, config, paths)
    except Exception as error:
        result = {"schema": "openh3.train_vae_cache_terminal_audit.v1", "complete": False,
                  "audit_complete": False, "engineering_verdict": "audit_failed",
                  "quality_verdict": "not_assessed", "errors": [repr(error)], "archive_arrays": {}}
    result["elapsed_seconds"] = time.monotonic() - start
    result["Torch_imported"] = "torch" in sys.modules; result["torch_npu_imported"] = "torch_npu" in sys.modules
    write_json(out / "terminal_audit.json", result)
    status = {"state": "completed" if result["complete"] else "failed", "complete": result["complete"],
              "audit_sha256": file_sha(out / "terminal_audit.json"), "quality_verdict": "not_assessed"}
    try:
        status["archive"] = package(root, paths["prep"], paths["run"], own, out, result)
    except Exception as error:
        status.update(state="failed", complete=False, archive_error=repr(error))
    write_json(out / "audit_status.json", status)
    print(json.dumps(status)); return 0 if status["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
