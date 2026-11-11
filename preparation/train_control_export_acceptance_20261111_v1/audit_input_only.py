"""One-shot CPU acceptance of eight completed TRAIN control RGB caches.

Reads the frozen checkpoint on CPU only to hash its state; constructs no model,
does no forward, scoring, VAE, H3, or training, and never queries/uses an NPU.
Original reference pixels are allowed ONLY for the explicitly labelled control
construction and this verification. The evidence archive excludes all NPZ/PT.
"""
import datetime
import json
import os
from pathlib import Path
import sys
import tarfile
import time

import numpy as np

ROOT = Path("/cache/zihaohe/omni_openh3_20261111")
PREP = ROOT / "preparation/train_control_export_20261111_v1"
OUT = Path(__file__).resolve().parent
RUN = ROOT / "runs/rphi_train_controls_legacy_smoke_20261111a"
QUEUE_SHA = "785ccf1f10da8a45cfb7ec655868f5a53e49ba4493e7b4bb8f67c95dc245979b"
sys.path.insert(0, str(PREP / "code"))
import export_rphi_train_controls as exporter
from rphi_data import RphiDataset, assert_disjoint
from eval_rphi import canonical_sha, causal_indices, sample_id, sha_file, TASKS


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def contained(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("export evidence must be a contained existing file: " + relative)
    return path


def main():
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        raise RuntimeError("run with TORCH_DEVICE_BACKEND_AUTOLOAD=0")
    if (OUT / "engineering_review.json").exists() or (OUT / "small_evidence.tar.gz").exists():
        raise FileExistsError("audit already exists; do not repeat completed acceptance")
    start = time.monotonic()
    queue_file = PREP / "queue_30794.json"
    assert sha_file(queue_file) == QUEUE_SHA
    plan, status = read(queue_file), read(RUN / "queue_status.json")
    assert status["state"] == "completed" and status["card"] == 6
    assert status["completed"] == [step["name"] for step in plan["steps"]]
    for relative, expected in plan["implementation_sha256"].items():
        assert sha_file(ROOT / relative) == expected, relative
    for guard in plan["required_json_values"]:
        assert read(Path(guard["path"])).get(guard["key"]) == guard["value"], guard
    launch = read(RUN / "launch_admission.json")
    for field in ("root_authorized_launch", "fresh_resource_check_passed",
                  "no_conflicting_owned_npu_processes", "prelaunch_review_allow"):
        assert launch[field] is True

    directory = RUN / "guides"
    manifest_file = directory / "manifest.json"
    m, prior = read(manifest_file), read(PREP / "inspection.json")
    planned_inputs = read(PREP / "selected_control_inputs.json")
    assert planned_inputs["passed"] and planned_inputs["samples"] == 8
    input_map = {row["sample_id"]: row for row in planned_inputs["per_sample"]}
    assert len(input_map) == 8
    assert m["schema"] == exporter.SCHEMA and m["complete"] and m["rgb_export_complete"]
    assert m["samples"] == 8 and len(m["per_sample"]) == 8
    assert m["role"] == m["split"] == "train" and m["source_ids"] == [exporter.SOURCE_ID]
    assert m["source_sha256"] == exporter.SOURCE_SHA256
    assert m["scope"] == "engineering_training_control_subset"
    assert m["controls"] == list(exporter.controls.CONTROLS)
    assert not m["ordinary_tasks_included"] and m["reference_derived_controls"]
    assert m["condition_mode"] == "observations_only" and m["model_config"] == exporter.ordinary.MODEL_CONFIG
    assert m["checkpoint_step"] == 3000 and m["frame_batch"] == 1 and m["model_forward_calls"] == 176
    assert m["all_parameters_frozen"] and not m["separate_supervision_array_cached"]
    assert not m["vae_encoded"] and m["latent_path"] is None
    assert m["guide_training_steps"] == m["h3_sampling_calls"] == 0
    assert m["semantics"] == exporter.SEMANTICS
    assert m["implementation_sha256"] == exporter._implementation()
    for key in ("checkpoint_sha256", "train_manifest_sha256", "dev_manifest_sha256", "source_isolation",
                "implementation_sha256", "training_contract", "semantics"):
        assert m[key] == prior[key], key

    # No model reload/constructor. CPU tensor deserialization verifies the actual
    # frozen checkpoint state, including buffers, against both export state hashes.
    checkpoint = Path(m["checkpoint"])
    assert sha_file(checkpoint) == exporter.ordinary.CHECKPOINT_SHA256 == m["checkpoint_sha256"]
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["model_config"] == m["model_config"] and payload["training_contract"] == m["training_contract"]
    state = {}
    for key, tensor in payload["state_dict"].items():
        assert tensor.device.type == "cpu"
        # Same floating-point dtype conversion as model.to(dtype=float32).
        value = tensor.float() if tensor.is_floating_point() else tensor
        state[key] = {"shape": list(value.shape), "dtype": str(value.dtype),
                      "sha256": exporter.ordinary.array_sha(value.numpy())}
    checkpoint_state_sha = canonical_sha(state)
    assert checkpoint_state_sha == m["frozen_state_sha256_before"] == m["frozen_state_sha256_after"]
    del payload, state

    train = RphiDataset(m["train_manifest"], verify_hashes=True, cache_bytes=256 * 1024**2)
    dev = RphiDataset(m["dev_manifest"], verify_hashes=True, cache_bytes=0)
    isolation = exporter.ordinary.validate_train_contract(
        {"schema": "rphi.checkpoint.v1", "step": 3000, "model_config": m["model_config"],
         "training_contract": m["training_contract"]},
        train.manifest_sha256, train.records, dev.manifest_sha256, dev.records)
    assert_disjoint(train, dev)
    assert isolation == m["source_isolation"]
    chosen = exporter.selected_records(train.records)
    originals = {sample_id(r): (train.records.index(r), r) for r in chosen}
    expected = {exporter.controls.control_id(r, c) for r in chosen for c in exporter.controls.CONTROLS}
    assert len(expected) == 8 and set(input_map) == expected
    assert {row["sample_id"] for row in m["per_sample"]} == expected
    assert {row["task"] for row in m["per_sample"]} == set(TASKS)
    rows, identity_hashes, near_hashes = [], {key: set() for key in exporter.ordinary.ARRAY_KEYS}, set()
    for row in m["per_sample"]:
        index, record = originals[row["parent_sample_id"]]
        before = input_map[row["sample_id"]]
        for key in ("sample_id", "parent_sample_id", "record_sha256", "source_id", "source_sha256", "split",
                    "task", "control", "target_key", "target_pixel_sha256", "target_file_sha256",
                    "original_observation_file_sha256", "control_recipe", "control_recipe_sha256",
                    "control_input_pixel_sha256", "reference_derived_control", "ordinary_observation_used"):
            assert row[key] == before[key], key
        for key in ("checkpoint_sha256", "checkpoint_step", "model_config", "train_manifest_sha256",
                    "dev_manifest_sha256", "implementation_sha256", "semantics"):
            assert row[key] == m[key], key
        assert row["role"] == "train" and row["condition_mode"] == "observations_only"
        assert row["target_role"] == "explicit_control_construction_and_supervision_reference_no_separate_target_model_input"
        assert row["frame_indices"] == list(range(22))
        assert row["history_indices"] == causal_indices(range(22), 3, 22).tolist()
        assert row["vae_encoded"] is False and row["latent_path"] is None
        sidecar = contained(directory, row["provenance_json"])
        assert sha_file(sidecar) == row["provenance_file_sha256"]
        assert read(sidecar) == {k: v for k, v in row.items() if k not in ("provenance_json", "provenance_file_sha256")}
        loaded = train.load(index)
        clean = loaded["target"]
        assert clean.shape == (22, 480, 832, 3)
        assert sha_file(train.resolve_path(record["target_npz"])) == row["target_file_sha256"]
        assert sha_file(train.resolve_path(record["observation_npz"])) == row["original_observation_file_sha256"]
        assert Path(row["target_npz"]) == train.resolve_path(record["target_npz"])
        assert Path(row["original_observation_npz"]) == train.resolve_path(record["observation_npz"])
        cache = contained(directory, row["guide_npz"])
        assert exporter.verify_control_cache(cache, row, record, clean)
        with np.load(cache, allow_pickle=False) as archive:
            assert set(archive.files) == exporter.ordinary.ARRAY_KEYS
            raw = archive["rphi_raw_rgb"]
            clipped = archive["rphi_rgb"]
            assert raw.shape == clipped.shape == (22, 480, 832, 3)
            clipping = {"component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                        "raw_min": float(raw.min()), "raw_max": float(raw.max())}
            assert clipping == row["clipping"]
            assert np.array_equal(clipped, np.clip(raw, 0, 1))
            if row["control"] == "identity":
                for key in identity_hashes:
                    identity_hashes[key].add(row["arrays"][key]["array_sha256"])
            else:
                near_hashes.add(row["control_input_pixel_sha256"])
            rows.append({key: row[key] for key in ("sample_id", "parent_sample_id", "task", "control", "source_id",
                "source_sha256", "target_key", "target_pixel_sha256", "control_recipe", "control_recipe_sha256",
                "control_input_pixel_sha256", "guide_file_sha256", "guide_file_bytes", "arrays", "clipping")})
            rows[-1].update(actual_npz_file_and_arrays_verified=True, actual_control_input_matches_prelaunch_hash=True,
                control_recipe_reconstructed=True, native_HR_even_for_nominal_SR=True,
                clamp_bitwise_verified=True, no_separate_target_or_latent_array=True)
        del raw, clipped, loaded, clean
        print(json.dumps({"verified": row["sample_id"], "task": row["task"], "control": row["control"]}), flush=True)
    assert len(identity_hashes["raw_draft"]) == 1 and len(near_hashes) == 4
    # Same deterministic blind model, HR identity pixels and scheduling must
    # yield the same predictions across nominal task labels too.
    assert all(len(values) == 1 for values in identity_hashes.values())
    assert sha_file(checkpoint) == m["checkpoint_sha256"]
    assert sha_file(train.manifest_path) == m["train_manifest_sha256"]
    assert sha_file(dev.manifest_path) == m["dev_manifest_sha256"]
    assert exporter._implementation() == m["implementation_sha256"]
    assert "torch_npu" not in sys.modules and "rphi_model" not in sys.modules
    assert not any(name == "comfy" or name.startswith("comfy.") for name in sys.modules)
    report = {"schema": "rphi.train_control_export.engineering_review.v1", "complete": True,
        "engineering_pass": True, "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "seconds": time.monotonic() - start, "run": str(RUN), "role": "train", "sources": [exporter.SOURCE_ID],
        "samples": 8, "frames_per_sample": 22, "guide_manifest_sha256": sha_file(manifest_file),
        "queue_status_sha256": sha_file(RUN / "queue_status.json"), "queue_plan_sha256": QUEUE_SHA,
        "checkpoint_sha256": m["checkpoint_sha256"], "checkpoint_cpu_state_sha256": checkpoint_state_sha,
        "checkpoint_cpu_state_equals_export_before_and_after": True, "source_isolation": isolation,
        "prelaunch_control_input_hashes_match_actual": True, "actual_eight_npz_verified": True,
        "all_raw_drafts_match_explicit_reference_derived_controls": True, "all_causal_K3_histories_verified": True,
        "identity_task_repetitions_bitwise_identical": {key: True for key in identity_hashes},
        "near_clean_unique_task_keyed_inputs": len(near_hashes), "original_observation_used_for_controls": False,
        "separate_GT_tensor_or_array_in_prediction_interface": False,
        "target_usage": exporter.SEMANTICS["target_usage"], "VAE_calls": 0, "H3_calls": 0,
        "training_steps": 0, "vae_encoded": False, "audit_model_forwards": 0, "audit_NPU_kernels": 0,
        "audit_torch_version": torch.__version__, "audit_torch_npu_imported": False,
        "model_constructed_in_audit": False, "per_sample": rows,
        "limitations": ["Engineering cache acceptance, not quality/heldout evidence or a training result.",
            "Identity intentionally uses clean reference pixels as its labelled observation. This is not an ordinary blind-restoration sample.",
            "Original video SHA remains the materialization attestation. Current selected NPZ/receipt/file/pixel bindings and complete train/dev metadata are reverified.",
            "Four identity task labels share one source/window; they are not independent sources. Near-clean is four task-seeded draws on that same source.",
            "Frozen model state compared to CPU checkpoint tensors without model construction or repeat inference.",
            "Causal history and no separate GT model input follow hash-pinned source plus exported provenance and synthetic CPU causality tests; this audit does not rerun the model.",
            "Only one training source exported; other31 sources and all VAE/H3/LoRA work remain outside this cache acceptance."]}
    write(OUT / "engineering_review.json", report)
    files = [RUN / name for name in ("queue_status.json", "launch_admission.json", "logs/export.log",
             "guides/manifest.json", "guides/inspection.json", "guides/partial_manifest.json")]
    files += [contained(directory, row["provenance_json"]) for row in m["per_sample"]]
    files += [PREP / name for name in ("queue_30794.json", "prepared_admission.json", "inspection.json",
              "inspection_receipt.json", "selected_control_inputs.json", "cpu_receipt.json", "cpu_tests.log", "source_pins.json")]
    files += [PREP / name for name in read(PREP / "source_pins.json")]
    files += [OUT / "engineering_review.json", Path(__file__).resolve()]
    assert len(files) == len(set(files))
    assert all(path.is_file() and path.suffix in (".json", ".log", ".py") for path in files)
    hashes = {str(path.relative_to(ROOT)): sha_file(path) for path in files}
    write(OUT / "evidence_files.json", hashes)
    archive = OUT / "small_evidence.tar.gz"
    with tarfile.open(archive, "x:gz") as bundle:
        for path in files + [OUT / "evidence_files.json"]:
            bundle.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
    for relative, expected in hashes.items():
        assert sha_file(ROOT / relative) == expected, "small evidence changed while archiving"
    receipt = {"complete": True, "archive": str(archive), "archive_sha256": sha_file(archive),
        "archive_bytes": archive.stat().st_size, "verified_files": len(files),
        "engineering_review_sha256": sha_file(OUT / "engineering_review.json"),
        "guide_manifest_sha256": report["guide_manifest_sha256"], "NPZ_or_checkpoint_archived": False,
        "actual_npz_verified_remotely": True, "audit_model_forwards": 0, "audit_NPU_calls": 0}
    write(OUT / "archive_receipt.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
