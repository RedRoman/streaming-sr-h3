"""Export current v1 R_phi RGB guides; scaffolding only, no H3/VAE execution.

Every original frame is inferred using the same consecutive causal K3 history
as eval_rphi.infer_selected. Only observations enter the model. Target pixels
are loaded by the dataset for provenance and geometry validation, never used
as guide pixels, labels, residual corrections, or identity-control inputs.

Each NPZ contains float32 THWC ``rgb`` clipped to [0,1], ``raw_rgb`` before
clipping, and optionally THW1 ``log_variance`` calibrated for raw RGB errors.
No RGB quantization or VAE encoding is performed. These are development-set
exports, not evidence of H3 integration, generalization, or improved quality.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_rphi import (TASKS, canonical_sha, causal_indices, infer_selected,
                       sample_id, sha_file, validate_contract)


SCHEMA = "rphi.guide_export.v1"


def array_sha(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()


def _write_json_new(path, value):
    """Publish complete JSON bytes only, in a newly owned output directory."""
    if path.exists():
        raise FileExistsError(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as stream:
        stream.write(payload)
    temporary.replace(path)


def verify_export_file(path, sample):
    """Verify saved NPZ bytes and each declared array; no model is loaded."""
    path = Path(path)
    if sha_file(path) != sample["guide_file_sha256"]:
        raise ValueError("export NPZ file SHA256 mismatch")
    expected = sample["arrays"]
    if set(expected) not in ({"rgb", "raw_rgb"}, {"rgb", "raw_rgb", "log_variance"}):
        raise ValueError("unsupported guide array fields")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("export NPZ array fields mismatch")
        for name, spec in expected.items():
            value = archive[name]
            if list(value.shape) != spec["shape"] or value.dtype != np.float32:
                raise ValueError("export array geometry/dtype mismatch: " + name)
            if spec["dtype"] != "float32" or not np.isfinite(value).all():
                raise ValueError("invalid export array: " + name)
            if array_sha(value) != spec["array_sha256"]:
                raise ValueError("export array SHA256 mismatch: " + name)
        if not np.array_equal(archive["rgb"], np.clip(archive["raw_rgb"], 0, 1)):
            raise ValueError("export RGB does not match clipped raw prediction")
    return True


def export_guides(checkpoint, dev_manifest, out, device="cpu", frame_batch=1,
                  max_sources=None):
    """Export a v1 checkpoint to a NEW directory and return manifest contents.

    ``frame_batch`` groups complete K3 histories; it never subsamples frames or
    resets history at batch boundaries. A failed run leaves failure.json and
    no completed manifest. Already written samples are not resumable implicitly.
    """
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if isinstance(frame_batch, bool) or not isinstance(frame_batch, int) or frame_batch < 1:
        raise ValueError("frame_batch must be a positive integer")
    if max_sources is not None and (isinstance(max_sources, bool) or
                                   not isinstance(max_sources, int) or max_sources < 1):
        raise ValueError("max_sources must be a positive integer")
    if device != "cpu" and not (isinstance(device, str) and device.startswith("npu:")
                                 and device[4:].isdigit()):
        raise ValueError("device must be cpu or npu:<logical-index>")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        import torch
        from rphi_data import RphiDataset, build_condition, prepare_observed
        from rphi_model import MODEL_SCHEMA, RphiModel

        checkpoint = Path(checkpoint).expanduser().resolve()
        dev_manifest = Path(dev_manifest).expanduser().resolve()
        checkpoint_sha = sha_file(checkpoint)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if sha_file(checkpoint) != checkpoint_sha:
            raise ValueError("checkpoint changed while loading")
        dataset = RphiDataset(dev_manifest, verify_hashes=True)
        manifest_sha = sha_file(dev_manifest)
        if dataset.manifest_sha256 != manifest_sha:
            raise ValueError("development manifest changed while loading")
        contract = validate_contract(payload, manifest_sha, dataset.records)
        config = payload["model_config"]
        if MODEL_SCHEMA != "rphi.canonical_rgb.v1" or config.get("k") != 3:
            raise ValueError("only the current v1 K3 RphiModel can be exported")
        sources = sorted({r["source_id"] for r in dataset.records})
        selected_sources = sources[:max_sources] if max_sources is not None else sources
        selected = [(i, r) for i, r in enumerate(dataset.records) if r["source_id"] in selected_sources]
        for source in selected_sources:
            if {r["task"] for _, r in selected if r["source_id"] == source} != set(TASKS):
                raise ValueError("each selected source must contain all four development tasks")
        if device.startswith("npu:"):
            import torch_npu  # noqa: F401 -- register NPU before selecting it
            torch.npu.set_device(device)
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
        model = RphiModel(**config)
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device).eval()
        condition_mode = "known" if config["variant"] == "known" else "observations_only"
        scope = "full_development_set" if len(selected_sources) == len(sources) else "engineering_smoke_subset"
        code_dir = Path(__file__).resolve().parent
        implementation = {"code/" + name: sha_file(code_dir / name) for name in
                          ("export_rphi_guides.py", "eval_rphi.py", "rphi_data.py", "rphi_model.py")}
        semantics = {
            "rgb": "float32 THWC clipped to [0,1]; no quantization; intended downstream RGB guide",
            "raw_rgb": "float32 THWC prediction before clipping; may be outside [0,1]",
            "log_variance": "optional float32 THW1 isotropic per-channel variance in log space for raw_rgb errors; calibration only",
            "input": "observed RGB only; SR uses prepare_observed bicubic HR grid; native observations unchanged",
            "target_usage": "dataset integrity and HR geometry validation only; no target pixels enter model or exported arrays",
            "temporal": "all original frames, consecutive t-2,t-1,t; first-frame repetition; no future frames",
            "scope_limit": "development export only; no VAE/H3 call, learned physical operator, or confidence-guided DC"}
        samples_dir = out / "samples"
        samples_dir.mkdir()
        for index, record in selected:
            observation_path = dataset.resolve_path(record["observation_npz"])
            target_path = dataset.resolve_path(record["target_npz"])
            input_sha, target_sha = sha_file(observation_path), sha_file(target_path)
            loaded = dataset.load(index)
            if loaded["record"] != record:
                raise ValueError("dataset returned a different sample record")
            observed = loaded["observed"]
            target_shape = loaded["target"].shape
            if len(target_shape) != 4 or target_shape[-1] != 3 or len(observed) != target_shape[0]:
                raise ValueError("target/observation geometry mismatch")
            guide = prepare_observed(observed, target_shape[1:3])
            frames = list(range(len(guide)))
            condition = build_condition(record["task"]) if condition_mode == "known" else None
            result = infer_selected(model, guide, condition, frames, device, batch_size=frame_batch)
            raw = np.ascontiguousarray(result["rgb"], dtype=np.float32)
            if raw.shape != target_shape or not np.isfinite(raw).all():
                raise ValueError("invalid full-frame guide prediction")
            arrays = {"rgb": np.clip(raw, 0, 1), "raw_rgb": raw}
            variance = result["log_variance"]
            if variance is not None:
                variance = np.ascontiguousarray(variance, dtype=np.float32)
                if variance.shape != raw.shape[:-1] + (1,) or not np.isfinite(variance).all():
                    raise ValueError("invalid guide log_variance")
                arrays["log_variance"] = variance
            if sha_file(observation_path) != input_sha or sha_file(target_path) != target_sha:
                raise ValueError("input/reference asset changed during export")
            identity = sample_id(record)
            relative_npz = Path("samples") / (identity + ".npz")
            npz_path = out / relative_npz
            with npz_path.open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            row = {key: record[key] for key in ("source_id", "source_sha256", "target_key", "task",
                   "split", "target_pixel_sha256", "observation_pixel_sha256")}
            row.update(schema=SCHEMA, complete=True, sample_id=identity, scope=scope,
                       checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
                       checkpoint_step=payload.get("step"), model_schema=MODEL_SCHEMA,
                       model_config=dict(config), condition_mode=condition_mode,
                       dev_manifest_sha256=manifest_sha, record_sha256=canonical_sha(record),
                       observation_npz=str(observation_path), observation_file_sha256=input_sha,
                       target_npz=str(target_path), target_file_sha256=target_sha,
                       geometry={"frames": raw.shape[0], "height": raw.shape[1], "width": raw.shape[2],
                                 "channels": 3, "layout": "THWC", "dtype": "float32"},
                       frame_indices=frames, history_indices=causal_indices(frames, 3, len(guide)).tolist(),
                       guide_npz=str(relative_npz), guide_file_sha256=sha_file(npz_path),
                       arrays={key: {"shape": list(value.shape), "dtype": "float32",
                                     "array_sha256": array_sha(value)} for key, value in arrays.items()},
                       clipping={"component_count": int(((raw < 0) | (raw > 1)).sum()),
                                 "component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                                 "raw_min": float(raw.min()), "raw_max": float(raw.max())},
                       semantics=semantics, implementation_sha256=implementation)
            verify_export_file(npz_path, row)
            sidecar = relative_npz.with_suffix(".json")
            _write_json_new(out / sidecar, row)
            rows.append({**row, "provenance_json": str(sidecar),
                         "provenance_file_sha256": sha_file(out / sidecar)})
        if sha_file(checkpoint) != checkpoint_sha or sha_file(dev_manifest) != manifest_sha:
            raise ValueError("checkpoint or development manifest changed during export")
        manifest = {"schema": SCHEMA, "complete": True, "scope": scope, "split": "dev",
                    "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
                    "checkpoint_step": payload.get("step"), "model_schema": MODEL_SCHEMA,
                    "model_config": dict(config), "condition_mode": condition_mode,
                    "dev_manifest": str(dev_manifest), "dev_manifest_sha256": manifest_sha,
                    "training_contract": contract, "checkpoint_implementation_sha256": payload.get("implementation_sha256"),
                    "implementation_sha256": implementation, "device": device, "frame_batch": frame_batch,
                    "source_ids": selected_sources, "available_sources": len(sources), "samples": len(rows),
                    "semantics": semantics, "dataset_provenance": dict(dataset.provenance),
                    "materialization_receipt_sha256": dict(dataset.receipt_hashes), "per_sample": rows}
        _write_json_new(out / "manifest.json", manifest)
        return manifest
    except Exception as error:
        _write_json_new(out / "failure.json", {"schema": SCHEMA, "complete": False,
                        "completed_samples": len(rows), "error": type(error).__name__ + ": " + str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-batch", type=int, default=1)
    parser.add_argument("--max-sources", type=int)
    args = parser.parse_args()
    manifest = export_guides(**vars(args))
    print(json.dumps({"manifest": str(args.out / "manifest.json"), "complete": manifest["complete"],
                      "samples": manifest["samples"], "scope": manifest["scope"]}))


if __name__ == "__main__":
    main()
