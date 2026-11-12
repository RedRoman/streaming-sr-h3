"""Explicit TRAIN-role RGB guide cache for the fixed legacy_g05 final3000 R_phi.

This is independent of the development exporter and its dev-only contract.
Both complete materialized manifests are required; source IDs AND source-byte
hashes must be disjoint and match the checkpoint's original training contract.
Only observations enter the frozen restorer. Original RGB targets are verified
and referenced for later supervision, never copied into guide arrays. No VAE,
H3, latent cache, loss, or training is executed here. --inspect-only verifies
the actual checkpoint and all train/dev assets without constructing a model.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eval_rphi import TASKS, canonical_sha, causal_indices, infer_selected, sample_id, sha_file
from rphi_data import RphiDataset, assert_disjoint, prepare_observed


SCHEMA = "rphi.train_rgb_guide_export.v1"
CHECKPOINT_SHA256 = "61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617"
MODEL_CONFIG = {"variant": "blind", "k": 3, "ch": 64, "blocks": 8,
                "condition_dim": 6, "observation_encoder": "legacy"}
ARRAY_KEYS = {"raw_draft", "rphi_raw_rgb", "rphi_rgb"}
SEMANTICS = {
    "raw_draft": "float32 THWC [0,1] observed HR input; SR Torch bicubic antialias x4, native unchanged; NOT pretrained latent upscaler output",
    "rphi_raw_rgb": "float32 THWC prediction before output clipping; may be outside [0,1]",
    "rphi_rgb": "float32 THWC clip(rphi_raw_rgb,0,1), no RGB8 quantization",
    "target_usage": "original-recording clean proxy; integrity/geometry verification here, reference for future RGB supervision only; no target pixels enter restorer or guide arrays",
    "temporal": "all 22 original frames; K3 histories [max(0,t-2),max(0,t-1),t], no future, no batch-boundary history reset",
    "condition": "blind observations only; no task label, severity, target statistic, or hidden recipe enters restorer",
    "scope": "ordinary four-task TRAIN cache from fixed reference candidate; not held-out evaluation, quality approval, VAE encoding, or LoRA training",
}


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def _write_json(path, value, *, replace=False):
    path = Path(path)
    if not replace and (path.exists() or path.is_symlink()):
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("x") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _role_records(records, role):
    if not records:
        raise ValueError(role + " manifest is empty")
    sources, source_names, targets, seen = {}, {}, {}, set()
    for record in records:
        RphiDataset._check_record(record)
        if record["split"] != role or record.get("latent_path") is not None:
            raise ValueError("expected explicit " + role + " materialized RGB records")
        identity = sample_id(record)
        if identity in seen:
            raise ValueError("duplicate " + role + " sample")
        seen.add(identity)
        source, digest = record["source_id"], record["source_sha256"]
        if source in sources and sources[source] != digest:
            raise ValueError("source ID bound to different source SHA256")
        if digest in source_names and source_names[digest] != source:
            raise ValueError("duplicate source bytes under different source IDs")
        sources[source], source_names[digest] = digest, source
        key = record["target_key"]
        binding = tuple(record[k] for k in ("source_id", "source_sha256", "target_npz", "target_pixel_sha256"))
        if key not in targets:
            targets[key] = {"binding": binding, "tasks": set()}
        if targets[key]["binding"] != binding or record["task"] in targets[key]["tasks"]:
            raise ValueError("inconsistent or duplicate target/task binding")
        targets[key]["tasks"].add(record["task"])
    if any(item["tasks"] != set(TASKS) for item in targets.values()):
        raise ValueError("every target must contain exactly all four tasks")
    return sources


def validate_train_contract(payload, train_sha, train_records, dev_sha, dev_records):
    """Pure metadata validation; never relabels training material as dev."""
    if payload.get("schema") != "rphi.checkpoint.v1":
        raise ValueError("expected Rphi v1 checkpoint")
    contract = payload.get("training_contract", {})
    if payload.get("step") != 3000 or contract.get("steps") != 3000:
        raise ValueError("only final3000 reference checkpoint is accepted")
    if payload.get("model_config") != MODEL_CONFIG:
        raise ValueError("expected fixed blind legacy K3 model configuration")
    if contract.get("target") != "clean_original_recording" or tuple(contract.get("task_order", [])) != TASKS:
        raise ValueError("checkpoint target/task contract mismatch")
    if contract.get("observation_encoder") != "legacy" or contract.get("k") != 3:
        raise ValueError("checkpoint observation/history contract mismatch")
    if contract.get("train_manifest_sha256") != train_sha or contract.get("dev_manifest_sha256") != dev_sha:
        raise ValueError("checkpoint train/dev manifest SHA256 mismatch")
    train_sources = _role_records(train_records, "train")
    dev_sources = _role_records(dev_records, "dev")
    if set(train_sources) & set(dev_sources):
        raise ValueError("train/dev source ID overlap")
    if set(train_sources.values()) & set(dev_sources.values()):
        raise ValueError("train/dev source SHA256 overlap")
    for field, actual in (("train_source_ids", set(train_sources)),
                          ("train_source_sha256", set(train_sources.values())),
                          ("dev_source_ids", set(dev_sources))):
        expected = contract.get(field)
        if not isinstance(expected, list) or len(expected) != len(set(expected)) or set(expected) != actual:
            raise ValueError("checkpoint complete source set mismatch: " + field)
    return {"train_source_ids": sorted(train_sources), "dev_source_ids": sorted(dev_sources),
            "train_source_sha256": sorted(train_sources.values()), "dev_source_sha256": sorted(dev_sources.values()),
            "source_id_overlap": 0, "source_sha256_overlap": 0,
            "train_samples": len(train_records), "dev_samples": len(dev_records),
            "scope": "complete contracted train/dev sets; no claim about arbitrary external datasets"}


def _implementation():
    import eval_rphi
    import rphi_data
    base = Path(__file__).resolve().parent
    paths = {"code/export_rphi_train_guides.py": Path(__file__).resolve(),
             "code/eval_rphi.py": Path(eval_rphi.__file__).resolve(),
             "code/rphi_data.py": Path(rphi_data.__file__).resolve(),
             "code/rphi_model.py": base / "rphi_model.py"}
    if any(path.parent != base for path in paths.values()):
        raise ValueError("inference dependency imported outside this code snapshot")
    return {key: sha_file(path) for key, path in paths.items()}


def inspect_inputs(checkpoint, train_manifest, dev_manifest, *, root_override=None):
    """CPU checkpoint deserialization + complete file/pixel preflight, no model."""
    checkpoint = Path(checkpoint).expanduser().resolve()
    digest = sha_file(checkpoint)
    if digest != CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 differs from fixed legacy_g05 final3000")
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if sha_file(checkpoint) != digest:
        raise ValueError("checkpoint changed while loading")
    train = RphiDataset(train_manifest, root_override=root_override, verify_hashes=True, cache_bytes=0)
    dev = RphiDataset(dev_manifest, root_override=root_override, verify_hashes=True, cache_bytes=0)
    isolation = validate_train_contract(payload, train.manifest_sha256, train.records,
                                        dev.manifest_sha256, dev.records)
    assert_disjoint(train, dev)
    implementation = _implementation()
    for key in ("code/rphi_data.py", "code/rphi_model.py"):
        if payload.get("implementation_sha256", {}).get(key) != implementation[key]:
            raise ValueError("checkpoint/inference implementation mismatch: " + key)
    preflight = {}
    for role, dataset in (("train", train), ("dev", dev)):
        preflight[role] = dataset.preflight()
        # Inspect all records, even when a later export requests a smoke subset.
        for index in range(len(dataset)):
            loaded = dataset.load(index)
            if len(loaded["target"]) != 22 or len(loaded["observed"]) != 22:
                raise ValueError("expected all 22 original frames")
        dataset.clear_cache()
    if sha_file(checkpoint) != digest or sha_file(train.manifest_path) != train.manifest_sha256 or sha_file(dev.manifest_path) != dev.manifest_sha256:
        raise ValueError("checkpoint or complete manifest changed during inspection")
    proof = {"schema": SCHEMA, "inspection_complete": True, "rgb_export_complete": False,
             "model_constructed": False, "model_forward_calls": 0, "vae_encoded": False, "latent_path": None,
             "checkpoint": str(checkpoint), "checkpoint_sha256": digest, "checkpoint_step": 3000,
             "candidate": "legacy_g05 final3000 fixed reference, not quality-approved", "model_config": dict(MODEL_CONFIG),
             "role": "train", "split": "train", "condition_mode": "observations_only",
             "train_manifest": str(train.manifest_path), "train_manifest_sha256": train.manifest_sha256,
             "dev_manifest": str(dev.manifest_path), "dev_manifest_sha256": dev.manifest_sha256,
             "source_isolation": isolation, "preflight": preflight,
             "training_contract": payload["training_contract"],
             "checkpoint_implementation_sha256": payload["implementation_sha256"],
             "implementation_sha256": implementation, "semantics": SEMANTICS}
    return proof, payload, train, dev


def predict_arrays(model, observed, target_shape, device="cpu", frame_batch=1):
    """The sole prediction boundary takes observations and geometry, no target pixels."""
    if len(target_shape) != 4 or target_shape[0] != 22 or target_shape[-1] != 3:
        raise ValueError("expected full 22-frame THWC target geometry")
    guide = prepare_observed(observed, target_shape[1:3])
    if len(guide) != 22:
        raise ValueError("observations must contain all 22 frames")
    result = infer_selected(model, guide, None, list(range(22)), device, batch_size=frame_batch)
    raw = np.ascontiguousarray(result["rgb"], dtype=np.float32)
    draft = np.ascontiguousarray(guide.permute(0, 2, 3, 1).cpu().numpy(), dtype=np.float32)
    if raw.shape != tuple(target_shape) or not np.isfinite(raw).all() or not np.isfinite(draft).all():
        raise ValueError("invalid full-frame prediction/input")
    if result["log_variance"] is not None:
        raise ValueError("fixed blind reference has no confidence head")
    return {"raw_draft": draft, "rphi_raw_rgb": raw, "rphi_rgb": np.clip(raw, 0, 1)}


def verify_cache(path, row):
    path = Path(path)
    if sha_file(path) != row["guide_file_sha256"] or path.stat().st_size != row["guide_file_bytes"]:
        raise ValueError("guide cache file SHA256/bytes mismatch")
    if set(row["arrays"]) != ARRAY_KEYS:
        raise ValueError("unexpected guide arrays; target/latent arrays are prohibited")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != ARRAY_KEYS:
            raise ValueError("unexpected guide cache fields")
        for name, spec in row["arrays"].items():
            value = archive[name]
            if value.dtype != np.float32 or spec["dtype"] != "float32" or list(value.shape) != spec["shape"] or not np.isfinite(value).all():
                raise ValueError("guide array dtype/geometry/finite mismatch")
            if array_sha(value) != spec["array_sha256"]:
                raise ValueError("guide array SHA256 mismatch")
        if not np.array_equal(archive["rphi_rgb"], np.clip(archive["rphi_raw_rgb"], 0, 1)):
            raise ValueError("clipped prediction mismatch")
        draft = archive["raw_draft"]
        if draft.min() < 0 or draft.max() > 1:
            raise ValueError("raw draft must be normalized observed HR RGB")
    return True


def _state_sha(model):
    # R_phi inference is explicitly float32; covers parameters AND buffers.
    return canonical_sha({key: {"shape": list(value.shape), "dtype": str(value.dtype),
                               "sha256": array_sha(value.detach().cpu().numpy())}
                          for key, value in model.state_dict().items()})


def export_train_guides(checkpoint, train_manifest, dev_manifest, out, *, device="cpu",
                        frame_batch=1, max_sources=None, root_override=None):
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    for name, value in (("frame_batch", frame_batch), ("max_sources", max_sources)):
        if value is None and name == "max_sources":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(name + " must be a positive integer")
    if device != "cpu" and not (isinstance(device, str) and device.startswith("npu:") and device[4:].isdigit()):
        raise ValueError("device must be cpu or npu:<logical-index>")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        proof, payload, train, dev = inspect_inputs(checkpoint, train_manifest, dev_manifest, root_override=root_override)
        _write_json(out / "inspection.json", proof)
        import torch
        import rphi_model
        if Path(rphi_model.__file__).resolve().parent != Path(__file__).resolve().parent or rphi_model.MODEL_SCHEMA != "rphi.canonical_rgb.v1":
            raise ValueError("unsupported or externally imported RphiModel")
        if device.startswith("npu:"):
            import torch_npu  # noqa: F401 -- register device only for an explicitly requested export
            torch.npu.set_device(device)
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
        model = rphi_model.RphiModel(**payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Rphi reference must be fully frozen")
        state_before = _state_sha(model)
        sources = proof["source_isolation"]["train_source_ids"]
        selected_sources = sources[:max_sources] if max_sources is not None else sources
        scope = "complete_contracted_training_set" if selected_sources == sources else "engineering_training_subset"
        (out / "samples").mkdir()
        for index, record in enumerate(train.records):
            if record["source_id"] not in selected_sources:
                continue
            observation_path, target_path = (train.resolve_path(record[key]) for key in ("observation_npz", "target_npz"))
            input_sha, target_sha = sha_file(observation_path), sha_file(target_path)
            loaded = train.load(index)
            target_shape = tuple(loaded["target"].shape)
            arrays = predict_arrays(model, loaded["observed"], target_shape, device, frame_batch)
            if sha_file(observation_path) != input_sha or sha_file(target_path) != target_sha:
                raise ValueError("input/supervision asset changed during export")
            identity = sample_id(record)
            relative = Path("samples") / (identity + ".npz")
            with (out / relative).open("xb") as stream:
                np.savez_compressed(stream, **arrays)
            raw = arrays["rphi_raw_rgb"]
            row = {key: record[key] for key in ("source_id", "source_sha256", "target_key", "task", "split",
                                              "target_pixel_sha256", "observation_pixel_sha256")}
            row.update(schema=SCHEMA, complete=True, role="train", scope=scope, sample_id=identity,
                       record_sha256=canonical_sha(record), train_manifest_sha256=train.manifest_sha256,
                       dev_manifest_sha256=dev.manifest_sha256, checkpoint=proof["checkpoint"],
                       checkpoint_sha256=proof["checkpoint_sha256"], checkpoint_step=3000,
                       model_config=payload["model_config"], condition_mode="observations_only",
                       observation_npz=str(observation_path), observation_file_sha256=input_sha,
                       target_npz=str(target_path), target_file_sha256=target_sha,
                       target_role="supervision_reference_only_never_model_input",
                       geometry={"frames": 22, "height": target_shape[1], "width": target_shape[2], "channels": 3,
                                 "layout": "THWC", "dtype": "float32", "observation_shape": list(loaded["observed"].shape)},
                       frame_indices=list(range(22)), history_indices=causal_indices(range(22), 3, 22).tolist(),
                       guide_npz=str(relative), guide_file_sha256=sha_file(out / relative),
                       guide_file_bytes=(out / relative).stat().st_size,
                       arrays={key: {"shape": list(value.shape), "dtype": "float32", "array_sha256": array_sha(value)}
                               for key, value in arrays.items()},
                       clipping={"component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                                 "component_count": int(((raw < 0) | (raw > 1)).sum()),
                                 "raw_min": float(raw.min()), "raw_max": float(raw.max())},
                       vae_encoded=False, latent_path=None, vae_roundtrip_path=None,
                       semantics=SEMANTICS, implementation_sha256=proof["implementation_sha256"])
            verify_cache(out / relative, row)
            sidecar = relative.with_suffix(".json")
            _write_json(out / sidecar, row)
            rows.append({**row, "provenance_json": str(sidecar), "provenance_file_sha256": sha_file(out / sidecar)})
            _write_json(out / "partial_manifest.json", {"schema": SCHEMA, "complete": False, "role": "train",
                        "completed_samples": len(rows), "per_sample": rows}, replace=True)
        if _state_sha(model) != state_before or any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError("frozen Rphi reference changed during export")
        if _implementation() != proof["implementation_sha256"]:
            raise ValueError("inference implementation changed during export")
        for path, expected in ((checkpoint, proof["checkpoint_sha256"]), (train.manifest_path, train.manifest_sha256),
                               (dev.manifest_path, dev.manifest_sha256)):
            if sha_file(path) != expected:
                raise ValueError("checkpoint or full manifest changed during export")
        for dataset in (train, dev):
            for path, expected in dataset.receipt_hashes.items():
                if sha_file(path) != expected:
                    raise ValueError("materialization receipt changed during export")
        manifest = {**proof, "complete": True, "rgb_export_complete": True,
                    "status": "rgb_guides_complete_not_vae_encoded", "scope": scope, "role": "train", "split": "train",
                    "model_constructed": True, "model_forward_calls": len(rows) * ((22 + frame_batch - 1) // frame_batch),
                    "device": device, "frame_batch": frame_batch, "source_ids": selected_sources,
                    "samples": len(rows), "per_sample": rows,
                    "frozen_state_sha256_before": state_before, "frozen_state_sha256_after": state_before,
                    "all_parameters_frozen": True, "original_rgb_supervision_cached": False,
                    "vae_encoded": False, "latent_path": None, "vae_roundtrip_path": None,
                    "controls_included": [], "guide_training_steps": 0, "h3_sampling_calls": 0}
        _write_json(out / "manifest.json", manifest)
        return manifest
    except Exception as error:
        _write_json(out / "failure.json", {"schema": SCHEMA, "complete": False, "role": "train",
                    "completed_samples": len(rows), "error": type(error).__name__ + ": " + str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="new directory; required only for actual RGB export")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-batch", default=1, type=int)
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--root-override", type=Path)
    args = parser.parse_args(argv)
    if args.inspect_only:
        if args.out is not None or args.device != "cpu":
            parser.error("--inspect-only requires CPU and no output cache directory")
        proof, _, _, _ = inspect_inputs(args.checkpoint, args.train_manifest, args.dev_manifest, root_override=args.root_override)
        print(json.dumps(proof, indent=2, allow_nan=False))
    else:
        if args.out is None:
            parser.error("--out is required for an actual export")
        kwargs = vars(args).copy()
        kwargs.pop("inspect_only")
        result = export_train_guides(**kwargs)
        print(json.dumps({"manifest": str(args.out / "manifest.json"), "complete": result["complete"],
                          "samples": result["samples"], "scope": result["scope"], "vae_encoded": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
