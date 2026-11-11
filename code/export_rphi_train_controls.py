"""TRAIN-only identity/near-clean RGB cache for one already-audited source.

This is deliberately separate from ordinary train guides and dev controls.
The complete train/dev sets and fixed legacy_g05 final3000 checkpoint are
verified with the existing train exporter. Reference pixels are used ONLY to
construct explicitly labelled controls (and as a future supervision reference).
The frozen blind restorer sees those control observations, never a separate GT
tensor, task label, severity, or future frame. No H3/VAE/LoRA work is performed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

import export_rphi_train_guides as ordinary
import rphi_h3_controls as controls
from eval_rphi import TASKS, canonical_sha, causal_indices, sample_id, sha_file


SCHEMA = "rphi.train_control_rgb_export.v1"
SOURCE_ID = "2BdYaGuVAKo"
SOURCE_SHA256 = "ab1d437859fa6de7a25d2235bd4c144e319749bee0249f9556f1c708b09f9394"
SEMANTICS = {
    "raw_draft": "float32 THWC control_input_uint8/255, native HR even for nominal sr; NOT an ordinary task observation or pretrained latent upscaler",
    "rphi_raw_rgb": "float32 THWC frozen blind Rphi output before clipping",
    "rphi_rgb": "float32 THWC clip(rphi_raw_rgb,0,1), no RGB8 quantization",
    "target_usage": "reference-derived identity/near-clean control construction plus future RGB supervision/reference; no separate target tensor enters the restorer",
    "ordinary_observation": "verified for full original-record binding, never used to construct or predict these controls",
    "temporal": "all 22 original frames; causal K3 [max(0,t-2),max(0,t-1),t]; no future or chunk-boundary reset",
    "condition": "blind control observations only; nominal task is metadata/near-clean seed, never model conditioning",
    "scope": "engineering subset of TRAIN, not dev, independent test, quality approval, VAE encoding or training",
}


def _implementation():
    """Bind reused recipe, train-role validator and transitive local imports."""
    base = Path(__file__).resolve().parent
    result = ordinary._implementation()
    for name in ("export_rphi_train_controls", "rphi_h3_controls", "export_rphi_guides_stage2",
                 "export_rphi_guides", "run_fixed_h3_rphi", "openh3_assets"):
        path = base / (name + ".py")
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve() != path:
            raise ValueError("control dependency imported outside this code snapshot: " + name)
        result["code/" + path.name] = sha_file(path)
    return result


def selected_records(train_records):
    """Exactly the fixed source/window with all four original task bindings."""
    chosen = [record for record in train_records if record["source_id"] == SOURCE_ID]
    if len(chosen) != 4 or {r["task"] for r in chosen} != set(TASKS):
        raise ValueError("fixed training source must have exactly four task records")
    if any(r["split"] != "train" or r["source_sha256"] != SOURCE_SHA256 for r in chosen):
        raise ValueError("fixed source must match train role and source SHA256")
    if len({(r["target_key"], r["target_npz"], r["target_pixel_sha256"]) for r in chosen}) != 1:
        raise ValueError("fixed source controls require one shared target window")
    return chosen


def inspect_inputs(checkpoint, train_manifest, dev_manifest, *, root_override=None):
    proof, payload, train, dev = ordinary.inspect_inputs(
        checkpoint, train_manifest, dev_manifest, root_override=root_override)
    chosen = selected_records(train.records)
    implementation = _implementation()
    for key, digest in proof["implementation_sha256"].items():
        if implementation[key] != digest:
            raise ValueError("implementation changed during input inspection")
    proof = {**proof, "schema": SCHEMA, "implementation_sha256": implementation,
             "semantics": SEMANTICS, "source_ids": [SOURCE_ID], "source_sha256": SOURCE_SHA256,
             "scope": "engineering_training_control_subset", "controls": list(controls.CONTROLS),
             "selected_parent_samples": len(chosen), "planned_control_samples": 8,
             "ordinary_tasks_included": False, "reference_derived_controls": True,
             "identity_repetition": "same source/window repeated under four nominal task labels; one statistical source",
             "near_clean_repetition": "four task-keyed deterministic noise draws; still one statistical source",
             "recipe": "exact rphi_h3_controls.control_input / eval_rphi.control_seed / rphi_data.make_near_clean",
             "original_source_verification": "materialization_attestation_only; not live source video rehash"}
    return proof, payload, train, dev


def _geometry(clean):
    if (not isinstance(clean, np.ndarray) or clean.dtype != np.uint8 or clean.ndim != 4
            or clean.shape[0] != 22 or clean.shape[-1] != 3 or min(clean.shape) < 1
            or clean.shape[1] % 4 or clean.shape[2] % 4):
        raise ValueError("controls require uint8 full22 THWC RGB, spatial dimensions divisible by4")


def verify_control_cache(path, row, record, clean):
    """Verify cache against an already receipt-verified original TRAIN record."""
    if (record["source_id"] != SOURCE_ID or record["source_sha256"] != SOURCE_SHA256
            or record["split"] != "train" or record["task"] not in TASKS):
        raise ValueError("control reference must belong to fixed training source")
    _geometry(clean)
    if ordinary.array_sha(clean) != record["target_pixel_sha256"]:
        raise ValueError("control reference pixel SHA256 mismatch")
    expected_fields = {key: record[key] for key in (
        "source_id", "source_sha256", "target_key", "task", "split", "target_pixel_sha256", "observation_pixel_sha256")}
    expected_fields.update(schema=SCHEMA, complete=True, role="train", record_sha256=canonical_sha(record),
                           parent_sample_id=sample_id(record), reference_derived_control=True,
                           ordinary_observation_used=False, condition_mode="observations_only")
    if any(row.get(key) != value for key, value in expected_fields.items()):
        raise ValueError("control cache TRAIN record/role binding mismatch")
    pixels, recipe = controls.control_input(clean, record, row["control"])
    if (row.get("sample_id") != controls.control_id(record, row["control"])
            or row.get("control_recipe") != recipe
            or row.get("control_recipe_sha256") != canonical_sha(recipe)
            or row.get("control_input_pixel_sha256") != ordinary.array_sha(pixels)
            or row.get("frame_indices") != list(range(22))
            or row.get("history_indices") != causal_indices(range(22), 3, 22).tolist()):
        raise ValueError("control recipe/input/causal history mismatch")
    ordinary.verify_cache(path, row)
    with np.load(path, allow_pickle=False) as cache:
        if any(cache[key].shape != clean.shape for key in ordinary.ARRAY_KEYS):
            raise ValueError("control cache geometry differs from reference")
        normalized = np.ascontiguousarray(pixels, dtype=np.float32) / np.float32(255)
        if not np.array_equal(cache["raw_draft"], normalized):
            raise ValueError("raw draft differs from deterministic control input")
    return True


def export_train_controls(checkpoint, train_manifest, dev_manifest, out, *, device="cpu",
                          frame_batch=1, root_override=None):
    out = Path(out).expanduser().absolute()
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if type(frame_batch) is not int or frame_batch < 1:
        raise ValueError("frame_batch must be a positive integer")
    if device != "cpu" and not (isinstance(device, str) and device.startswith("npu:") and device[4:].isdigit()):
        raise ValueError("device must be cpu or npu:<logical-index>")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    try:
        proof, payload, train, dev = inspect_inputs(
            checkpoint, train_manifest, dev_manifest, root_override=root_override)
        ordinary._write_json(out / "inspection.json", proof)
        import torch
        import rphi_model
        if Path(rphi_model.__file__).resolve().parent != Path(__file__).resolve().parent or rphi_model.MODEL_SCHEMA != "rphi.canonical_rgb.v1":
            raise ValueError("unsupported or externally imported RphiModel")
        if device.startswith("npu:"):
            import torch_npu  # noqa: F401 -- only explicit actual export registers NPU
            torch.npu.set_device(device)
            torch.npu.set_compile_mode(jit_compile=False)
            torch.npu.matmul.allow_hf32 = False
            torch.npu.conv.allow_hf32 = False
        model = rphi_model.RphiModel(**payload["model_config"])
        model.load_state_dict(payload["state_dict"], strict=True)
        model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError("Rphi reference must be fully frozen")
        state_before = ordinary._state_sha(model)
        (out / "samples").mkdir()
        for index, record in enumerate(train.records):
            if record["source_id"] != SOURCE_ID:
                continue
            observation_path, target_path = (train.resolve_path(record[key]) for key in ("observation_npz", "target_npz"))
            observation_sha, target_sha = sha_file(observation_path), sha_file(target_path)
            clean = train.load(index)["target"]
            _geometry(clean)
            for control in controls.CONTROLS:
                pixels, recipe = controls.control_input(clean, record, control)
                # Sole inference boundary: explicit control observations and geometry.
                # No ordinary observation, GT tensor, recipe, task or control label.
                arrays = ordinary.predict_arrays(model, pixels, clean.shape, device, frame_batch)
                identity = controls.control_id(record, control)
                relative = Path("samples") / (identity + ".npz")
                with (out / relative).open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                raw = arrays["rphi_raw_rgb"]
                row = {key: record[key] for key in ("source_id", "source_sha256", "target_key", "task", "split",
                                                  "target_pixel_sha256", "observation_pixel_sha256")}
                row.update(schema=SCHEMA, complete=True, role="train", scope=proof["scope"],
                    sample_id=identity, parent_sample_id=sample_id(record), control=control,
                    record_sha256=canonical_sha(record), reference_derived_control=True,
                    ordinary_observation_used=False, control_recipe=recipe,
                    control_recipe_sha256=canonical_sha(recipe), control_input_pixel_sha256=ordinary.array_sha(pixels),
                    train_manifest_sha256=train.manifest_sha256, dev_manifest_sha256=dev.manifest_sha256,
                    checkpoint=proof["checkpoint"], checkpoint_sha256=proof["checkpoint_sha256"], checkpoint_step=3000,
                    model_config=payload["model_config"], condition_mode="observations_only",
                    original_observation_npz=str(observation_path), original_observation_file_sha256=observation_sha,
                    target_npz=str(target_path), target_file_sha256=target_sha,
                    target_role="explicit_control_construction_and_supervision_reference_no_separate_target_model_input",
                    frame_indices=list(range(22)), history_indices=causal_indices(range(22), 3, 22).tolist(),
                    guide_npz=str(relative), guide_file_sha256=sha_file(out / relative),
                    guide_file_bytes=(out / relative).stat().st_size,
                    arrays={key: {"shape": list(value.shape), "dtype": "float32", "array_sha256": ordinary.array_sha(value)}
                            for key, value in arrays.items()},
                    clipping={"component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                              "raw_min": float(raw.min()), "raw_max": float(raw.max())},
                    semantics=SEMANTICS, implementation_sha256=proof["implementation_sha256"],
                    vae_encoded=False, latent_path=None)
                verify_control_cache(out / relative, row, record, clean)
                sidecar = relative.with_suffix(".json")
                ordinary._write_json(out / sidecar, row)
                rows.append({**row, "provenance_json": str(sidecar), "provenance_file_sha256": sha_file(out / sidecar)})
                ordinary._write_json(out / "partial_manifest.json", {
                    "schema": SCHEMA, "complete": False, "role": "train", "completed_samples": len(rows), "per_sample": rows}, replace=True)
            if sha_file(observation_path) != observation_sha or sha_file(target_path) != target_sha:
                raise ValueError("reference/original observation changed during control export")
        expected = {controls.control_id(r, c) for r in selected_records(train.records) for c in controls.CONTROLS}
        if len(rows) != 8 or {r["sample_id"] for r in rows} != expected:
            raise ValueError("incomplete fixed-source control matrix")
        state_after = ordinary._state_sha(model)
        if state_after != state_before or any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError("frozen Rphi state changed during control export")
        if _implementation() != proof["implementation_sha256"]:
            raise ValueError("implementation changed during control export")
        for path, expected in ((checkpoint, proof["checkpoint_sha256"]), (train.manifest_path, train.manifest_sha256),
                               (dev.manifest_path, dev.manifest_sha256)):
            if sha_file(path) != expected:
                raise ValueError("checkpoint or complete train/dev manifest changed during export")
        for dataset in (train, dev):
            for path, expected in dataset.receipt_hashes.items():
                if sha_file(path) != expected:
                    raise ValueError("materialization receipt changed during export")
        manifest = {**proof, "complete": True, "rgb_export_complete": True,
            "status": "train_control_rgb_complete_not_vae_encoded", "model_constructed": True,
            "model_forward_calls": len(rows) * ((22 + frame_batch - 1) // frame_batch),
            "device": device, "frame_batch": frame_batch, "samples": len(rows), "per_sample": rows,
            "frozen_state_sha256_before": state_before, "frozen_state_sha256_after": state_after,
            "all_parameters_frozen": True, "separate_supervision_array_cached": False,
            "vae_encoded": False, "latent_path": None, "guide_training_steps": 0, "h3_sampling_calls": 0}
        ordinary._write_json(out / "manifest.json", manifest)
        return manifest
    except Exception as error:
        ordinary._write_json(out / "failure.json", {"schema": SCHEMA, "complete": False, "role": "train",
            "completed_samples": len(rows), "error": type(error).__name__ + ": " + str(error)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--out", type=Path, help="new directory, actual RGB export only")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-batch", default=1, type=int)
    parser.add_argument("--root-override", type=Path)
    args = parser.parse_args(argv)
    if args.inspect_only:
        if args.out is not None or args.device != "cpu":
            parser.error("--inspect-only requires CPU and no output directory")
        proof, _, _, _ = inspect_inputs(args.checkpoint, args.train_manifest, args.dev_manifest, root_override=args.root_override)
        print(json.dumps(proof, indent=2, allow_nan=False))
    else:
        if args.out is None:
            parser.error("--out is required for actual export")
        kwargs = vars(args).copy()
        kwargs.pop("inspect_only")
        result = export_train_controls(**kwargs)
        print(json.dumps({"manifest": str(args.out / "manifest.json"), "complete": result["complete"],
                          "samples": result["samples"], "role": "train", "vae_encoded": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
