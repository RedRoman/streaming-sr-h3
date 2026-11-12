"""Separate identity/near-clean R_phi exports and fixed-H3 evaluation.

These explicitly reference-derived controls are never ordinary task observations.
Near-clean uses eval_rphi.control_seed and rphi_data.make_near_clean unchanged.
The original recording remains the RGB reference. All 22 frames are exported;
each R_phi prediction has consecutive causal K3 context. H3 remains a full-block
generator, with one fixed encoded control-input draft across its two guide arms.
"""
import argparse
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from eval_rphi import (TASKS, canonical_sha, causal_indices, control_seed,
                       error_attribution, flow_support, infer_selected,
                       load_local_lpips, rgb_metrics, sample_id, sha_file,
                       source_summary, temporal_metrics, validate_contract)
from export_rphi_guides import array_sha
from export_rphi_guides_stage2 import verify_checkpoint_implementation
from openh3_assets import NATIVE_SIGMAS, read_manifest
from rphi_data import RphiDataset, make_near_clean
from run_fixed_h3_rphi import contained_file, generate_pair, validate_h3_geometry


SCHEMA = "rphi.h3_control_export.v1"
EVAL_SCHEMA = "openh3.rphi_controls_eval.v1"
CONTROLS = ("identity", "near_clean")
PAIRINGS = (("rphi_only", "input"), ("h3_raw", "input"),
            ("h3_rphi", "h3_raw"), ("h3_raw_anchor", "h3_raw"),
            ("h3_rphi_anchor", "h3_rphi"),
            ("h3_rphi_anchor", "h3_raw_anchor"),
            ("h3_rphi", "rphi_only"), ("h3_rphi_anchor", "rphi_only"))


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def write_partial_atomic(path, value):
    """Publish a complete snapshot, retaining the prior one if encoding fails."""
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                prefix="." + path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def control_id(record, control):
    if control not in CONTROLS:
        raise ValueError("only explicit identity/near_clean controls are accepted")
    return sample_id(record) + "-" + control


def control_input(clean, record, control):
    """Deterministic uint8 input and recipe; no ordinary degraded pixels read."""
    if clean.dtype != np.uint8 or clean.ndim != 4 or clean.shape[-1] != 3:
        raise ValueError("clean control reference must be uint8 THWC RGB")
    control_id(record, control)
    if control == "identity":
        pixels, seed = clean.copy(), None
    else:
        seed = control_seed(record)
        pixels = make_near_clean(clean, seed=seed, sigma_8bit=1.0)
    recipe = {"control": control, "reference": "clean_original_recording",
              "reference_derived_control": True, "ordinary_observation_used": False,
              "seed": seed, "sigma_8bit": 0.0 if control == "identity" else 1.0,
              "implementation": "identity uint8 copy" if control == "identity" else
                  "rphi_data.make_near_clean: PCG64 float32 Gaussian, rint ties-even, clip, uint8",
              "geometry": "native HR control input for every task label; no SR downsample",
              "task_label_usage": "near-clean seed and task-specific anchor policy only; blind model receives no task label"}
    return pixels, recipe


def _normalized(pixels):
    return np.ascontiguousarray(pixels, dtype=np.float32) / np.float32(255)


def _sources(dataset, max_sources):
    if max_sources is not None and (type(max_sources) is not int or max_sources < 1):
        raise ValueError("max_sources must be a positive integer")
    sources = sorted({r["source_id"] for r in dataset.records})
    chosen = sources if max_sources is None else sources[:max_sources]
    for source in chosen:
        if {r["task"] for r in dataset.records if r["source_id"] == source} != set(TASKS):
            raise ValueError("each source must have all four original task records")
    return sources, chosen


def _model_proof(payload, code_dir):
    implementation = verify_checkpoint_implementation(payload, code_dir)
    return {"inference_implementation_verified": True,
            "implementation_sha256": implementation,
            "fidelity_state_sha256": payload.get("fidelity_state_sha256"),
            "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
            "calibration_contract": payload.get("calibration_contract"),
            "calibration_step": payload.get("calibration_step"),
            "fidelity_invariance": payload.get("fidelity_invariance")}


def export_controls(checkpoint, dev_manifest, out, device="cpu", frame_batch=1,
                    max_sources=None):
    out, checkpoint, dev_manifest = map(lambda x: Path(x).expanduser().absolute(),
                                        (out, checkpoint, dev_manifest))
    if out.exists() or out.is_symlink():
        raise FileExistsError(out)
    if type(frame_batch) is not int or frame_batch < 1:
        raise ValueError("frame_batch must be a positive integer")
    if device != "cpu" and not (device.startswith("npu:") and device[4:].isdigit()):
        raise ValueError("device must be cpu or npu:<logical-index>")
    import torch
    import rphi_model
    checkpoint_sha = sha_file(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    proof = _model_proof(payload, Path(rphi_model.__file__).resolve().parent)
    dataset = RphiDataset(dev_manifest, verify_hashes=True)
    contract = validate_contract(payload, dataset.manifest_sha256, dataset.records)
    sources, chosen = _sources(dataset, max_sources)
    if sha_file(checkpoint) != checkpoint_sha:
        raise ValueError("checkpoint changed while loading")
    model = rphi_model.RphiModel(**payload["model_config"])
    model.load_state_dict(payload["state_dict"], strict=True)
    if device.startswith("npu:"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    model.to(device).eval()
    implementation_files = ["rphi_h3_controls.py", "eval_rphi.py", "rphi_data.py",
                            "rphi_model.py", "export_rphi_guides_stage2.py",
                            "export_rphi_guides.py", "run_fixed_h3_rphi.py", "openh3_assets.py"]
    if payload["model_config"].get("observation_encoder") == "native":
        implementation_files.append("rphi_condition_v2.py")
    code_dir = Path(__file__).resolve().parent
    implementation = {"code/" + n: sha_file(code_dir / n) for n in implementation_files}
    manifest = {"schema": SCHEMA, "complete": False, "controls": list(CONTROLS),
        "condition_mode": "observations_only", "split": "dev", "ordinary_tasks_included": False,
        "scope": "control_only_full_development_set" if chosen == sources else "control_engineering_subset",
        "checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": payload["step"], "model_config": dict(payload["model_config"]),
        "model_schema": rphi_model.MODEL_SCHEMA, "training_contract": contract,
        "checkpoint_implementation_sha256": payload["implementation_sha256"],
        "stage2_export_proof": {**proof, "checkpoint_sha256": checkpoint_sha},
        "implementation_sha256": implementation, "dev_manifest": str(dev_manifest),
        "dev_manifest_sha256": dataset.manifest_sha256, "source_ids": chosen,
        "available_sources": len(sources), "frame_batch": frame_batch, "device": device,
        "identity_repetition": "same source/target under four task labels; never count these as independent sources",
        "near_clean_repetition": "four deterministic task-keyed noise draws per source/target; source remains statistical unit",
        "rgb_semantics": "float32 THWC rgb=clip(raw_rgb,0,1); input_rgb=uint8 control input/255; no output quantization",
        "log_variance_semantics": "optional isotropic per-channel log variance for raw_rgb; calibration only",
        "dataset_provenance": dataset.provenance, "per_sample": []}
    out.mkdir(parents=True, exist_ok=False)
    try:
        for index, record in enumerate(dataset.records):
            if record["source_id"] not in chosen:
                continue
            loaded = dataset.load(index)
            clean = loaded["target"]
            validate_h3_geometry(clean.shape)
            target_path = dataset.resolve_path(record["target_npz"])
            observation_path = dataset.resolve_path(record["observation_npz"])
            target_sha, observation_sha = sha_file(target_path), sha_file(observation_path)
            for control in CONTROLS:
                pixels, recipe = control_input(clean, record, control)
                input_rgb = _normalized(pixels)
                guide = torch.from_numpy(input_rgb.transpose(0, 3, 1, 2).copy())
                frames = list(range(len(pixels)))
                prediction = infer_selected(model, guide, None, frames, device, frame_batch)
                raw = np.ascontiguousarray(prediction["rgb"], dtype=np.float32)
                if raw.shape != input_rgb.shape or not np.isfinite(raw).all():
                    raise ValueError("invalid control prediction")
                arrays = {"input_rgb": input_rgb, "rgb": np.clip(raw, 0, 1), "raw_rgb": raw}
                if prediction["log_variance"] is not None:
                    arrays["log_variance"] = np.ascontiguousarray(prediction["log_variance"], dtype=np.float32)
                sid = control_id(record, control)
                file = out / (sid + ".npz")
                with file.open("xb") as stream:
                    np.savez_compressed(stream, **arrays)
                row = {k: record[k] for k in ("source_id", "source_sha256", "target_key", "task", "split",
                       "target_pixel_sha256", "observation_pixel_sha256")}
                row.update(sample_id=sid, parent_sample_id=sample_id(record), control=control,
                    record_sha256=canonical_sha(record), control_recipe=recipe,
                    control_input_pixel_sha256=array_sha(pixels), target_file_sha256=target_sha,
                    original_observation_file_sha256=observation_sha,
                    checkpoint_sha256=checkpoint_sha, checkpoint_step=payload["step"],
                    model_config=dict(payload["model_config"]), condition_mode="observations_only",
                    dev_manifest_sha256=dataset.manifest_sha256, frame_indices=frames,
                    history_indices=causal_indices(frames, 3, len(pixels)).tolist(),
                    guide_npz=file.name, guide_file_sha256=sha_file(file),
                    arrays={k: {"shape": list(v.shape), "dtype": "float32", "array_sha256": array_sha(v)}
                            for k, v in arrays.items()},
                    clipping={"component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                              "raw_min": float(raw.min()), "raw_max": float(raw.max())})
                verify_control_file(file, row, pixels)
                sidecar = out / (sid + ".json")
                write_new(sidecar, row)
                manifest["per_sample"].append({**row, "provenance_json": sidecar.name,
                    "provenance_file_sha256": sha_file(sidecar)})
            if sha_file(target_path) != target_sha or sha_file(observation_path) != observation_sha:
                raise ValueError("source assets changed during export")
        if sha_file(checkpoint) != checkpoint_sha or sha_file(dev_manifest) != dataset.manifest_sha256:
            raise ValueError("checkpoint/manifest changed during export")
        for name, digest in implementation.items():
            if sha_file(code_dir / Path(name).name) != digest:
                raise ValueError("inference implementation changed during export")
        manifest.update(complete=True, samples=len(manifest["per_sample"]),
                        materialization_receipt_sha256=dict(dataset.receipt_hashes))
        write_new(out / "manifest.json", manifest)
        return manifest
    except Exception as error:
        write_new(out / "failure.json", {"schema": SCHEMA, "complete": False,
                  "completed_samples": len(manifest["per_sample"]), "error": type(error).__name__ + ": " + str(error)})
        raise


def verify_control_file(path, row, expected_input=None):
    if sha_file(path) != row["guide_file_sha256"]:
        raise ValueError("control NPZ file SHA mismatch")
    expected = row["arrays"]
    if set(expected) not in ({"input_rgb", "rgb", "raw_rgb"},
                             {"input_rgb", "rgb", "raw_rgb", "log_variance"}):
        raise ValueError("invalid control arrays")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(expected):
            raise ValueError("control NPZ array fields mismatch")
        for name, spec in expected.items():
            value = archive[name]
            if (value.dtype != np.float32 or spec["dtype"] != "float32" or
                    list(value.shape) != spec["shape"] or not np.isfinite(value).all() or
                    array_sha(value) != spec["array_sha256"]):
                raise ValueError("control array hash/dtype/geometry mismatch: " + name)
        shape = archive["rgb"].shape
        validate_h3_geometry(shape)
        if archive["input_rgb"].shape != shape or archive["raw_rgb"].shape != shape:
            raise ValueError("control RGB shapes differ")
        if "log_variance" in archive and archive["log_variance"].shape != shape[:-1] + (1,):
            raise ValueError("control variance shape mismatch")
        if not np.array_equal(archive["rgb"], np.clip(archive["raw_rgb"], 0, 1)):
            raise ValueError("control clipping mismatch")
        if expected_input is not None and not np.array_equal(archive["input_rgb"], _normalized(expected_input)):
            raise ValueError("control input is not the declared deterministic reference-derived control")
    return True


def inspect_controls(guide_manifest, dev_manifest, max_sources=None):
    guide_manifest, dev_manifest = Path(guide_manifest), Path(dev_manifest)
    initial_sha = sha_file(guide_manifest)
    manifest = json.loads(guide_manifest.read_text())
    if (manifest.get("schema") != SCHEMA or manifest.get("complete") is not True or
            manifest.get("ordinary_tasks_included") is not False or manifest.get("controls") != list(CONTROLS)):
        raise ValueError("expected complete separate identity/near-clean control export")
    config = manifest.get("model_config", {})
    if (manifest.get("model_schema") != "rphi.canonical_rgb.v1" or
            config.get("variant") not in ("blind", "blind_confidence") or config.get("k") != 3):
        raise ValueError("controls require blind K3 model")
    if manifest.get("condition_mode") != "observations_only":
        raise ValueError("control conditions must be observations_only")
    if manifest.get("checkpoint_step") != manifest["training_contract"].get("steps") or not manifest.get("checkpoint_step"):
        raise ValueError("control checkpoint must be the final contracted step")
    proof = manifest.get("stage2_export_proof", {})
    if proof.get("checkpoint_sha256") != manifest.get("checkpoint_sha256") or proof.get("inference_implementation_verified") is not True:
        raise ValueError("control checkpoint implementation proof missing")
    names = ["rphi_model.py", "rphi_data.py"] + (["rphi_condition_v2.py"] if config.get("observation_encoder") == "native" else [])
    for name in names:
        key = "code/" + name
        value = proof.get("implementation_sha256", {}).get(key)
        if not value or value != manifest["checkpoint_implementation_sha256"].get(key) or value != manifest["implementation_sha256"].get(key):
            raise ValueError("control inference implementation is not checkpoint-bound: " + key)
    dataset = RphiDataset(dev_manifest, verify_hashes=True)
    validate_contract({"schema": "rphi.checkpoint.v1", "training_contract": manifest["training_contract"]},
                      dataset.manifest_sha256, dataset.records)
    if dataset.manifest_sha256 != manifest["dev_manifest_sha256"]:
        raise ValueError("control development manifest mismatch")
    sources, _ = _sources(dataset, max_sources)
    selected = manifest["source_ids"]
    if sorted(set(selected)) != selected or not selected or not set(selected) <= set(sources):
        raise ValueError("invalid control source coverage")
    expected_scope = "control_only_full_development_set" if selected == sources else "control_engineering_subset"
    if (manifest.get("scope") != expected_scope or manifest.get("available_sources") != len(sources)
            or manifest.get("split") != "dev"):
        raise ValueError("control scope/source count mismatch")
    chosen = selected if max_sources is None else selected[:max_sources]
    indexed = {sample_id(r): (i, r) for i, r in enumerate(dataset.records)}
    rows, seen = [], set()
    for row in manifest["per_sample"]:
        index, record = indexed[row["parent_sample_id"]]
        sid = control_id(record, row["control"])
        if row["sample_id"] != sid or sid in seen or record["source_id"] not in selected:
            raise ValueError("duplicate or inconsistent control identity")
        seen.add(sid)
        if row["record_sha256"] != canonical_sha(record):
            raise ValueError("control record binding mismatch")
        for key in ("source_id", "source_sha256", "target_key", "task", "split", "target_pixel_sha256", "observation_pixel_sha256"):
            if row.get(key) != record[key]:
                raise ValueError("control source binding mismatch: " + key)
        for key in ("checkpoint_sha256", "checkpoint_step", "model_config", "condition_mode", "dev_manifest_sha256"):
            if row.get(key) != manifest.get(key):
                raise ValueError("control row/manifest mismatch: " + key)
        sidecar = contained_file(guide_manifest.parent, row["provenance_json"])
        if sha_file(sidecar) != row["provenance_file_sha256"] or json.loads(sidecar.read_text()) != {
                k: v for k, v in row.items() if k not in ("provenance_json", "provenance_file_sha256")}:
            raise ValueError("control sidecar mismatch")
        loaded = dataset.load(index)
        pixels, recipe = control_input(loaded["target"], record, row["control"])
        frames = list(range(len(pixels)))
        if (row["control_recipe"] != recipe or row["control_input_pixel_sha256"] != array_sha(pixels) or
                row["frame_indices"] != frames or row["history_indices"] != causal_indices(frames, 3, len(pixels)).tolist()):
            raise ValueError("control recipe/input/history mismatch")
        for field, key in (("target_file_sha256", "target_npz"), ("original_observation_file_sha256", "observation_npz")):
            if row[field] != sha_file(dataset.resolve_path(record[key])):
                raise ValueError("control source file mismatch")
        path = contained_file(guide_manifest.parent, row["guide_npz"])
        verify_control_file(path, row, pixels)
        if record["source_id"] in chosen:
            rows.append({"index": index, "record": record, "export": row, "guide_file": str(path)})
    required = {control_id(r, c) for r in dataset.records if r["source_id"] in selected for c in CONTROLS}
    if seen != required or manifest.get("samples") != len(seen):
        raise ValueError("incomplete control coverage")
    if sha_file(guide_manifest) != initial_sha or sha_file(dev_manifest) != dataset.manifest_sha256:
        raise ValueError("control manifest changed during inspection")
    return dataset, manifest, rows


def score_control_pair(h3, input_rgb, rphi_rgb, target, task, alpha=.25, seed=101,
                       frame_stride=1, lpips_metric=None, require_flow=True):
    """Target enters metrics only; fake-H3 CPU tests exercise this same path."""
    import torch
    from rphi_soft_anchor import apply_soft_anchor
    if not np.isfinite(alpha) or not 0 < alpha < 1 or type(frame_stride) is not int or frame_stride < 1:
        raise ValueError("invalid anchor alpha/frame stride")
    if target.dtype != np.uint8 or target.shape != input_rgb.shape:
        raise ValueError("control reference must be same-geometry uint8 RGB")
    raw, restored = torch.from_numpy(input_rgb.copy()), torch.from_numpy(rphi_rgb.copy())
    decoded, sampling = generate_pair(h3, raw, restored, seed)
    arrays, audits = {"input": input_rgb, "rphi_only": rphi_rgb}, {}
    for label, pixels in decoded.items():
        arrays["h3_" + label] = pixels.clamp(0, 1).numpy()
        corrected, audits[label] = apply_soft_anchor(pixels.permute(0, 3, 1, 2).contiguous(),
            restored.permute(0, 3, 1, 2).contiguous(), task, alpha)
        arrays["h3_" + label + "_anchor"] = corrected.permute(0, 2, 3, 1).numpy()
    frames = list(range(0, len(input_rgb), frame_stride))
    gt = _normalized(target)[frames]
    support, note = flow_support(gt, enabled=require_flow)
    if require_flow and support is None:
        raise RuntimeError("required flow support unavailable: " + note)
    scores = {key: {"quality": rgb_metrics(value[frames], gt, lpips_metric),
                   "temporal": temporal_metrics(value[frames], gt, support)} for key, value in arrays.items()}
    pairs = {arm + "_vs_" + base: {"model": scores[arm]["quality"], "baseline": scores[base]["quality"],
              "baseline_arm": base, "temporal_model": scores[arm]["temporal"],
              "temporal_baseline": scores[base]["temporal"],
              "attribution": error_attribution(arrays[arm][frames], arrays[base][frames], gt)} for arm, base in PAIRINGS}
    return {"arms": scores, "paired": pairs, "soft_anchor": audits, "sampling": sampling,
            "frame_indices": frames, "flow": note}, decoded


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    for name in ("checkpoint", "dev-manifest", "out"):
        export.add_argument("--" + name, required=True, type=Path)
    export.add_argument("--device", default="cpu")
    export.add_argument("--frame-batch", type=int, default=1)
    export.add_argument("--max-sources", type=int)
    for command in ("inspect", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--guide-manifest", required=True, type=Path)
        sub.add_argument("--dev-manifest", required=True, type=Path)
        sub.add_argument("--max-sources", type=int)
        if command == "evaluate":
            sub.add_argument("--pretrained-manifest", required=True, type=Path)
            sub.add_argument("--out", required=True, type=Path)
            sub.add_argument("--seed", type=int, default=101)
            sub.add_argument("--alpha", type=float, default=.25)
            sub.add_argument("--frame-stride", type=int, default=1)
            sub.add_argument("--lpips-weights", type=Path)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_controls(**{k: v for k, v in vars(args).items() if k != "command"})
        print(json.dumps({"complete": True, "samples": result["samples"], "scope": result["scope"]}))
        return 0
    if args.command == "evaluate" and (args.out.exists() or args.out.is_symlink()):
        raise FileExistsError(args.out)
    manifest_sha = sha_file(args.guide_manifest)
    dataset, manifest, samples = inspect_controls(args.guide_manifest, args.dev_manifest, args.max_sources)
    if args.command == "inspect":
        print(json.dumps({"verified": True, "schema": SCHEMA, "controls": list(CONTROLS),
            "samples": len(samples), "unique_sources": len({x["record"]["source_id"] for x in samples}),
            "h3_sample_calls": 2 * len(samples), "denoiser_evaluations": 6 * len(samples),
            "ordinary_tasks_included": False, "H3_VAE_executed": False}))
        return 0
    if not np.isfinite(args.alpha) or not 0 < args.alpha < 1 or args.frame_stride < 1:
        raise ValueError("invalid alpha or frame stride")
    pretrained = read_manifest(args.pretrained_manifest, verify=False)
    import torch
    from h3_comfy import H3
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    metric, lpips_info = load_local_lpips(args.lpips_weights)
    if metric is None:
        raise RuntimeError("required local LPIPS unavailable")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": EVAL_SCHEMA, "complete": False, "ordinary_tasks_included": False,
        "scope": manifest["scope"] if args.max_sources is None else "control_engineering_subset",
        "contract": {"guide_manifest_sha256": manifest_sha, "dev_manifest_sha256": dataset.manifest_sha256,
            "checkpoint_sha256": manifest["checkpoint_sha256"], "pretrained_fingerprint": pretrained["_fingerprint"],
            "sigma_schedule": list(NATIVE_SIGMAS), "mode": "base", "adaptation": None,
            "seed": args.seed, "alpha": args.alpha, "frame_stride": args.frame_stride,
            "controls": list(CONTROLS), "audio_init": "zeros", "condition_mode": "observations_only",
            "warm_start": "fixed encoded control input in both H3 calls",
            "anchor_policy": "original task label: canonical Q4 for sr, full RGB otherwise; control input remains native HR",
            "metric": "float32 clipped RGB, no output quantization; same eval_rphi metrics and flow",
            "future_frames": "full22 H3 block, no end-to-end causal claim",
            "control_source": "reference-derived diagnostic only; never ordinary task evidence",
            "source_weighting": "per task/control source averages; identity labels are duplicates, near-clean labels have task-keyed noise"},
        "lpips": lpips_info, "implementation_sha256": {name: sha_file(Path(__file__).parent / name) for name in (
            "rphi_h3_controls.py", "run_fixed_h3_rphi.py", "h3_comfy.py", "openh3_assets.py", "openh3_runtime.py",
            "rphi_soft_anchor.py", "pixel_consistency.py", "rphi_model.py", "rphi_data.py", "eval_rphi.py")}, "per_sample": []}
    try:
        h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS)
        for sample in samples:
            record, row = sample["record"], sample["export"]
            loaded = dataset.load(sample["index"])
            pixels, _ = control_input(loaded["target"], record, row["control"])
            verify_control_file(sample["guide_file"], row, pixels)
            with np.load(sample["guide_file"], allow_pickle=False) as archive:
                scored, decoded = score_control_pair(h3, archive["input_rgb"], archive["rgb"],
                    loaded["target"], record["task"], args.alpha, args.seed, args.frame_stride, metric)
            file = args.out / (row["sample_id"] + ".npz")
            with file.open("xb") as stream:
                np.savez_compressed(stream, **{k: v.numpy() for k, v in decoded.items()})
            result["per_sample"].append({**{k: row[k] for k in ("sample_id", "parent_sample_id", "source_id",
                "source_sha256", "target_key", "task", "split", "control", "record_sha256", "target_pixel_sha256",
                "control_input_pixel_sha256", "control_recipe", "guide_file_sha256")}, **scored,
                "decoded_file": {"path": file.name, "sha256": sha_file(file),
                    "arrays": {k: array_sha(v.numpy()) for k, v in decoded.items()}}})
            write_partial_atomic(args.out / "partial_metrics.json", result)
            print(json.dumps({"done": len(result["per_sample"]), "total": len(samples), "control": row["control"]}), flush=True)
        if sha_file(args.guide_manifest) != manifest_sha or sha_file(args.dev_manifest) != dataset.manifest_sha256:
            raise ValueError("control manifests changed during evaluation")
        result["comparisons"] = {control: {task: {arm + "_vs_" + base: source_summary([
            {"source_id": row["source_id"], **row["paired"][arm + "_vs_" + base]} for row in result["per_sample"]
            if row["control"] == control and row["task"] == task]) for arm, base in PAIRINGS}
            for task in TASKS} for control in CONTROLS}
        result.update(complete=True, unique_sources=len({r["source_id"] for r in result["per_sample"]}),
                      model_record=h3.record, freeze_audit=h3.freeze_audit(),
                      identity_metric_note="Identity input PSNR is capped at 120 dB; interpret MAE/MSE, not the huge negative PSNR delta.")
        write_new(args.out / "metrics.json", result)
    except Exception as error:
        write_new(args.out / "failure.json", {"complete": False, "completed_samples": len(result["per_sample"]),
                  "error": type(error).__name__ + ": " + str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
