"""One-source, paired VAE-only diagnostic for completed ordinary/control H3 runs.

Loads the exact existing H3 VAE without DiT, sampler or adaptation. Each encoded
latent must match the previous H3 guide latent SHA. An instance-local observer
captures RGB before MiniMax's internal clamp while preserving the original
decode return. Targets enter scoring only. Existing outputs are never replaced.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import time

import numpy as np

from eval_rphi import flow_support, load_local_lpips, rgb_metrics, sha_file, temporal_metrics
from export_rphi_guides import array_sha
from openh3_assets import NATIVE_SIGMAS, read_manifest
from rphi_data import prepare_observed
from rphi_h3_controls import inspect_controls, write_new, write_partial_atomic
from run_fixed_h3_rphi import inspect_inputs, tensor_sha, validate_h3_geometry

SCHEMA = "openh3.vae_roundtrip_diagnostic.v1"


def validate_reference(metrics, manifest, manifest_sha, scope):
    """Check prior sampling evidence without reading targets or generating pixels."""
    if not metrics.get("complete") or not manifest.get("complete"):
        raise ValueError("completed reference metrics and guide export required")
    c = metrics["contract"]
    checkpoint_key = "rphi_checkpoint_sha256" if scope == "ordinary" else "checkpoint_sha256"
    if (c["guide_manifest_sha256"] != manifest_sha or
            c[checkpoint_key] != manifest["checkpoint_sha256"] or
            c["dev_manifest_sha256"] != manifest["dev_manifest_sha256"] or
            manifest["checkpoint_step"] != manifest["training_contract"]["steps"]):
        raise ValueError("reference checkpoint/manifest binding mismatch")
    if c["frame_stride"] != 1 or c["mode"] != "base" or c["adaptation"] is not None:
        raise ValueError("reference must use full-frame fixed base H3")
    if not metrics.get("lpips", {}).get("available"):
        raise ValueError("reference local LPIPS must be available")
    expected = {r["sample_id"]: r for r in manifest["per_sample"]}
    rows = {r["sample_id"]: r for r in metrics["per_sample"]}
    if len(rows) != len(metrics["per_sample"]) or set(rows) != set(expected):
        raise ValueError("reference sample coverage mismatch")
    for sid, row in rows.items():
        exported = expected[sid]
        for key in ("source_id", "source_sha256", "task", "record_sha256", "target_pixel_sha256"):
            if row[key] != exported[key]:
                raise ValueError("reference source binding mismatch: " + key)
        extra = ("control", "control_recipe", "control_input_pixel_sha256") if scope == "controls" else ("observation_pixel_sha256",)
        if any(row[key] != exported[key] for key in extra):
            raise ValueError("reference observation/control binding mismatch")
        if row["frame_indices"] != list(range(22)):
            raise ValueError("reference requires all 22 consecutive frames")
        if any(a["quality"]["frames"] != 22 or a["temporal"]["pairs"] != 21 or
               a["temporal"]["valid_pairs"] != 21 for a in row["arms"].values()):
            raise ValueError("reference requires 22 quality frames and 21 flow pairs")
        s = row["sampling"]
        if s["guide_pixel_sha256"]["rphi"] != exported["arrays"]["rgb"]["array_sha256"]:
            raise ValueError("reference Rphi pixel SHA mismatch")
        if scope == "controls" and s["guide_pixel_sha256"]["raw"] != exported["arrays"]["input_rgb"]["array_sha256"]:
            raise ValueError("reference control pixel SHA mismatch")
        calls = s["calls"]
        if not (calls["raw"]["warm_start_latent_sha256_float32"] ==
                calls["rphi"]["warm_start_latent_sha256_float32"] ==
                calls["raw"]["guide_latent_sha256_float32"]):
            raise ValueError("reference warm starts differ")
        for call in calls.values():
            r = call["sampling_record"]
            if (r["steps"] != 3 or r["seed"] != c["seed"] or r["mode"] != "base" or
                    r["audio_init"] != "zeros" or r["correction_events"] or
                    r["pretrained"]["fingerprint"] != c["pretrained_fingerprint"] or
                    len(r["sigmas"]) != 4 or any(abs(a-b) > 1e-7 for a, b in zip(r["sigmas"], NATIVE_SIGMAS))):
                raise ValueError("reference fixed-H3 sampling contract mismatch")
    return rows


def inspect_bundle(args):
    sources, items, manifests, metrics, hashes = [], [], {}, {}, {}
    for scope, inspect_fn in (("ordinary", inspect_inputs), ("controls", inspect_controls)):
        guide_path = getattr(args, scope + "_guides")
        metric_path = getattr(args, scope + "_metrics")
        hashes[scope + "_guides"] = sha_file(guide_path)
        hashes[scope + "_metrics"] = sha_file(metric_path)
        dataset, manifest, selected = inspect_fn(guide_path, args.dev_manifest)
        prior = json.loads(metric_path.read_text())
        rows = validate_reference(prior, manifest, hashes[scope + "_guides"], scope)
        if len(manifest["source_ids"]) != 1:
            raise ValueError("this diagnostic is bounded to exactly one source")
        sources.append(manifest["source_ids"])
        for item in selected:
            validate_h3_geometry(item["export"]["arrays"]["rgb"]["shape"])
            items.append({**item, "dataset": dataset, "scope": scope,
                          "reference": rows[item["export"]["sample_id"]]})
        manifests[scope], metrics[scope] = manifest, prior
    if sources[0] != sources[1] or len(items) != 12:
        raise ValueError("requires the same one-source ordinary4 + controls8")
    for key in ("checkpoint_sha256", "dev_manifest_sha256", "model_config"):
        if manifests["ordinary"][key] != manifests["controls"][key]:
            raise ValueError("ordinary/control model or data differ: " + key)
    for key in ("pretrained_fingerprint", "sigma_schedule", "seed", "frame_stride"):
        if metrics["ordinary"]["contract"][key] != metrics["controls"]["contract"][key]:
            raise ValueError("ordinary/control generator contracts differ: " + key)
    if metrics["ordinary"]["lpips"] != metrics["controls"]["lpips"]:
        raise ValueError("ordinary/control LPIPS contracts differ")
    if len({i["record"]["source_sha256"] for i in items}) != 1 or len({i["record"]["target_pixel_sha256"] for i in items}) != 1:
        raise ValueError("source/reference pixels differ")
    hashes["dev_manifest"] = sha_file(args.dev_manifest)
    hashes["pretrained_manifest"] = sha_file(args.pretrained_manifest)
    pretrained = read_manifest(args.pretrained_manifest, verify=False)
    if pretrained["_fingerprint"] != metrics["ordinary"]["contract"]["pretrained_fingerprint"]:
        raise ValueError("VAE asset manifest differs from prior H3")
    return items, hashes, pretrained


def decode_with_preclip_capture(h3, latent, expected_shape):
    """Observe MiniMax's streaming finalizer; never change its returned pixels."""
    import torch
    stage = h3.vae.first_stage_model
    original = stage._finalize_pixels
    had_instance_method = "_finalize_pixels" in stage.__dict__
    previous_instance_method = stage.__dict__.get("_finalize_pixels")
    chunks = []

    def capture(part):
        # Exact frozen MiniMax _finalize_pixels scale/mean, before its clamp.
        raw = part * stage.pixel_std.to(device=part.device, dtype=torch.float32)
        raw.add_(stage.pixel_mean.to(device=part.device, dtype=torch.float32))
        chunks.append(raw.detach().cpu().contiguous())
        return original(part)

    stage._finalize_pixels = capture
    try:
        decoded = h3.decode(latent).detach().float().cpu().contiguous()
    finally:
        if had_instance_method:
            stage._finalize_pixels = previous_instance_method
        else:
            delattr(stage, "_finalize_pixels")
    if not chunks or any(p.ndim != 5 or p.shape[0] != 1 or p.shape[1] != 3 for p in chunks):
        raise ValueError("unexpected MiniMax streaming finalizer layout")
    raw = torch.cat(chunks, dim=2)[:, :, :expected_shape[0]].movedim(1, -1)[0].contiguous()
    if list(raw.shape) != list(expected_shape) or decoded.shape != raw.shape:
        raise ValueError("pre-clamp capture frame/geometry mismatch")
    parity_error = float((raw.clamp(0, 1) - decoded).abs().max())
    if not torch.isfinite(raw).all() or not torch.equal(raw.clamp(0, 1), decoded):
        raise ValueError("captured pre-clamp RGB does not reproduce original H3.decode; max_abs=" + str(parity_error))
    return raw.numpy(), decoded.numpy(), {"captured_parts": len(chunks),
        "captured_frames_before_final_crop": sum(p.shape[2] for p in chunks),
        "clamp_parity_exact": True, "clamp_parity_max_abs_error": parity_error,
        "capture_order": "post temporal overlap/blending and pre-finalizer; concatenate write_part order, truncate to original output frames",
        "pre_internal_clamp_min": float(raw.min()), "pre_internal_clamp_max": float(raw.max()),
        "outside_01_fraction": float(((raw < 0) | (raw > 1)).float().mean())}


def roundtrip_pair(h3, raw_rgb, rphi_rgb, reference):
    """No target argument: only the two exact guides enter encode/decode."""
    import torch
    outputs, records = {}, {}
    for label, array in (("raw", raw_rgb), ("rphi", rphi_rgb)):
        validate_h3_geometry(array.shape)
        if array.dtype != np.float32 or not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("guides require finite float32 RGB in [0,1]")
        expected = reference["sampling"]
        if array_sha(array) != expected["guide_pixel_sha256"][label]:
            raise ValueError("prior H3 input pixel SHA mismatch: " + label)
        start = time.monotonic()
        latent = h3.encode_guide_pixels(torch.from_numpy(array.copy()), list(array.shape[1:3]))
        encode_seconds = time.monotonic() - start
        latent_sha = tensor_sha(latent)
        if latent_sha != expected["calls"][label]["guide_latent_sha256_float32"]:
            raise ValueError("prior H3 encoded latent SHA mismatch: " + label)
        if list(latent.shape) != [1, 24, 7, array.shape[1] // 16, array.shape[2] // 16]:
            raise ValueError("unexpected VAE latent geometry")
        start = time.monotonic()
        unbounded, clipped, audit = decode_with_preclip_capture(h3, latent, array.shape)
        outputs[label + "_pre_internal_clamp"] = unbounded
        outputs[label + "_as_returned_internal_clipped"] = clipped
        records[label] = {"guide_pixel_sha256": array_sha(array), "encoded_latent_sha256_float32": latent_sha,
            "encoded_latent_dtype": str(latent.dtype), "encoded_latent_shape": list(latent.shape),
            "latent_matches_prior_H3_exactly": True, "encode_seconds": encode_seconds,
            "decode_seconds": time.monotonic() - start, **audit}
    return outputs, records


def score_outputs(outputs, raw_rgb, rphi_rgb, target, prior, lpips_metric=None, require_flow=True):
    if target.dtype != np.uint8 or target.shape != raw_rgb.shape:
        raise ValueError("target must be same-geometry uint8 reference")
    gt = np.ascontiguousarray(target, dtype=np.float32) / np.float32(255)
    support, flow_note = flow_support(gt, enabled=require_flow)
    if require_flow and support is None:
        raise RuntimeError("required flow support unavailable: " + flow_note)
    arrays = {"input": raw_rgb, "rphi_only": rphi_rgb, **outputs}
    scores = {name: {"quality": rgb_metrics(pixels, gt, lpips_metric),
        "temporal": temporal_metrics(pixels, gt, support)} for name, pixels in arrays.items()}
    for arm in ("input", "rphi_only"):
        for key in ("mse", "mae", "psnr", "lpips"):
            old, new = prior["arms"][arm]["quality"].get(key), scores[arm]["quality"].get(key)
            if old is not None and new is not None and abs(old-new) > (1e-6 if key == "lpips" else 1e-10):
                raise ValueError("reconstructed baseline metric differs from prior H3: " + arm + "/" + key)
    paired = {}
    for label in ("raw", "rphi"):
        vae = scores[label + "_as_returned_internal_clipped"]["quality"]
        for prior_arm in ("h3_" + label, "h3_" + label + "_anchor"):
            paired[prior_arm + "_minus_vae_" + label] = {k: prior["arms"][prior_arm]["quality"][k] - vae[k]
                for k in ("psnr", "mse", "mae", "lpips") if vae[k] is not None}
    return {"arms": scores, "prior_H3_minus_VAE": paired, "flow": flow_note,
        "frame_indices": list(range(22)), "baseline_metric_parity": True}


def verify_vae_binding(pretrained, implementation):
    """Bind identity by bytes/hash, never assume the new checkout has asset links."""
    verified = {}
    for role, default in (("vae", implementation.VAE_PATH), ("context", implementation.CONTEXT_PATH)):
        record, path = pretrained[role], Path(pretrained["_paths"][role]).resolve()
        if record["sha256"] != implementation.SHA[role]:
            raise ValueError("manifest component differs from frozen loader identity: " + role)
        if role == "vae" and record["bytes"] != implementation.SIZE[role]:
            raise ValueError("manifest VAE size differs from frozen loader")
        if not path.is_file() or path.stat().st_size != record["bytes"]:
            raise ValueError("manifest component missing/size mismatch: " + role)
        if sha_file(path) != record["sha256"]:
            raise ValueError("manifest component actual SHA mismatch: " + role)
        verified[role] = {"path": str(path), "bytes": record["bytes"], "sha256": record["sha256"],
            "original_default": str(default), "original_default_exists": Path(default).is_file(),
            "same_resolved_default_path": Path(default).resolve() == path}
    return verified


@contextmanager
def manifest_vae_binding(implementation, verified):
    """Only this isolated process's two component constants change, then restore."""
    old = implementation.VAE_PATH, implementation.CONTEXT_PATH
    implementation.VAE_PATH = Path(verified["vae"]["path"])
    implementation.CONTEXT_PATH = Path(verified["context"]["path"])
    try:
        yield
    finally:
        implementation.VAE_PATH, implementation.CONTEXT_PATH = old


def load_vae_only(pretrained):
    import torch
    import h3_comfy as implementation
    binding = verify_vae_binding(pretrained, implementation)
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    with manifest_vae_binding(implementation, binding):
        model = implementation.H3(load_dit=False, verify_sha=True)
    if model.patcher is not None or hasattr(model, "frozen"):
        raise ValueError("VAE-only loader unexpectedly loaded DiT")
    stage = model.vae.first_stage_model
    if type(stage).__name__ != "MiniMaxH3VideoVAE" or not hasattr(stage, "_finalize_pixels"):
        raise ValueError("unexpected VAE architecture/finalizer")
    model.record["vae_only"] = {"weights": pretrained["vae"], "component_binding": binding,
        "component_binding_restored_after_load": True, "dtype": str(model.vae.vae_dtype),
        "scale": "normalized native VAE latent, no external scaling; exact prior encode SHA required",
        "padding": "unmodified H3 encode_guide_pixels / MiniMax internal chunk padding and overlap",
        "finalizer_source_sha256": hashlib.sha256(inspect.getsource(type(stage)._finalize_pixels).encode()).hexdigest(),
        "sampler_calls": 0, "denoiser_evaluations": 0, "dit_loaded": False,
        "note": "Default H3 loader also reads the unchanged small context; no diffusion weights/components loaded."}
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("ordinary-metrics", "ordinary-guides", "controls-metrics", "controls-guides",
                 "dev-manifest", "pretrained-manifest", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--lpips-weights", type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    items, hashes, pretrained = inspect_bundle(args)
    if args.inspect_only:
        print(json.dumps({"verified": True, "samples": 12, "unique_sources": 1,
            "roundtrip_budget": 24, "H3_sampler_calls": 0, "NFE": 0, "hashes": hashes}))
        return 0
    metric, lpips_info = load_local_lpips(args.lpips_weights)
    if metric is None:
        raise RuntimeError("required local LPIPS unavailable")
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "complete": False, "contract": {"input_hashes": hashes,
        "source_ids": sorted({i["record"]["source_id"] for i in items}), "frame_stride": 1,
        "geometry": [22, 480, 832, 3], "samples": 12, "roundtrip_budget": 24,
        "H3_sampler_calls": 0, "NFE": 0, "input_dtype": "float32 THWC [0,1]",
        "cache_dtype": "float32 THWC; both true pre-internal-clamp RGB and original H3.decode return",
        "target_usage": "reference geometry/integrity and metrics only; no target supplied to roundtrip_pair",
        "scope": "one-source mechanism diagnostic; controls separate, identity task labels are repeats",
        "raw_lpips": "evaluated directly on pre-clamp RGB; may be outside LPIPS usual [0,1] input domain",
        "no_restoration_compensation_implemented": True}, "lpips": lpips_info,
        "implementation_sha256": {n: sha_file(Path(__file__).parent / n) for n in (
            "run_vae_roundtrip_diagnostic.py", "h3_comfy.py", "run_fixed_h3_rphi.py", "rphi_h3_controls.py",
            "rphi_data.py", "eval_rphi.py", "openh3_assets.py")}, "per_sample": []}
    try:
        h3 = load_vae_only(pretrained)
        frozen = [(n, p, p._version) for n, p in h3.vae.first_stage_model.named_parameters()]
        for item in items:
            loaded, row = item["dataset"].load(item["index"]), item["export"]
            with np.load(item["guide_file"], allow_pickle=False) as archive:
                restored = archive["rgb"].copy()
                raw = (archive["input_rgb"].copy() if item["scope"] == "controls" else
                    prepare_observed(loaded["observed"], restored.shape[1:3]).permute(0, 2, 3, 1).contiguous().numpy())
            outputs, observations = roundtrip_pair(h3, raw, restored, item["reference"])
            scored = score_outputs(outputs, raw, restored, loaded["target"], item["reference"], metric)
            name = item["scope"] + "-" + row["sample_id"] + ".npz"
            with (args.out / name).open("xb") as stream:
                np.savez_compressed(stream, **outputs)
            result["per_sample"].append({"scope": item["scope"], **{k: row[k] for k in (
                "sample_id", "source_id", "source_sha256", "task", "target_pixel_sha256", "guide_file_sha256")},
                "control": row.get("control"), "control_recipe": row.get("control_recipe"),
                "observations": observations, **scored, "cache": {"path": name, "file_sha256": sha_file(args.out / name),
                    "arrays": {k: {"shape": list(v.shape), "dtype": str(v.dtype), "array_sha256": array_sha(v)} for k, v in outputs.items()}}})
            write_partial_atomic(args.out / "partial_metrics.json", result)
            print(json.dumps({"done": len(result["per_sample"]), "total": len(items), "scope": item["scope"],
                              "control": row.get("control")}), flush=True)
        for name, digest in hashes.items():
            if sha_file(getattr(args, name)) != digest:
                raise ValueError("input manifest/metrics changed during diagnostic: " + name)
        if any(p._version != version or p.requires_grad or p.grad is not None for _, p, version in frozen):
            raise ValueError("VAE parameters changed")
        result.update(complete=True, model_record=h3.record,
            freeze_audit={"VAE_parameter_versions_unchanged": True, "tensors": len(frozen)})
        write_new(args.out / "metrics.json", result)
    except Exception as error:
        write_new(args.out / "failure.json", {"complete": False, "completed_samples": len(result["per_sample"]),
            "error": type(error).__name__ + ": " + str(error)})
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
