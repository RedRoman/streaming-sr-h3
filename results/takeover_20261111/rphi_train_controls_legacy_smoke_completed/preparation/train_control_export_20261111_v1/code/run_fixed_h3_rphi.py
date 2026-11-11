"""Fixed Open-H3 guide ablation on verified, unquantized four-task RGB exports.

Two H3 calls per sample: raw-guide/raw-draft and R_phi-guide/raw-draft.
The draft, weights, seed, sigma schedule and audio initialization stay fixed.
Soft anchors reuse those decoded outputs; they do not trigger another H3 call.
This first bridge handles ordinary task observations only. It cannot establish
clean-input protection, long-video stability or an end-to-end causal pipeline.
No NPU/H3 import occurs in --inspect-only mode. Use a reserved idle card via
the project queue; this program does not allocate or preempt cluster jobs.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from eval_rphi import (TASKS, canonical_sha, causal_indices, error_attribution, flow_support,
                       load_local_lpips, rgb_metrics, sample_id, sha_file,
                       source_summary, temporal_metrics, validate_contract)
from export_rphi_guides import array_sha, verify_export_file
from openh3_assets import NATIVE_SIGMAS, read_manifest
from rphi_data import RphiDataset, prepare_observed

SCHEMA = "openh3.fixed_rphi_guides.v1"


def contained_file(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError("guide asset must be contained in its export directory")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def inspect_inputs(guide_manifest, dev_manifest, max_sources=None):
    """Bind every exported prediction to its original manifest and input pixels."""
    guide_manifest, dev_manifest = Path(guide_manifest), Path(dev_manifest)
    if max_sources is not None and (isinstance(max_sources, bool) or max_sources < 1):
        raise ValueError("max_sources must be positive")
    guide_sha = sha_file(guide_manifest)
    export = json.loads(guide_manifest.read_text())
    if export.get("schema") != "rphi.guide_export.v1" or export.get("complete") is not True:
        raise ValueError("expected a complete R_phi guide export")
    dataset = RphiDataset(dev_manifest, verify_hashes=True)
    validate_contract({"schema": "rphi.checkpoint.v1", "training_contract": export["training_contract"]},
                      dataset.manifest_sha256, dataset.records)
    if export.get("dev_manifest_sha256") != dataset.manifest_sha256:
        raise ValueError("guide export/development manifest mismatch")
    if export.get("condition_mode") != "observations_only":
        raise ValueError("fixed blind comparison requires observations-only R_phi guides")
    if export.get("checkpoint_step") != export["training_contract"].get("steps") or not export.get("checkpoint_step"):
        raise ValueError("fixed candidate must be the final contracted checkpoint")
    config = export.get("model_config", {})
    if config.get("variant") not in ("blind", "blind_confidence") or config.get("k") != 3:
        raise ValueError("expected the declared blind K3 restoration model")
    proof = export.get("stage2_export_proof", {})
    if proof.get("checkpoint_sha256") != export.get("checkpoint_sha256") or not proof.get("inference_implementation_verified"):
        raise ValueError("stage2 export requires verified inference code and checkpoint binding")
    if config.get("observation_encoder") == "native":
        native_sha = proof.get("implementation_sha256", {}).get("code/rphi_condition_v2.py")
        if not native_sha or native_sha != export.get("checkpoint_implementation_sha256", {}).get("code/rphi_condition_v2.py"):
            raise ValueError("native observation encoder SHA is not bound to the checkpoint")
    rows = export["per_sample"]
    if export.get("samples") != len(rows) or not rows:
        raise ValueError("incomplete guide sample count")
    records = {sample_id(row): (i, row) for i, row in enumerate(dataset.records)}
    seen, verified = set(), []
    for row in rows:
        sid = row["sample_id"]
        if sid in seen or sid not in records:
            raise ValueError("duplicate or unknown guide sample")
        seen.add(sid)
        index, record = records[sid]
        if row.get("record_sha256") != canonical_sha(record):
            raise ValueError("guide record mismatch")
        for key in ("source_id", "source_sha256", "target_key", "task", "split",
                    "target_pixel_sha256", "observation_pixel_sha256"):
            if row.get(key) != record[key]:
                raise ValueError("guide source binding mismatch: " + key)
        for key in ("checkpoint_sha256", "model_config", "condition_mode", "dev_manifest_sha256"):
            if row.get(key) != export.get(key):
                raise ValueError("guide row/manifest mismatch: " + key)
        sidecar = contained_file(guide_manifest.parent, row["provenance_json"])
        if sha_file(sidecar) != row["provenance_file_sha256"]:
            raise ValueError("guide provenance hash mismatch")
        expected = {k: v for k, v in row.items() if k not in ("provenance_json", "provenance_file_sha256")}
        if json.loads(sidecar.read_text()) != expected:
            raise ValueError("guide provenance content mismatch")
        guide_file = contained_file(guide_manifest.parent, row["guide_npz"])
        verify_export_file(guide_file, row)
        loaded = dataset.load(index)  # Verifies receipt, NPZ bytes and RGB pixels.
        if list(loaded["target"].shape) != row["arrays"]["rgb"]["shape"]:
            raise ValueError("guide/target geometry mismatch")
        if row["frame_indices"] != list(range(len(loaded["target"]))):
            raise ValueError("H3 requires every consecutive original frame")
        if row.get("history_indices") != causal_indices(row["frame_indices"], 3, len(loaded["target"])).tolist():
            raise ValueError("guide history must be consecutive causal K3")
        for prefix in ("observation", "target"):
            path = dataset.resolve_path(record[prefix + "_npz"])
            if sha_file(path) != row[prefix + "_file_sha256"]:
                raise ValueError("guide/current asset mismatch: " + prefix)
        verified.append({"index": index, "record": record, "export": row, "guide_file": str(guide_file)})
    sources = sorted({row["record"]["source_id"] for row in verified})
    if sources != export["source_ids"]:
        raise ValueError("guide source list mismatch")
    required = {sid for sid, (_, row) in records.items() if row["source_id"] in sources}
    if required != seen:
        raise ValueError("guide export does not cover every task/window of its sources")
    for source in sources:
        if {row["record"]["task"] for row in verified if row["record"]["source_id"] == source} != set(TASKS):
            raise ValueError("each source must cover all four tasks")
    chosen = sources[:max_sources] if max_sources is not None else sources
    verified = [row for row in verified if row["record"]["source_id"] in chosen]
    if sha_file(guide_manifest) != guide_sha or sha_file(dev_manifest) != dataset.manifest_sha256:
        raise ValueError("manifest changed during inspection")
    return dataset, export, verified


def tensor_sha(value):
    return array_sha(value.detach().cpu().float().contiguous().numpy())


def validate_h3_geometry(shape):
    if len(shape) != 4 or shape[0] != 22 or shape[-1] != 3 or min(shape) < 1 or shape[1] % 32 or shape[2] % 32:
        raise ValueError("this H3 bridge supports 22 frames and dimensions divisible by 32 for spatial patchify")


def generate_pair(h3, raw_rgb, rphi_rgb, seed):
    """Two calls with independent clones of one fixed raw warm start."""
    import torch
    for rgb in (raw_rgb, rphi_rgb):
        if (rgb.device.type != "cpu" or rgb.dtype != torch.float32 or rgb.ndim != 4
                or rgb.shape[-1] != 3 or not torch.isfinite(rgb).all()
                or rgb.min() < 0 or rgb.max() > 1):
            raise ValueError("guides must be finite CPU float32 THWC RGB in [0,1]")
    if raw_rgb.shape != rphi_rgb.shape:
        raise ValueError("raw and R_phi guide geometries differ")
    frames, height, width, _ = raw_rgb.shape
    validate_h3_geometry(raw_rgb.shape)
    before = time.monotonic()
    input_hashes = {"raw": tensor_sha(raw_rgb), "rphi": tensor_sha(rphi_rgb)}
    guides = {"raw": h3.encode_guide_pixels(raw_rgb.clone(), [height, width]),
              "rphi": h3.encode_guide_pixels(rphi_rgb.clone(), [height, width])}
    encode_seconds = time.monotonic() - before
    for latent in guides.values():
        if list(latent.shape) != [1, 24, 7, height // 16, width // 16] or not torch.isfinite(latent).all():
            raise ValueError("unexpected H3 guide latent geometry or nonfinite values")
    latent_hashes = {key: tensor_sha(value) for key, value in guides.items()}
    outputs, calls = {}, {}
    for label in ("raw", "rphi"):
        guide, draft = guides[label].clone(), guides["raw"].clone()
        start = time.monotonic()
        generated, timing = h3.sample(guide, "base", hw=[height, width], seed=seed,
                                      warm_start=draft, audio_init="zeros")
        decoded = h3.decode(generated["video"]).detach().float().cpu()
        if decoded.shape != raw_rgb.shape or not torch.isfinite(decoded).all():
            raise ValueError("invalid decoded RGB output")
        if tensor_sha(guide) != latent_hashes[label] or tensor_sha(draft) != latent_hashes["raw"]:
            raise ValueError("H3 mutated the supplied guide or fixed draft")
        outputs[label] = decoded
        calls[label] = {"guide_latent_sha256_float32": latent_hashes[label],
                       "warm_start_latent_sha256_float32": latent_hashes["raw"],
                       "sampling_record": timing, "sampling_decode_seconds": time.monotonic() - start}
    return outputs, {"guide_pixel_sha256": input_hashes, "encode_both_seconds": encode_seconds,
                     "calls": calls, "h3_sample_calls": 2, "denoiser_evaluations": 2 * (len(NATIVE_SIGMAS) - 1), "warm_start": "fixed encoded raw RGB for both arms",
                     "guide_and_draft_joint_effect_tested": False}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--guide-manifest", required=True, type=Path)
    ap.add_argument("--dev-manifest", required=True, type=Path)
    ap.add_argument("--pretrained-manifest", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--alpha", type=float, default=.25)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--max-sources", type=int)
    ap.add_argument("--lpips-weights", type=Path)
    ap.add_argument("--inspect-only", action="store_true")
    args = ap.parse_args(argv)
    if not np.isfinite(args.alpha) or not 0 < args.alpha < 1 or args.frame_stride < 1:
        ap.error("alpha must be strictly between zero and one; stride must be positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    export_sha = sha_file(args.guide_manifest)
    dataset, export, samples = inspect_inputs(args.guide_manifest, args.dev_manifest, args.max_sources)
    for sample in samples:
        validate_h3_geometry(sample["export"]["arrays"]["rgb"]["shape"])
    pretrained = read_manifest(args.pretrained_manifest, verify=False)
    contract = {"guide_manifest_sha256": export_sha, "dev_manifest_sha256": dataset.manifest_sha256,
                "rphi_checkpoint_sha256": export["checkpoint_sha256"], "model_config": export["model_config"],
                "pretrained_fingerprint": pretrained["_fingerprint"], "sigma_schedule": list(NATIVE_SIGMAS),
                "seed": args.seed, "audio_init": "zeros", "mode": "base", "adaptation": None,
                "raw_guide_recipe": dataset.provenance["sr_input_preparation"],
                "warm_start": "same E(raw HR RGB) in both H3 calls; pretrained latent upscaler unused",
                "rphi_guide": "clipped float32 export rgb; no uint8/MP4 intermediate",
                "alpha": args.alpha, "anchor": "clipped R_phi RGB; canonical Q4 SR, full RGB native",
                "anchor_stage": "post-decode only; same generated output reused; no confidence weighting",
                "metric": "eval_rphi RGB/LPIPS/flow on float32 clipped output, no prediction quantization",
                "frame_stride": args.frame_stride, "controls_included": [],
                "future_frames": "full 22-frame H3/VAE block; no whole-pipeline causality claim",
                "scope": "development task observations; no clean-protection or frozen-test conclusion"}
    if args.inspect_only:
        print(json.dumps({"contract": contract, "sources": sorted({s["record"]["source_id"] for s in samples}),
                          "samples": len(samples), "h3_sample_calls": 2 * len(samples), "denoiser_evaluations": 2 * len(samples) * (len(NATIVE_SIGMAS) - 1),
                          "asset_hashes_verified": False, "guide_and_data_hashes_verified": True}, indent=2))
        return 0
    import torch
    from h3_comfy import H3
    from rphi_soft_anchor import apply_soft_anchor
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    metric, lpips_info = load_local_lpips(args.lpips_weights)
    if metric is None:
        raise RuntimeError("required local LPIPS unavailable: " + str(lpips_info))
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "complete": False, "contract": contract, "lpips": lpips_info,
              "implementation_sha256": {name: sha_file(Path(__file__).parent / name) for name in (
                  "run_fixed_h3_rphi.py", "h3_comfy.py", "openh3_runtime.py", "openh3_assets.py",
                  "eval_rphi.py", "restoration_metrics.py", "rphi_data.py", "rphi_model.py", "rphi_soft_anchor.py",
                  "pixel_consistency.py", "export_rphi_guides.py")}, "per_sample": []}
    try:
        h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS)
        for sample in samples:
            record, provenance = sample["record"], sample["export"]
            loaded = dataset.load(sample["index"])
            raw = prepare_observed(loaded["observed"], loaded["target"].shape[1:3]).permute(0, 2, 3, 1).contiguous()
            verify_export_file(sample["guide_file"], provenance)
            with np.load(sample["guide_file"], allow_pickle=False) as npz:
                rphi = torch.from_numpy(npz["rgb"].copy())
            decoded, sampling = generate_pair(h3, raw, rphi, args.seed)
            arrays = {"input": raw.numpy(), "rphi_only": rphi.numpy()}
            anchor_reports = {}
            for label, pixels in decoded.items():
                arrays["h3_" + label] = pixels.clamp(0, 1).numpy()
                anchored, audit = apply_soft_anchor(pixels.permute(0, 3, 1, 2).contiguous(),
                    rphi.permute(0, 3, 1, 2).contiguous(), record["task"], args.alpha)
                arrays["h3_" + label + "_anchor"] = anchored.permute(0, 2, 3, 1).numpy()
                anchor_reports[label] = audit
            frames = list(range(0, len(raw), args.frame_stride))
            gt = loaded["target"][frames].astype(np.float32) / 255
            support, flow_note = flow_support(gt)
            if support is None:
                raise RuntimeError("flow support required for this comparison: " + flow_note)
            scored = {key: {"quality": rgb_metrics(value[frames], gt, metric),
                            "temporal": temporal_metrics(value[frames], gt, support)} for key, value in arrays.items()}
            sid = sample_id(record)
            row = {"sample_id": sid, "source_id": record["source_id"], "task": record["task"],
                   "record_sha256": canonical_sha(record), "source_sha256": record["source_sha256"],
                   "observation_pixel_sha256": record["observation_pixel_sha256"],
                   "target_pixel_sha256": record["target_pixel_sha256"], "frame_indices": frames,
                   "sampling": sampling, "soft_anchor": anchor_reports, "flow": flow_note, "arms": scored,
                   "paired": {}}
            for arm, baseline in (("rphi_only", "input"), ("h3_raw", "input"), ("h3_rphi", "h3_raw"),
                                  ("h3_raw_anchor", "h3_raw"), ("h3_rphi_anchor", "h3_rphi"),
                                  ("h3_rphi_anchor", "h3_raw_anchor"), ("h3_rphi", "rphi_only"),
                                  ("h3_rphi_anchor", "rphi_only")):
                row["paired"][arm + "_vs_" + baseline] = {"model": scored[arm]["quality"], "baseline": scored[baseline]["quality"],
                    "baseline_arm": baseline, "temporal_model": scored[arm]["temporal"],
                    "temporal_baseline": scored[baseline]["temporal"],
                    "attribution": error_attribution(arrays[arm][frames], arrays[baseline][frames], gt)}
            # Preserve both unquantized decoded sequences for later controls/visual review.
            pixel_file = args.out / (sid + ".npz")
            with pixel_file.open("xb") as stream:
                np.savez_compressed(stream, **{key: value.numpy() for key, value in decoded.items()})
            row["decoded_file"] = {"path": pixel_file.name, "sha256": sha_file(pixel_file),
                                   "arrays": {key: array_sha(value.numpy()) for key, value in decoded.items()}}
            result["per_sample"].append(row)
            (args.out / "partial_metrics.json").write_text(json.dumps(result, allow_nan=False))
            print(json.dumps({"sample_id": sid, "done": len(result["per_sample"]), "total": len(samples)}), flush=True)
        if sha_file(args.guide_manifest) != export_sha or sha_file(args.dev_manifest) != dataset.manifest_sha256:
            raise ValueError("manifest changed during evaluation")
        result["comparisons"] = {arm: {task: source_summary([
            {"source_id": row["source_id"], **row["paired"][arm]} for row in result["per_sample"] if row["task"] == task])
            for task in TASKS} for arm in result["per_sample"][0]["paired"]}
        result.update(complete=True, model_record=h3.record, freeze_audit=h3.freeze_audit())
        (args.out / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    except Exception as error:
        (args.out / "failure.json").write_text(json.dumps({"complete": False, "completed_samples": len(result["per_sample"]),
            "error": type(error).__name__ + ": " + str(error)}))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
