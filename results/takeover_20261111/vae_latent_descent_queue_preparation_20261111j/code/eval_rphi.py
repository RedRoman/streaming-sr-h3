"""Standalone R_phi RGB evaluation. No H3/VAE, GT conditions, or downloads.

--out must be a new directory. Metrics compare RGB against the original-recording
proxy at its original crop resolution; no x4 band substitution is performed.
Identity/near-clean inputs are explicitly separate controls, never ordinary dev.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

SCHEMA = "rphi.rgb_eval.v1"
TASKS = ("sr", "deblur", "denoise", "compression")
VARIANCE_EDGES = np.array([-np.inf, -12, -10, -8, -6, -4, -2, 0, np.inf])
METRICS = {"psnr": "higher", "mae": "lower", "mse": "lower", "lpips": "lower"}


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def sample_id(record):
    """Stable identity even when the materialized manifest has no sample_id."""
    return "rphi-" + canonical_sha({key: record[key] for key in ("source_id", "target_key", "task")})


def _matching_rgb(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 4 or a.shape[-1] != 3:
        raise ValueError("Expected matching [T,H,W,3] RGB arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("RGB arrays must be finite")
    return a, b


def rgb_metrics(prediction, target, lpips_metric=None):
    prediction, target = _matching_rgb(prediction, target)
    frame_mse = np.square(prediction - target).mean(axis=(1, 2, 3))
    result = {"psnr": float(np.mean(-10 * np.log10(np.maximum(frame_mse, 1e-12)))),
              "psnr_from_mean_mse": float(-10 * np.log10(max(float(frame_mse.mean()), 1e-12))),
              "mse": float(frame_mse.mean()), "mae": float(np.abs(prediction - target).mean()),
              "perfect_frames": int((frame_mse == 0).sum()), "frames": len(frame_mse), "lpips": None}
    if lpips_metric is not None:
        import torch
        distances = []
        with torch.no_grad():
            for pred, truth in zip(prediction, target):
                p = torch.from_numpy(pred.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
                g = torch.from_numpy(truth.astype(np.float32)).permute(2, 0, 1)[None] * 2 - 1
                distances.append(float(lpips_metric(p, g).item()))
        result["lpips"] = float(np.mean(distances))
    return result


def error_attribution(prediction, baseline, target):
    """Exact RGB MSE change; no orthogonality or physical null-space assumption."""
    prediction, target = _matching_rgb(prediction, target)
    baseline, _ = _matching_rgb(baseline, target)
    error, correction = baseline - target, prediction - baseline
    base = float(np.square(error).mean())
    actual = float(np.square(prediction - target).mean())
    energy = float(np.square(correction).mean())
    cross = float(2 * (error * correction).mean())
    return {"baseline_mse": base, "model_mse": actual, "mse_change": actual - base,
            "correction_energy": energy, "cross_term": cross,
            "identity_abs_error": abs(actual - base - energy - cross)}


def calibration_statistics(raw_prediction, target, log_variance):
    """Gaussian variance calibration of RAW errors, never a quality-weighted score.

    One predicted variance per RGB pixel is interpreted as per-channel variance.
    Coverage counts RGB components, bins count pixels. Fixed log-variance bins
    make statistics exactly poolable across samples without quantile tie bias.
    """
    pred, truth = _matching_rgb(raw_prediction, target)
    lv = np.asarray(log_variance, dtype=np.float64)
    if lv.shape == pred.shape[:-1]:
        lv = lv[..., None]
    if lv.shape != pred.shape[:-1] + (1,):
        raise ValueError("Expected scalar log variance [T,H,W,1]")
    if not np.isfinite(lv).all() or np.abs(lv).max() > 80:
        raise ValueError("Invalid or numerically unbounded log variance")
    error = pred - truth
    variance = np.exp(lv[..., 0])
    mse = np.square(error).mean(-1)
    sigma = np.sqrt(variance)[..., None]
    index = np.searchsorted(VARIANCE_EDGES[1:-1], lv[..., 0], side="right").ravel()
    def bins(values):
        return np.bincount(index, weights=np.asarray(values).ravel(), minlength=8).tolist()
    return {"bin_count": np.bincount(index, minlength=8).tolist(),
            "bin_variance_sum": bins(variance), "bin_mse_sum": bins(mse),
            "bin_coverage_1sigma_sum": bins((np.abs(error) <= sigma).mean(-1)),
            "bin_coverage_1_96sigma_sum": bins((np.abs(error) <= 1.96 * sigma).mean(-1)),
            "bin_nll_sum": bins(.5 * (mse / variance + lv[..., 0]))}


def summarize_calibration(statistics):
    if not statistics:
        return {"available": False, "reason": "model has no uncertainty head"}
    totals = {key: np.sum([np.asarray(row[key]) for row in statistics], axis=0)
              for key in statistics[0]}
    count = totals["bin_count"]
    n = int(count.sum())
    rows = []
    calibration_error = 0.0
    for i, c in enumerate(count):
        row = {"log_variance_low": None if i == 0 else float(VARIANCE_EDGES[i]),
               "log_variance_high": None if i == 7 else float(VARIANCE_EDGES[i + 1]),
               "pixels": int(c)}
        if c:
            pv = float(totals["bin_variance_sum"][i] / c)
            mse = float(totals["bin_mse_sum"][i] / c)
            row.update(predicted_variance=pv, empirical_mse=mse,
                       predicted_rms_sigma=math.sqrt(pv), empirical_rmse=math.sqrt(mse),
                       coverage_1sigma=float(totals["bin_coverage_1sigma_sum"][i] / c),
                       coverage_1_96sigma=float(totals["bin_coverage_1_96sigma_sum"][i] / c))
            calibration_error += c * abs(pv - mse)
        rows.append(row)
    return {"available": True, "error_space": "raw unclamped RGB; per-channel variance",
            "pixels": n, "bins": rows,
            "variance_calibration_mae": float(calibration_error / n),
            "gaussian_nll_without_constant": float(totals["bin_nll_sum"].sum() / n),
            "coverage_1sigma": float(totals["bin_coverage_1sigma_sum"].sum() / n),
            "coverage_1_96sigma": float(totals["bin_coverage_1_96sigma_sum"].sum() / n),
            "nominal_gaussian_coverage": [0.682689492, 0.95000421],
            "interpretation": "Calibration diagnostic; no confidence weighting of RGB quality metrics."}


def causal_indices(frame_indices, k, frame_count):
    if k < 1 or frame_count < 1 or any(t < 0 or t >= frame_count for t in frame_indices):
        raise ValueError("Invalid causal context indices")
    return np.array([[max(0, t - j) for j in range(k - 1, -1, -1)] for t in frame_indices], dtype=np.int64)


def control_seed(record):
    # No hidden degradation or GT-derived statistic enters the conditioning.
    key = [record["source_id"], record["target_key"], record["task"], "near_clean_eval_v1"]
    return int(canonical_sha(key)[:16], 16) % (2**63)


def validate_contract(checkpoint, manifest_sha256, records):
    if checkpoint.get("schema") != "rphi.checkpoint.v1":
        raise ValueError("Expected rphi.checkpoint.v1")
    contract = checkpoint.get("training_contract", {})
    if contract.get("dev_manifest_sha256") != manifest_sha256:
        raise ValueError("Checkpoint/dev manifest SHA mismatch")
    if contract.get("target") != "clean_original_recording":
        raise ValueError("Checkpoint target must be clean_original_recording RGB proxy")
    if tuple(contract.get("task_order", [])) != TASKS:
        raise ValueError("Checkpoint task order mismatch")
    ids, hashes = contract.get("train_source_ids"), contract.get("train_source_sha256")
    if not ids or not hashes:
        raise ValueError("Checkpoint must record train source IDs and content hashes")
    hashes = set(hashes.values()) if isinstance(hashes, dict) else set(hashes)
    if set(ids) & {r["source_id"] for r in records}:
        raise ValueError("Training/dev source ID overlap")
    if hashes & {r["source_sha256"] for r in records}:
        raise ValueError("Training/dev source content overlap")
    samples, targets = set(), {}
    for record in records:
        if record.get("split") != "dev" or record.get("task") not in TASKS:
            raise ValueError("Expected only four-task development records")
        sample = sample_id(record)
        if sample in samples:
            raise ValueError("Duplicate development sample ID")
        samples.add(sample)
        for field in ("source_sha256", "target_pixel_sha256", "observation_pixel_sha256"):
            value = record.get(field, "")
            if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
                raise ValueError("Missing/invalid sample SHA: " + field)
        key = (record["source_id"], record["target_key"])
        old = targets.setdefault(key, record["target_pixel_sha256"])
        if old != record["target_pixel_sha256"]:
            raise ValueError("Tasks do not share an identical RGB target")
    return contract


def source_summary(rows):
    """Equal weight per source, first average its windows. Controls stay separate."""
    result = {"sources": len({r["source_id"] for r in rows}), "samples": len(rows),
              "model": {}, "baseline": {}, "model_minus_baseline": {}}
    for key, direction in METRICS.items():
        by_source = {}
        for row in rows:
            a, b = row["model"].get(key), row["baseline"].get(key)
            if a is not None and b is not None:
                by_source.setdefault(row["source_id"], []).append([a, b])
        if not by_source:
            result["model"][key] = result["baseline"][key] = None
            result["model_minus_baseline"][key] = None
            continue
        values = np.array([np.mean(by_source[s], axis=0) for s in sorted(by_source)])
        diff = values[:, 0] - values[:, 1]
        result["model"][key], result["baseline"][key] = map(float, values.mean(axis=0))
        rng = np.random.default_rng(101)
        ci = np.quantile(diff[rng.integers(0, len(diff), (2000, len(diff)))].mean(1), [.025, .975]) if len(diff) > 1 else None
        result["model_minus_baseline"][key] = {
            "mean": float(diff.mean()), "median_source_delta": float(np.median(diff)),
            "ci95_source_bootstrap": ci.tolist() if ci is not None else None,
            "direction": direction, "model_better_sources": int((diff > 0 if direction == "higher" else diff < 0).sum()),
            "ties": int((diff == 0).sum()), "n_sources": len(diff)}
    if rows:
        damage = np.array([r["attribution"]["mse_change"] for r in rows])
        result["damage"] = {"threshold_mse": 1e-12, "damaged_samples": int((damage > 1e-12).sum()),
                            "samples": len(rows), "max_mse_increase": float(damage.max())}
    result["temporal"] = {}
    for key in ("unwarped_rgb_error_change_mae", "flow_aligned_rgb_error_change_mae"):
        grouped = {}
        for row in rows:
            a, b = row.get("temporal_model", {}).get(key), row.get("temporal_baseline", {}).get(key)
            if a is not None and b is not None:
                grouped.setdefault(row["source_id"], []).append([a, b])
        if grouped:
            values = np.array([np.mean(v, axis=0) for v in grouped.values()])
            result["temporal"][key] = {"model": float(values[:, 0].mean()), "baseline": float(values[:, 1].mean()),
                                        "model_minus_baseline": float((values[:, 0] - values[:, 1]).mean()),
                                        "n_sources": len(values), "direction": "lower"}
        else:
            result["temporal"][key] = None
    return result


def flow_support(target, enabled=True):
    if not enabled:
        return None, "disabled"
    try:
        import cv2
        from restoration_metrics import flow_visibility_mask
    except ImportError:
        return None, "cv2 unavailable; unwarped RGB error-change diagnostic remains available"
    flows = []
    for a, b in zip(target[:-1], target[1:]):
        gray = lambda x: cv2.cvtColor(np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        a, b = gray(a), gray(b)
        fw = cv2.calcOpticalFlowFarneback(a, b, None, .5, 3, 15, 3, 5, 1.2, 0)
        bw = cv2.calcOpticalFlowFarneback(b, a, None, .5, 3, 15, 3, 5, 1.2, 0)
        flows.append((fw, flow_visibility_mask(fw, bw)))
    return flows, "GT-proxy Farneback forward flow; in-bounds + forward/backward consistency mask"


def temporal_metrics(prediction, target, support):
    pred, gt = _matching_rgb(prediction, target)
    error = pred - gt
    result = {"pairs": max(0, len(pred) - 1), "unwarped_rgb_error_change_mae":
              float(np.abs(np.diff(error, axis=0)).mean()) if len(pred) > 1 else None,
              "flow_aligned_rgb_error_change_mae": None, "valid_coverage": None}
    if support:
        from restoration_metrics import masked_warp_mae
        values = [masked_warp_mae(a.astype(np.float32), b.astype(np.float32), f, m)
                  for a, b, (f, m) in zip(error[:-1], error[1:], support)]
        valid = [x for x in values if x is not None]
        result.update(flow_aligned_rgb_error_change_mae=float(np.mean(valid)) if valid else None,
                      valid_coverage=float(np.mean([m.mean() for _, m in support])),
                      valid_pairs=len(valid))
    return result


def load_local_lpips(weights=None):
    """Construct random feature topology, then load LOCAL AlexNet weights strictly.

    pnet_rand=True prevents torchvision's implicit pretrained-weight download.
    Every feature parameter is immediately replaced; random LPIPS is never used.
    """
    import torch
    path = Path(weights) if weights else Path(torch.hub.get_dir()) / "checkpoints/alexnet-owt-7be5be79.pth"
    if not path.is_file():
        return None, {"available": False, "reason": "local pretrained AlexNet weights missing", "looked_at": str(path)}
    try:
        import lpips
        with torch.device("cpu"):
            metric = lpips.LPIPS(net="alex", pnet_rand=True, pretrained=True, verbose=False).cpu().eval()
        state = torch.load(path, map_location="cpu", weights_only=True)
        features = {name: state["features." + name.split(".", 1)[1]] for name in metric.net.state_dict()}
        metric.net.load_state_dict(features, strict=True)
        if any(p.device.type != "cpu" for p in metric.parameters()):
            raise RuntimeError("LPIPS must remain on CPU")
        return metric, {"available": True, "network": "alex", "device": "cpu",
                        "alexnet_sha256": sha_file(path), "weights_path": str(path), "downloads": False}
    except (ImportError, KeyError, RuntimeError, OSError) as error:
        return None, {"available": False, "reason": type(error).__name__ + ": " + str(error), "downloads": False}


def infer_selected(model, guide, condition, frames, device, batch_size=1):
    import torch
    if batch_size < 1:
        raise ValueError("Inference batch size must be positive")
    indices = causal_indices(frames, model.k, len(guide))
    output = {"rgb": [], "log_variance": []}
    with torch.no_grad():
        for start in range(0, len(frames), batch_size):
            ix = torch.from_numpy(indices[start:start + batch_size])
            observed = guide[ix].to(device)
            cond = condition[None].expand(len(ix), -1).to(device) if condition is not None else None
            value = model(observed, cond)
            rgb = value["rgb"]
            if rgb.shape != observed[:, -1].shape or not torch.isfinite(rgb).all():
                raise ValueError("Invalid model RGB output")
            output["rgb"].append(rgb.detach().float().cpu().permute(0, 2, 3, 1).numpy())
            lv = value.get("log_variance")
            if lv is not None:
                if lv.shape != rgb[:, :1].shape or not torch.isfinite(lv).all():
                    raise ValueError("Invalid scalar-per-pixel log_variance")
                output["log_variance"].append(lv.detach().float().cpu().permute(0, 2, 3, 1).numpy())
    return {key: np.concatenate(parts) if parts else None for key, parts in output.items()}


def verify_chunk_equivalence(model, guide, condition, frames, device, tolerance=1e-5):
    # Uneven grouping of complete histories emulates stream chunks without dropping
    # t-2/t-1 at a boundary. Check adjacent and strided positions, not subsampled history.
    chosen = sorted(set([0, min(1, len(guide) - 1), min(2, len(guide) - 1), *frames[:3]]))
    single = infer_selected(model, guide, condition, chosen, device, 1)
    chunked = infer_selected(model, guide, condition, chosen, device, 2)
    differences = {}
    for key in ("rgb", "log_variance"):
        if single[key] is not None:
            differences[key] = float(np.max(np.abs(single[key] - chunked[key])))
            if differences[key] > tolerance:
                raise ValueError(f"Chunk equivalence failed for {key}: {differences[key]}")
    return {"checked_frames": chosen, "tolerance_max_abs": tolerance, "max_abs_differences": differences,
            "passed": True, "context": "absolute original frame t-2..t; history preserved across groups"}


def write_examples(directory, frames, truth, baseline, prediction):
    from PIL import Image
    directory.mkdir(parents=True, exist_ok=False)
    chosen = sorted(set([0, len(frames) // 2, len(frames) - 1]))
    for i in chosen:
        panel = np.concatenate([baseline[i], prediction[i], truth[i]], axis=1)
        Image.fromarray(np.rint(np.clip(panel, 0, 1) * 255).astype(np.uint8)).save(directory / f"frame_{frames[i]:04d}.png")
    return {"frames": [frames[i] for i in chosen], "panel_order": ["identity_or_bicubic", "model", "original_RGB_proxy"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dev-manifest", required=True, type=Path)
    parser.add_argument("--root-override", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-stride", default=4, type=int)
    parser.add_argument("--condition-mode", choices=["known", "missing"], default="known")
    parser.add_argument("--max-sources", type=int)
    parser.add_argument("--lpips-weights", type=Path)
    parser.add_argument("--skip-flow", action="store_true")
    args = parser.parse_args()
    if args.frame_stride < 1 or (args.max_sources is not None and args.max_sources < 1):
        parser.error("stride and max-sources must be positive")
    if args.out.exists():
        raise FileExistsError(f"Evaluation output already exists: {args.out}")
    import torch
    from rphi_data import RphiDataset, build_condition, prepare_observed, make_near_clean
    from rphi_model import RphiModel
    checkpoint_sha = sha_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if sha_file(args.checkpoint) != checkpoint_sha:
        raise ValueError("Checkpoint changed while loading")
    dataset = RphiDataset(args.dev_manifest, root_override=args.root_override, verify_hashes=True)
    contract = validate_contract(checkpoint, sha_file(args.dev_manifest), dataset.records)
    config = checkpoint["model_config"]
    if args.condition_mode == "missing" and config["variant"] != "known":
        parser.error("--condition-mode missing is only a known-model interference control")
    sources = sorted({r["source_id"] for r in dataset.records})
    selected_sources = sources[:args.max_sources] if args.max_sources else sources
    selected = [(i, r) for i, r in enumerate(dataset.records) if r["source_id"] in selected_sources]
    for source in selected_sources:
        if {r["task"] for _, r in selected if r["source_id"] == source} != set(TASKS):
            raise ValueError("Each selected development source must cover all four tasks")
    if args.device.startswith("npu:"):
        import torch_npu  # noqa: F401 -- register device, never import H3
        torch.npu.set_device(args.device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
    elif args.device != "cpu":
        parser.error("--device must be cpu or npu:<logical-index>")
    model = RphiModel(**config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(args.device).eval()
    lpips_metric, lpips_info = load_local_lpips(args.lpips_weights)
    args.out.mkdir(parents=True, exist_ok=False)
    result = {"schema": SCHEMA, "complete": False, "checkpoint_sha256": checkpoint_sha,
              "fidelity_state_sha256": checkpoint.get("fidelity_state_sha256"),
              "initial_fidelity_sha256": checkpoint.get("initial_fidelity_sha256"),
              "source_checkpoint_sha256": checkpoint.get("source_checkpoint_sha256"),
              "calibration_contract": checkpoint.get("calibration_contract"),
              "calibration_step": checkpoint.get("calibration_step"),
              "fidelity_invariance": checkpoint.get("fidelity_invariance"),
              "initial_backbone_sha256": contract.get("initial_backbone_sha256"),
              "checkpoint_step": checkpoint.get("step"), "model_config": config, "training_contract": contract,
              "dev_manifest_sha256": sha_file(args.dev_manifest), "device": args.device,
              "condition_mode": args.condition_mode if config["variant"] == "known" else "observations_only",
              "scope": "engineering_smoke_subset" if len(selected_sources) < len(sources) else "full_development_set",
              "source_ids": selected_sources, "available_sources": len(sources), "frame_stride": args.frame_stride,
              "metric_contract": {"rgb": "float32 clipped [0,1], no prediction quantization; original recording RGB proxy",
                  "baseline": "SR: torch bicubic antialias=True align_corners=False; native tasks: observed RGB identity",
                  "psnr": "mean per-frame RGB PSNR, unit peak; exact-zero MSE capped at 120 dB",
                  "aggregation": "windows averaged within source, then equal source weights; controls excluded from task scores",
                  "temporal": "selected frame pairs t,t+frame_stride; full RGB error-change, no x4 band/physical-null claim",
                  "calibration": "raw unclamped RGB errors, separate from quality metrics; nominal Gaussian coverage is a diagnostic",
                  "causal_context": f"original consecutive frames t-{config['k'] - 1}..t with first-frame repetition",
                  "control_inputs": "explicit identity and sigma=1/255 near-clean RGB controls from target; never ordinary dev input",
                  "comparison_limit": "development ablations, not an untouched generalization test"},
              "dataset_provenance": dict(dataset.provenance),
              "lpips": lpips_info, "per_sample": []}
    started = time.monotonic()
    examples_written = set()
    try:
        with (args.out / "progress.jsonl").open("x") as progress:
            for index, record in selected:
                loaded = dataset.load(index)
                observed, target = loaded["observed"], loaded["target"]
                if loaded["record"] != record:
                    raise ValueError("Dataset returned a different sample record")
                frames = list(range(0, len(target), args.frame_stride))
                gt = target[frames].astype(np.float32) / 255
                support, flow_note = flow_support(gt, not args.skip_flow)
                condition = build_condition(record["task"], known=args.condition_mode == "known") if config["variant"] == "known" else None
                seed = control_seed(record)
                row = {key: record[key] for key in ("source_id", "source_sha256", "target_key", "task", "target_pixel_sha256", "observation_pixel_sha256")}
                row.update(sample_id=sample_id(record), record_sha256=canonical_sha(record), frame_indices=frames, controls={}, flow=flow_note)
                inputs = [("task", observed), ("identity", target), ("near_clean", make_near_clean(target, seed=seed, sigma_8bit=1.0))]
                for control, pixels in inputs:
                    guide = prepare_observed(pixels, target.shape[1:3])
                    baseline = guide[frames].permute(0, 2, 3, 1).numpy()
                    prediction = infer_selected(model, guide, condition, frames, args.device)
                    raw, clipped = prediction["rgb"], np.clip(prediction["rgb"], 0, 1)
                    scored = {"model": rgb_metrics(clipped, gt, lpips_metric),
                              "baseline": rgb_metrics(baseline, gt, lpips_metric),
                              "raw_model_mse": float(np.square(raw.astype(np.float64) - gt).mean()),
                              "clipped_component_fraction": float(((raw < 0) | (raw > 1)).mean()),
                              "attribution": error_attribution(clipped, baseline, gt),
                              "temporal_model": temporal_metrics(clipped, gt, support),
                              "temporal_baseline": temporal_metrics(baseline, gt, support)}
                    if prediction["log_variance"] is not None:
                        scored["calibration_sufficient_statistics"] = calibration_statistics(raw, gt, prediction["log_variance"])
                        scored["calibration"] = summarize_calibration([scored["calibration_sufficient_statistics"]])
                    else:
                        scored["calibration"] = summarize_calibration([])
                    if control == "task":
                        scored["chunk_equivalence"] = verify_chunk_equivalence(model, guide, condition, frames, args.device)
                        row.update(scored)
                        example_key = (record["source_id"], record["task"])
                        if example_key not in examples_written:
                            safe_id = canonical_sha(list(example_key))[:16]
                            row["examples"] = write_examples(args.out / "examples" / record["task"] / safe_id, frames, gt, baseline, clipped)
                            examples_written.add(example_key)
                    else:
                        row["controls"][control] = scored
                row["controls"]["near_clean"]["seed"] = seed
                row["controls"]["near_clean"]["sigma_8bit"] = 1.0
                result["per_sample"].append(row)
                message = {"sample_id": sample_id(record), "task": record["task"], "done": len(result["per_sample"]), "total": len(selected), "elapsed_seconds": round(time.monotonic() - started, 3)}
                progress.write(json.dumps(message) + "\n"); progress.flush()
                print(json.dumps(message), flush=True)
        rows = result["per_sample"]
        result["per_task"] = {task: source_summary([r for r in rows if r["task"] == task]) for task in TASKS}
        result["controls"] = {control: {task: source_summary([{"source_id": r["source_id"], **r["controls"][control]} for r in rows if r["task"] == task]) for task in TASKS} for control in ("identity", "near_clean")}
        result["calibration_per_task"] = {task: summarize_calibration([r["calibration_sufficient_statistics"] for r in rows if r["task"] == task and "calibration_sufficient_statistics" in r]) for task in TASKS}
        result["materialization_receipt_sha256"] = dict(dataset.receipt_hashes)
        result["complete"] = True
        result["elapsed_seconds"] = time.monotonic() - started
        with (args.out / "metrics.json").open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    except Exception as error:
        with (args.out / "failure.json").open("x") as stream:
            json.dump({"schema": SCHEMA, "complete": False, "completed_samples": len(result["per_sample"]), "error": type(error).__name__ + ": " + str(error)}, stream, indent=2)
        raise


if __name__ == "__main__":
    main()
