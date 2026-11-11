"""Plan a small real-video OpenH3 pilot from a takeover_data_audit JSON.

This writes references and deterministic degradation recipes only. It does not
decode media, synthesize observations, encode latents, or start training. Native
non-SR tasks keep target resolution. Every task shares the same clean target for
a source/window. All historically evaluated source IDs and paths are excluded.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


TASKS = ("sr", "deblur", "denoise", "compression")


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def recipe(task, seed):
    fraction = (seed % 1000001) / 1000000
    if task == "sr":
        return {"operator": "bicubic_antialias_downsample", "scale": 4,
                "observation_resolution": "target_divided_by_4", "backend": "pending_pinned_converter"}
    if task == "deblur":
        return {"operator": "isotropic_gaussian_blur", "sigma_pixels": round(0.8 + 1.7 * fraction, 6),
                "observation_resolution": "target", "boundary": "reflect"}
    if task == "denoise":
        return {"operator": "additive_gaussian_noise", "sigma_8bit": round(5 + 20 * fraction, 6),
                "observation_resolution": "target", "clip_range": [0, 1]}
    if task == "compression":
        return {"operator": "h264_roundtrip", "codec": "libx264", "crf": 20 + seed % 16,
                "pixel_format": "yuv420p", "observation_resolution": "target"}
    raise ValueError(f"unknown task: {task}")


def make_plan(audit, train_sources=32, dev_sources=8, windows=1,
              frames=22, fps=24, height=480, width=832, seed=20261111):
    if audit.get("schema") != "openh3.takeover_data_audit.v1":
        raise ValueError("unsupported or missing audit schema")
    if min(train_sources, dev_sources, windows, frames, fps, height, width) <= 0:
        raise ValueError("counts and geometry must be positive")
    if height % 16 or width % 16 or height % 4 or width % 4:
        raise ValueError("pilot geometry must be divisible by 16 and by SR scale 4")
    blocked_ids = {str(i) for e in audit["evaluations"] for i in e["source_ids"]}
    blocked_paths = {str(p) for e in audit["evaluations"] for p in e.get("source_video_paths", {}).values()}
    candidates = audit["real_candidates"]
    duplicate_ids = [k for k, n in collections.Counter(c["source_id"] for c in candidates).items() if n > 1]
    if duplicate_ids:
        raise ValueError("duplicate candidate source IDs: " + ", ".join(duplicate_ids))
    pool = {"train": [], "dev": []}
    rejected = collections.Counter()
    for c in candidates:
        ident, path = c["source_id"], c["source_video"]
        if c.get("historical_eval") or ident in blocked_ids or path in blocked_paths:
            rejected["historical_evaluation"] += 1
            continue
        if c.get("source_kind") != "captured_video":
            rejected["not_captured_video"] += 1
            continue
        if not c.get("available"):
            rejected["source_unavailable_at_audit"] += 1
            continue
        split = {"train": "train", "val": "dev"}.get(c.get("historical_split"))
        if not split:
            rejected["unassigned_or_conflicting_split"] += 1
            continue
        wh = c.get("width_height") or [0, 0]
        duration = float(c.get("duration_s") or 0)
        if min(wh) <= 0 or wh[0] < width or wh[1] < height or not math.isfinite(duration) or duration < frames / fps + 0.2:
            rejected["insufficient_geometry_or_duration"] += 1
            continue
        pool[split].append(c)
    selected = {}
    for split, count in (("train", train_sources), ("dev", dev_sources)):
        ranked = sorted(pool[split], key=lambda c: (stable_seed(seed, c["source_group"]), c["source_id"]))
        if len(ranked) < count:
            raise ValueError(f"need {count} {split} sources, only {len(ranked)} available after exclusions")
        selected[split] = ranked[:count]
    # Groups, paths and IDs all remain disjoint, including aliased filenames.
    for field in ("source_id", "source_group", "source_video"):
        overlap = {c[field] for c in selected["train"]} & {c[field] for c in selected["dev"]}
        if overlap:
            raise ValueError(f"train/dev {field} overlap: {sorted(overlap)}")
    records = []
    for split in ("train", "dev"):
        for c in selected[split]:
            for window in range(windows):
                ws = stable_seed(seed, c["source_id"], window, "target")
                span = float(c["duration_s"]) - frames / fps - 0.2
                start = 0.1 + span * (window + ((ws % 1000001) / 1000000)) / windows
                target = {"source_video": c["source_video"], "start_s": round(start, 8),
                          "frames": frames, "fps": fps, "height": height, "width": width,
                          "spatial_transform": "center_crop_no_resize", "pixel_target": "clean_original_recording",
                          "clean_reference_caveat": c["clean_reference"], "seed": ws}
                target_key = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
                for task in TASKS:
                    ds = stable_seed(seed, c["source_id"], window, task)
                    records.append({"schema": "openh3.omni_pilot_sample.v1",
                                    "sample_id": f"{c['source_id']}__w{window}__{task}",
                                    "source_id": c["source_id"], "source_group": c["source_group"],
                                    "split": split, "task": task, "target_key": target_key,
                                    "target": target, "degradation": {**recipe(task, ds), "seed": ds},
                                    "audio": {"mode": "preserve_source_passthrough_for_video_pilot", "restoration_supervision": False},
                                    "status": "planned_requires_decode_degrade_and_encode",
                                    "preencoded_latents": None})
    summary = {"schema": "openh3.omni_pilot_plan.v1", "seed": seed,
               "available_after_exclusions": {s: len(v) for s, v in pool.items()},
               "selected_source_ids": {s: [c["source_id"] for c in cs] for s, cs in selected.items()},
               "samples_by_split": dict(collections.Counter(r["split"] for r in records)),
               "samples_by_task": dict(collections.Counter(r["task"] for r in records)),
               "excluded_candidates": dict(rejected), "historical_eval_ids_excluded": len(blocked_ids),
               "frozen_test": {"samples": 0, "status": "unavailable_requires_new_unseen_source_manifest"},
               "training_started": False, "media_generated": False,
               "requirements_before_training": ["Pin and verify OpenH3-Regenerate base, VAE and conditioning API.",
                                                "Implement and version recipes, then decode/encode the planned observations and common clean targets.",
                                                "Verify actual source hashes and output lengths at conversion time.",
                                                "Establish a new unseen frozen test before reporting generalization."]}
    return records, summary


def existing_latent_manifests(audit, summary):
    """Reference only verified existing Stage7 paths; no tensor content conversion."""
    manifests = {}
    for split, old_split in (("train", "train"), ("dev", "val")):
        selected = set(summary["selected_source_ids"][split])
        inventory = audit["inventory"].get("stage7_latents_" + old_split, {})
        samples = [s for s in inventory.get("samples", []) if s["source_id"] in selected]
        found = {s["source_id"] for s in samples}
        manifests[split] = {
            "schema": "openh3.existing_stage7_latents.v1", "split": split,
            "samples": sorted(samples, key=lambda s: (s["source_id"], s["latent_path"])),
            "selected_sources": len(selected), "available_sources": len(found),
            "missing_source_ids": sorted(selected - found),
            "target_key": "target", "guide_key": "guide_raw",
            "draft_mode": "pixel_upsample_guide_fallback",
            "draft_tensor_key": "guide_raw",
            "pretrained_latent_upsampler_used": False,
            "scope": "Legacy x4 engineering smoke only; not the four-task raw-video pilot.",
            "compatibility": "requires explicit trainer fallback and validation against the pinned OpenH3 VAE",
            "historical_evaluation_sources_excluded": True,
            "frozen_test": False,
        }
    return manifests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    for name, default in (("train-sources", 32), ("dev-sources", 8), ("windows", 1),
                          ("frames", 22), ("fps", 24), ("height", 480), ("width", 832), ("seed", 20261111)):
        parser.add_argument("--" + name, type=int, default=default)
    args = parser.parse_args()
    payload = args.audit.read_bytes()
    records, summary = make_plan(json.loads(payload), **{k: getattr(args, k) for k in
        ("train_sources", "dev_sources", "windows", "frames", "fps", "height", "width", "seed")})
    summary["audit_sha256"] = hashlib.sha256(payload).hexdigest()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: args.output_dir / f"{split}.jsonl" for split in ("train", "dev")}
    outputs["summary"] = args.output_dir / "summary.json"
    for split in ("train", "dev"):
        outputs[split + "_latents"] = args.output_dir / f"{split}_latents.json"
    if any(path.exists() for path in outputs.values()):
        raise SystemExit("refusing to overwrite an existing pilot manifest; choose a new output directory")
    for split in ("train", "dev"):
        outputs[split].write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in records if r["split"] == split))
        summary[split + "_manifest_sha256"] = hashlib.sha256(outputs[split].read_bytes()).hexdigest()
    for split, manifest in existing_latent_manifests(json.loads(payload), summary).items():
        outputs[split + "_latents"].write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        summary[split + "_legacy_latent_samples"] = len(manifest["samples"])
        summary[split + "_legacy_latent_manifest_sha256"] = hashlib.sha256(outputs[split + "_latents"].read_bytes()).hexdigest()
    outputs["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output_dir": str(args.output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
