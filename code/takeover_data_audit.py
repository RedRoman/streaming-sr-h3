"""Read-only, CPU-only inventory and source overlap audit for the OpenH3 takeover.

Only JSON metadata, directory entries and safetensors headers are read. No torch,
video decoding, tensor loading, recursive media scan, or accelerator initialization.
All historical evaluation sets are development evidence, never a new frozen test.
Run on the asset host; stdout is JSON unless --output is explicitly supplied.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import struct


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_id(stem):
    return re.sub(r"_?_w\d+$", "", stem)


def read_jsonl(path):
    rows, errors = [], []
    for line, raw in enumerate(Path(path).read_text().splitlines(), 1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
            if not isinstance(row, dict) or not row.get("id"):
                raise ValueError("expected an object with an id")
            rows.append(row)
        except (ValueError, TypeError) as exc:
            errors.append({"line": line, "error": str(exc)})
    return rows, errors


def safetensors_header(path):
    with Path(path).open("rb") as stream:
        n = struct.unpack("<Q", stream.read(8))[0]
        if n > 16 * 1024 * 1024:
            raise ValueError("unreasonably large safetensors header")
        return json.loads(stream.read(n))


def audit(project_root, unified_manifests):
    root, unified = Path(project_root), Path(unified_manifests)
    sets, inventory, issues = {}, {}, []
    unified_rows = {}
    for split in ("train", "val"):
        path = unified / f"eligible.{split}.jsonl"
        if not path.exists():
            issues.append({"missing": str(path)})
            continue
        rows, errors = read_jsonl(path)
        name = "unified_" + split
        unified_rows[split] = rows
        sets[name] = {str(row["id"]) for row in rows}
        inventory[name] = {
            "manifest": str(path), "manifest_sha256": digest(path),
            "rows": len(rows), "unique_source_ids": len(sets[name]),
            "invalid_lines": errors,
            "source_counts": dict(collections.Counter(r.get("source", "unknown") for r in rows)),
            "available_raw_videos": sum(Path(r["video_path"]).is_file() for r in rows),
        }
        if errors:
            issues.append({"invalid_manifest": str(path), "lines": [e["line"] for e in errors]})
    split_dirs = {
        "pilot_latents": "*.safetensors", "stage7_train": "*.npz",
        "stage7_latents": "*.safetensors", "stage7_teacher": "*.safetensors",
        "range_pairs": "*.npz", "stage8_audio": "*.npz",
    }
    pilot_sources = {}
    for kind, pattern in split_dirs.items():
        for split in ("train", "val"):
            directory = root / "data" / kind / split
            files = sorted(directory.glob(pattern))
            name = kind + "_" + split
            ids, metadata_sources = set(), collections.Counter()
            for path in files:
                sidecar = path.with_suffix(".json")
                if kind == "pilot_latents" and sidecar.is_file():
                    row = json.loads(sidecar.read_text())
                    ids.add(str(row["source_id"]))
                    metadata_sources[row.get("source", "unknown")] += 1
                else:
                    ids.add(source_id(path.stem))
            sets[name] = ids
            entry = {"directory": str(directory), "exists": directory.is_dir(),
                     "files": len(files), "unique_source_ids": len(ids),
                     "source_ids": sorted(ids), "identity_evidence": "sidecars" if kind == "pilot_latents" else "window filenames"}
            if metadata_sources:
                entry["source_counts"] = dict(metadata_sources)
                pilot_sources[split] = dict(metadata_sources)
            if files and pattern.endswith("safetensors"):
                header = safetensors_header(files[0])
                entry["sample_tensor_header"] = {
                    "path": str(files[0]), "tensors": {k: {"shape": v["shape"], "dtype": v["dtype"]}
                    for k, v in header.items() if k != "__metadata__"},
                    "metadata": header.get("__metadata__", {}),
                }
            inventory[name] = entry
            if kind == "stage7_latents":
                entry["samples"] = [{"source_id": source_id(p.stem), "latent_path": str(p)} for p in files]
    evaluations = []
    eval_paths = sorted((root / "data/real_eval").glob("*/manifest.json"))
    stage2 = root / "data/stage2_eval/manifest.json"
    if stage2.exists():
        eval_paths.append(stage2)
    for path in eval_paths:
        manifest = json.loads(path.read_text())
        name = "eval_" + path.parent.name
        ids, video_paths = set(), {}
        for item in manifest.get("clips", []):
            ident = str(item if isinstance(item, str) else item["id"])
            ids.add(ident)
            meta_path = path.parent / ident / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                video = meta.get("video") or meta.get("video_path")
                if video:
                    video_paths[ident] = str(Path(video).resolve())
        sets[name] = ids
        evaluations.append({"name": name, "manifest": str(path), "manifest_sha256": digest(path),
                            "declared_n": manifest.get("n"), "clips": len(ids), "source_ids": sorted(ids),
                            "source_video_paths": video_paths, "role": "historical_dev_excluded_from_pilot",
                            "source": manifest.get("source"), "has_hr": manifest.get("has_hr", True)})
        if manifest.get("n", len(ids)) != len(ids):
            issues.append({"manifest_count_mismatch": str(path), "declared": manifest["n"], "actual": len(ids)})
    pairs = []
    names = sorted(sets)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            # Keep training/evaluation and within-family train/val checks, not duplicate eval variants.
            if left.startswith("eval_") and right.startswith("eval_"):
                continue
            if not (left.startswith("eval_") or right.startswith("eval_") or
                    left.rsplit("_", 1)[0] == right.rsplit("_", 1)[0]):
                continue
            common = sorted(sets[left] & sets[right])
            pairs.append({"left": left, "right": right, "count": len(common), "source_ids": common})
    candidate_path = root / "data/real_av/candidates_big.json"
    candidates = []
    excluded_ids = set().union(*(sets[e["name"]] for e in evaluations)) if evaluations else set()
    excluded_paths = {v for e in evaluations for v in e["source_video_paths"].values()}
    if candidate_path.exists():
        for row in json.loads(candidate_path.read_text()).get("selected", []):
            ident = Path(row["file"]).stem
            path = (root / "data/real_av/_cand" / row["file"]).resolve()
            train = ident in sets.get("stage7_train_train", set())
            val = ident in sets.get("stage7_train_val", set())
            excluded = ident in excluded_ids or str(path) in excluded_paths
            candidates.append({"source_id": ident, "source_video": str(path), "available": path.is_file(),
                               "source_kind": "captured_video", "clean_reference": "original_recording_proxy_not_pristine_truth",
                               "historical_split": "conflict" if train and val else "train" if train else "val" if val else "unassigned",
                               "historical_eval": excluded, "duration_s": row.get("duration"),
                               "width_height": row.get("wh"), "sharp_ok": row.get("sharp_ok"),
                               "audio_rms": row.get("rms"), "source_group": "audioset:" + ident})
    same_source = []
    by_id = {c["source_id"]: c for c in candidates}
    for ev in evaluations:
        for ident, path in ev["source_video_paths"].items():
            c = by_id.get(ident)
            if c and c["historical_split"] in ("train", "val"):
                same_source.append({"source_id": ident, "historical_train_split": c["historical_split"],
                                    "eval_manifest": ev["manifest"], "eval_source_video": path,
                                    "training_source_video": c["source_video"],
                                    "same_resolved_source_path": path == c["source_video"]})
    unified_overlap = {}
    if len(unified_rows) == 2:
        for field in ("id", "original_id", "video_sha256", "source_group"):
            left = {str(r[field]) for r in unified_rows["train"] if r.get(field)}
            right = {str(r[field]) for r in unified_rows["val"] if r.get(field)}
            unified_overlap[field] = sorted(left & right)
    return {
        "schema": "openh3.takeover_data_audit.v1", "project_root": str(root),
        "read_only": True, "media_or_tensors_decoded": False,
        "inventory": inventory, "evaluations": evaluations, "overlaps": pairs,
        "same_source_path_evidence": same_source, "unified_split_overlap": unified_overlap,
        "real_candidate_manifest": str(candidate_path),
        "real_candidate_manifest_sha256": digest(candidate_path) if candidate_path.exists() else None,
        "real_candidates": candidates,
        "eligible_real_sources_after_all_historical_eval_exclusions": dict(collections.Counter(
            c["historical_split"] for c in candidates if c["available"] and not c["historical_eval"])),
        "frozen_test": {"available": False, "reason": "No unseen test split is established by this audit; all existing eval sets are historical development evidence."},
        "limitations": ["IDs and resolved source paths establish provenance; no perceptual or content deduplication is performed.",
                        "Existing target latents require a codec compatibility check before reuse with OpenH3-Regenerate.",
                        "Source overlap of Stage7 does not establish which checkpoint trained on Stage7; inspect checkpoint provenance separately."],
        "issues": issues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path("/cache/zihaohe/streaming_sr0"))
    parser.add_argument("--unified-manifests", type=Path, default=Path("/temp/h3_unified_dataset_20260918/manifests"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.project_root, args.unified_manifests)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
