#!/usr/bin/env python3
"""Bounded source-only audit. Never assigns dataset roles or evaluates a model."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(path):
    value = path.stat()
    return {"bytes": value.st_size, "mtime_ns": value.st_mtime_ns}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def alarm_handler(signum, frame):
    raise TimeoutError("single source sparse decoding exceeded 12 seconds")


def signatures(path):
    import av
    import cv2
    import numpy as np
    cv2.setNumThreads(1)
    result = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 1
        duration = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
        start_time = float((stream.start_time or 0) * stream.time_base)
        metadata = {"duration_s": duration, "width": stream.width, "height": stream.height,
                    "codec": stream.codec_context.name, "time_base": str(stream.time_base),
                    "start_time_s": start_time, "fractions": [0.1, 0.5, 0.9]}
        for fraction in metadata["fractions"]:
            requested = start_time + duration * fraction
            container.seek(int(requested / stream.time_base), stream=stream, backward=True, any_frame=False)
            selected = None
            for frame in container.decode(stream):
                if frame.pts is not None and float(frame.pts * stream.time_base) >= requested:
                    selected = frame
                    break
            if selected is None:
                raise ValueError("no decoded frame at or after requested timestamp")
            rgb = selected.to_ndarray(format="rgb24")
            small = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32)
            coefficients = cv2.dct(gray)[:8, :8].ravel()[1:]
            median = np.median(coefficients)
            phash = sum(int(bit) << index for index, bit in enumerate(coefficients > median))
            dh = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
            dhash = sum(int(bit) << index for index, bit in enumerate((dh[:, 1:] > dh[:, :-1]).ravel()))
            result.append({"fraction": fraction, "requested_seconds": requested,
                           "decoded_seconds": float(selected.pts * stream.time_base),
                           "thumbnail_rgb32_sha256": hashlib.sha256(small.tobytes()).hexdigest(),
                           "phash63": f"{phash:016x}", "dhash64": f"{dhash:016x}",
                           "rgb_mean_8bit": small.mean(axis=(0, 1)).tolist(),
                           "gray_std_8bit": float(gray.std())})
    return metadata, result


def comparison(a, b):
    distances = []
    exact = []
    informative_hits = []
    for ai, af in enumerate(a["frames"]):
        for bi, bf in enumerate(b["frames"]):
            ph = (int(af["phash63"], 16) ^ int(bf["phash63"], 16)).bit_count()
            dh = (int(af["dhash64"], 16) ^ int(bf["dhash64"], 16)).bit_count()
            mean_delta = sum(abs(x - y) for x, y in zip(af["rgb_mean_8bit"], bf["rgb_mean_8bit"])) / 3
            row = {"a_frame": ai, "b_frame": bi, "phash_hamming": ph,
                   "dhash_hamming": dh, "rgb_mean_absolute_difference_8bit": mean_delta}
            distances.append(row)
            if af["thumbnail_rgb32_sha256"] == bf["thumbnail_rgb32_sha256"]:
                exact.append(row)
            if min(af["gray_std_8bit"], bf["gray_std_8bit"]) >= 8 and ph <= 6 and dh <= 8 and mean_delta <= 12:
                informative_hits.append(row)
    distances.sort(key=lambda r: (r["phash_hamming"], r["dhash_hamming"], r["rgb_mean_absolute_difference_8bit"], r["a_frame"], r["b_frame"]))
    return {"other_source_id": b["source_id"], "closest_frame_pair": distances[0],
            "informative_near_frame_hits": informative_hits, "identical_thumbnail_hits": exact,
            "at_least_two_distinct_times_match": len({r['a_frame'] for r in informative_hits}) >= 2 and len({r['b_frame'] for r in informative_hits}) >= 2}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs = json.loads(args.inputs.read_text())
    started = time.monotonic()
    limits = inputs["limits"]
    records = []
    output = {"schema": "openh3.candidate_source_identity_audit.v1",
              "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "status": "running", "inputs_sha256": sha256(args.inputs),
              "script_sha256": sha256(Path(__file__)), "parent_evidence": inputs["parent_evidence"],
              "limits": limits, "records": records, "bytes_newly_hashed": 0,
              "roles_assigned": 0, "npu_calls": 0, "model_calls": 0,
              "source_root": inputs["source_root"]}
    root = Path(inputs["source_root"])
    for category, identifiers in (("candidate", inputs["candidate_ids"]), ("historical_stage7", inputs["comparison_ids"])):
        for sid in identifiers:
            path = root / (sid + ".mp4")
            row = {"source_id": sid, "category": category, "path": str(path),
                   "source_group_hint": "audioset-youtube:" + sid,
                   "semantic_group_status": "literal_video_id_only_reuploads_and_other_excerpts_unknown"}
            records.append(row)
            if not path.is_file():
                row["status"] = "missing"
                continue
            before = stat(path)
            row.update(before)
            cache = inputs["reusable_candidate_cache"].get(sid)
            if cache and all(before[k] == cache[k] for k in ("bytes", "mtime_ns")):
                row["source_sha256"] = cache["source_sha256"]
                row["hash_provenance"] = "reused_prior_full_file_sha256_after_size_mtime_ns_match"
            elif output["bytes_newly_hashed"] + before["bytes"] <= limits["max_new_hash_bytes"] and time.monotonic() - started < limits["max_elapsed_seconds"]:
                row["source_sha256"] = sha256(path)
                row["hash_provenance"] = "new_full_file_read"
                output["bytes_newly_hashed"] += before["bytes"]
            else:
                row["status"] = "not_hashed_budget"
                continue
            row["unchanged_during_hash"] = before == stat(path)
            row["status"] = "hashed" if row["unchanged_during_hash"] else "changed_during_hash"
            if sid in inputs["current_materialized_sources"]:
                expected = inputs["current_materialized_sources"][sid]
                row["current_materialized_binding"] = expected
                row["matches_current_materialized_sha256"] = row["source_sha256"] == expected["source_sha256"]
            if category == "candidate":
                row["source_group_evidence"] = inputs["candidate_source_group_evidence"][sid]
    atomic_json(args.out, output)
    print(json.dumps({"phase": "hash_complete", "records": len(records), "bytes_newly_hashed": output["bytes_newly_hashed"], "seconds": time.monotonic() - started}), flush=True)
    groups = {}
    for row in records:
        if row.get("status") == "hashed":
            groups.setdefault(row["source_sha256"], []).append({"source_id": row["source_id"], "category": row["category"]})
    output["exact_file_duplicate_groups"] = [{"sha256": key, "members": val} for key, val in groups.items() if len(val) > 1]
    output["current_40_sha_mismatches"] = [r["source_id"] for r in records if r.get("matches_current_materialized_sha256") is False]
    import av
    import cv2
    import numpy as np
    output["software"] = {"python": sys.version, "numpy": np.__version__, "opencv": cv2.__version__, "pyav": av.__version__, "libav_versions": {k: list(v) for k, v in av.library_versions.items()}}
    signal.signal(signal.SIGALRM, alarm_handler)
    ordered = sorted(records, key=lambda r: (0 if r["category"] == "candidate" else 1 if r["source_id"] in inputs["current_materialized_sources"] else 2, r["source_id"]))
    for index, row in enumerate(ordered):
        if row.get("status") != "hashed":
            row["perceptual_status"] = "unavailable_source_identity"
            continue
        if time.monotonic() - started > limits["max_elapsed_seconds"] - 15:
            row["perceptual_status"] = "not_run_time_budget"
            continue
        rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss_bytes > limits["max_resident_memory_bytes"]:
            row["perceptual_status"] = "not_run_rss_budget"
            continue
        try:
            before = stat(Path(row["path"]))
            if before != {k: row[k] for k in ("bytes", "mtime_ns")}:
                raise ValueError("source changed since identity check")
            signal.alarm(12)
            row["video_metadata"], row["frames"] = signatures(Path(row["path"]))
            row["unchanged_during_decode"] = before == stat(Path(row["path"]))
            if not row["unchanged_during_decode"]:
                raise ValueError("source changed during decoding")
            row["perceptual_status"] = "three_frames_decoded"
        except Exception as error:
            row["perceptual_status"] = "decode_failed"
            row["decode_error"] = type(error).__name__ + ": " + str(error)
        finally:
            signal.alarm(0)
        if index % 20 == 0:
            atomic_json(args.out, output)
            print(json.dumps({"phase": "sparse_decode", "processed": index + 1, "total": len(ordered), "seconds": time.monotonic() - started}), flush=True)
    candidates = [r for r in records if r["category"] == "candidate" and r.get("perceptual_status") == "three_frames_decoded"]
    history = [r for r in records if r["category"] == "historical_stage7" and r.get("perceptual_status") == "three_frames_decoded"]
    output["perceptual_protocol"] = {"fractions_of_container_video_duration": [0.1, 0.5, 0.9], "selection": "first decoded presentation timestamp at or after requested timestamp; backward keyframe seek then normal decode", "normalization": "RGB24 resize32x32 INTER_AREA; RGB-to-gray OpenCV", "phash": "63 AC bits from top-left8x8 DCT of gray32x32; strict above median; bit index is flattened AC position", "dhash": "64 comparisons right greater than left on gray9x8 INTER_AREA", "flag_heuristic": "gray std both>=8; phash distance<=6 AND dhash<=8 AND mean RGB absolute difference<=12 on 8bit scale; these are predeclared triage heuristics, not validated duplicate-detector thresholds", "scope": "candidate-candidate and candidate-versus-enumerated-Stage7 sources only; prior sparse32 thumbnail hashes not treated as compatible with this decoder/positions", "not_a_proof": "Sparse frames can miss temporal offsets, crops, reuploads, overlays and unenumerated histories. Similar scenes can false-positive. No clean-quality verdict or calibration/test eligibility follows."}
    output["perceptual_candidate_nearest_historical"] = []
    output["perceptual_flagged_pairs"] = []
    for a in candidates:
        comparisons = [comparison(a, b) for b in history]
        comparisons.sort(key=lambda c: (c["closest_frame_pair"]["phash_hamming"], c["closest_frame_pair"]["dhash_hamming"], c["other_source_id"]))
        output["perceptual_candidate_nearest_historical"].append({"source_id": a["source_id"], "top3": comparisons[:3]})
        for entry in comparisons:
            if entry["informative_near_frame_hits"] or entry["identical_thumbnail_hits"]:
                output["perceptual_flagged_pairs"].append({"source_id": a["source_id"], "comparison_scope": "candidate_historical", **entry})
    for ai, a in enumerate(candidates):
        for b in candidates[ai + 1:]:
            entry = comparison(a, b)
            if entry["informative_near_frame_hits"] or entry["identical_thumbnail_hits"]:
                output["perceptual_flagged_pairs"].append({"source_id": a["source_id"], "comparison_scope": "candidate_candidate", **entry})
    output["summary"] = {"candidate_total": len(inputs["candidate_ids"]), "historical_total": len(inputs["comparison_ids"]), "hashed": sum(r.get("status") == "hashed" for r in records), "hash_reused": sum(r.get("hash_provenance", "").startswith("reused") for r in records), "candidate_sparse_decoded": len(candidates), "historical_sparse_decoded": len(history), "historical_current40_sparse_decoded": sum(r["source_id"] in inputs["current_materialized_sources"] for r in history), "candidate_historical_pairs_screened": len(candidates) * len(history), "candidate_candidate_pairs_screened": len(candidates) * (len(candidates) - 1) // 2, "exact_duplicate_groups": len(output["exact_file_duplicate_groups"]), "perceptual_flagged_pairs": len(output["perceptual_flagged_pairs"]), "candidate_content_quality_approved": 0, "calibration_test_assigned": 0}
    output["required_before_qualification"] = ["Resolve any exact/perceptual duplicate flags by source-only manual review; source-group all confirmed reuploads/excerpts together.", "Verify complete deterministic 96-frame clean target window readability, timestamp and decoded pixel hashes; three sparse frames do not establish this.", "Source-only visual quality review under frozen sharpness, existing compression, resampling, title/black/static and content criteria; historical bandpass gate is insufficient.", "Resolve source/uploader/reupload group relationships beyond literal YouTube ID; historical unlogged aliases and upstream pretraining remain unknown.", "Extend semantic/source and duplicate crosscheck to historical generated, unified and benchmark assets outside the enumerated AudioSet/Stage7 pool before stronger nonoverlap claims.", "Approve eligibility before any calibration/test split; current pilot dev remains tuning data."]
    output["elapsed_seconds"] = time.monotonic() - started
    output["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    output["status"] = "bounded_source_audit_complete_no_roles_assigned"
    output["completed_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    atomic_json(args.out, output)
    print(json.dumps(output["summary"]), flush=True)


if __name__ == "__main__":
    main()
