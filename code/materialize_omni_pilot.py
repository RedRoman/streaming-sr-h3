"""CPU-only conversion of audited raw pilot plans into clean/observed video NPZs.

Each source/window is decoded once and has exactly one immutable clean target.
Three native-resolution tasks and x4 SR reference that same target. No VAE,
OpenH3, torch, training, or accelerator is loaded. Completed groups are resumed
only after plan, source and output hashes are checked. A final JSONL is emitted
only for verified materialized samples; the original plan is never changed.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

VERSION = "omni-pilot-materializer-v1"
TASKS = {"sr", "deblur", "denoise", "compression"}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)


def parse_plans(paths):
    grouped, sample_ids, source_splits = {}, set(), {}
    for path in paths:
        for line_number, text in enumerate(Path(path).read_text().splitlines(), 1):
            if not text.strip():
                continue
            row = json.loads(text)
            if row.get("schema") != "openh3.omni_pilot_sample.v1":
                raise ValueError(f"unsupported schema: {path}:{line_number}")
            if row.get("status") != "planned_requires_decode_degrade_and_encode":
                raise ValueError("expected an unmaterialized pilot plan")
            if row["task"] not in TASKS or row["split"] not in ("train", "dev"):
                raise ValueError("unsupported task or split")
            if row["sample_id"] in sample_ids:
                raise ValueError("duplicate sample ID: " + row["sample_id"])
            sample_ids.add(row["sample_id"])
            previous = source_splits.setdefault(row["source_id"], row["split"])
            if previous != row["split"]:
                raise ValueError("source present in both train and dev")
            target = row["target"]
            if target["pixel_target"] != "clean_original_recording" or target["spatial_transform"] != "center_crop_no_resize":
                raise ValueError("target must be the unchanged source crop, not a degraded or resized target")
            key = hashlib.sha256(json.dumps(target, sort_keys=True).encode()).hexdigest()
            if key != row["target_key"]:
                raise ValueError("target key does not match target specification")
            grouped.setdefault((row["split"], key), []).append(row)
    groups = []
    for (split, key), rows in sorted(grouped.items()):
        if len(rows) != 4 or {row["task"] for row in rows} != TASKS:
            raise ValueError("each source/window must have all four tasks exactly once")
        first = rows[0]
        if any(row["source_id"] != first["source_id"] or row["target"] != first["target"] for row in rows):
            raise ValueError("tasks do not share an identical clean target")
        groups.append({"split": split, "target_key": key, "rows": sorted(rows, key=lambda row: row["task"])})
    if not groups:
        raise ValueError("empty plans")
    return groups


def run_ffmpeg(command, payload=None):
    completed = subprocess.run(command, input=payload, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=180)
    if completed.returncode:
        raise RuntimeError("ffmpeg failed: " + completed.stderr.decode(errors="replace")[-4000:])
    return completed.stdout


def pixels_from_raw(raw, shape):
    import numpy as np
    expected = math.prod(shape)
    if len(raw) != expected:
        raise ValueError(f"decoded length mismatch: expected {expected} RGB bytes, got {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()


def decode_target(target, ffmpeg, threads):
    shape = (int(target["frames"]), int(target["height"]), int(target["width"]), 3)
    filters = f"fps=fps={target['fps']}:round=near,crop={target['width']}:{target['height']}"
    command = [ffmpeg, "-nostdin", "-v", "error", "-threads", str(threads),
               "-filter_threads", "1", "-ss", str(target["start_s"]), "-i", target["source_video"],
               "-vf", filters, "-frames:v", str(target["frames"]), "-an", "-sn",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    frames = pixels_from_raw(run_ffmpeg(command), shape)
    return frames, {"command": command, "accurate_seek": True,
                    "frame_selection": "ffmpeg fps filter round=near after accurate input seek",
                    "frames_verified": shape[0]}


def degrade_pixels(clean, row, ffmpeg, threads, directory):
    import cv2
    import numpy as np
    from PIL import Image
    spec, task = row["degradation"], row["task"]
    actual = {"task": task, "planned": spec, "materializer_version": VERSION}
    if task == "sr":
        scale = int(spec["scale"])
        if scale != 4 or clean.shape[1] % scale or clean.shape[2] % scale:
            raise ValueError("expected exactly x4 SR with divisible geometry")
        size = (clean.shape[2] // scale, clean.shape[1] // scale)
        observed = np.stack([np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BICUBIC)) for frame in clean])
        actual.update({"implementation": "Pillow.Image.resize", "resample": "BICUBIC",
                       "antialias": "Pillow downsampling filter support", "scale": scale,
                       "quantization": "Pillow uint8 output", "size_wh": list(size)})
    elif task == "deblur":
        sigma = float(spec["sigma_pixels"])
        kernel = 2 * math.ceil(3 * sigma) + 1
        observed = np.stack([cv2.GaussianBlur(frame, (kernel, kernel), sigmaX=sigma, sigmaY=sigma,
                                            borderType=cv2.BORDER_REFLECT_101) for frame in clean])
        actual.update({"implementation": "cv2.GaussianBlur", "kernel": [kernel, kernel],
                       "sigma_pixels": sigma, "border": "BORDER_REFLECT_101", "quantization": "OpenCV uint8 output"})
    elif task == "denoise":
        rng = np.random.Generator(np.random.PCG64(int(spec["seed"])))
        noise = rng.standard_normal(clean.shape, dtype=np.float32) * float(spec["sigma_8bit"])
        observed = np.clip(np.rint(clean.astype(np.float32) + noise), 0, 255).astype(np.uint8)
        actual.update({"implementation": "numpy.Generator.PCG64.standard_normal", "noise_dtype": "float32",
                       "sigma_8bit": float(spec["sigma_8bit"]), "seed": int(spec["seed"]),
                       "quantization": "rint ties-to-even then clip [0,255]"})
    elif task == "compression":
        import av
        with tempfile.TemporaryDirectory(prefix="codec_", dir=directory) as temporary:
            encoded = str(Path(temporary) / "observation.mp4")
            with av.open(encoded, mode="w", format="mp4") as container:
                stream = container.add_stream("libx264", rate=int(row["target"]["fps"]))
                stream.width, stream.height, stream.pix_fmt = clean.shape[2], clean.shape[1], "yuv420p"
                stream.codec_context.thread_count = threads
                stream.codec_context.gop_size = len(clean)
                stream.codec_context.max_b_frames = 0
                stream.options = {"crf": str(int(spec["crf"])), "preset": "medium"}
                for frame in clean:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            with av.open(encoded) as container:
                container.streams.video[0].codec_context.thread_count = threads
                decoded = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
            if len(decoded) != len(clean):
                raise ValueError("H264 roundtrip frame count mismatch")
            observed = np.stack(decoded)
            actual.update({"implementation": "PyAV libx264 roundtrip", "crf": int(spec["crf"]),
                           "preset": "medium", "pixel_format": "yuv420p", "gop": len(clean), "bframes": 0,
                           "threads": threads, "PyAV": av.__version__, "libav_versions": av.library_versions})
    else:
        raise ValueError("unknown task")
    expected = (len(clean), clean.shape[1] // 4, clean.shape[2] // 4, 3) if task == "sr" else clean.shape
    if observed.shape != expected or observed.dtype != np.uint8:
        raise ValueError("observation geometry/dtype does not match task")
    return observed, actual


def write_pixels(path, frames, metadata):
    import numpy as np
    with tempfile.NamedTemporaryFile(dir=Path(path).parent, suffix=".npz", delete=False) as stream:
        np.savez_compressed(stream, frames=frames, metadata=canonical(metadata))
        temporary = Path(stream.name)
    os.replace(temporary, path)
    return {"path": str(path), "file_sha256": sha_file(path),
            "pixel_sha256": hashlib.sha256(frames.tobytes(order="C")).hexdigest(),
            "shape": list(frames.shape), "dtype": str(frames.dtype), "bytes_on_disk": Path(path).stat().st_size}


def verify_group(manifest, expected_plan_sha, expected_source_sha):
    import numpy as np
    if manifest["plan_sha256"] != expected_plan_sha or manifest["source_sha256"] != expected_source_sha:
        raise ValueError("existing group plan/source changed; use a new versioned output directory")
    if manifest["materializer_version"] != VERSION or manifest["status"] != "materialized":
        raise ValueError("existing group has an incompatible version or incomplete status")
    for record in [manifest["target"], *manifest["observations"].values()]:
        if sha_file(record["path"]) != record["file_sha256"]:
            raise ValueError("materialized file hash mismatch: " + record["path"])
        with np.load(record["path"], allow_pickle=False) as archive:
            pixels = archive["frames"]
            if list(pixels.shape) != record["shape"] or str(pixels.dtype) != record["dtype"]:
                raise ValueError("materialized shape/dtype mismatch")
            if hashlib.sha256(pixels.tobytes(order="C")).hexdigest() != record["pixel_sha256"]:
                raise ValueError("materialized pixel hash mismatch")
            meta = json.loads(str(archive["metadata"]))
            if meta["source_sha256"] != expected_source_sha:
                raise ValueError("materialized source metadata mismatch")
            if meta.get("role") == "observation" and meta["target_pixel_sha256"] != manifest["target"]["pixel_sha256"]:
                raise ValueError("observation points to a different clean target")


def materialize(group, output, libraries, ffmpeg, threads):
    first, rows = group["rows"][0], group["rows"]
    source = Path(first["target"]["source_video"])
    if not source.is_file():
        raise ValueError("source does not exist: " + str(source))
    source_sha = sha_file(source)
    plan_sha = hashlib.sha256(canonical(rows).encode()).hexdigest()
    directory = output / group["split"] / group["target_key"]
    directory.mkdir(parents=True, exist_ok=True)
    completed = directory / "complete.json"
    if completed.exists():
        result = json.loads(completed.read_text())
        if canonical(result["libraries"]) != canonical(libraries):
            raise ValueError("materializer libraries changed; use a new versioned output directory")
        verify_group(result, plan_sha, source_sha)
        return result, True
    clean, decoding = decode_target(first["target"], ffmpeg, threads)
    target_sha = hashlib.sha256(clean.tobytes(order="C")).hexdigest()
    provenance = {"source_id": first["source_id"], "source_video": str(source), "source_sha256": source_sha,
                  "source_bytes": source.stat().st_size, "target_key": group["target_key"],
                  "materializer_version": VERSION, "libraries": libraries, "target_specification": first["target"]}
    target = write_pixels(directory / "target.npz", clean, {**provenance, "role": "clean_target", "decoding": decoding})
    observations = {}
    for row in rows:
        observed, actual = degrade_pixels(clean, row, ffmpeg, threads, directory)
        observations[row["task"]] = write_pixels(directory / (row["task"] + ".npz"), observed,
            {**provenance, "role": "observation", "task": row["task"], "actual_recipe": actual,
             "target_pixel_sha256": target_sha, "target_npz": target["path"], "sample_id": row["sample_id"]})
        observations[row["task"]]["actual_recipe"] = actual
        del observed
    # Detect accidental in-place degradation of the shared clean target.
    if hashlib.sha256(clean.tobytes(order="C")).hexdigest() != target_sha:
        raise AssertionError("clean target was modified by a degradation")
    result = {**provenance, "plan_sha256": plan_sha, "split": group["split"], "status": "materialized",
              "target": target, "observations": observations, "decoding": decoding,
              "VAE_encoded": False, "training_started": False}
    verify_group(result, plan_sha, source_sha)
    atomic_json(completed, result)
    return result, False


def main():
    import cv2
    import numpy as np
    import PIL
    import av
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-groups", type=int, default=0, help="0 materializes every supplied group")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--minimum-free-gb", type=float, default=10)
    args = parser.parse_args()
    if args.threads < 1 or args.threads > 4 or args.limit_groups < 0:
        raise SystemExit("threads must be 1..4; limit-groups must be nonnegative")
    cv2.setNumThreads(args.threads)
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not found")
    libraries = {"numpy": np.__version__, "Pillow": PIL.__version__, "OpenCV": cv2.__version__,
                 "PyAV": av.__version__, "libav_versions": av.library_versions,
                 "ffmpeg": subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, check=True).stdout.splitlines(),
                 "ffmpeg_executable": ffmpeg, "ffmpeg_executable_sha256": sha_file(ffmpeg)}
    all_groups = parse_plans(args.plan)
    groups = all_groups[:args.limit_groups] if args.limit_groups else all_groups
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < args.minimum_free_gb * 1024 ** 3:
        raise SystemExit("insufficient free disk for conversion")
    started, results, reused = time.monotonic(), [], 0
    for index, group in enumerate(groups, 1):
        result, was_reused = materialize(group, output, libraries, ffmpeg, args.threads)
        reused += int(was_reused)
        results.append(result)
        print(json.dumps({"group": index, "total": len(groups), "source_id": result["source_id"],
                          "split": result["split"], "reused_verified": was_reused,
                          "target_pixel_sha256": result["target"]["pixel_sha256"],
                          "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
    manifest_paths = {}
    for split in ("train", "dev"):
        rows = []
        for result in results:
            if result["split"] != split:
                continue
            for task, observed in result["observations"].items():
                rows.append({"schema": "openh3.materialized_omni_pilot.v1", "source_id": result["source_id"],
                             "target_key": result["target_key"], "split": split, "task": task,
                             "target_npz": result["target"]["path"], "observation_npz": observed["path"],
                             "target_pixel_sha256": result["target"]["pixel_sha256"],
                             "observation_pixel_sha256": observed["pixel_sha256"], "source_sha256": result["source_sha256"],
                             "status": "materialized_not_vae_encoded", "latent_path": None})
        filename = ("smoke_" if args.limit_groups else "") + split + "_materialized.jsonl"
        path = output / filename
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        manifest_paths[split] = {"path": str(path), "sha256": sha_file(path), "samples": len(rows)}
    summary = {"schema": "openh3.materialization_summary.v1", "status": "smoke_complete" if args.limit_groups else "complete",
               "planned_groups": len(all_groups), "materialized_verified_groups": len(results), "reused_verified_groups": reused,
               "source_ids": sorted({r["source_id"] for r in results}), "VAE_encoded": False, "training_started": False,
               "input_plan_sha256": {str(p.resolve()): sha_file(p) for p in args.plan}, "libraries": libraries,
               "manifests": manifest_paths, "seconds": round(time.monotonic() - started, 2),
               "total_npz_bytes": sum(r["target"]["bytes_on_disk"] + sum(o["bytes_on_disk"] for o in r["observations"].values()) for r in results)}
    atomic_json(output / ("smoke_summary.json" if args.limit_groups else "summary.json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
