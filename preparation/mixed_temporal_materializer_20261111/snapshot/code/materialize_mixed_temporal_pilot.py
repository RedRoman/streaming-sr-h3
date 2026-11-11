"""One-source CPU engineering materializer; never imports a model or accelerator.

Default mode is inspection. The only executable cohort is the predeclared train
source 2BdYaGuVAKo, all six frozen profiles and exactly 96 frames. Output is new,
versioned and incompatible with the existing four-task loader by design.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import io
import json
import math
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time

VERSION = "openh3.mixed_temporal_materializer.v1"
SOURCE = "2BdYaGuVAKo"
PROFILE_IDS = (
    "native_static_blur_noise_h264", "native_static_noise_blur_h264",
    "native_smooth_blur_noise_h264", "native_abrupt_blur_noise_h264",
    "native_smooth_jpeg_quality", "sr_smooth_blur_noise_h264",
)
NATIVE = ["gaussian_blur_float_rgb", "gaussian_noise_float_rgb", "quantize_rint_clip_uint8", "h264_roundtrip"]
ORDERS = {
    PROFILE_IDS[0]: NATIVE,
    PROFILE_IDS[1]: [NATIVE[1], NATIVE[0], *NATIVE[2:]],
    PROFILE_IDS[2]: NATIVE, PROFILE_IDS[3]: NATIVE,
    PROFILE_IDS[4]: [*NATIVE[:3], "jpeg_per_frame_roundtrip"],
    PROFILE_IDS[5]: ["gaussian_blur_float_rgb_hr", "quantize_rint_clip_uint8", "bicubic_downsample_x4_uint8", "gaussian_noise_float_rgb_lr", "quantize_rint_clip_uint8", "h264_roundtrip_lr"],
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def value_sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pixel_sha(pixels):
    digest = hashlib.sha256()
    for frame in pixels:
        digest.update(memoryview(frame).cast("B"))
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_registry(train_path, dev_path, plan):
    """Reject IDs AND full source hashes crossing train/dev, including aliases."""
    roles, hashes, groups, rows = {}, {}, {}, {}
    bindings = {}
    for split, path, role_key in (("train", train_path, "existing_train"), ("dev", dev_path, "existing_dev")):
        path = Path(path)
        expected = plan["source_roles"][role_key]
        actual_sha = file_sha(path)
        if actual_sha != expected["manifest_sha256"]:
            raise ValueError(f"{split} manifest differs from frozen plan")
        bindings[split] = {"plan_manifest_reference": expected.get("manifest", path.name), "sha256": actual_sha}
        source_set = set()
        for text in path.read_text().splitlines():
            if not text.strip():
                continue
            row = json.loads(text)
            sid, sha = row["source_id"], row["source_sha256"]
            if row["split"] != split or not re.fullmatch(r"[0-9a-f]{64}", sha):
                raise ValueError("invalid source/split binding")
            group = row.get("source_group_id", "audioset-youtube:" + sid)
            for mapping, key in ((roles, sid), (hashes, sha), (groups, group)):
                if key in mapping and mapping[key] != split:
                    raise ValueError("source ID/hash/group overlaps across splits")
                mapping[key] = split
            if sid in rows and (rows[sid]["source_sha256"], rows[sid]["source_group_id"]) != (sha, group):
                raise ValueError("one source ID has conflicting source bytes/group")
            rows[sid] = {"source_id": sid, "source_sha256": sha, "source_group_id": group, "split": split}
            source_set.add(sid)
        if source_set != set(expected["source_ids"]):
            raise ValueError("manifest source membership differs from frozen role")
    if SOURCE not in rows or rows[SOURCE]["split"] != "train":
        raise ValueError("predeclared smoke source is not a train source")
    registry = {"schema": "openh3.mixed_temporal_smoke_registry.v1", "manifest_bindings": bindings,
                "sources": [rows[k] for k in sorted(rows)], "authorized_smoke_sources": [SOURCE],
                "role_assignment": "existing_train_engineering_only_no_new_split"}
    return registry, rows[SOURCE]


def validate_plan(plan, contract):
    target = plan["target_contract"]
    if (target["frames"], target["fps"], target["target_shape_thwc"], target["dtype"]) != (96, 24, [96, 480, 832, 3], "uint8"):
        raise ValueError("only the declared 96x480x832 RGB smoke is supported")
    if contract.get("schema") != "openh3.mixed_temporal_execution_contract.v1" or contract.get("status") != "root_confirmed_engineering_only":
        raise ValueError("explicit confirmed execution contract required")
    required = {"frame_index_zero_pad_width": 6, "window_key": "canonical_target_spec_sha256",
                "canonical_json": "sort_keys_compact_utf8_ensure_ascii_false_no_nan",
                "rgb_conversion_before_crop": True, "center_crop_rounding": "integer_floor",
                "allow_resize": False, "natural_cut_status": "unknown_engineering_only",
                "replay_count": 2, "allowed_source_ids": [SOURCE]}
    if any(contract.get(k) != v for k, v in required.items()):
        raise ValueError("execution contract differs from supported frozen conventions")
    if plan["resource_limits"]["planned_memory_budget_mib"] != 768 or plan["optional_codec_segment_control"]["enabled"]:
        raise ValueError("unsupported resource or segmented-codec plan")
    profiles = plan["profiles"]
    if tuple(p["profile_id"] for p in profiles) != PROFILE_IDS:
        raise ValueError("six profile identities/order must match the declared smoke")
    for profile in profiles:
        pid = profile["profile_id"]
        if profile["scale"] != (4 if pid == PROFILE_IDS[5] else 1) or profile["task"] != ("sr" if pid == PROFILE_IDS[5] else "mixed_native"):
            raise ValueError("profile task/scale must preserve fixed native or x4 geometry")
        if profile["ordered_operators"] != ORDERS[pid]:
            raise ValueError("operator order mismatch: " + pid)
        params = profile["frame_parameters"]
        if len(params) != 96 or [p["frame"] for p in params] != list(range(96)):
            raise ValueError("every frame requires exactly one ordered parameter row")
        for p in params:
            if not all(math.isfinite(float(p[k])) and p[k] >= 0 for k in ("blur_sigma_hr_px", "noise_sigma_8bit")):
                raise ValueError("invalid blur/noise schedule")
            if "jpeg_quality" in p and (type(p["jpeg_quality"]) is not int or not 1 <= p["jpeg_quality"] <= 100):
                raise ValueError("JPEG quality must be an explicit integer")
        if "h264" in pid and len({p["h264_crf"] for p in params}) != 1:
            raise ValueError("primary H264 CRF must remain constant across the clip")
        if profile.get("codec_reset_frames", []) or profile.get("scene_cut_frames", []):
            raise ValueError("scene-cut/codec-reset controls are outside this one-source smoke")
    if profiles[0]["noise_stream"] != profiles[1]["noise_stream"] or profiles[0]["frame_parameters"] != profiles[1]["frame_parameters"]:
        raise ValueError("static order controls must share noise and parameters")
    if profiles[2]["noise_stream"] != profiles[3]["noise_stream"]:
        raise ValueError("smooth/abrupt native controls must share unit noise")
    if value_sha(plan["operator_contract"]) != contract.get("operator_contract_sha256") or value_sha(profiles) != contract.get("profile_recipes_sha256"):
        raise ValueError("operator definitions or full profile recipes differ from execution contract")


def noise_seed(source_sha, window_key, noise_stream, frame_index, width=6):
    if width != 6 or not 0 <= frame_index < 96:
        raise ValueError("unsupported frame index or seed padding")
    text = f"omni-mixed-v1|{source_sha}|{window_key}|{noise_stream}|{frame_index:06d}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def unit_noise(shape, seed):
    import numpy as np
    return np.random.Generator(np.random.PCG64(seed)).standard_normal(shape, dtype=np.float32)


def quantize(pixels):
    import numpy as np
    rounded = np.rint(pixels)
    np.clip(rounded, 0, 255, out=rounded)
    return rounded.astype(np.uint8)


def blur_float(pixels, sigma):
    if sigma == 0:
        return pixels
    import cv2
    kernel = 2 * math.ceil(3 * sigma) + 1
    return cv2.GaussianBlur(pixels, (kernel, kernel), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)


def precodec_frame(clean, profile, index, source_sha, window_key):
    import numpy as np
    from PIL import Image
    if clean.dtype != np.uint8 or clean.ndim != 3 or clean.shape[-1] != 3:
        raise ValueError("clean input must be uint8 HWC RGB")
    params = profile["frame_parameters"][index]
    value = clean.astype(np.float32)
    noise_info = None
    for operator in profile["ordered_operators"]:
        if operator.startswith("gaussian_blur"):
            value = blur_float(value, float(params["blur_sigma_hr_px"]))
        elif operator.startswith("gaussian_noise"):
            if value.dtype != np.float32:
                value = value.astype(np.float32)
            seed = noise_seed(source_sha, window_key, profile["noise_stream"], index)
            noise = unit_noise(value.shape, seed)
            noise_info = {"frame": index, "seed_uint64": seed, "unit_noise_shape": list(noise.shape),
                          "unit_noise_pixel_sha256": hashlib.sha256(memoryview(noise).cast("B")).hexdigest()}
            value = value + noise * np.float32(params["noise_sigma_8bit"])
        elif operator == "quantize_rint_clip_uint8":
            value = quantize(value)
        elif operator == "bicubic_downsample_x4_uint8":
            if value.dtype != np.uint8 or value.shape[0] % 4 or value.shape[1] % 4:
                raise ValueError("SR boundary requires divisible uint8 RGB")
            value = np.asarray(Image.fromarray(value).resize((value.shape[1] // 4, value.shape[0] // 4), Image.Resampling.BICUBIC))
        elif operator.startswith("h264") or operator == "jpeg_per_frame_roundtrip":
            break
        else:
            raise ValueError("unknown ordered operator")
    if value.dtype != np.uint8 or noise_info is None:
        raise ValueError("incomplete quantization/noise receipt")
    return value, noise_info


def checked_shape(path, shape):
    expected = math.prod(shape)
    if Path(path).stat().st_size != expected:
        raise ValueError(f"decoded RGB length mismatch, expected exactly {expected} bytes")


def peak_rss():
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def peak_child_rss():
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def process_rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ProcessLookupError):
        return 0
    return 0


class Budget:
    def __init__(self, seconds, mib=768):
        self.started = time.monotonic()
        self.seconds, self.bytes = seconds, mib * 1024**2
        self.max_combined_rss = 0
        self.conservative_peak_sum = 0

    def check(self, child=None):
        combined = (process_rss(os.getpid()) or peak_rss()) + (process_rss(child) if child else 0)
        self.max_combined_rss = max(self.max_combined_rss, combined, peak_rss())
        self.conservative_peak_sum = max(self.conservative_peak_sum, peak_rss() + peak_child_rss())
        if max(self.max_combined_rss, self.conservative_peak_sum) > self.bytes:
            raise MemoryError("single-source CPU memory budget exceeded")
        if time.monotonic() - self.started > self.seconds:
            raise TimeoutError("single-source CPU wall budget exceeded")


def run_owned(command, log, budget):
    with Path(log).open("wb") as handle:
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=handle, start_new_session=True)
        try:
            while process.poll() is None:
                budget.check(process.pid)
                time.sleep(0.1)
            if process.returncode:
                raise RuntimeError("FFmpeg failed; see " + str(log))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
    budget.check()


def source_probe(source, ffprobe):
    data = json.loads(subprocess.check_output([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", str(source)], timeout=30))
    stream = data["streams"][0]
    if int(stream["width"]) < 832 or int(stream["height"]) < 480:
        raise ValueError("source cannot support center crop without resizing")
    duration = Decimal(data["format"]["duration"])
    if duration < Decimal(4):
        raise ValueError("source metadata is shorter than 4s; actual decode is also checked")
    return data


def target_specification(binding, probe, contract):
    duration = Decimal(probe["format"]["duration"])
    start = max(Decimal(0), (duration - Decimal(4)) / 2)
    return {"schema": "openh3.mixed_temporal_target.v1", **binding,
            "frames": 96, "fps": 24, "height": 480, "width": 832, "channels": 3,
            "dtype": "uint8", "color": "RGB24", "start_seconds": format(start, "f"),
            "duration_definition": "ffprobe format.duration",
            "source_format_start_time": probe["format"].get("start_time", "0"),
            "spatial_transform": "RGB24_then_integer_floor_center_crop_no_resize",
            "frame_selection": "accurate_input_seek_then_fps24_round_near_on_window_relative_grid",
            "execution_contract_sha256": value_sha(contract)}


SHOWINFO = re.compile(r"\[showinfo@(source|target)[^\]]*\].*?\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([\d.eE+\-]+)")
TIMEBASE = re.compile(r"\[showinfo@(source|target)[^\]]*\].*?config in time_base:\s*(\d+/\d+)")


def parse_timestamps(log_text, frames=96):
    rows = {"source": [], "target": []}
    timebases = {}
    for line in log_text.splitlines():
        configuration = TIMEBASE.search(line)
        if configuration:
            timebases[configuration[1]] = configuration[2]
        match = SHOWINFO.search(line)
        if match:
            tag, n, pts, seconds = match.groups()
            rows[tag].append({"index": int(n), "pts": int(pts), "pts_time_seconds": float(seconds)})
    if set(timebases) != {"source", "target"}:
        raise ValueError("source and output filter time bases must both be recorded")
    for tag in rows:
        for row in rows[tag]:
            row["time_base"] = timebases[tag]
    if len(rows["target"]) != frames or [r["index"] for r in rows["target"]] != list(range(frames)):
        raise ValueError(f"fps output timestamps do not contain exactly {frames} frames: got {len(rows['target'])}")
    if len(rows["source"]) < frames:
        raise ValueError("fewer than 96 actual source frames reached the filter; metadata is insufficient")
    if [r["index"] for r in rows["source"]] != list(range(len(rows["source"]))):
        raise ValueError("source decoder timestamp log has missing/duplicate indices")
    if any(b["pts_time_seconds"] <= a["pts_time_seconds"] for a, b in zip(rows["source"], rows["source"][1:])):
        raise ValueError("source decoded presentation timestamps are not strictly increasing")
    if any((b["pts"] - a["pts"]) * Fraction(timebases["target"]) != Fraction(1, 24) for a, b in zip(rows["target"], rows["target"][1:])):
        raise ValueError("fps output does not follow the 24fps grid")
    return rows


def target_command(source, spec, raw, ffmpeg):
    start = spec["start_seconds"]
    shift = Decimal(start) + Decimal(spec["source_format_start_time"])
    # copyts exposes source PTS; subtract the same seek+format origin before fps
    # to retain the plan's window-relative grid instead of shifting its phase.
    graph = f"format=pix_fmts=rgb24,showinfo@source,setpts=PTS-({shift})/TB,fps=fps=24:round=near,trim=end_frame={spec['frames']},crop={spec['width']}:{spec['height']}:floor((iw-ow)/2):floor((ih-oh)/2):exact=1,showinfo@target"
    command = [ffmpeg, "-nostdin", "-hide_banner", "-v", "info", "-threads", "1", "-filter_threads", "1", "-copyts", "-ss", start,
               "-i", str(source), "-map", "0:v:0", "-vf", graph, "-frames:v", str(spec['frames']), "-vsync", "0", "-an", "-sn", "-dn", "-threads", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw)]
    return command, graph


def decode_target(source, spec, directory, ffmpeg, budget):
    import numpy as np
    raw = directory / "target.rgb"
    log = directory / "target_decode.log"
    command, graph = target_command(source, spec, raw, ffmpeg)
    run_owned(command, log, budget)
    shape = (96, 480, 832, 3)
    checked_shape(raw, shape)
    timestamps = parse_timestamps(log.read_text(errors="replace"))
    clean = np.memmap(raw, mode="r", dtype=np.uint8, shape=shape)
    return clean, {"command": command, "filter_graph": graph, "accurate_input_seek": True,
                   "source_pts_semantic": "original presentation timestamps retained with copyts before relative-grid setpts",
                   "target_pts_semantic": "24fps window-relative output grid after explicit setpts",
                   "timestamps": timestamps, "decoded_source_count": len(timestamps["source"]),
                   "decoded_target_count": len(timestamps["target"]), "log_sha256": file_sha(log)}


def color_info(context):
    return {key: str(getattr(context, key, "unavailable")) for key in ("color_range", "colorspace", "color_primaries", "color_trc", "pix_fmt")}


def observation(clean, profile, binding, window_key, directory, replay, budget):
    import av
    import numpy as np
    from PIL import Image
    pid = profile["profile_id"]
    shape = (len(clean), clean.shape[1] // profile["scale"], clean.shape[2] // profile["scale"], 3)
    raw = directory / f"{pid}.replay{replay}.rgb"
    encoded = directory / f"{pid}.replay{replay}.mp4"
    receipt = {"profile_id": pid, "noise_frames": [], "keyframe_indices": [], "decoded_pts": []}

    def frames():
        for index, frame in enumerate(clean):
            budget.check()
            result, noise = precodec_frame(frame, profile, index, binding["source_sha256"], window_key)
            receipt["noise_frames"].append(noise)
            yield index, result
            budget.check()

    if pid == "native_smooth_jpeg_quality":
        receipt["codec"] = {"implementation": "Pillow JPEG encode/decode", "subsampling": 2, "optimize": False, "progressive": False}
        with raw.open("wb") as out:
            for index, frame in frames():
                payload = io.BytesIO()
                Image.fromarray(frame).save(payload, format="JPEG", quality=profile["frame_parameters"][index]["jpeg_quality"], subsampling=2, optimize=False, progressive=False)
                with Image.open(io.BytesIO(payload.getvalue())) as image:
                    rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
                    out.write(memoryview(rgb).cast("B"))
                receipt["decoded_pts"].append({"index": index, "pts": index, "time_base": "1/24"})
    else:
        crf = profile["frame_parameters"][0]["h264_crf"]
        options = {"crf": str(crf), "preset": "medium", "x264-params": "scenecut=0"}
        with av.open(str(encoded), "w", format="mp4") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width, stream.height, stream.pix_fmt = shape[2], shape[1], "yuv420p"
            stream.codec_context.thread_count = 1
            stream.codec_context.gop_size = 96
            stream.codec_context.max_b_frames = 0
            stream.options = options
            for index, rgb in frames():
                frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
                frame.pts, frame.time_base = index, Fraction(1, 24)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
            receipt["encoder_color"] = color_info(stream.codec_context)
        receipt["codec"] = {"implementation": "PyAV libx264", "options": options, "fps": 24, "gop_size": 96, "max_b_frames": 0, "threads": 1, "pix_fmt": "yuv420p", "bitstream_sha256": file_sha(encoded)}
        with av.open(str(encoded)) as container, raw.open("wb") as out:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            receipt["decoder_color"] = color_info(stream.codec_context)
            for index, frame in enumerate(container.decode(stream)):
                if index >= len(clean):
                    raise ValueError("H264 decoder produced extra frames")
                rgb = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                if rgb.shape != shape[1:]:
                    raise ValueError("codec changed observation geometry")
                out.write(memoryview(rgb).cast("B"))
                if frame.key_frame:
                    receipt["keyframe_indices"].append(index)
                receipt["decoded_pts"].append({"index": index, "pts": frame.pts, "time_base": str(frame.time_base)})
                if frame.pts is None or frame.pts * frame.time_base != Fraction(index, 24):
                    raise ValueError("H264 presentation timestamp mismatch")
                budget.check()
        if receipt["keyframe_indices"] != [0]:
            raise ValueError("unexpected H264 reset/keyframe in primary continuous clip")
        encoded.unlink()
    checked_shape(raw, shape)
    if len(receipt["decoded_pts"]) != len(clean):
        raise ValueError("observation decode count mismatch")
    observed = np.memmap(raw, mode="r", dtype=np.uint8, shape=shape)
    receipt["observation_pixel_sha256"] = pixel_sha(observed)
    receipt["noise_schedule_sha256"] = value_sha(receipt["noise_frames"])
    budget.check()
    return observed, raw, receipt


def save_npz(path, pixels, metadata):
    import numpy as np
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, frames=pixels, metadata=canonical(metadata))
    return {"path": str(path), "file_sha256": file_sha(path), "pixel_sha256": pixel_sha(pixels),
            "shape": list(pixels.shape), "dtype": str(pixels.dtype), "bytes_on_disk": Path(path).stat().st_size}


def software_receipt(ffmpeg, ffprobe):
    import av
    import cv2
    import numpy as np
    import PIL
    cv2.setNumThreads(1)
    return {"python": sys.version, "platform": sys.platform, "byteorder": sys.byteorder, "numpy": np.__version__, "opencv": cv2.__version__, "pillow": PIL.__version__,
            "pyav": av.__version__, "libav_versions": {k: list(v) for k, v in av.library_versions.items()},
            "ffmpeg_version": subprocess.check_output([ffmpeg, "-version"], text=True, timeout=10),
            "ffprobe_version": subprocess.check_output([ffprobe, "-version"], text=True, timeout=10),
            "byte_replay_scope": "recorded build and settings only; pixel replay is verified in this smoke", "threads": 1}


def execute(args, plan, contract, registry, binding):
    if args.source_id != SOURCE or Path(args.source).stem != SOURCE:
        raise ValueError("only the predeclared single train source is authorized by this script")
    if args.output is None or args.max_seconds is None or not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("materialization needs explicit new output and wall budget")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("use a new output directory; no overwrite or implicit resume")
    source = Path(args.source)
    if file_sha(source) != binding["source_sha256"]:
        raise ValueError("source bytes differ from existing train manifest")
    source_stat = source.stat()
    probe = source_probe(source, args.ffprobe)
    spec = target_specification(binding, probe, contract)
    key = value_sha(spec)
    software = software_receipt(args.ffmpeg, args.ffprobe)
    output.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.max_seconds)
    receipt = {"schema": VERSION, "status": "started_not_complete", "source": binding, "source_path": str(source),
               "source_stat": {"bytes": source_stat.st_size, "mtime_ns": source_stat.st_mtime_ns},
               "input_manifest_paths": {"train": str(args.train_manifest.resolve()), "dev": str(args.dev_manifest.resolve())},
               "source_probe": probe, "target_specification": spec, "target_specification_sha256": key,
               "window_key": key, "plan_file_sha256": file_sha(args.plan), "execution_contract": contract,
               "execution_contract_file_sha256": file_sha(args.execution_contract), "registry": registry,
               "registry_sha256": value_sha(registry), "software": software, "software_sha256": value_sha(software),
               "materializer_sha256": file_sha(Path(__file__)), "observations": {},
               "source_natural_cut_status": "unknown_engineering_only", "scientific_continuity_approved": False,
               "new_split_assignment": False, "model_calls": 0, "npu_calls": 0}
    atomic_json(output / "receipt.partial.json", receipt)
    with tempfile.TemporaryDirectory(prefix="scratch_", dir=output) as temporary:
        scratch = Path(temporary)
        clean, decode = decode_target(source, spec, scratch, args.ffmpeg, budget)
        receipt["decode"] = decode
        # Preserve exact decoder diagnostics outside scratch without rewriting them.
        shutil.copyfile(scratch / "target_decode.log", output / "target_decode.log")
        target_sha = pixel_sha(clean)
        common = {"schema": VERSION, **binding, "window_key": key, "target_pixel_sha256": target_sha,
                  "plan_file_sha256": receipt["plan_file_sha256"], "registry_sha256": receipt["registry_sha256"],
                  "materializer_sha256": receipt["materializer_sha256"], "software_sha256": receipt["software_sha256"]}
        receipt["target"] = save_npz(output / "target.npz", clean, {**common, "role": "target", "target_specification": spec})
        noise_receipts = {}
        for profile in plan["profiles"]:
            pid = profile["profile_id"]
            replays = []
            for replay in range(2):
                observed, raw, replay_receipt = observation(clean, profile, binding, key, scratch, replay, budget)
                replays.append(replay_receipt)
                if replay == 0:
                    artifact = save_npz(output / (pid + ".npz"), observed, {**common, "role": "observation", "profile": profile,
                        "parameter_schedule_sha256": value_sha(profile["frame_parameters"]),
                        "noise_stream": profile["noise_stream"], "blind_model_condition": "not_supplied_new_loader_required"})
                del observed
                raw.unlink()
                budget.check()
            if replays[0]["observation_pixel_sha256"] != replays[1]["observation_pixel_sha256"] or replays[0]["noise_schedule_sha256"] != replays[1]["noise_schedule_sha256"]:
                raise ValueError("pixel/noise replay mismatch: " + pid)
            noise_receipts[pid] = replays[0]["noise_frames"]
            receipt["observations"][pid] = {"artifact": artifact, "task": profile["task"], "scale": profile["scale"],
                "parameter_schedule_sha256": value_sha(profile["frame_parameters"]), "replay_count": 2, "pixel_replay_equal": True, "replays": replays}
            if pixel_sha(clean) != target_sha:
                raise ValueError("immutable target was changed by a profile")
            atomic_json(output / "receipt.partial.json", receipt)
        for a, b in ((PROFILE_IDS[0], PROFILE_IDS[1]), (PROFILE_IDS[2], PROFILE_IDS[3])):
            if noise_receipts[a] != noise_receipts[b]:
                raise ValueError("paired profiles did not use identical unit-noise arrays")
        del clean
    after = source.stat()
    if (source_stat.st_size, source_stat.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_sha(source) != binding["source_sha256"]:
        raise ValueError("source changed during materialization")
    budget.check()
    receipt.update(status="engineering_smoke_complete_not_training_or_scientific_acceptance", elapsed_seconds=time.monotonic() - budget.started,
                   peak_process_rss_bytes=peak_rss(), max_sampled_parent_plus_ffmpeg_rss_bytes=budget.max_combined_rss,
                   conservative_nonconcurrent_parent_plus_child_peak_bytes=budget.conservative_peak_sum,
                   memory_budget_bytes=budget.bytes, paired_noise_equal=True, source_unchanged=True)
    atomic_json(output / "complete.json", receipt)
    (output / "receipt.partial.json").unlink()
    print(json.dumps({"status": receipt["status"], "output": str(output), "complete_sha256": file_sha(output / "complete.json"), "profiles": 6}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-id", default=SOURCE)
    parser.add_argument("--mode", choices=("inspect", "materialize"), default="inspect")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    contract = json.loads(args.execution_contract.read_text())
    validate_plan(plan, contract)
    registry, binding = load_registry(args.train_manifest, args.dev_manifest, plan)
    if args.source_id != SOURCE or args.source.stem != SOURCE:
        raise ValueError("only the declared train smoke source is supported")
    if args.mode == "inspect":
        print(json.dumps({"status": "inspected_not_materialized", "source": binding, "source_path": str(args.source),
                          "registry_sha256": value_sha(registry), "profiles": list(PROFILE_IDS),
                          "frame_count": 96, "plan_sha256": file_sha(args.plan), "contract_sha256": file_sha(args.execution_contract),
                          "source_io": "none", "natural_cut_status": "unknown_engineering_only", "output_created": False}, indent=2))
    else:
        execute(args, plan, contract, registry, binding)


if __name__ == "__main__":
    main()
