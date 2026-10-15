"""Observe the unmodified official RelayVSR entrypoint without changing predictions.

Only host timing, CUDA timing resolution, and input-consumption counters are added.
CUDA synchronization before yielding is explicit and recorded in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=31)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--flash-checkpoint", type=Path)
    parser.add_argument("--wan-checkpoint", type=Path)
    parser.add_argument("--fps")
    parser.add_argument("--protocol", choices=["official", "causal_repeat_start"], default="official")
    parser.add_argument("--png-dir", type=Path, help="Lossless outputs for quality evaluation; wall times then include PNG writing")
    args = parser.parse_args()
    root = args.repository.resolve()
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root))
    import torch
    import inference as official

    rows = []
    meta = {
        "protocol": "official_bounded_lookahead" if args.protocol == "official" else "causal_repeat_start_ablation",
        "repository": str(root),
        "commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "source_status": subprocess.check_output(["git", "-C", str(root), "status", "--short"], text=True),
        "max_frames": args.max_frames,
        "window_size": args.window_size,
        "seed": 42,
        "dtype": "bf16",
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "packages": {name: importlib.metadata.version(name) for name in (
            "diffusers", "accelerate", "triton", "einops", "av", "safetensors", "numpy", "pillow")},
        "instrumentation": "Unmodified official inference.main with observer subclass; CUDA synchronization and event resolution before each output; input consumption tracked; no prediction, seed, weight, or sampler changes.",
    }
    if args.png_dir:
        from PIL import Image
        base_writer = official.VideoWriter
        args.png_dir.mkdir(parents=True, exist_ok=True)

        class LosslessWriter(base_writer):
            def write(self, frame):
                index = self.frames
                super().write(frame)
                Image.fromarray(frame.numpy()).save(args.png_dir / f"{index:05d}.png")

        official.VideoWriter = LosslessWriter
        meta["png_dir"] = str(args.png_dir)
        meta["wall_time_includes_png_writing"] = True
    original_loader = official.load_checkpoints
    original_stream = official.CollaborativeFrameStream
    if args.protocol == "causal_repeat_start":
        from causal_stream import CausalRepeatStartStream
        original_stream = CausalRepeatStartStream
        meta["protocol_changes"] = (
            "Strictly causal input; repeat the latest Wan reference in both endpoint slots; "
            "refresh every K-1 frames; reset Flash history at refresh without overlapping re-decode. "
            "Uses pretrained official weights under a changed conditioning protocol; no claim of official equivalence."
        )
    load_started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def measured_loader(*a, **kw):
        start = time.perf_counter()
        result = original_loader(*a, **kw)
        torch.cuda.synchronize()
        meta["model_load_seconds"] = time.perf_counter() - start
        meta["memory_after_load_allocated_bytes"] = torch.cuda.memory_allocated()
        return result

    class MeasuredStream(original_stream):
        def frames(self, source):
            started = previous = time.perf_counter()
            last_consumed = -1

            def observed_source():
                nonlocal last_consumed
                for packet in source:
                    last_consumed = packet.index
                    yield packet

            with args.metrics.with_suffix(".frames.jsonl").open("w") as handle:
                for output in super().frames(observed_source()):
                    torch.cuda.synchronize()
                    completed = time.perf_counter()
                    stages = {}
                    for name, span in output.timings:
                        stages[name] = stages.get(name, 0.0) + span.milliseconds()
                    row = {
                        "index": output.index,
                        "window_start": output.window_start,
                        "window_end": output.window_end,
                        "last_consumed_input_index": last_consumed,
                        "lookahead_frames": last_consumed - output.index,
                        "completion_from_stream_start_seconds": completed - started,
                        "since_previous_completion_seconds": completed - previous,
                        "cuda_stage_ms": stages,
                        "allocated_bytes": torch.cuda.memory_allocated(),
                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "output_shape_bchw": list(output.rgb.shape),
                    }
                    rows.append(row)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()
                    print(json.dumps({"event": "output", "index": output.index,
                                      "lookahead_frames": row["lookahead_frames"],
                                      "completion_seconds": round(completed - started, 4)}), flush=True)
                    previous = completed
                    yield output
            meta["wan_keyframe_indices"] = list(self.last_keyframe_indices)
            meta["wan_keyframe_count"] = self.keyframe_count
            meta["stream_seconds"] = time.perf_counter() - started

    official.load_checkpoints = measured_loader
    official.CollaborativeFrameStream = MeasuredStream
    original_argv = sys.argv
    sys.argv = [str(root / "inference.py"), "--output", str(args.output),
                "--max-frames", str(args.max_frames), "--window-size", str(args.window_size)]
    if args.input:
        sys.argv.extend(["--input", str(args.input)])
    if args.flash_checkpoint:
        sys.argv.extend(["--flash-checkpoint", str(args.flash_checkpoint)])
    if args.wan_checkpoint:
        sys.argv.extend(["--wan-checkpoint", str(args.wan_checkpoint)])
    if args.fps:
        sys.argv.extend(["--fps", args.fps])
    try:
        official_args = official.parse_args()
        meta["input_path"] = str(official_args.input)
        meta["input_fps"] = str(official.input_fps(official_args.input, official_args.fps))
        meta["wan_checkpoint"] = str(official_args.wan_checkpoint)
        meta["flash_checkpoint"] = str(official_args.flash_checkpoint)
        official.main(official_args)
        meta["status"] = "complete"
    except Exception as exc:
        meta["status"] = "failed"
        meta["exception"] = repr(exc)
        raise
    finally:
        sys.argv = original_argv
        meta["total_process_seconds"] = time.perf_counter() - load_started
        meta["output_frames"] = len(rows)
        meta["max_lookahead_frames"] = max((r["lookahead_frames"] for r in rows), default=None)
        meta["first_output_seconds"] = rows[0]["completion_from_stream_start_seconds"] if rows else None
        meta["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        meta["peak_reserved_bytes"] = torch.cuda.max_memory_reserved()
        for label, low, high in [("first_window_after_first", 1, 16), ("second_window", 16, 31)]:
            group = [r["since_previous_completion_seconds"] for r in rows if low <= r["index"] < high]
            meta[label + "_mean_completion_interval_seconds"] = statistics.mean(group) if group else None
        if args.output.exists():
            meta["output_path"] = str(args.output)
            meta["output_bytes"] = args.output.stat().st_size
            meta["output_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        args.metrics.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
