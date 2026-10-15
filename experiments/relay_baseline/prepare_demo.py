"""Prepare explicitly cropped and self-degraded inputs from the official demo.

The native real demo has no HR target. Only self_degraded_lr has a known target,
namely losslessly decoded source frames cropped to 512x256, stored as PNG.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=31)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.repository.resolve()))
    from refvsr_inference.media import input_fps, iter_frames
    source = args.repository / "assets/demo/input.mp4"
    directories = {key: args.output_root / key for key in [
        "native_lr_center128", "self_degraded_hr512x256", "self_degraded_lr128x64", "bicubic512x256"]}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, rgb in enumerate(iter_frames(source, max_frames=args.frames)):
        height, width = rgb.shape[-2:]
        if height < 256 or width < 512:
            raise ValueError("Source too small for declared crops")
        x, y = (width - 128) // 2, (height - 128) // 2
        native = rgb[:, y:y + 128, x:x + 128]
        gx, gy = (width - 512) // 2, (height - 256) // 2
        gt = rgb[:, gy:gy + 256, gx:gx + 512]
        lr = F.interpolate(gt[None].float() / 255, size=(64, 128), mode="bicubic",
                           align_corners=False, antialias=True)[0].mul(255).round().clamp(0, 255).to(torch.uint8)
        bicubic = F.interpolate(lr[None].float() / 255, size=(256, 512), mode="bicubic",
                                align_corners=False, antialias=True)[0].mul(255).round().clamp(0, 255).to(torch.uint8)
        frames = {"native_lr_center128": native, "self_degraded_hr512x256": gt,
                  "self_degraded_lr128x64": lr, "bicubic512x256": bicubic}
        for key, frame in frames.items():
            Image.fromarray(frame.permute(1, 2, 0).contiguous().numpy()).save(directories[key] / f"{index:05d}.png")
        rows.append({"frame": index, "native_crop_xywh": [x, y, 128, 128],
                     "hr_crop_xywh": [gx, gy, 512, 256], "source_size_wh": [width, height]})
    manifest = {
        "source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "fps": str(input_fps(source)), "frames": len(rows),
        "native_lr_center128_has_hr_ground_truth": False,
        "self_degraded_ground_truth": "Decoded official demo frames; center crop 512x256; lossless PNG. Not a separately captured genuine-HR scene.",
        "degradation": "RGB float in [0,1]; torch bicubic x4 downsample with align_corners=False and antialias=True; multiply255, round, clamp, uint8; save lossless PNG.",
        "directories": {key: str(value) for key, value in directories.items()},
        "frame_metadata": rows,
    }
    (args.output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
