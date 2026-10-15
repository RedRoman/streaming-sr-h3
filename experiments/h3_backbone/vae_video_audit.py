"""Isolate temporal and spatial effects with fixed official H3 VAE wrapper."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=["256x5", "256x1", "512x1", "128x5", "512x5", "256x22"])
    parser.add_argument("--memory-fraction", type=float, default=.35)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    import comfy.sd
    import comfy.utils
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    args.output.mkdir(parents=True, exist_ok=True)
    sd, metadata = comfy.utils.load_torch_file(str(args.weights), return_metadata=True)
    vae = comfy.sd.VAE(sd=sd, device=torch.device("cuda"), dtype=torch.float16, metadata=metadata)
    summary = {"scope": "Fixed native H3 FP16 VAE E-D audit; same source image repeated for T frames; no DiT and no SR quality claim",
               "api": "comfy.sd.VAE.encode(float [T,H,W,C] in[0,1]); decode(latent)",
               "weights": str(args.weights), "weights_sha256_verified_by": "../vae_smoke_128/result.json",
               "source_image": str(args.image), "source_sha256": hashlib.sha256(args.image.read_bytes()).hexdigest(),
               "memory_fraction": args.memory_fraction, "cases": []}
    original = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        for name in args.cases:
            size, frames = map(int, name.split("x"))
            folder = args.output / name
            folder.mkdir(exist_ok=True)
            rgb = np.array(original.resize((size, size), Image.Resampling.BICUBIC), dtype=np.uint8)
            Image.fromarray(rgb).save(folder / "input.png")
            pixels = torch.from_numpy(rgb.copy()).float().div(255)[None].repeat(frames, 1, 1, 1)
            torch.cuda.synchronize()
            began = time.perf_counter()
            latent = vae.encode(pixels)
            torch.cuda.synchronize()
            encoded = time.perf_counter()
            reconstructed = vae.decode(latent).float()
            torch.cuda.synchronize()
            ended = time.perf_counter()
            if reconstructed.ndim == 5:
                reconstructed = reconstructed[0]
            if reconstructed.shape != pixels.shape:
                raise ValueError(f"Unexpected reconstruction shape {list(reconstructed.shape)} for {name}")
            errors = (reconstructed.cpu() - pixels).square().mean(dim=(1, 2, 3))
            psnr = (-10 * errors.clamp_min(1e-12).log10()).tolist()
            selected = sorted(set([0, frames // 2, frames - 1]))
            for index in selected:
                value = reconstructed[index].cpu().numpy()
                Image.fromarray((value.clip(0, 1) * 255).round().astype(np.uint8)).save(folder / f"reconstruction_{index:02d}.png")
            row = {"name": name, "input_shape": list(pixels.shape), "latent_shape": list(latent.shape),
                   "output_shape": list(reconstructed.shape), "psnr_rgb_per_frame_db": psnr,
                   "mean_psnr_db": float(np.mean(psnr)), "min_psnr_db": min(psnr), "max_psnr_db": max(psnr),
                   "output_range": [float(reconstructed.min()), float(reconstructed.max())],
                   "encode_seconds": encoded - began, "decode_seconds": ended - encoded,
                   "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                   "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            summary["cases"].append(row)
            (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({"event": "case_complete", **row}), flush=True)
            del pixels, latent, reconstructed
    summary["status"] = "complete"
    (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
