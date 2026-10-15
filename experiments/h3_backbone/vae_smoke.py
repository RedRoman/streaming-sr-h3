"""Real pretrained H3 FP16 VAE single-frame smoke; no generative-backbone claim."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file, save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--memory-fraction", type=float, default=0.35)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy))
    import comfy.ops
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with args.weights.open("rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    expected = "7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522"
    if checksum != expected:
        raise ValueError("VAE failed canonical SHA256 verification")
    vae = MiniMaxH3VideoVAE(num_layers=36, operations=comfy.ops.disable_weight_init)
    vae.load_state_dict(load_file(str(args.weights), device="cpu"), strict=True, assign=True)
    vae = vae.to(device="cuda", dtype=torch.float16).eval().requires_grad_(False)
    torch.cuda.synchronize()
    loaded = time.perf_counter()
    image = Image.open(args.image).convert("RGB")
    image = image.resize((args.size, args.size), Image.Resampling.BICUBIC)
    rgb = np.array(image, dtype=np.uint8)
    image.save(args.output / "input.png")
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None, :, None].to("cuda", torch.float16) / 127.5 - 1
    with torch.no_grad():
        latent = vae.encode(x.clone())
        torch.cuda.synchronize()
        encoded = time.perf_counter()
        reconstruction = vae.decode(latent.to(torch.float16))
        torch.cuda.synchronize()
        decoded = time.perf_counter()
    if not torch.isfinite(latent).all() or not torch.isfinite(reconstruction).all():
        raise ValueError("Non-finite real VAE result")
    y = reconstruction[0, :, 0].permute(1, 2, 0).float().cpu().numpy()
    Image.fromarray((y.clip(0, 1) * 255).round().astype(np.uint8)).save(args.output / "reconstruction.png")
    mse = float(np.mean((y - rgb.astype(np.float32) / 255)**2))
    save_file({"latent": latent.float().contiguous().cpu(),
               "input_rgb": torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous(),
               "reconstruction": reconstruction.float().contiguous().cpu()}, str(args.output / "smoke_tensors.safetensors"))
    result = {"scope": "Actual pretrained MiniMax-H3 FP16 VAE only; generative BF16 DiT not executed",
              "status": "passed", "weights": str(args.weights), "verified_sha256": checksum,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "memory_fraction_limit": args.memory_fraction, "source_image": str(args.image),
              "input_shape": list(x.shape), "latent_shape": list(latent.shape),
              "output_shape": list(reconstruction.shape), "load_and_sha_seconds": loaded-started,
              "encode_seconds": encoded-loaded, "decode_seconds": decoded-encoded,
              "psnr_rgb_db": -10*np.log10(mse) if mse else float("inf"),
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
              "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
