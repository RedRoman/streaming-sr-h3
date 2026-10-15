"""Actual native MiniMax-H3 short sampling with cached real text and LR guides.

This is an explicitly quantized integration smoke, not a foundation-quality or
super-resolution benchmark. No LoRA, VDN branch, or external service is changed.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import save_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--frames", type=int, default=5)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--memory-fraction", type=float, default=.5)
    args = parser.parse_args()
    if args.frames % 17 != 5:
        raise ValueError("Native H3 duration must be 17k+5")
    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram = 10.0
    import comfy.sd
    import comfy.sample
    import comfy.utils
    import comfy.nested_tensor
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = {"status": "running", "scope": "Real quantized native MiniMax-H3 DiT plus guide-conditioned sampling integration smoke; not a quality benchmark",
              "model_path": str(args.model), "precision_label": "Native NVFP4-all pruned FL2VA DiT; BF16 compute and independently cached genuine BF16 Qwen3-VL text; FP16 VAE",
              "steps": args.steps, "sampler": "euler", "scheduler": "simple", "cfg": 1.0,
              "seed": 101, "frames": args.frames, "size": args.size,
              "max_observed_input_index": args.frames - 1, "max_input_wait_frames": args.frames - 1,
              "future_beyond_block_used": False, "memory_fraction_limit": args.memory_fraction,
              "comfy_reserved_vram_gb": 10.0, "forward_events": []}
    try:
        cached = torch.load(args.context, map_location="cpu", weights_only=True)
        context, tags = cached["prompt_embeds"], cached["text_token_tags"]
        if context.ndim != 2 or context.shape[-1] != 5120 or not torch.isfinite(context).all():
            raise ValueError("Expected genuine raw layer-50 H3 Qwen context [L,5120]")
        if tags.shape != context.shape[:1] or not (tags == 1).all():
            raise ValueError("This smoke expects text-only real context with all modality tags=1")
        result.update(prompt=cached["prompt"], context_shape=list(context.shape),
                      context_sha256=hashlib.sha256(args.context.read_bytes()).hexdigest(),
                      context_protocol="Qwen3-VL32B BF16 raw prompt without chat/special tokens; unnormalized hidden_states[50]; TEXT_TAG=1; cached source provenance archived separately")
        rgb = np.array(Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.Resampling.BICUBIC))
        source = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None].float() / 255
        low = F.interpolate(source, size=(args.size // 4, args.size // 4), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        upsampled = F.interpolate(low, size=(args.size, args.size), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
        Image.fromarray(rgb).save(args.output / "source_hr_for_evaluation_only.png")
        Image.fromarray((low[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / "observed_lr.png")
        Image.fromarray((upsampled[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / "lr_upsampled_guide.png")
        pixels = upsampled[0].permute(1, 2, 0)[None].repeat(args.frames, 1, 1, 1)
        sd, metadata = comfy.utils.load_torch_file(str(args.vae), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, device=torch.device("cuda"), dtype=torch.float16, metadata=metadata)
        del sd
        with torch.no_grad():
            guide = vae.encode(pixels)
        result["guide_latent_shape"] = list(guide.shape)
        positive = [[context[None], {"minimax_token_tags": tags,
                    "minimax_keyframes": [{"resolved_frame_index": 0, "latent": guide}]}]]
        model = comfy.sd.load_diffusion_model(str(args.model), model_options={"dtype": torch.bfloat16})
        result["native_model_class"] = type(model.model.diffusion_model).__name__
        result["native_block_count"] = len(model.model.diffusion_model.blocks)

        def observed_forward(module, inputs, kwargs, output):
            x = inputs[0] if inputs else kwargs["x"]
            timestep = inputs[1] if len(inputs) > 1 else kwargs["timestep"]
            ctx = kwargs.get("context", inputs[2] if len(inputs) > 2 else None)
            event = {"call": len(result["forward_events"]), "video_shape": list(x[0].shape),
                     "audio_shape": list(x[1].shape), "context_shape": list(ctx.shape),
                     "timestep": timestep.detach().float().cpu().flatten().tolist(),
                     "video_output_finite": bool(torch.isfinite(output[0]).all()),
                     "audio_output_finite": bool(torch.isfinite(output[1]).all())}
            result["forward_events"].append(event)
            print(json.dumps({"event": "native_dit_forward", **event}), flush=True)

        hook = model.model.diffusion_model.register_forward_hook(observed_forward, with_kwargs=True)
        temporal = 2 if args.frames <= 5 else ((args.frames - 5) // 17) * 5 + 2
        audio_t = round(args.frames / 24 * 40)
        latent = comfy.nested_tensor.NestedTensor((
            torch.zeros(1, 24, temporal, args.size // 16, args.size // 16),
            torch.zeros(1, 32, 2, audio_t)))
        noise = comfy.sample.prepare_noise(latent, 101)
        with torch.no_grad():
            began = time.perf_counter()
            samples = comfy.sample.sample(model, noise, args.steps, 1.0, "euler", "simple",
                                          positive, [], latent, seed=101, disable_pbar=True)
            torch.cuda.synchronize()
            result["sampling_seconds_including_offload_and_observers"] = time.perf_counter() - began
            video, audio = samples.unbind()
            save_file({"video": video.contiguous().cpu(), "audio": audio.contiguous().cpu()},
                      str(args.output / "sampled_av_latents.safetensors"))
            decoded = vae.decode(video).float().cpu()
        hook.remove()
        if decoded.ndim == 5:
            decoded = decoded[0]
        if not torch.isfinite(decoded).all():
            raise ValueError("Nonfinite decoded sample")
        for index, frame in enumerate(decoded):
            Image.fromarray((frame.numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(args.output / f"output_{index:02d}.png")
        result.update(status="complete", output_shape=list(decoded.shape), output_range=[float(decoded.min()), float(decoded.max())])
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        result.update(total_seconds=time.perf_counter() - started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
