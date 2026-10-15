"""Separate-process VAE/DiT/VAE stages to bound original-machine CPU memory.

The guide is one observed LR repeated over a native five-frame block. This is a
static interface smoke. Only this process can be terminated by its watchdog.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("phase", choices=["encode", "sample", "decode"])
    p.add_argument("--comfy", type=Path, required=True)
    p.add_argument("--vae", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--context", type=Path)
    p.add_argument("--image", type=Path)
    p.add_argument("--lr-directory", type=Path)
    p.add_argument("--frames", type=int, default=5)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--memory-fraction",type=float,default=.5)
    p.add_argument("--reserve-vram",type=float,default=20.)
    p.add_argument("--cfg", type=float, default=1.0)
    p.add_argument("--scheduler", default="simple")
    p.add_argument("--official-grid", action="store_true", help="Use steps+1 exact shifted sigma points; official normal is 50 points /49 evaluations")
    p.add_argument("--decode-guide", action="store_true", help="Also decode the observed LR guide as a VAE-only comparison")
    p.add_argument("--guide-only", action="store_true", help="Decode only the matched guide, preserving future DiT output names")
    args = p.parse_args()
    if args.frames < 5 or args.frames % 17 != 5:
        raise ValueError("Native H3 durations are 5/22/39/... frames")
    assert args.width%32==args.height%32==0
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    sys.path.insert(0, str(args.comfy))
    from comfy.cli_args import args as comfy_args
    comfy_args.reserve_vram = args.reserve_vram
    import comfy.sd
    import comfy.utils
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    result = {"status": "running", "phase": args.phase, "pid": os.getpid(),
              "scope": "Static repeated-image 5-frame 256-square LR-guide integration smoke, not motion SR quality or streaming latency",
              "dit_precision": "Full pruned BF16 foundation weights, no DiT quantization/LoRA/VDN",
              "text_precision": "real Qwen3VL32B NVFP4-AWQ text encoder, BF16 hidden-state cache",
              "vae_precision": "FP16 stored weights and computation", "gpu_memory_fraction": args.memory_fraction, "comfy_reserved_vram_gb": args.reserve_vram,
              "observed_distinct_frame_indices": [0], "actual_latest_distinct_observation_index": 0,
              "future_gt_semantics_used": False, "forward_events": []}
    result.update(canvas_width=args.width,canvas_height=args.height)
    if args.lr_directory is not None:
        result.update(scope="Real DAVIS motion clip, synthetic4x LR, single observed native block; no GT pixels enter restoration",
                      observed_distinct_frame_indices=list(range(args.frames)),
                      actual_latest_distinct_observation_index=args.frames-1,
                      block_max_wait_frames=args.frames-1,
                      acquisition_wait_note="Offline known-input budget only; no real-time acquisition waiting measured")
    if (args.output / "guide_manifest.json").exists():
        diagnostic=json.loads((args.output/"guide_manifest.json").read_text())
        result.update(scope="Guide detail-carrying diagnosis, fixed BSDS train source; HR oracle only as interface upper bound, never causal method input",
                      oracle_HR_guide=diagnostic['oracle_HR_guide'],guide_quality=diagnostic['guide_quality'],guide_mode=diagnostic['guide_mode'],
                      actual_latest_distinct_observation_index=0,observed_distinct_frame_indices=[0],
                      official_AddGuide_parity_max_abs=diagnostic['official_AddGuide_execute_vs_direct_encode_max_abs'])
    record = args.output / (("guide_decode" if args.guide_only else args.phase) + "_result.json")

    def watchdog():
        while True:
            try:
                rows = dict(line.split() for line in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines())
                anon = int(rows["anon"])
                limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
                if anon > limit - 6 * 1024**3:
                    result.update(status="stopped_owned_process_by_memory_watchdog", cgroup_anon=anon, cgroup_limit=limit)
                    record.write_text(json.dumps(result, indent=2) + "\n")
                    print(json.dumps({"event": "own_process_memory_guard", "anon": anon, "limit": limit}), flush=True)
                    os._exit(75)
            except (OSError, ValueError, KeyError):
                pass
            time.sleep(.5)
    threading.Thread(target=watchdog, daemon=True).start()

    def get_vae():
        sd, metadata = comfy.utils.load_torch_file(str(args.vae), return_metadata=True)
        vae = comfy.sd.VAE(sd=sd, device=torch.device("cuda"), dtype=torch.float16, metadata=metadata)
        del sd
        return vae

    try:
        if args.phase == "encode":
            if args.lr_directory is None:
                rgb = np.asarray(Image.open(args.image).convert("RGB").resize((args.width, args.height), Image.Resampling.BICUBIC)).copy()
                source = torch.from_numpy(rgb).permute(2, 0, 1)[None].float() / 255
                low = F.interpolate(source, size=(args.height//4, args.width//4), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
                Image.fromarray(rgb).save(args.output / "source_hr_for_evaluation_only.png")
            else:
                files = sorted(args.lr_directory.glob("*.png"))[:args.frames]
                assert len(files) == args.frames
                low = torch.stack([torch.from_numpy(np.array(Image.open(f).convert("RGB"))).permute(2, 0, 1) for f in files]).float()/255
                assert low.shape == (args.frames, 3, args.height//4, args.width//4)
                result["observed_input_sha256"] = {str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
            up = F.interpolate(low, size=(args.height, args.width), mode="bicubic", align_corners=False, antialias=True).clamp(0, 1)
            for name, tensor in [("observed_lr", low), ("lr_upsampled_guide", up)]:
                Image.fromarray((tensor[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(args.output / (name + ".png"))
            pixels = up.permute(0, 2, 3, 1)
            if args.lr_directory is None:
                pixels = pixels.repeat(args.frames, 1, 1, 1)
            # Compile this exact upstream helper without importing UI/server API
            # modules, which are unnecessary for a source-only H100 runtime.
            node_source=args.comfy/'comfy_extras/nodes_minimax_h3.py'
            resize_ast=next(n for n in ast.parse(node_source.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_resize')
            namespace={'comfy':types.SimpleNamespace(utils=comfy.utils)}
            exec(compile(ast.Module(body=[resize_ast],type_ignores=[]),str(node_source),'exec'),namespace)
            _resize=namespace['_resize']
            official_pixels=_resize(pixels,args.width,args.height,'center')
            result['official_resize_source_sha256']=hashlib.sha256(node_source.read_bytes()).hexdigest()
            result['official_AddGuide_same_canvas_quantization_max_abs']=float((official_pixels-pixels).abs().max())
            result['guide_preprocessing']='Torch AA bicubic LR-upsample followed by actual official AddGuide lanczos/center helper (includes8bit truncation), then standardVAE encode'
            with torch.no_grad():
                guide = get_vae().encode(official_pixels)
            save_file({"guide": guide.cpu().contiguous()}, str(args.output / "guide.safetensors"))
            result["guide_shape"] = list(guide.shape)
        elif args.phase == "sample":
            import comfy.sample
            import comfy.nested_tensor
            cache = torch.load(args.context, weights_only=True, map_location="cpu")
            context, tags = cache["prompt_embeds"], cache["text_token_tags"]
            assert context.ndim == 2 and context.shape[-1] == 5120 and torch.isfinite(context).all()
            assert tags.shape == context.shape[:1] and bool(((tags == 1)|(tags == 0)).all())
            result.update(prompt=cache["prompt"], context_shape=list(context.shape),
                          context_sha256=hashlib.sha256(args.context.read_bytes()).hexdigest(),
                          context_vision_tag_count=int((tags==0).sum()),context_text_tag_count=int((tags==1).sum()),
                          model_path=str(args.model), model_bytes=args.model.stat().st_size,
                          steps=args.steps, cfg=args.cfg, scheduler=args.scheduler, sampler="euler", seed=args.seed)
            positive = [[context[None], {"minimax_token_tags": tags, "minimax_keyframes": [
                {"resolved_frame_index": 0, "latent": load_file(str(args.output / "guide.safetensors"))["guide"]}]}]]
            print(json.dumps({"event": "loading_bf16_dit", "pid": os.getpid()}), flush=True)
            model = comfy.sd.load_diffusion_model(str(args.model), model_options={"dtype": torch.bfloat16})
            result.update(native_model_class=type(model.model.diffusion_model).__name__,
                          native_block_count=len(model.model.diffusion_model.blocks))
            result["model_sampling"] = {"video_shift": model.model.model_sampling.shift,
                                        "audio_shift": model.model.model_sampling.audio_shift}
            sigmas = None
            if args.official_grid:
                base = torch.linspace(1, 0, args.steps + 1, dtype=torch.float32)
                sigmas = 12 * base / (1 + 11 * base)
                result.update(sigma_grid=sigmas.tolist(), schedule_protocol="Exact official base50 sigma points including terminal0;49 Euler model evaluations; video shift12/audio3")
            def observed_forward(module, inputs, kwargs, output):
                x = inputs[0] if inputs else kwargs["x"]
                ts = inputs[1] if len(inputs) > 1 else kwargs["timestep"]
                ctx = kwargs.get("context", inputs[2] if len(inputs) > 2 else None)
                event = {"call": len(result["forward_events"]), "video_shape": list(x[0].shape), "audio_shape": list(x[1].shape),
                         "context_shape": list(ctx.shape), "timestep": ts.detach().float().cpu().flatten().tolist(),
                         "video_finite": bool(torch.isfinite(output[0]).all()), "audio_finite": bool(torch.isfinite(output[1]).all())}
                result["forward_events"].append(event)
                print(json.dumps({"event": "native_bf16_forward", **event}), flush=True)
            hook = model.model.diffusion_model.register_forward_hook(observed_forward, with_kwargs=True)
            temporal = ((args.frames-5)//17)*5+2
            latent = comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, temporal, args.height//16, args.width//16), torch.zeros(1, 32, 2, round(args.frames/24*40))))
            with torch.no_grad():
                began = time.perf_counter()
                samples = comfy.sample.sample(model, comfy.sample.prepare_noise(latent, args.seed), args.steps, args.cfg,
                    "euler", args.scheduler, positive, [], latent, seed=args.seed, disable_pbar=True, sigmas=sigmas)
                torch.cuda.synchronize()
                result["sampling_seconds_including_offload_and_observers"] = time.perf_counter() - began
            hook.remove()
            video, audio = samples.unbind()
            save_file({"video": video.cpu().contiguous(), "audio": audio.cpu().contiguous()}, str(args.output / "sampled_av_latents.safetensors"))
        else:
            vae = get_vae()
            with torch.no_grad():
                latent_file='guide.safetensors' if args.guide_only else 'sampled_av_latents.safetensors'
                latent_key='guide' if args.guide_only else 'video'
                decoded = vae.decode(load_file(str(args.output / latent_file))[latent_key]).float().cpu()
            if decoded.ndim == 5:
                decoded = decoded[0]
            assert torch.isfinite(decoded).all()
            for i, frame in enumerate(decoded):
                prefix='guide_roundtrip' if args.guide_only else 'output'
                Image.fromarray((frame.numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(args.output / f"{prefix}_{i:02d}.png")
            result.update(output_shape=list(decoded.shape), output_range=[float(decoded.min()), float(decoded.max())])
            if args.decode_guide and not args.guide_only:
                with torch.no_grad():
                    guide_rgb = vae.decode(load_file(str(args.output / "guide.safetensors"))["guide"]).float().cpu()
                if guide_rgb.ndim == 5:
                    guide_rgb = guide_rgb[0]
                assert torch.isfinite(guide_rgb).all()
                for i, frame in enumerate(guide_rgb):
                    Image.fromarray((frame.numpy().clip(0, 1) * 255).round().astype(np.uint8)).save(args.output / f"guide_roundtrip_{i:02d}.png")
                result["vae_only_guide_output_shape"] = list(guide_rgb.shape)
        result["status"] = "complete"
    except Exception as exc:
        result.update(status="failed", exception=repr(exc))
        raise
    finally:
        result.update(seconds=time.perf_counter() - started, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        record.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
