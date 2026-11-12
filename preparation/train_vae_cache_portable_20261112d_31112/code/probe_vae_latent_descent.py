"""Prepared, bounded VAE latent-descent diagnostic; never an automatic extension.

Fixed seed102/full geometry/scale1024/RMS step1e-3. At most eight decoder
forwards and three backwards. No DiT, LoRA, learned-weight update, LR search,
quality claim or revision of the earlier exact-repeatability failure.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from compare_vae_gradient_repeatability import (
    array_sha, array_statistics, atomic_json, checkpoint_matches, save_array, sha_file,
)
from probe_vae_gradient_repeatability import implementation_hashes as prior_hashes, runtime_observation


SHAPE = (1, 24, 7, 30, 52)
RGB_SHAPE = (22, 480, 832, 3)
SCALE, RHO, STEPS = 1024.0, .001, 3
INPUT_SHA = "f235d9af325c56cc8f42a41587ddb532031cbd232eb0a976cb9ef6fbc2e65ab2"


def contract():
    hashes = prior_hashes()
    hashes[Path(__file__).name] = sha_file(__file__)
    return {"schema": "openh3.vae_latent_descent_contract.v1", "seed": 102,
            "input_shape": list(SHAPE), "RGB_shape": list(RGB_SHAPE), "input_sha256": INPUT_SHA,
            "input_dtype": "float32", "decoder_dtype": "float16", "loss_scale": SCALE,
            "rho_latent_rms": RHO, "steps": STEPS,
            "update": "FP64 RMS of FP32 unscaled gradient; direction, rho multiplication and latent subtraction in FP32",
            "target": "CPU torch.float32 linspace(.2,.8,832)[None,None,:,None], broadcast THWC",
            "loss": "CPU float32 MSE of as_returned_internal_clipped VAE RGB",
            "grad_rope": "ascend_native", "checkpoint_blocks": 36,
            "forward_atol": .0002, "forward_rtol": .0001,
            "budget": {"forwards": 8, "backwards": 3, "learned_weight_updates": 0, "H3_NFE": 0},
            "acceptance": ["finite nonzero gradients and finite observations", "deployment/direct forward parity at each current point",
                           "first central directional difference > 0", "all three negative steps strictly lower the deployment loss",
                           "unchanged frozen weights/runtime and restored checkpoint contexts", "exact declared execution budget"],
            "descriptive_only": ["central difference versus gradient dot direction magnitude error", "Lplus > L0", "bitwise gradient repeatability"],
            "scope": "local synthetic latent optimization only; not adapter training, true RGB quality, or joint H3 acceptance",
            "implementation_sha256": hashes}


def normalized_step(source, raw_gradient):
    if source.dtype != np.float32 or raw_gradient.dtype != np.float32 or source.shape != raw_gradient.shape:
        raise ValueError("same-shape float32 latent and raw gradient required")
    if not np.isfinite(source).all() or not np.isfinite(raw_gradient).all():
        raise ValueError("nonfinite latent/gradient")
    gradient = np.ascontiguousarray(raw_gradient / np.float32(SCALE))
    rms = float(np.sqrt(np.mean(gradient.astype(np.float64) ** 2)))
    if rms == 0 or not np.isfinite(rms):
        raise ValueError("zero/nonfinite gradient RMS")
    direction = np.ascontiguousarray(gradient / np.float32(rms))
    delta = np.float32(RHO) * direction
    minus, plus = np.ascontiguousarray(source - delta), np.ascontiguousarray(source + delta)
    if not np.isfinite(minus).all() or not np.isfinite(plus).all():
        raise ValueError("nonfinite fixed update")
    actual = source.astype(np.float64) - minus.astype(np.float64)
    return minus, plus, {"gradient_rms_float64": rms, "rho": RHO,
            "actual_float32_step_rms": float(np.sqrt(np.mean(actual ** 2))),
            "gradient_dot_direction": float(np.sum(gradient.astype(np.float64) * direction.astype(np.float64))),
            "predicted_first_order_decrease": float(np.sum(gradient.astype(np.float64) * actual))}


def cast_change(before, after):
    if before.dtype != np.float16 or after.dtype != np.float16 or before.shape != after.shape:
        raise ValueError("actual same-shape float16 decoder inputs required")
    change = after.astype(np.float64) - before.astype(np.float64)
    return {"before_sha256": array_sha(before), "after_sha256": array_sha(after), "dtype": "float16",
            "changed_coordinates": int(np.count_nonzero(change)), "elements": int(change.size),
            "changed_fraction": float(np.count_nonzero(change) / change.size),
            "actual_cast_step_rms": float(np.sqrt(np.mean(change ** 2))),
            "actual_cast_max_abs_change": float(np.max(np.abs(change)))}


def new_report():
    return {"schema": "openh3.vae_latent_descent_probe.v1", "complete": False, "protocol_complete": False,
            "phase": "prepared", "forward_attempts": 0, "forwards_completed": 0,
            "backward_attempts": 0, "backwards_completed": 0,
            "evaluations": {}, "gradients": {}, "checkpoint": {}, "arrays": {}, "steps": [], "error": None}


def run_protocol(backend, source, out, report, update):
    """Backend invokes events only at actual operation boundaries; no retries."""
    def event(kind, label, value=None):
        if kind in ("forward_attempt", "backward_attempt"):
            key, maximum = ("forward_attempts", 8) if kind == "forward_attempt" else ("backward_attempts", 3)
            if report[key] >= maximum:
                raise RuntimeError("fixed operation budget exhausted")
            report[key] += 1
        elif kind == "forward_completed":
            report["forwards_completed"] += 1
            report["evaluations"][label] = value
        elif kind == "backward_completed":
            report["backwards_completed"] += 1
            if value is None:
                raise ValueError("input gradient missing")
            report["arrays"][label + "_raw"] = save_array(out, label + "_raw.npy", value)
            unscaled = np.ascontiguousarray(value / np.float32(SCALE))
            report["arrays"][label + "_unscaled"] = save_array(out, label + "_unscaled.npy", unscaled)
            report["gradients"][label] = {"raw": array_statistics(value), "unscaled": array_statistics(unscaled)}
        elif kind == "checkpoint":
            report["checkpoint"][label] = value
        else:
            raise ValueError("unknown operation event")
        report["phase"] = label + ":" + kind
        update()

    report["arrays"]["latent_0"] = save_array(out, "latent_0.npy", source)
    event("forward_attempt", "initial_reference")
    current = backend.no_grad(source, "initial_reference", event)
    for index in range(STEPS):
        label = "step" + str(index + 1)
        row = {"step": index + 1, "loss_before": current["loss"], "accepted": False}
        report["steps"].append(row)
        event("forward_attempt", label + "_gradient")
        gradient = backend.gradient(source, current, label + "_gradient", event)
        minus, plus, update_record = normalized_step(source, gradient)
        row.update(update_record)
        report["arrays"]["latent_" + str(index + 1)] = save_array(out, "latent_" + str(index + 1) + ".npy", minus)
        event("forward_attempt", label + "_next")
        following = backend.no_grad(minus, label + "_next", event)
        row.update(loss_after=following["loss"], actual_loss_decrease=current["loss"] - following["loss"],
                   float16_cast_change=cast_change(current["cast"], following["cast"]))
        if not np.isfinite(row["actual_loss_decrease"]) or row["actual_loss_decrease"] <= 0:
            update()
            raise ValueError("predeclared descent gate failed at " + label + "; stop without further forwards")
        if index == 0:
            report["arrays"]["first_positive_latent"] = save_array(out, "first_positive_latent.npy", plus)
            event("forward_attempt", "first_positive")
            positive = backend.no_grad(plus, "first_positive", event)
            central = (positive["loss"] - following["loss"]) / (2 * RHO)
            dot = row["gradient_dot_direction"]
            row["directional_check"] = {"loss_positive": positive["loss"], "central_difference": central,
                "gradient_dot_direction": dot, "central_minus_gradient_dot": central - dot,
                "relative_magnitude_error_descriptive_only": abs(central - dot) / abs(dot),
                "positive_loss_minus_initial": positive["loss"] - current["loss"],
                "positive_loss_increased_observation_only": positive["loss"] > current["loss"],
                "positive_float16_cast_change": cast_change(current["cast"], positive["cast"]),
                "central_positive_gate": central > 0}
        row["accepted"] = (np.isfinite(row["actual_loss_decrease"]) and row["actual_loss_decrease"] > 0
                           and (index != 0 or row["directional_check"]["central_positive_gate"]))
        row["accepted"] = bool(row["accepted"])
        update()
        if not row["accepted"]:
            raise ValueError("predeclared directional/descent gate failed at " + label + "; no step-size search or continuation")
        source, current = minus, following
    report["protocol_complete"] = True
    update()


class NativeBackend:
    def __init__(self, torch, h3):
        from check_vae_decode_grad import decoder_math
        from h3_comfy import native_training_math
        from vae_checkpoint_context import checkpoint_decoder_blocks
        self.torch, self.h3, self.vae = torch, h3, h3.vae
        self.native_math, self.decoder_math, self.checkpoint_scope = native_training_math, decoder_math, checkpoint_decoder_blocks
        self.target = torch.linspace(.2, .8, 832, dtype=torch.float32)[None, None, :, None]

    def observe(self, pixels, source, cast):
        torch = self.torch
        finite = bool(torch.isfinite(pixels).all())
        loss = (pixels - self.target).square().mean() if tuple(pixels.shape) == RGB_SHAPE else None
        loss_value = float(loss.detach()) if loss is not None and bool(torch.isfinite(loss)) else None
        record = {"input_sha256": array_sha(source), "actual_cast_sha256": array_sha(cast),
                  "actual_cast_dtype": str(cast.dtype), "output_sha256": array_sha(pixels.detach().numpy()),
                  "output_shape": list(pixels.shape), "output_dtype": str(pixels.dtype),
                  "finite": finite, "loss": loss_value, "target_sha256": array_sha(self.target.numpy())}
        return loss, record

    def no_grad(self, source, label, event):
        torch = self.torch
        tensor = torch.from_numpy(source).to(device=self.h3.dev, dtype=self.vae.vae_dtype)
        cast = tensor.detach().cpu().numpy().copy()
        with self.native_math(), torch.no_grad():
            pixels = self.h3.decode(tensor)
        _, record = self.observe(pixels, source, cast)
        event("forward_completed", label, record)
        if not record["finite"] or record["loss"] is None or pixels.requires_grad:
            raise ValueError("invalid no-grad deployment reference")
        return {"pixels": pixels, "cast": cast, "loss": record["loss"]}

    def gradient(self, source, reference, label, event):
        torch, audit = self.torch, None
        latent = torch.from_numpy(source).to(device=self.h3.dev, dtype=torch.float32).requires_grad_(True)
        try:
            with torch.enable_grad(), self.decoder_math(self.native_math, "ascend_native"), self.checkpoint_scope(self.vae.first_stage_model, True) as audit:
                tensor = latent.to(dtype=self.vae.vae_dtype)
                cast = tensor.detach().cpu().numpy().copy()
                pixels = self.vae.decode(tensor).float().cpu()
                pixels = pixels[0] if pixels.ndim == 5 else pixels
                loss, record = self.observe(pixels, source, cast)
                shape_ok = tuple(pixels.shape) == RGB_SHAPE and pixels.shape == reference["pixels"].shape
                record["parity"] = {"cast_exact": array_sha(cast) == array_sha(reference["cast"]),
                    "requires_grad": pixels.requires_grad,
                    "bitwise_equal": shape_ok and bool(torch.equal(pixels.detach(), reference["pixels"])),
                    "max_abs_difference": float((pixels.detach() - reference["pixels"]).abs().max()) if shape_ok and record["finite"] else None,
                    "allclose": shape_ok and record["finite"] and bool(torch.allclose(pixels.detach(), reference["pixels"], atol=.0002, rtol=.0001))}
                event("forward_completed", label, record)
                if not record["parity"]["allclose"] or not record["parity"]["cast_exact"] or not pixels.requires_grad or loss is None or record["loss"] is None:
                    raise ValueError("direct decoder parity/gradient gate failed")
                event("backward_attempt", label)
                (loss * SCALE).backward()
                raw = None if latent.grad is None else latent.grad.detach().float().cpu().numpy().copy()
                event("backward_completed", label, raw)
        finally:
            if audit is not None:
                event("checkpoint", label, audit)
        if not checkpoint_matches(audit, "single"):
            raise ValueError("checkpoint execution/restoration gate failed")
        return raw


def execute(args):
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        raise ValueError("explicit single independently admitted physical NPU required")
    args.out.mkdir(parents=True, exist_ok=False)
    report, began = new_report(), time.monotonic()
    report["contract"] = contract()
    stage = h3 = torch = None
    before_versions = None
    def update():
        report["seconds"] = time.monotonic() - began
        atomic_json(args.out / "report.json", report)
    update()
    try:
        import torch as torch_module
        torch = torch_module
        from openh3_assets import read_manifest
        from check_vae_decode_grad import parameter_fingerprint
        from run_vae_roundtrip_diagnostic import load_vae_only
        import h3_comfy
        import comfy_kitchen.backends.ascend.rope
        manifest = read_manifest(args.pretrained_manifest, verify=False)
        report["contract"].update(manifest_sha256=sha_file(args.pretrained_manifest), pretrained_fingerprint=manifest["_fingerprint"])
        source = torch.randn(SHAPE, generator=torch.Generator(device="cpu").manual_seed(102), dtype=torch.float32).numpy()
        if array_sha(source) != INPUT_SHA:
            raise ValueError("seed102 input differs from predeclared CPU bytes")
        h3 = load_vae_only(manifest)
        vae, stage = h3.vae, h3.vae.first_stage_model
        if vae.vae_dtype != torch.float16:
            raise ValueError("fixed FP16 decoder required")
        vae.device = vae.offload_device = h3.dev
        vae.disable_offload = True
        stage.eval().requires_grad_(False).to(h3.dev)
        before_hash, before_versions = parameter_fingerprint(stage, torch)
        report["frozen_before"] = {"sha256": before_hash, "parameter_count": len(before_versions)}
        report["model"] = h3.record
        report["runtime_before"] = runtime_observation(torch, h3)
        torch.npu.reset_peak_memory_stats()
        update()
        run_protocol(NativeBackend(torch, h3), source, args.out, report, update)
    except BaseException as error:
        report["error"] = repr(error)
    finally:
        if stage is not None:
            try:
                torch.npu.synchronize()
                after_hash, after_versions = parameter_fingerprint(stage, torch)
                report["frozen_after"] = {"sha256": after_hash, "parameter_count": len(after_versions)}
                report["frozen_versions_unchanged"] = before_versions == after_versions
                report["runtime_after"] = runtime_observation(torch, h3)
                report["memory"] = {"peak_allocated_bytes": torch.npu.max_memory_allocated(), "peak_reserved_bytes": torch.npu.max_memory_reserved()}
            except BaseException as error:
                report["after_observation_error"] = repr(error)
        report["gates"] = {"protocol": report["protocol_complete"] and report["error"] is None,
            "frozen": report.get("frozen_before") is not None and report.get("frozen_before") == report.get("frozen_after") and report.get("frozen_versions_unchanged", False),
            "runtime": report.get("runtime_before") is not None and report.get("runtime_before") == report.get("runtime_after"),
            "budget": report["forward_attempts"] == report["forwards_completed"] == 8 and report["backward_attempts"] == report["backwards_completed"] == 3,
            "restoration": len(report["checkpoint"]) == 3 and all(checkpoint_matches(a, "single") for a in report["checkpoint"].values())}
        report["complete"] = all(report["gates"].values())
        report["phase"] = "completed_local_descent" if report["complete"] else "stopped_evidence_retained"
        update()
    print(json.dumps({"complete": report["complete"], "report": str(args.out / "report.json"), "error": report["error"]}))
    return 0 if report["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError(args.out)
    if not args.execute:
        print(json.dumps({"inspection_only": True, "contract": contract(), "NPU_calls": 0, "weights_loaded": False}, indent=2))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
