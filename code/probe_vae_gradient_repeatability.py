"""Bounded fresh-process VAE gradient diagnostic; no DiT/optimizer/training.

A1/A2 use the identical --mode single path; --label is metadata only. The
paired_history mode requires a revalidated exact A1/A2 admission. Execution is
opt-in, on one independently admitted card, in a new output directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

import numpy as np

from compare_vae_gradient_repeatability import (
    EXPECTED_INPUT_SHA, array_sha, array_statistics, atomic_json, checkpoint_matches,
    save_array, sha_file, validate_admission,
)


SHAPE = (1, 24, 7, 30, 52)
SCALE = 1024.0


def implementation_hashes():
    code = Path(__file__).resolve().parent
    names = ("probe_vae_gradient_repeatability.py", "compare_vae_gradient_repeatability.py",
             "check_vae_decode_grad.py", "vae_checkpoint_context.py", "h3_comfy.py",
             "run_vae_roundtrip_diagnostic.py", "openh3_assets.py", "eval_rphi.py",
             "export_rphi_guides.py", "export_rphi_guides_stage2.py", "lora.py",
             "openh3_runtime.py", "pixel_consistency.py", "restoration_metrics.py",
             "rphi_condition_v2.py", "rphi_data.py", "rphi_h3_controls.py",
             "rphi_model.py", "rphi_soft_anchor.py", "run_fixed_h3_rphi.py")
    return {name: sha_file(code / name) for name in names}


def runtime_observation(torch, h3):
    """Read-only flags. Avoid torch_npu.npu._get_deterministic_level (it mutates)."""
    import torch_npu
    native_getter = getattr(torch_npu._C, "_npu_get_deterministic_level", None)
    level = native_getter() if native_getter is not None else None
    modules = {}
    for name in ("comfy.ldm.minimax.vae", "comfy.ldm.modules.attention", "comfy.ops",
                 "comfy.model_management", "comfy_kitchen.backends.ascend.rope"):
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__file__", None):
            modules[name] = {"path": str(Path(module.__file__).resolve()), "sha256": sha_file(module.__file__)}
    return {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "native_deterministic_level_read_only_private_getter": level,
            "matmul_allow_hf32": bool(torch.npu.matmul.allow_hf32),
            "conv_allow_hf32": bool(torch.npu.conv.allow_hf32),
            "jit_compile_false": bool(torch.npu.is_jit_compile_false()),
            "mm_bmm_format_nd": bool(torch.npu.get_mm_bmm_format_nd()),
            "vae_dtype": str(h3.vae.vae_dtype), "device": str(h3.dev),
            "environment": {k: os.environ.get(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_HOME_PATH", "CANN", "ASCEND_OPP_PATH")},
            "imported_source": modules}


def run_backwards(loss, latent, torch, paired, persist):
    """Same leaf-grad clearing and scale for both modes; persist each observation."""
    latent.grad = None
    if paired:
        loss.backward(retain_graph=True)
        persist("prior_unscaled", latent.grad)
        latent.grad = None
    (loss * SCALE).backward()
    persist("scaled_raw", latent.grad)


def execute(args):
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError(args.out)
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if not visible.isdecimal(): raise ValueError("explicit single admitted physical NPU required")
    admission = validate_admission(args.admission) if args.mode == "paired_history" else None
    args.out.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    report = {"schema": "openh3.vae_gradient_repeatability_probe.v1", "complete": False,
              "observations_complete": False, "execution_id": str(uuid.uuid4()),
              "pid": os.getpid(), "label": args.label, "mode": args.mode,
              "phase": "initializing", "actual_completed_forwards": 0,
              "actual_completed_backwards": 0, "arrays": {}, "gradient_statistics": {},
              "frozen": {}, "admission": None if admission is None else {
                  "path": admission["path"], "sha256": admission["sha256"],
                  "a1": admission["comparison"]["a1"], "a2": admission["comparison"]["a2"]},
              "gates": {}, "checkpoint": {}, "error": None}
    report_path = args.out / "report.json"
    stage = h3 = torch = None
    before_versions = None
    checkpoint_audit = None
    error = None

    def update():
        report["seconds"] = time.monotonic() - began
        atomic_json(report_path, report)

    update()
    try:
        import torch as torch_module
        torch = torch_module
        from openh3_assets import read_manifest
        manifest = read_manifest(args.pretrained_manifest, verify=False)
        # A1/A2 execute this same generation path; labels never touch the RNG.
        source = torch.randn(SHAPE, generator=torch.Generator(device="cpu").manual_seed(101), dtype=torch.float32)
        if array_sha(source.numpy()) != EXPECTED_INPUT_SHA:
            raise ValueError("CPU seed101 input differs from predeclared historical bytes")
        if args.input_latent is not None:
            array = np.load(args.input_latent, allow_pickle=False)
            if array.dtype != np.float32 or array.shape != SHAPE or not np.isfinite(array).all():
                raise ValueError("input latent must be finite float32 with the fixed full shape")
            if array_sha(array) != array_sha(source.numpy()):
                raise ValueError("optional input artifact differs from regenerated seed101 input")
        source_array = source.numpy()
        report["arrays"]["input"] = save_array(args.out, "latent_input.npy", source_array)
        report["common_contract"] = {
            "implementation_sha256": implementation_hashes(),
            "manifest_sha256": sha_file(args.pretrained_manifest), "pretrained_fingerprint": manifest["_fingerprint"],
            "seed": 101, "input_shape": list(SHAPE), "input_dtype": "float32",
            "input_array_sha256": array_sha(source_array), "RGB_shape": [22, 480, 832, 3],
            "expected_seed101_input_sha256": EXPECTED_INPUT_SHA,
            "input_generation": "CPU torch.Generator seed101 float32 randn, identical path in all modes",
            "loss_scale": SCALE, "target": "torch float32 linspace(.2,.8,832)[None,None,:,None], broadcast to THWC",
            "loss": "CPU float32 mean((as_returned_internal_clipped_decoder_RGB-target)**2)",
            "grad_rope_mode": "ascend_native", "checkpoint_blocks": 36, "use_reentrant": False,
            "forward_atol": 0.0002, "forward_rtol": 0.0001,
            "optimizer_steps": 0, "H3_denoiser_calls": 0,
            "scope": "synthetic VAE input-gradient repeatability only; no restoration or joint H3 acceptance"}
        if admission is not None and report["common_contract"] != admission["comparison"]["common_contract"]:
            raise ValueError("paired input/runtime code contract differs from admitted A1/A2")
        report["phase"] = "loading_verified_vae"
        update()
        from check_vae_decode_grad import decoder_math, parameter_fingerprint
        from h3_comfy import native_training_math
        from run_vae_roundtrip_diagnostic import load_vae_only
        from vae_checkpoint_context import checkpoint_decoder_blocks
        # Ensure the exact native functional module is imported before both
        # runtime snapshots; importing it does not change deterministic flags.
        import comfy_kitchen.backends.ascend.rope
        h3 = load_vae_only(manifest)
        vae, stage = h3.vae, h3.vae.first_stage_model
        vae.device = vae.offload_device = h3.dev
        vae.disable_offload = True
        stage.eval().requires_grad_(False).to(h3.dev)
        before_hash, before_versions = parameter_fingerprint(stage, torch)
        report["frozen"]["before"] = {"sha256": before_hash, "parameter_count": len(before_versions)}
        report["model"] = h3.record
        report["runtime_before"] = runtime_observation(torch, h3)
        if admission is not None:
            update()
            if (report["runtime_before"] != admission["comparison"]["runtime_before"]
                    or report["frozen"]["before"] != admission["comparison"]["frozen_before"]):
                raise ValueError("paired runtime or loaded weights differ from admitted A1/A2; stopped before forward")
        report["phase"] = "reference_forward"
        torch.npu.reset_peak_memory_stats()
        update()
        with native_training_math(), torch.no_grad():
            reference = h3.decode(source.to(device=h3.dev, dtype=vae.vae_dtype))
        report["actual_completed_forwards"] += 1
        report["reference"] = {"shape": list(reference.shape), "dtype": str(reference.dtype),
                               "array_sha256": array_sha(reference.detach().float().cpu().numpy())}
        report["phase"] = "gradient_forward"
        update()
        latent = source.to(device=h3.dev, dtype=torch.float32).requires_grad_(True)
        with torch.enable_grad(), decoder_math(native_training_math, "ascend_native"), checkpoint_decoder_blocks(stage, True) as checkpoint_audit:
            pixels = vae.decode(latent.to(dtype=vae.vae_dtype)).float().cpu()
            pixels = pixels[0] if pixels.ndim == 5 else pixels
            report["actual_completed_forwards"] += 1
            target = torch.linspace(.2, .8, 832, dtype=torch.float32)[None, None, :, None]
            shape_ok = list(pixels.shape) == [22, 480, 832, 3] and pixels.shape == reference.shape
            finite = bool(torch.isfinite(pixels).all() and torch.isfinite(reference).all())
            delta = pixels.detach() - reference if shape_ok else None
            report["forward"] = {
                "shape": list(pixels.shape), "dtype": str(pixels.dtype), "finite": finite,
                "requires_grad": pixels.requires_grad, "shape_ok": shape_ok,
                "reference_sha256": report["reference"]["array_sha256"],
                "prediction_sha256": array_sha(pixels.detach().numpy()),
                "target_sha256": array_sha(target.numpy()),
                "bitwise_equal": shape_ok and bool(torch.equal(pixels.detach(), reference)),
                "max_abs_difference": float(delta.abs().max()) if shape_ok and finite else None,
                "allclose": shape_ok and finite and bool(torch.allclose(pixels.detach(), reference, atol=.0002, rtol=.0001))}
            report["phase"] = "forward_observed"
            update()
            if not report["forward"]["allclose"] or not pixels.requires_grad:
                raise ValueError("forward numerical/gradient gate failed; evidence retained")
            loss = (pixels - target).square().mean()
            report["synthetic_loss"] = float(loss.detach())
            if not np.isfinite(report["synthetic_loss"]):
                report["synthetic_loss"] = None
                raise ValueError("synthetic loss is nonfinite")

            def persist(name, gradient):
                report["actual_completed_backwards"] += 1
                if gradient is None:
                    report["gradient_statistics"][name] = {"present": False}
                    update()
                    raise ValueError("input gradient missing")
                array = gradient.detach().float().cpu().numpy().copy()
                report["arrays"][name] = save_array(args.out, name + ".npy", array)
                report["gradient_statistics"][name] = array_statistics(array)
                if name == "scaled_raw":
                    unscaled = np.ascontiguousarray(array / np.float32(SCALE), dtype=np.float32)
                    report["arrays"]["scaled_unscaled"] = save_array(args.out, "scaled_unscaled.npy", unscaled)
                    report["gradient_statistics"]["scaled_unscaled"] = array_statistics(unscaled)
                report["phase"] = "saved_" + name
                update()

            run_backwards(loss, latent, torch, args.mode == "paired_history", persist)
        report["observations_complete"] = True
    except BaseException as caught:
        error = caught
        report["error"] = repr(caught)
    finally:
        # This executes before *any* exact-gradient comparison verdict. A failed
        # observation must not discard the after-fingerprint or restore audit.
        if checkpoint_audit is not None:
            report["checkpoint"] = checkpoint_audit
        if stage is not None:
            try:
                torch.npu.synchronize()
                after_hash, after_versions = parameter_fingerprint(stage, torch)
                report["frozen"]["after"] = {"sha256": after_hash, "parameter_count": len(after_versions)}
                report["frozen"]["versions_unchanged"] = before_versions == after_versions
            except BaseException as caught:
                report["frozen"]["after_error"] = repr(caught)
            try:
                report["runtime_after"] = runtime_observation(torch, h3)
                report["memory"] = {"peak_allocated_bytes": torch.npu.max_memory_allocated(),
                                    "peak_reserved_bytes": torch.npu.max_memory_reserved()}
            except BaseException as caught:
                report["runtime_after_error"] = repr(caught)
        report["gates"] = {
            "observation_completed": report["observations_complete"] and error is None,
            "forward_parity": report.get("forward", {}).get("allclose", False),
            "finite_nonzero_scaled_gradient": report.get("gradient_statistics", {}).get("scaled_raw", {}).get("finite_count") == int(np.prod(SHAPE)) and report.get("gradient_statistics", {}).get("scaled_raw", {}).get("nonzero_count", 0) > 0,
            "frozen_hash_unchanged": report["frozen"].get("before", {}).get("sha256") is not None and report["frozen"].get("before", {}).get("sha256") == report["frozen"].get("after", {}).get("sha256"),
            "frozen_versions_unchanged": report["frozen"].get("versions_unchanged", False),
            "runtime_unchanged": report.get("runtime_before") is not None and report.get("runtime_before") == report.get("runtime_after"),
            "checkpoint_restored": report["checkpoint"].get("restored", False),
            "checkpoint_execution": checkpoint_matches(report["checkpoint"], args.mode),
            "bounded_calls": report["actual_completed_forwards"] == 2 and report["actual_completed_backwards"] == (1 if args.mode == "single" else 2),
        }
        report["complete"] = all(report["gates"].values())
        report["phase"] = "completed_observation" if report["complete"] else "failed_observation_evidence_retained"
        update()
    print(json.dumps({"complete": report["complete"], "mode": args.mode, "report": str(report_path), "error": report["error"]}))
    return 0 if report["complete"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("single", "paired_history"), default="single")
    parser.add_argument("--label", default="unlabeled", help="metadata only; does not alter execution")
    parser.add_argument("--input-latent", type=Path)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if (args.mode == "paired_history") != (args.admission is not None):
        parser.error("only paired_history mode requires --admission")
    if args.out.exists() or args.out.is_symlink(): raise FileExistsError(args.out)
    if not args.execute:
        print(json.dumps({"inspection_only": True, "mode": args.mode, "label_metadata_only": args.label,
                          "shape": list(SHAPE), "scale": SCALE, "planned_forwards": 2,
                          "planned_backwards": 1 if args.mode == "single" else 2,
                          "actual_asset_verification": False, "NPU_calls": 0,
                          "implementation_sha256": implementation_hashes()}, indent=2))
        return 0
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
