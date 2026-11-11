"""Bounded synthetic same-graph H3 -> frozen VAE RGB gradient probe.

No optimizer, dataset, quality claim or launch policy. The external admitted
queue owns resource checks/locks/timeout. Inspection imports no torch/models.
All new guards are scoped; no existing Comfy/H3 source is edited.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

from check_h3_trajectory_grad import (MANIFEST_SHA, SEED, SHAPE, atomic_json,
    sha_file, tensor_sha, tensor_record, canonical_base_items, parity_metrics,
    adapter_audit, observe_noise, memory_record, runtime_imports, verify_pins)

SOURCE_CONTRACT_SHA = "1d9acc5451dce4f0454832ff86dffc98eb85f8ed07c9d13c73b776367f560cc7"
LOSS_SCALE = 1024.0
RGB_SHAPE = (5, 128, 128, 3)
ATOL, RTOL = .0002, .0001


class Evidence:
    def __init__(self, path, contract):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path / "arrays").mkdir()
        self.started = time.monotonic()
        self.data = {"schema": "openh3.joint_rgb_probe.v1", "complete": False,
            "contract": contract, "phase": "created", "events": [], "arrays": {},
            "actual": {"reference_NFE": 0, "candidate_NFE": 0,
                "reference_VAE_decode_calls": 0, "candidate_VAE_decode_calls": 0,
                "stage_decode_calls": {}, "forbidden_calls": [], "RGB_loss_expressions": 0,
                "backward_attempts": 0, "completed_backward_calls": 0,
                "optimizer_steps": 0, "adapter_updates": 0},
            "block_forward_calls": {"DiT": {}, "VAE": {}},
            "math_observations": [], "decoder_context_restorations": [],
            "noise_observations": [], "state_checks": {}}
        self.save()

    def save(self):
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.path / "partial.json", self.data)

    def phase(self, label, torch=None):
        self.data["phase"] = label
        event = {"phase": label, "elapsed_seconds": time.monotonic() - self.started,
                 "max_rss_KiB_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        status = Path("/proc/self/status")
        if status.exists():
            event["rss"] = {line.split(":", 1)[0]: line.split(":", 1)[1].strip()
                            for line in status.read_text().splitlines()
                            if line.startswith(("VmRSS:", "VmHWM:"))}
        if torch is not None:
            event["memory"] = memory_record(torch)
        self.data["events"].append(event)
        self.save()
        print(json.dumps(event), flush=True)


def require(value, message, evidence):
    evidence.save()
    if not value:
        raise RuntimeError(message)


def save_array(evidence, name, value, torch):
    """Preserve actual dtype. All this probe's persisted arrays are FP16/FP32."""
    import numpy as np
    array = value.detach().cpu().contiguous().numpy()
    path = evidence.path / "arrays" / (name + ".npy")
    np.save(path, array, allow_pickle=False)
    row = {"path": str(path.relative_to(evidence.path)), "file_sha256": sha_file(path),
           "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
           "dtype": str(array.dtype), "shape": list(array.shape), "role": name}
    evidence.data["arrays"][name] = row
    evidence.save()
    return row


def fixed_target(torch):
    return torch.linspace(.2, .8, 128, dtype=torch.float32, device="cpu")[None, None, :, None]


def frozen_inventory(model, torch, adapted_names=()):
    """Separate named parameters and ALL registered buffers, including nonpersistent.

    Does not claim to inventory unregistered tensors or Python caches.
    """
    groups = {"parameters": [], "registered_buffers": []}
    for canonical, value in canonical_base_items(model, adapted_names):
        kind, name = canonical.split(":", 1)
        row = {"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
               "sha256_bytes": tensor_sha(value, torch), "device": str(value.device),
               "version": value._version, "requires_grad": value.requires_grad,
               "grad_is_none": value.grad is None}
        groups["parameters" if kind == "parameter" else "registered_buffers"].append(row)
    for key in ("parameters", "registered_buffers"):
        # Object/version/device are lifecycle facts, not the canonical byte identity.
        content = [{k: r[k] for k in ("name", "shape", "dtype", "sha256_bytes")} for r in groups[key]]
        groups[key + "_sha256"] = hashlib.sha256(json.dumps(content, sort_keys=True,
                                                      separators=(",", ":")).encode()).hexdigest()
    groups["scope"] = "named parameters and separately named registered buffers; no untracked Python caches"
    return groups


def freeze_snapshot(model, torch, adapted_names=()):
    return {name: {"object": value, "version": value._version,
                   "device": str(value.device), "sha256_bytes": tensor_sha(value, torch)}
            for name, value in canonical_base_items(model, adapted_names)}


def check_snapshot(model, snapshot, torch, adapted_names=(), device=None):
    actual = dict(canonical_base_items(model, adapted_names))
    failures = []
    if actual.keys() != snapshot.keys():
        failures.append("named parameter/buffer set changed")
    for name, prior in snapshot.items():
        value = actual.get(name)
        if value is None:
            continue
        if value is not prior["object"]: failures.append(name + ": object replaced")
        if value._version != prior["version"]: failures.append(name + ": version changed")
        if str(value.device) != prior["device"] or (device is not None and value.device != torch.device(device)):
            failures.append(name + ": device changed/not resident")
        if value.requires_grad or value.grad is not None: failures.append(name + ": frozen/gradient violation")
        if tensor_sha(value, torch) != prior["sha256_bytes"]: failures.append(name + ": bytes changed")
    return {"passed": not failures, "failures": failures, "tensors_checked": len(snapshot),
            "checks": ["canonical name", "object", "version", "device", "bytes", "frozen", "grad None"]}


def adapter_snapshot(model, torch):
    return {n: {"object": p, "version": p._version, "device": str(p.device),
                "sha256_bytes": tensor_sha(p, torch)} for n, p in model.named_parameters()
            if n.endswith((".lora_a", ".lora_b"))}


def check_adapters(model, snapshot, torch, device):
    current = {n: p for n, p in model.named_parameters() if n.endswith((".lora_a", ".lora_b"))}
    failures = []
    if current.keys() != snapshot.keys(): failures.append("adapter names changed")
    for n, prior in snapshot.items():
        p = current.get(n)
        if p is None: continue
        if (p is not prior["object"] or p._version != prior["version"]
                or str(p.device) != prior["device"] or p.device != torch.device(device)
                or not p.requires_grad or p.dtype != torch.float32
                or tensor_sha(p, torch) != prior["sha256_bytes"]):
            failures.append(n)
    return {"passed": not failures, "failures": failures, "tensors_checked": len(snapshot)}


def gradient_stats(value, torch):
    if value is None: return {"present": False}
    g = value.detach().float().cpu()
    finite = torch.isfinite(g)
    valid = g[finite]
    return {"present": True, "elements": g.numel(), "finite_count": int(finite.sum()),
            "all_finite": bool(finite.all()), "nonzero_finite_count": int(torch.count_nonzero(valid)),
            "finite_l2": float(valid.double().norm()), "nan_count": int(torch.isnan(g).sum()),
            "inf_count": int(torch.isinf(g).sum())}


def save_gradients(dit, video, evidence, torch, expected=400):
    rows, b_squared, b_nonzero = {}, 0., 0
    for name, p in dit.named_parameters():
        if not name.endswith((".lora_a", ".lora_b")): continue
        stats = gradient_stats(p.grad, torch)
        row = {"statistics_raw": stats, "present": stats["present"]}
        if p.grad is not None:
            raw = p.grad.detach().float().cpu()
            row["raw"] = save_array(evidence, "gradient_raw__" + name, raw, torch)
            row["unscaled"] = save_array(evidence, "gradient_unscaled__" + name, raw / LOSS_SCALE, torch)
            row["statistics_unscaled"] = gradient_stats(raw / LOSS_SCALE, torch)
            if name.endswith(".lora_b"):
                b_squared += stats["finite_l2"] ** 2
                b_nonzero += stats["nonzero_finite_count"] > 0
        rows[name] = row
    present = sum(r["present"] for r in rows.values())
    finite = all(r["present"] and r["statistics_raw"]["all_finite"] for r in rows.values())
    evidence.data["adapter_gradients"] = {"tensors": rows, "expected": expected,
        "present_tensors": present, "all_finite": finite, "B_l2_raw": math.sqrt(b_squared),
        "B_l2_unscaled": math.sqrt(b_squared) / LOSS_SCALE, "B_nonzero_tensors": b_nonzero,
        "passed": len(rows) == expected and present == expected and finite and b_squared > 0 and b_nonzero > 0,
        "A_zero_is_legal": True, "every_B_nonzero_not_required": True}
    stats = gradient_stats(None if video is None else video.grad, torch)
    row = {"statistics_raw": stats, "passed": stats["present"] and stats["all_finite"]
           and stats["nonzero_finite_count"] > 0}
    if stats["present"]:
        raw = video.grad.detach().float().cpu()
        row["raw"] = save_array(evidence, "video_gradient_raw", raw, torch)
        row["unscaled"] = save_array(evidence, "video_gradient_unscaled", raw / LOSS_SCALE, torch)
        row["statistics_unscaled"] = gradient_stats(raw / LOSS_SCALE, torch)
    evidence.data["retained_video_gradient"] = row
    evidence.save()


def save_latent_parity(reference, candidate, evidence, torch):
    for arm, values in (("reference", reference), ("candidate", candidate)):
        for stream in ("video", "audio"):
            save_array(evidence, arm + "_" + stream, values[stream].float(), torch)
    evidence.data["latent_parity"] = {s: parity_metrics(reference[s], candidate[s], torch)
                                      for s in ("video", "audio")}
    require(all(row["torch_equal"] for row in evidence.data["latent_parity"].values()),
            "strict video/audio parity failed; decode/backward forbidden", evidence)


def rgb_backward(reference_rgb, rgb, video, evidence, torch, expected_shape=RGB_SHAPE):
    """Exactly one RGB-only loss/backward; must be inside retained session."""
    save_array(evidence, "reference_rgb", reference_rgb.float(), torch)
    save_array(evidence, "candidate_rgb", rgb.float(), torch)
    metrics = parity_metrics(reference_rgb, rgb, torch)
    metrics.update({"atol": ATOL, "rtol": RTOL,
        "allclose": metrics["same_shape"] and metrics["finite"] and bool(torch.allclose(
            rgb.detach().float().cpu(), reference_rgb.detach().float().cpu(), atol=ATOL, rtol=RTOL)),
        "acceptance": "unchanged VAE allclose atol=.0002 rtol=.0001; torch.equal measured"})
    evidence.data["rgb_parity"] = metrics
    evidence.data["graph"] = {"video_requires_grad": video.requires_grad,
        "video_has_grad_fn": video.grad_fn is not None, "rgb_requires_grad": rgb.requires_grad,
        "rgb_has_grad_fn": rgb.grad_fn is not None}
    require(tuple(rgb.shape) == tuple(expected_shape) and metrics["allclose"],
            "RGB geometry/parity failed; backward forbidden", evidence)
    require(all(evidence.data["graph"].values()), "joint retained graph absent", evidence)
    target = fixed_target(torch)
    save_array(evidence, "target", target, torch)
    require(rgb.shape[-2] == 128, "target width is frozen at128", evidence)
    pixels = rgb.float().cpu()  # differentiable device copy, never detach the loss
    evidence.data["actual"]["RGB_loss_expressions"] += 1
    loss = (pixels - target).square().mean()
    import numpy as np
    evidence.data["loss"] = {"torch_float32_mse": float(loss.detach()) if bool(torch.isfinite(loss)) else None,
        "numpy_float64_mse": float(np.mean((pixels.detach().numpy().astype(np.float64) -
                                              target.numpy().astype(np.float64)) ** 2)),
        "scale": LOSS_SCALE, "recipe": "mean((candidate_RGB.float().cpu()-target)**2)",
        "target_recipe": "torch.linspace(.2,.8,128,dtype=float32,device=cpu)[None,None,:,None]",
        "RGB_semantics": "as_returned_internal_clipped; no extra clipping/quantization",
        "auxiliary_latent_audio_loss": False}
    require(bool(torch.isfinite(loss)) and loss.requires_grad, "RGB loss nonfinite/disconnected", evidence)
    evidence.phase("backward", torch if hasattr(torch, "npu") else None)
    evidence.data["actual"]["backward_attempts"] += 1
    require(evidence.data["actual"]["backward_attempts"] == 1, "second backward forbidden", evidence)
    (loss * LOSS_SCALE).backward()
    evidence.data["actual"]["completed_backward_calls"] += 1
    evidence.save()


@contextmanager
def override_method(owner, name, replacement):
    had = name in vars(owner)
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        if had: setattr(owner, name, original)
        else: delattr(owner, name)


@contextmanager
def guarded_vae(h3, evidence, torch):
    """No sd.VAE OOM fallback; preserve first_stage_model native tiling.

    Real stage.decode counted once per reference/candidate; block recomputation
    does not call stage.decode again. Every override restores on failure.
    """
    vae, stage = h3.vae, h3.vae.first_stage_model
    with ExitStack() as scope:
        for owner, methods in ((h3, ("encode", "encode_guide", "encode_guide_pixels", "ref_block")),
                               (vae, ("encode", "decode_tiled_1d", "_decode_tiled_owned", "decode_tiled_", "decode_tiled_3d"))):
            for name in methods:
                if not hasattr(owner, name): continue
                def forbidden(*args, _name=name, **kwargs):
                    evidence.data["actual"]["forbidden_calls"].append(_name)
                    evidence.save()
                    raise RuntimeError("probe forbids encode/OOM fallback: " + _name)
                scope.enter_context(override_method(owner, name, forbidden))
        original_decode = vae.decode
        def wrapper_decode(value, *args, **kwargs):
            phase = evidence.data["phase"]
            if phase not in ("reference_decode", "candidate_decode"):
                raise RuntimeError("unexpected VAE decode phase: " + phase)
            arm = phase.split("_")[0]
            key = arm + "_VAE_decode_calls"
            evidence.data["actual"][key] += 1
            save_array(evidence, arm + "_decoder_input", value, torch)
            require(evidence.data["actual"][key] == 1, "second VAE decode forbidden", evidence)
            return original_decode(value, *args, **kwargs)
        scope.enter_context(override_method(vae, "decode", wrapper_decode))
        original_stage_decode = stage.decode
        def stage_decode(*args, **kwargs):
            phase = evidence.data["phase"]
            counts = evidence.data["actual"]["stage_decode_calls"]
            counts[phase] = counts.get(phase, 0) + 1
            require(phase in ("reference_decode", "candidate_decode") and counts[phase] == 1,
                    "second whole stage.decode/retry forbidden", evidence)
            return original_stage_decode(*args, **kwargs)
        scope.enter_context(override_method(stage, "decode", stage_decode))
        yield


@contextmanager
def counted_blocks(blocks, group, evidence, observe_math):
    """Captured bound-forward wrapper executes inside checkpoint context_fn."""
    with ExitStack() as scope:
        for i, block in enumerate(blocks):
            original = block.forward
            def wrapped(*args, _i=i, _original=original, **kwargs):
                phase = evidence.data["phase"]
                counts = evidence.data["block_forward_calls"][group].setdefault(phase, {})
                counts[str(_i)] = counts.get(str(_i), 0) + 1
                observe_math(group, phase, _i)
                return _original(*args, **kwargs)
            scope.enter_context(override_method(block, "forward", wrapped))
        yield


def function_id(fn):
    return {"id": id(fn), "module": getattr(fn, "__module__", None),
            "qualname": getattr(fn, "__qualname__", None)}


def actual_math_state(torch):
    import comfy.model_management as mm
    import comfy.quant_ops
    import comfy.ldm.minimax.model as implementation
    return {"in_training": bool(mm.in_training),
        "rope": function_id(comfy.quant_ops.ck.rms_rope_split_half),
        "scale_shift": function_id(implementation._mod_scale_shift),
        "gate": function_id(implementation._mod_gate),
        "jit_compile_false": bool(torch.npu.is_jit_compile_false()),
        "matmul_allow_hf32": bool(torch.npu.matmul.allow_hf32),
        "conv_allow_hf32": bool(torch.npu.conv.allow_hf32),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled()}


def wrapper_snapshot(objects):
    return [(owner, name, getattr(owner, name), name in vars(owner)) for owner, name in objects]


def wrappers_restored(snapshot):
    return all(getattr(owner, name) == original and (name in vars(owner)) == had
               for owner, name, original, had in snapshot)


def restore_patcher_state(model, trajectory, saved_patcher, saved_shapes):
    """Audit session restoration before repairing the probe's own entry state."""
    initialized = trajectory is not None and all(hasattr(trajectory, n)
        for n in ("saved_patcher", "had_shapes", "saved_shapes"))
    row = {"session_state_initialized": initialized,
        "session_current_patcher_restored": (getattr(model, "current_patcher", None) is trajectory.saved_patcher)
                                            if initialized else None,
        "session_shapes_restored": (hasattr(model, "latent_shapes") == trajectory.had_shapes and
            (not trajectory.had_shapes or getattr(model, "latent_shapes") == trajectory.saved_shapes)) if initialized else None,
        "session_scope": "actual saved session entry state" if initialized else "not applicable: session entry did not initialize state",
        "whole_run_had_shapes": saved_shapes[0]}
    model.current_patcher = saved_patcher
    if saved_shapes[0]: model.latent_shapes = saved_shapes[1]
    elif hasattr(model, "latent_shapes"): del model.latent_shapes
    row["whole_run_current_patcher_restored"] = getattr(model, "current_patcher", None) is saved_patcher
    row["whole_run_shapes_restored"] = (hasattr(model, "latent_shapes") == saved_shapes[0] and
        (not saved_shapes[0] or model.latent_shapes == saved_shapes[1]))
    return row


def execute(args, contract, evidence):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    from h3_comfy import H3, native_training_math
    from check_vae_decode_grad import decoder_math
    from openh3_assets import NATIVE_SIGMAS
    from openh3_runtime import inject_adaptation
    from openh3_train_trajectory import ResidentH3Trajectory
    import comfy.sample
    import comfy.utils
    from comfy_kitchen.backends.ascend.rope import rms_rope_split_half as native_rope
    evidence.data["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "python": sys.version, "visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "synthetic_no_dataset": True}
    evidence.phase("load_and_verify_assets", torch)
    h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS,
            adaptation_path=None, lora_path=None)
    dit, vae = h3.patcher.model.diffusion_model, h3.vae
    stage = vae.first_stage_model
    evidence.data["pretrained"] = h3.record
    evidence.data["asset_bytes_sha_verified"] = True
    dit.eval().requires_grad_(False).to(h3.dev)
    stage.eval().requires_grad_(False).to(h3.dev)
    vae.device = vae.offload_device = h3.dev
    vae.disable_offload = True
    require(vae.vae_dtype == torch.float16, "bound VAE must be FP16", evidence)
    preflight = ResidentH3Trajectory(h3, checkpoint_blocks=True, checkpoint_decoder=True)
    preflight._validate_resident()
    preflight._validate_vae()
    evidence.data["actual_imports_before_sampling"] = runtime_imports(evidence.data["verified_runtime_files"])
    require(evidence.data["actual_imports_before_sampling"]["passed"], "unbound runtime import", evidence)
    initial_math = actual_math_state(torch)
    evidence.data["math_before"] = initial_math
    require(initial_math["jit_compile_false"] and not initial_math["matmul_allow_hf32"]
            and not initial_math["conv_allow_hf32"], "undeclared JIT/HF32 mode", evidence)
    require(len(dit.blocks) == 50 and len(stage.decoder.transformer_blocks) == 36, "wrong block inventory", evidence)
    evidence.data["vae_native_plan"] = {"tiling": stage.tiling, "tile_size": stage.tile_size,
        "vae_ratio": stage.vae_ratio, "vae_ratio_t": stage.vae_ratio_t,
        "temporal_padding_and_chunks": list(stage._decode_temporal_chunks(2)),
        "height_tile_plan": stage.split_tiles(128), "width_tile_plan": stage.split_tiles(128),
        "output_shape": list(stage.decode_output_shape(SHAPE)), "changed": False}
    torch.npu.reset_peak_memory_stats()
    evidence.phase("initial_resident_fingerprints", torch)
    baseline = {"DiT": frozen_inventory(dit, torch), "VAE": frozen_inventory(stage, torch)}
    atomic_json(evidence.path / "frozen_initial.json", baseline)
    evidence.data["frozen_initial"] = {"path": "frozen_initial.json", "sha256": sha_file(evidence.path / "frozen_initial.json")}
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    guide = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    raw = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    save_array(evidence, "guide", guide, torch)
    save_array(evidence, "raw_warm_start", raw, torch)
    evidence.data["inputs"] = {"seed": SEED, "audio_init": "zeros", "context_sha256": contract["context_sha256"],
        "distinct": not bool(torch.equal(guide, raw)), "target_is_never_model_input": True}
    require(evidence.data["inputs"]["distinct"], "guide/raw must differ", evidence)
    math_expected = {}
    def observe_math(group, phase, index):
        current = actual_math_state(torch)
        evidence.data["math_observations"].append({"group": group, "phase": phase, "block": index, "state": current})
        if not current["in_training"]: raise RuntimeError("block escaped training math")
        for flag in ("jit_compile_false", "matmul_allow_hf32", "conv_allow_hf32",
                     "deterministic_algorithms", "deterministic_warn_only"):
            if current[flag] != initial_math[flag]: raise RuntimeError("block runtime flag changed: " + flag)
        if group == "VAE" and phase != "reference_decode":
            if current["rope"] != function_id(native_rope): raise RuntimeError("VAE not in native decoder math")
        else:
            key = "DiT_active" if phase in ("candidate", "backward") else group + ":" + phase
            if key not in math_expected: math_expected[key] = current
            if current != math_expected[key]: raise RuntimeError("block math/flags changed: " + key)
    @contextmanager
    def decoder_context():
        before = actual_math_state(torch)
        try:
            with decoder_math(native_training_math, "ascend_native"):
                yield
        finally:
            after = actual_math_state(torch)
            row = {"phase": evidence.data["phase"], "before": before, "after": after, "restored": before == after}
            evidence.data["decoder_context_restorations"].append(row)
            if before != after: raise RuntimeError("decoder context leaked into DiT")
    phase = lambda: evidence.data["phase"]
    original_methods = [(b, "forward") for b in list(dit.blocks) + list(stage.decoder.transformer_blocks)]
    original_methods += [(vae, n) for n in ("decode", "encode", "decode_tiled_1d", "_decode_tiled_owned", "decode_tiled_", "decode_tiled_3d") if hasattr(vae, n)]
    original_methods += [(stage, "decode"), (comfy.sample, "prepare_noise")]
    before_wrappers = wrapper_snapshot(original_methods)
    names, snapshots, adapters, trajectory, candidate = [], None, None, None, None
    model = h3.patcher.model
    saved_patcher = getattr(model, "current_patcher", None)
    saved_shapes = (hasattr(model, "latent_shapes"), getattr(model, "latent_shapes", None))
    def audit_state(label):
        evidence.phase("state_hash_" + label, torch)
        checks = {"DiT": check_snapshot(dit, snapshots["DiT"], torch, names, h3.dev),
                  "VAE": check_snapshot(stage, snapshots["VAE"], torch, device=h3.dev)}
        if adapters is not None: checks["adapters"] = check_adapters(dit, adapters, torch, h3.dev)
        evidence.data["state_checks"][label] = checks
        require(all(r["passed"] for r in checks.values()), "frozen/residency gate failed: " + label, evidence)
    def nfe_observer(module, inputs):
        label = phase()
        if label not in ("reference", "candidate"): raise RuntimeError("unexpected whole DiT call: " + label)
        key = label + "_NFE"
        evidence.data["actual"][key] += 1
        if evidence.data["actual"][key] > 3: raise RuntimeError("NFE budget exceeded")
    hook = dit.register_forward_pre_hook(nfe_observer)
    try:
        with ExitStack() as scope:
            scope.enter_context(guarded_vae(h3, evidence, torch))
            scope.enter_context(observe_noise(comfy.sample, comfy.utils.pack_latents, phase, evidence.data["noise_observations"], torch))
            scope.enter_context(counted_blocks(dit.blocks, "DiT", evidence, observe_math))
            scope.enter_context(counted_blocks(stage.decoder.transformer_blocks, "VAE", evidence, observe_math))
            evidence.phase("reference", torch)
            reference, evidence.data["reference_record"] = h3.sample(guide, mode="base", hw=(128, 128), seed=SEED,
                refs=[], audio_init="zeros", warm_start=raw)
            for stream in ("video", "audio"): save_array(evidence, "reference_" + stream, reference[stream].float(), torch)
            evidence.phase("restore_reference_residency", torch)
            needs_restore = getattr(model, "model_lowvram", False) or any(v.device != torch.device(h3.dev)
                for _, v in canonical_base_items(dit))
            evidence.data["reference_residency_restore_called"] = bool(needs_restore)
            if needs_restore: h3.patcher.patch_model(device_to=h3.dev, lowvram_model_memory=0, load_weights=True)
            restored = {"DiT": frozen_inventory(dit, torch), "VAE": frozen_inventory(stage, torch)}
            evidence.data["reference_bytes_restored"] = all(restored[g][k] == baseline[g][k]
                for g in baseline for k in ("parameters_sha256", "registered_buffers_sha256"))
            require(evidence.data["reference_bytes_restored"], "reference changed canonical bytes", evidence)
            snapshots = {"DiT": freeze_snapshot(dit, torch), "VAE": freeze_snapshot(stage, torch)}
            audit_state("before_reference_decode")
            evidence.phase("reference_decode", torch)
            with native_training_math(), torch.no_grad():
                reference_rgb = h3.decode(reference["video"].to(device=h3.dev, dtype=vae.vae_dtype))
            save_array(evidence, "reference_rgb", reference_rgb, torch)
            audit_state("after_reference_decode")
            evidence.phase("inject_zero_B", torch)
            names = inject_adaptation(dit, rank=8, seed=SEED)
            evidence.data["adapter_initial"] = adapter_audit(dit, names, torch)
            adapters = adapter_snapshot(dit, torch)
            audit_state("after_injection")
            trace = evidence.data["candidate_euler_trace"] = []
            def callback(event):
                trace.append({"step": int(event["i"]), "sigma": float(event["sigma"]),
                              "x": tensor_record(event["x"], torch), "denoised_x0": tensor_record(event["denoised"], torch)})
                evidence.save()
            trajectory = ResidentH3Trajectory(h3, checkpoint_blocks=True, checkpoint_decoder=True, decoder_context=decoder_context)
            with trajectory:
                math_expected["DiT_active"] = actual_math_state(torch)
                evidence.data["DiT_math_active"] = math_expected["DiT_active"]
                evidence.phase("candidate", torch)
                candidate = trajectory.sample(guide, raw, seed=SEED, audio_init="zeros", callback=callback)
                evidence.data["candidate_contract"] = candidate["contract"]
                save_latent_parity(reference, candidate, evidence, torch)
                noise = evidence.data["noise_observations"]
                evidence.data["noise_and_initial_match"] = len(noise) == 2 and ({k:v for k,v in noise[0].items() if k != "phase"} == {k:v for k,v in noise[1].items() if k != "phase"})
                require(evidence.data["noise_and_initial_match"] and len(trace) == 3, "noise/trace identity failed", evidence)
                candidate["video"].retain_grad()
                audit_state("before_candidate_decode")
                evidence.phase("candidate_decode", torch)
                rgb = trajectory.decode(candidate["video"])
                save_array(evidence, "candidate_rgb", rgb.float(), torch)
                evidence.data["decoder_input_exact"] = (evidence.data["arrays"]["reference_decoder_input"]["array_sha256"] ==
                    evidence.data["arrays"]["candidate_decoder_input"]["array_sha256"])
                require(evidence.data["decoder_input_exact"], "actual FP16 decoder input mismatch", evidence)
                audit_state("after_candidate_decode")
                evidence.data["DiT_math_after_decode"] = actual_math_state(torch)
                require(evidence.data["DiT_math_after_decode"] == math_expected["DiT_active"], "decoder math leaked", evidence)
                audit_state("before_backward")
                rgb_backward(reference_rgb, rgb, candidate["video"], evidence, torch)
                evidence.phase("gradient_and_frozen_audit", torch)
                save_gradients(dit, candidate["video"], evidence, torch)
                audit_state("after_backward")
                require(evidence.data["adapter_gradients"]["passed"] and evidence.data["retained_video_gradient"]["passed"],
                        "joint RGB gradient gate failed", evidence)
            try: trajectory._guard_backward(torch.zeros(()), trajectory.generation)
            except RuntimeError: evidence.data["stale_backward_guard_rejected"] = True
            else: evidence.data["stale_backward_guard_rejected"] = False
    finally:
        if sys.exc_info()[0] is not None:
            evidence.data["failure_origin_phase"] = evidence.data["phase"]
        hook.remove()
        evidence.data["wrapper_attributes_restored"] = wrappers_restored(before_wrappers)
        evidence.data["math_after"] = actual_math_state(torch)
        evidence.data["math_restored"] = evidence.data["math_after"] == initial_math
        evidence.data["session_inactive"] = trajectory is None or not trajectory.active
        evidence.data["patcher_restoration"] = restore_patcher_state(model, trajectory, saved_patcher, saved_shapes)
        for label, action in (("final_frozen", lambda: {"DiT": frozen_inventory(dit, torch, names), "VAE": frozen_inventory(stage, torch)}),
                              ("actual_imports", lambda: runtime_imports(evidence.data["verified_runtime_files"]))):
            try:
                value = action()
                if label == "final_frozen":
                    atomic_json(evidence.path / "frozen_final.json", value)
                    evidence.data[label] = {"path": "frozen_final.json", "sha256": sha_file(evidence.path / "frozen_final.json"),
                        "canonical_bytes_match": all(value[g][k] == baseline[g][k] for g in baseline
                                                     for k in ("parameters_sha256", "registered_buffers_sha256"))}
                else: evidence.data[label] = value
            except BaseException as error: evidence.data[label + "_error"] = repr(error)
        if snapshots is not None:
            try: audit_state("finally")
            except BaseException as error: evidence.data["finally_state_error"] = repr(error)
        if evidence.data["actual"]["backward_attempts"] and "adapter_gradients" not in evidence.data:
            try: save_gradients(dit, None if candidate is None else candidate["video"], evidence, torch)
            except BaseException as error: evidence.data["gradient_save_error"] = repr(error)
        evidence.phase("finally_saved", torch)
    counts = evidence.data["block_forward_calls"]
    evidence.data["block_budget_passed"] = (all(counts["DiT"].get(p) == {str(i): 3 for i in range(50)}
        for p in ("reference", "candidate", "backward")) and all(counts["VAE"].get(p) == {str(i): 1 for i in range(36)}
        for p in ("reference_decode", "candidate_decode", "backward")))
    actual = evidence.data["actual"]
    evidence.data["budget_passed"] = (actual["reference_NFE"] == actual["candidate_NFE"] == 3 and
        actual["reference_VAE_decode_calls"] == actual["candidate_VAE_decode_calls"] == 1 and
        actual["stage_decode_calls"] == {"reference_decode": 1, "candidate_decode": 1} and
        actual["backward_attempts"] == actual["completed_backward_calls"] == actual["RGB_loss_expressions"] == 1
        and actual["optimizer_steps"] == actual["adapter_updates"] == 0 and not actual["forbidden_calls"])
    require(evidence.data["block_budget_passed"] and evidence.data["budget_passed"], "execution budget mismatch", evidence)
    require(evidence.data["wrapper_attributes_restored"] and evidence.data["math_restored"] and
        evidence.data["session_inactive"] and evidence.data["stale_backward_guard_rejected"] and
        evidence.data["patcher_restoration"]["session_shapes_restored"] and
        evidence.data["patcher_restoration"]["session_current_patcher_restored"] and
        evidence.data["patcher_restoration"]["whole_run_current_patcher_restored"] and
        evidence.data["patcher_restoration"]["whole_run_shapes_restored"] and
        all(x["restored"] for x in evidence.data["decoder_context_restorations"]), "restoration gate failed", evidence)
    require(evidence.data.get("actual_imports", {}).get("passed", False) and
        evidence.data.get("final_frozen", {}).get("canonical_bytes_match", False) and
        "finally_state_error" not in evidence.data, "terminal identity/frozen gate failed", evidence)
    evidence.data["complete"] = True
    evidence.data["scope"] = "small synthetic same-graph RGB connectivity only; no optimizer/quality/full22 acceptance"
    evidence.phase("completed", torch)
    atomic_json(evidence.path / "report.json", evidence.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.execution_contract.read_text())
    if contract["schema"] != "openh3.joint_rgb_probe_execution.v1": raise ValueError("wrong execution schema")
    if sha_file(__file__) != contract["probe_sha256"]: raise ValueError("new entrypoint not pinned")
    if sha_file(args.pretrained_manifest) != MANIFEST_SHA or contract["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("wrong pretrained manifest")
    root = Path(contract["root"])
    source_contract = (root / contract["source_contract"]).resolve()
    if contract["source_contract_sha256"] != SOURCE_CONTRACT_SHA or sha_file(source_contract) != SOURCE_CONTRACT_SHA:
        raise ValueError("wrong frozen joint source contract")
    verified = verify_pins(root, contract["implementation_sha256"])
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    if sha_file(manifest["_paths"]["context"]) != contract["context_sha256"]:
        raise ValueError("context asset differs from execution contract")
    inspection = {"inspection_only": not args.execute, "source_contract_sha256": SOURCE_CONTRACT_SHA,
        "contract_sha256": sha_file(args.execution_contract), "runtime_files_verified": len(verified),
        "assets_size_only_verified": True, "asset_bytes_sha_verified": False,
        "pretrained_fingerprint": manifest["_fingerprint"],
        "planned": {"explicit_NFE": 6, "VAE_decodes": 2, "RGB_backward": 1,
                    "loss_scale": LOSS_SCALE, "optimizer_steps": 0}}
    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "inspection.json", inspection)
        print(json.dumps(inspection), flush=True)
        return
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        raise ValueError("one explicitly admitted physical card required")
    evidence = Evidence(args.out, {**contract, "execution_contract_sha256": sha_file(args.execution_contract)})
    evidence.data["inspection"] = inspection
    evidence.data["verified_runtime_files"] = verified
    try:
        execute(args, contract, evidence)
    except BaseException as error:
        evidence.data["failure"] = {"type": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "phase": evidence.data.get("failure_origin_phase", evidence.data["phase"])}
        evidence.save()
        atomic_json(evidence.path / "failure.json", evidence.data)
        raise


if __name__ == "__main__":
    main()
