"""Finite synthetic SGD / strict unmerged reload / descriptive BF16 merge probe.

No dataset, learning-rate search, production checkpoint or launch policy. The
admitted external queue owns resource checks, lock and timeout. Existing pinned
H3/Comfy implementations are reused without editing them. CPU inspection never
loads model weights; --execute is a separate explicit operation.
"""
import argparse
from contextlib import contextmanager, ExitStack, nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import check_h3_joint_rgb_grad as J
from check_h3_trajectory_grad import (MANIFEST_SHA, SEED, SHAPE, TARGETS,
    atomic_json, sha_file, tensor_sha, tensor_record, canonical_base_items,
    parity_metrics, observe_noise, runtime_imports, verify_pins)

SOURCE_CONTRACT_SHA = "d3148504db9fbee2925448ea69e5f2ba83da5eba2127bf3229c9964f445bc08a"
LR, LOSS_SCALE = 1e-4, 1024.0
MODES = ("zero", "updated", "reloaded", "merged")


def finite_stat(value):
    result = float(value)
    return result if math.isfinite(result) else None


def stable_pretrained(record):
    """Timings and pre-patch precision are observations, not recipe identity."""
    return {k: record[k] for k in ("openh3_pretrained", "comfy_commit", "dit", "guide_helper_sha256",
        "jit_compile_false", "mm_bmm_format_nd", "precision_used", "torch", "torch_npu",
        "torch_npu_module_to_cast_weight_dropped")}


class Evidence(J.Evidence):
    def __init__(self, path, contract):
        super().__init__(path, contract)
        self.data["schema"] = "openh3.nonzero_adapter_probe.v1"
        self.data["actual"] = {"NFE": {m: 0 for m in MODES},
            "VAE_decode_calls": {m: 0 for m in MODES}, "stage_decode_calls": {},
            "forbidden_calls": [], "RGB_loss_expressions": 0, "backward_attempts": 0,
            "completed_backward_calls": 0, "optimizer_steps": 0, "adapter_updates": 0,
            "checkpoint_saves": 0, "clear_reload_cycles": 0, "merge_calls": 0,
            "unscale_calls": 0, "VAE_encode_calls": 0}
        self.data["session_restorations"] = []
        self.data["mode_outputs"] = {}
        self.save()


def require(value, message, evidence):
    J.require(value, message, evidence)


def expected_adapters(model, torch, blocks=50, rank=8):
    """Inspect actual target linears, not merely key suffix/rank heuristics."""
    import lora as L
    expected = {}
    if len(model.blocks) != blocks:
        raise ValueError("wrong block count")
    modules = {f"blocks.{i}.{t}" for i in range(blocks) for t in TARGETS}
    actual_modules = {n for n, m in model.named_modules() if isinstance(m, L.LoRALinear)}
    if modules != actual_modules:
        raise ValueError("wrong adapter module set")
    for name in sorted(modules):
        module = model.get_submodule(name)
        if module.rank != rank or module.scale != 1.0:
            raise ValueError("wrong rank/alpha/scale: " + name)
        expected[name + ".lora_a"] = (rank, module.base.in_features)
        expected[name + ".lora_b"] = (module.base.out_features, rank)
    params = dict(L.lora_parameters(model))
    validate_state(params, expected, torch)
    if {n for n, p in model.named_parameters() if p.requires_grad} != set(expected):
        raise ValueError("trainable parameter outside exact adapter set")
    if len({id(p) for p in params.values()}) != len(params):
        raise ValueError("aliased adapter parameter objects")
    return expected


def validate_state(state, expected, torch):
    if set(state) != set(expected):
        raise ValueError("adapter key coverage differs")
    for name, shape in expected.items():
        value = state[name]
        if tuple(value.shape) != tuple(shape):
            raise ValueError("exact adapter shape required (no broadcasting): " + name)
        if value.dtype != torch.float32:
            raise ValueError("adapter must be FP32: " + name)
        if not bool(torch.isfinite(value).all()):
            raise ValueError("adapter is nonfinite: " + name)


def save_parameters(model, evidence, torch, label):
    import lora as L
    rows = {n: J.save_array(evidence, label + "__" + n, p, torch)
            for n, p in L.lora_parameters(model)}
    evidence.data[label] = rows
    evidence.save()
    return rows


def one_sgd_update(model, expected, evidence, torch):
    """One real torch.optim.SGD step; no clipping, filling, search or reset.

    Expected arithmetic uses the same device's FP32 add_(alpha=-lr) primitive,
    matching SGD. Persisted before/gradient/after arrays allow separate audit.
    """
    import lora as L
    params = dict(L.lora_parameters(model))
    validate_state(params, expected, torch)
    require(evidence.data["actual"]["optimizer_steps"] == 0 and
            evidence.data["actual"]["unscale_calls"] == 0, "second update/unscale forbidden", evidence)
    for name, value in params.items():
        require(value.grad is not None and value.grad.dtype == torch.float32 and
                bool(torch.isfinite(value.grad).all()), "missing/nonfinite FP32 gradient: " + name, evidence)
    before = {n: p.detach().clone() for n, p in params.items()}
    save_parameters(model, evidence, torch, "parameters_before_update")
    with torch.no_grad():
        for name, value in params.items():
            value.grad.div_(LOSS_SCALE)
            row = evidence.data["adapter_gradients"]["tensors"][name]["unscaled"]
            require(tensor_sha(value.grad, torch) == row["array_sha256"],
                    "actual unscale differs from saved gradient: " + name, evidence)
    evidence.data["actual"]["unscale_calls"] += 1
    optimizer = torch.optim.SGD(list(params.values()), lr=LR, momentum=0., dampening=0.,
        weight_decay=0., nesterov=False, maximize=False, foreach=False, fused=False)
    require(len(optimizer.state) == 0, "unexpected SGD state", evidence)
    evidence.data["optimizer"] = {"type": "torch.optim.SGD", "lr": LR, "momentum": 0.,
        "dampening": 0., "weight_decay": 0., "nesterov": False, "maximize": False,
        "foreach": False, "fused": False, "loss_scale": LOSS_SCALE,
        "expected_arithmetic": "device FP32 before.clone().add_(unscaled_grad, alpha=-1e-4)",
        "loss_descent_required": False}
    evidence.data["actual"]["optimizer_steps"] += 1
    optimizer.step()
    evidence.data["actual"]["adapter_updates"] += 1
    rows, nonzero_b = {}, 0
    with torch.no_grad():
        for name, value in params.items():
            wanted = before[name].clone().add_(value.grad, alpha=-LR)
            # Statistics only: never require unsupported NPU FP64 arithmetic.
            delta = (value - before[name]).detach().cpu().double()
            count = int(torch.count_nonzero(delta))
            rows[name] = {"before_sha256": tensor_sha(before[name], torch),
                "gradient_unscaled_sha256": tensor_sha(value.grad, torch),
                "after_sha256": tensor_sha(value, torch), "expected_after_sha256": tensor_sha(wanted, torch),
                "exact_SGD_arithmetic": bool(torch.equal(value, wanted)), "changed_coordinates": count,
                "update_l2": finite_stat(delta.norm()), "update_rms": finite_stat(delta.square().mean().sqrt()),
                "all_finite": bool(torch.isfinite(value).all()), "version": value._version}
            if name.endswith(".lora_b"): nonzero_b += int(torch.count_nonzero(value))
    evidence.data["update"] = {"tensors": rows, "B_nonzero_coordinates": nonzero_b,
        "optimizer_state_entries_after": len(optimizer.state)}
    save_parameters(model, evidence, torch, "parameters_after_update")
    require(nonzero_b > 0 and all(r["exact_SGD_arithmetic"] and r["all_finite"] for r in rows.values())
            and len(optimizer.state) == 0, "single SGD arithmetic gate failed", evidence)
    del optimizer


def strict_saved_state(path, expected, pretrained, diagnostic, torch):
    from safetensors.torch import load_file
    from openh3_runtime import check_adapter
    metadata = check_adapter(path, pretrained, rank=8, training=diagnostic)
    state = load_file(str(path), device="cpu")
    validate_state(state, expected, torch)
    return state, metadata


def save_checkpoint(model, expected, pretrained, diagnostic, evidence, torch):
    import lora as L
    from safetensors.torch import save_file
    from openh3_runtime import adapter_metadata
    path = evidence.path / "diagnostic_synthetic_SGD_step1.safetensors"
    require(not path.exists() and evidence.data["actual"]["checkpoint_saves"] == 0,
            "checkpoint overwrite/second save forbidden", evidence)
    state = L.state_dict(model)
    validate_state(state, expected, torch)
    save_file(state, str(path), metadata=adapter_metadata(pretrained, diagnostic, rank=8))
    evidence.data["actual"]["checkpoint_saves"] += 1
    loaded, meta = strict_saved_state(path, expected, pretrained, diagnostic, torch)
    equality = {name: bool(torch.equal(state[name], loaded[name])) for name in state}
    evidence.data["checkpoint"] = {"path": path.name, "sha256": sha_file(path),
        "bytes": path.stat().st_size, "metadata": meta, "expected_shapes": {n:list(s) for n,s in expected.items()},
        "exact_saved_live_tensors": equality, "role": "diagnostic_synthetic_plain_SGD_step1_not_a_restoration_checkpoint"}
    require(all(equality.values()), "saved adapter differs from actual updated tensor", evidence)
    return path


def clear_reload(model, expected, path, pretrained, diagnostic, evidence, torch):
    import lora as L
    require(evidence.data["actual"]["clear_reload_cycles"] == 0, "second reload forbidden", evidence)
    require(sha_file(path) == evidence.data["checkpoint"]["sha256"], "checkpoint file changed", evidence)
    params = dict(L.lora_parameters(model))
    live_sha = {n: tensor_sha(p, torch) for n,p in params.items()}
    with torch.no_grad():
        for p in params.values(): p.zero_()
    cleared = {n: {"all_zero": not bool(torch.count_nonzero(p)), "version": p._version,
                   "sha256": tensor_sha(p, torch)} for n,p in params.items()}
    evidence.data["reload"] = {"cleared": cleared}
    evidence.save()
    require(all(r["all_zero"] for r in cleared.values()), "destructive clear failed", evidence)
    state, metadata = strict_saved_state(path, expected, pretrained, diagnostic, torch)
    require(all(tensor_sha(state[n], torch) == live_sha[n] for n in params),
            "saved contents differ from pre-clear live tensors", evidence)
    loaded_count = L.load(model, state, strict=True, strict_unused=True)
    reloaded = {n: {"sha256": tensor_sha(p, torch), "equals_saved": bool(torch.equal(p.detach().cpu(), state[n])),
                    "version": p._version} for n,p in params.items()}
    evidence.data["actual"]["clear_reload_cycles"] += 1
    evidence.data["reload"].update({"loaded_count": loaded_count, "reloaded": reloaded,
        "metadata": metadata, "file_sha_after": sha_file(path)})
    require(loaded_count == len(expected) and all(r["equals_saved"] for r in reloaded.values()) and
            sha_file(path) == evidence.data["checkpoint"]["sha256"], "strict reload gate failed", evidence)


def merge_once(model, names, frozen_snapshot, evidence, torch):
    """Declared 200 base writes, not an erroneous all-base-unchanged gate.

    Record the intended same-device addmm + BF16 cast hash before calling the
    existing merge. Keep only hashes/statistics, never duplicate the 20B base.
    """
    from openh3_runtime import merge_adaptation_for_inference
    import lora as L
    require(evidence.data["actual"]["merge_calls"] == 0, "second merge forbidden", evidence)
    rows = {}
    with torch.no_grad():
        for name in names:
            mod = model.get_submodule(name)
            weight = mod.base.weight
            require(weight.dtype == torch.bfloat16, "merge storage must be BF16: " + name, evidence)
            w = weight.float()
            delta = (mod.lora_b.float() @ mod.lora_a.float()) * mod.scale
            intended = w.clone().addmm_(mod.lora_b.float(), mod.lora_a.float(), alpha=mod.scale).to(weight.dtype)
            storage_delta = intended.float() - w
            rounding = storage_delta - delta
            dn = finite_stat(delta.detach().cpu().double().norm())
            rn = finite_stat(rounding.detach().cpu().double().norm())
            rows[name] = {"base_before_sha256": tensor_sha(weight, torch),
                "expected_after_sha256": tensor_sha(intended, torch), "base_before_version": weight._version,
                "shape": list(weight.shape), "dtype": str(weight.dtype), "scale": mod.scale,
                "ideal_delta_l2": dn, "storage_delta_l2": finite_stat(storage_delta.detach().cpu().double().norm()),
                "changed_weight_count": int(torch.count_nonzero(storage_delta)),
                "changed_weight_fraction": float(torch.count_nonzero(storage_delta)) / weight.numel(),
                "nonzero_ideal_rounded_to_zero_count": int(torch.count_nonzero((delta != 0) & (storage_delta == 0))),
                "rounding_residual_l2": rn, "rounding_residual_relative_l2": rn / dn if dn and rn is not None else None,
                "zero_denominator": dn == 0, "all_finite": bool(torch.isfinite(intended).all())}
            # Bound temporary layer storage; do not retain full weight matrices.
            del w, delta, intended, storage_delta, rounding
    evidence.data["merge"] = {"layers": rows, "expected_writes": len(names),
        "rounding_scope": "ideal scale*(B@A); storage from runtime-identical FP32 addmm then one BF16 cast",
        "equivalence_claim": False}
    evidence.save()
    require(all(r["all_finite"] for r in rows.values()), "nonfinite expected merged weights", evidence)
    evidence.data["actual"]["merge_calls"] += 1
    receipt = merge_adaptation_for_inference(model)
    expected_snapshot = {n: dict(r) for n,r in frozen_snapshot.items()}
    for name in names:
        weight = model.get_submodule(name).weight
        row = rows[name]
        row["actual_after_sha256"] = tensor_sha(weight, torch)
        row["actual_after_version"] = weight._version
        row["exact_declared_write"] = (row["actual_after_sha256"] == row["expected_after_sha256"] and
                                        weight._version == row["base_before_version"] + 1)
        entry = expected_snapshot["parameter:" + name + ".weight"]
        entry["sha256_bytes"] = row["expected_after_sha256"]
        entry["version"] = row["base_before_version"] + 1
    evidence.data["merge"]["runtime_receipt"] = receipt
    evidence.data["merge"]["remaining_adapter_tensors"] = len(L.lora_parameters(model))
    check = J.check_snapshot(model, expected_snapshot, torch, names)
    evidence.data["merge"]["declared_mutation_check"] = check
    require(receipt["merged_modules"] == len(names) and not L.lora_parameters(model) and
            all(r["exact_declared_write"] for r in rows.values()) and check["passed"],
            "merge changed an undeclared tensor or missed an expected write", evidence)
    return expected_snapshot


def load_array(directory, row):
    import numpy as np
    path = (Path(directory) / row["path"]).resolve()
    if not path.is_relative_to(Path(directory).resolve()): raise ValueError("array escapes report directory")
    if sha_file(path) != row["file_sha256"]: raise ValueError("array file SHA mismatch")
    arr = np.load(path, allow_pickle=False)
    if (str(arr.dtype) != row["dtype"] or list(arr.shape) != row["shape"] or
        hashlib.sha256(arr.tobytes()).hexdigest() != row["array_sha256"] or not np.isfinite(arr).all()):
        raise ValueError("array content/dtype/shape/finite mismatch")
    return arr


def inspect_prerequisite(contract):
    root, bound = Path(contract["root"]), contract["joint_prerequisite"]
    paths = {}
    for key in ("report", "audit", "archive"):
        paths[key] = root / bound[key]["path"]
        if sha_file(paths[key]) != bound[key]["sha256"]: raise ValueError("joint prerequisite SHA: " + key)
    report, audit = (json.loads(paths[k].read_text()) for k in ("report", "audit"))
    if (not report.get("complete") or not audit.get("complete") or not audit.get("audit_complete") or
        audit.get("errors") or audit.get("numerical_acceptance") != "passed_small_joint_RGB_only" or
        audit.get("report_sha256") != bound["report"]["sha256"]):
        raise ValueError("joint prerequisite not independently accepted")
    if (report["contract"]["manifest_sha256"] != MANIFEST_SHA or
        report["contract"]["context_sha256"] != contract["context_sha256"]):
        raise ValueError("joint prerequisite recipe/context differs")
    arrays = {n: load_array(paths["report"].parent, report["arrays"][n]) for n in
        ("guide", "raw_warm_start", "target", "reference_video", "reference_audio", "reference_rgb", "reference_decoder_input")}
    inventory_path = paths["report"].parent / report["frozen_initial"]["path"]
    if sha_file(inventory_path) != report["frozen_initial"]["sha256"]:
        raise ValueError("joint initial inventory SHA differs")
    inventory = json.loads(inventory_path.read_text())
    return report, arrays, inventory


@contextmanager
def guarded_vae(h3, evidence, torch):
    vae, stage = h3.vae, h3.vae.first_stage_model
    with ExitStack() as scope:
        for owner, methods in ((h3, ("encode", "encode_guide", "encode_guide_pixels", "ref_block", "sample")),
                (vae, ("encode", "decode_tiled_1d", "_decode_tiled_owned", "decode_tiled_", "decode_tiled_3d"))):
            for name in methods:
                if not hasattr(owner, name): continue
                def forbidden(*a, _name=name, **kw):
                    evidence.data["actual"]["forbidden_calls"].append(_name)
                    evidence.save()
                    raise RuntimeError("encode/deployed sampler/OOM fallback forbidden: " + _name)
                scope.enter_context(J.override_method(owner, name, forbidden))
        original = vae.decode
        def decode(value, *a, **kw):
            phase = evidence.data["phase"]
            mode = phase.removesuffix("_decode")
            require(phase == mode + "_decode" and mode in MODES, "unexpected decode phase", evidence)
            evidence.data["actual"]["VAE_decode_calls"][mode] += 1
            require(evidence.data["actual"]["VAE_decode_calls"][mode] == 1, "second decode forbidden", evidence)
            J.save_array(evidence, mode + "_decoder_input", value, torch)
            return original(value, *a, **kw)
        scope.enter_context(J.override_method(vae, "decode", decode))
        original_stage = stage.decode
        def stage_decode(*a, **kw):
            phase = evidence.data["phase"]
            counts = evidence.data["actual"]["stage_decode_calls"]
            counts[phase] = counts.get(phase, 0) + 1
            require(phase in {m + "_decode" for m in MODES} and counts[phase] == 1,
                    "second whole stage decode forbidden", evidence)
            return original_stage(*a, **kw)
        scope.enter_context(J.override_method(stage, "decode", stage_decode))
        yield


def save_mode(mode, output, rgb, evidence, torch):
    for stream in ("video", "audio"):
        J.save_array(evidence, mode + "_" + stream, output[stream].float(), torch)
    J.save_array(evidence, mode + "_rgb", rgb.float(), torch)
    target = J.fixed_target(torch)
    import numpy as np
    pixels = rgb.detach().float().cpu()
    valid = (tuple(output["video"].shape) == SHAPE and tuple(output["audio"].shape) == (1,32,2,8)
        and tuple(pixels.shape) == J.RGB_SHAPE and all(bool(torch.isfinite(v).all()) for v in
            (output["video"], output["audio"], pixels)))
    evidence.data["mode_outputs"][mode] = {"geometry_and_finite": valid,
        "torch_float32_mse": float((pixels-target).square().mean()) if valid else None,
        "numpy_float64_mse": float(np.mean((pixels.numpy().astype(np.float64)-target.numpy().astype(np.float64))**2)) if valid else None,
        "clipping": {"at_zero": int((pixels == 0).sum()), "at_one": int((pixels == 1).sum()),
            "outside_unit_interval": int(((pixels < 0) | (pixels > 1)).sum()),
            "semantics": "as_returned_internal_clipped; no extra clamp or preclamp claim"}}
    require(valid, "nonfinite/wrong output geometry: " + mode, evidence)


def compare_modes(left, right, evidence, torch, strict=False):
    metrics = {}
    for stream in ("video", "audio", "rgb"):
        a, b = (torch.from_numpy(load_array(evidence.path, evidence.data["arrays"][m + "_" + stream])) for m in (left,right))
        metrics[stream] = parity_metrics(a, b, torch)
        metrics[stream]["acceptance"] = "torch.equal required" if strict else "descriptive only; no equivalence tolerance"
        metrics[stream]["changed_coordinates"] = int(torch.count_nonzero(a != b))
    evidence.data.setdefault("comparisons", {})[left + "__" + right] = metrics
    if strict:
        require(all(r["torch_equal"] for r in metrics.values()), "same-mode unmerged reload is not exact", evidence)
    evidence.save()
    return metrics


def check_lifecycle(model, snapshot, torch, names=(), device=None):
    """Cheap per-boundary object/version/device gate; full bytes at stage gates."""
    current = dict(canonical_base_items(model, names))
    bad = []
    if current.keys() != snapshot.keys(): bad.append("names changed")
    for name, before in snapshot.items():
        p = current.get(name)
        if p is None: continue
        if (p is not before["object"] or p._version != before["version"] or p.requires_grad or
            p.grad is not None or str(p.device) != before["device"] or
            (device is not None and p.device != torch.device(device))): bad.append(name)
    return {"passed": not bad, "failures": bad, "tensors_checked": len(snapshot),
            "byte_hash": "separately checked at full stage gates, not recomputed at this boundary"}


def execute(args, contract, evidence, prior, prior_arrays, prior_inventory):
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
        "python": sys.version, "visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "synthetic_no_dataset": True}
    require(all(evidence.data["environment"][k] == prior["environment"][k] for k in ("torch","torch_npu","python")),
            "accepted joint runtime versions differ", evidence)
    evidence.phase("load_and_verify_assets", torch)
    h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS, adaptation_path=None, lora_path=None)
    dit, vae = h3.patcher.model.diffusion_model, h3.vae
    stage, model = vae.first_stage_model, h3.patcher.model
    dit.eval().requires_grad_(False).to(h3.dev)
    stage.eval().requires_grad_(False).to(h3.dev)
    vae.device = vae.offload_device = h3.dev
    vae.disable_offload = True
    evidence.data["pretrained"] = h3.record
    evidence.data["asset_bytes_sha_verified"] = True
    require(stable_pretrained(h3.record) == stable_pretrained(prior["pretrained"]) and vae.vae_dtype == torch.float16,
            "joint recipe/VAE precision differs", evidence)
    preflight = ResidentH3Trajectory(h3, checkpoint_blocks=True, checkpoint_decoder=True)
    preflight._validate_resident(); preflight._validate_vae()
    require(len(dit.blocks) == 50 and len(stage.decoder.transformer_blocks) == 36, "wrong block inventory", evidence)
    evidence.data["vae_native_plan"] = {"tiling": stage.tiling, "tile_size": stage.tile_size,
        "vae_ratio": stage.vae_ratio, "vae_ratio_t": stage.vae_ratio_t,
        "temporal_padding_and_chunks": list(stage._decode_temporal_chunks(2)),
        "height_tile_plan": stage.split_tiles(128), "width_tile_plan": stage.split_tiles(128),
        "output_shape": list(stage.decode_output_shape(SHAPE)), "changed": False}
    require(json.loads(json.dumps(evidence.data["vae_native_plan"])) == prior["vae_native_plan"],
            "native VAE geometry/padding differs", evidence)
    initial_math = J.actual_math_state(torch)
    evidence.data["math_before"] = initial_math
    flags = ("in_training", "jit_compile_false", "matmul_allow_hf32", "conv_allow_hf32",
             "deterministic_algorithms", "deterministic_warn_only")
    require(all(initial_math[k] == prior["math_before"][k] for k in flags), "joint runtime math flags differ", evidence)
    evidence.data["actual_imports_before_sampling"] = runtime_imports(evidence.data["verified_runtime_files"])
    require(evidence.data["actual_imports_before_sampling"]["passed"], "unbound runtime import", evidence)
    torch.npu.reset_peak_memory_stats()
    names, session, snapshots, adapters = [], None, None, None
    original_patcher = getattr(model, "current_patcher", None)
    original_shapes = (hasattr(model, "latent_shapes"), getattr(model, "latent_shapes", None))
    methods = [(b,"forward") for b in list(dit.blocks)+list(stage.decoder.transformer_blocks)]
    methods += [(owner,n) for owner, ns in ((h3,("encode","encode_guide","encode_guide_pixels","ref_block","sample")),
        (vae,("decode","encode","decode_tiled_1d","_decode_tiled_owned","decode_tiled_","decode_tiled_3d")),
        (stage,("decode",)), (comfy.sample,("prepare_noise",))) for n in ns if hasattr(owner,n)]
    wrappers = J.wrapper_snapshot(methods)
    guide, raw = (torch.from_numpy(prior_arrays[n].copy()) for n in ("guide","raw_warm_start"))
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    require(torch.equal(guide, torch.randn(SHAPE,generator=generator)) and
        torch.equal(raw, torch.randn(SHAPE,generator=generator)) and
        torch.equal(J.fixed_target(torch), torch.from_numpy(prior_arrays["target"])), "saved inputs/target recipe mismatch", evidence)
    for n,v in (("guide",guide),("raw_warm_start",raw),("target",J.fixed_target(torch))): J.save_array(evidence,n,v,torch)
    for n in ("reference_video","reference_audio","reference_rgb","reference_decoder_input"):
        J.save_array(evidence,n,torch.from_numpy(prior_arrays[n]),torch)
    evidence.data["inputs"] = {"seed": SEED, "audio_init": "zeros", "target_used_as_model_input": False,
        "source": "accepted joint saved arrays, independently hashed and regenerated CPU input recipe verified"}
    math_expected = {}
    def observe_math(group, phase, index):
        current = J.actual_math_state(torch)
        evidence.data["math_observations"].append({"group":group,"phase":phase,"block":index,"state":current})
        if not current["in_training"]: raise RuntimeError("block escaped training math")
        if any(current[k] != initial_math[k] for k in flags if k != "in_training"):
            raise RuntimeError("block runtime flags changed")
        if group == "VAE":
            if current["rope"] != J.function_id(native_rope): raise RuntimeError("VAE native math absent")
        elif current != math_expected["DiT_active"]: raise RuntimeError("decoder math leaked to DiT")
    @contextmanager
    def decoder_context():
        before = J.actual_math_state(torch)
        try:
            with decoder_math(native_training_math,"ascend_native"): yield
        finally:
            after = J.actual_math_state(torch)
            evidence.data["decoder_context_restorations"].append({"phase":evidence.data["phase"],
                "before":before,"after":after,"restored":before==after})
            if before != after: raise RuntimeError("native decoder context did not restore")
    def audit_state(label, full=False):
        evidence.phase(("full_state_hash_" if full else "lifecycle_")+label, torch)
        check = J.check_snapshot if full else check_lifecycle
        result = {"DiT":check(dit,snapshots["DiT"],torch,names,h3.dev),
                  "VAE":check(stage,snapshots["VAE"],torch,device=h3.dev)}
        if adapters is not None: result["adapters"] = J.check_adapters(dit,adapters,torch,h3.dev)
        evidence.data["state_checks"][label] = result
        require(all(r["passed"] for r in result.values()), "frozen state failed: "+label,evidence)
    def nfe_hook(module, inputs):
        mode = evidence.data["phase"]
        require(mode in MODES,"unexpected whole DiT phase",evidence)
        evidence.data["actual"]["NFE"][mode] += 1
        require(evidence.data["actual"]["NFE"][mode] <= 3,"NFE budget exceeded",evidence)
    hook = dit.register_forward_pre_hook(nfe_hook)
    current_candidate = None
    try:
        evidence.phase("initial_resident_fingerprints",torch)
        initial = {"DiT":J.frozen_inventory(dit,torch),"VAE":J.frozen_inventory(stage,torch)}
        atomic_json(evidence.path/"frozen_initial.json",initial)
        evidence.data["frozen_initial"]={"path":"frozen_initial.json","sha256":sha_file(evidence.path/"frozen_initial.json")}
        require(all(initial[g][k]==prior_inventory[g][k] for g in initial for k in
            ("parameters_sha256","registered_buffers_sha256")),"original base/VAE differs from accepted joint",evidence)
        snapshots={"DiT":J.freeze_snapshot(dit,torch),"VAE":J.freeze_snapshot(stage,torch)}
        evidence.phase("inject_zero_B",torch)
        names=inject_adaptation(dit,rank=8,seed=SEED)
        expected=expected_adapters(dit,torch)
        evidence.data["adapter_initial"]=J.adapter_audit(dit,names,torch)
        adapters=J.adapter_snapshot(dit,torch)
        audit_state("after_injection")
        with ExitStack() as scope:
            scope.enter_context(guarded_vae(h3,evidence,torch))
            scope.enter_context(observe_noise(comfy.sample,comfy.utils.pack_latents,lambda:evidence.data["phase"],evidence.data["noise_observations"],torch))
            scope.enter_context(J.counted_blocks(dit.blocks,"DiT",evidence,observe_math))
            scope.enter_context(J.counted_blocks(stage.decoder.transformer_blocks,"VAE",evidence,observe_math))
            for mode in MODES:
                if mode=="updated":
                    evidence.phase("single_SGD_update",torch)
                    one_sgd_update(dit,expected,evidence,torch)
                    diagnostic={"role":"diagnostic_synthetic_plain_SGD_step1_not_a_restoration_checkpoint",
                        "execution_contract_sha256":evidence.data["contract"]["execution_contract_sha256"],
                        "source_contract_sha256":SOURCE_CONTRACT_SHA,"manifest_sha256":MANIFEST_SHA,
                        "implementation_sha256":contract["implementation_sha256"],"pretrained":h3.record,
                        "input_arrays":{n:evidence.data["arrays"][n] for n in ("guide","raw_warm_start","target")},
                        "noise":evidence.data["noise_observations"][0],"sigmas":list(NATIVE_SIGMAS),
                        "sigmas_float32_sha256":tensor_sha(torch.tensor(NATIVE_SIGMAS,dtype=torch.float32),torch),
                        "rank":8,"alpha":8,"lr":LR,"steps":1,"loss_scale":LOSS_SCALE,
                        "objective":"synthetic RGB MSE only","optimizer":evidence.data["optimizer"]}
                    checkpoint_path=save_checkpoint(dit,expected,h3.record["openh3_pretrained"],diagnostic,evidence,torch)
                    adapters=J.adapter_snapshot(dit,torch)
                    audit_state("after_update_and_save",full=True)
                elif mode=="reloaded":
                    evidence.phase("clear_and_strict_reload",torch)
                    clear_reload(dit,expected,checkpoint_path,h3.record["openh3_pretrained"],diagnostic,evidence,torch)
                    adapters=J.adapter_snapshot(dit,torch)
                    audit_state("after_reload")
                elif mode=="merged":
                    evidence.phase("descriptive_BF16_merge",torch)
                    snapshots["DiT"]=merge_once(dit,names,snapshots["DiT"],evidence,torch)
                    adapters=None
                    audit_state("after_declared_merge")
                session=ResidentH3Trajectory(h3,checkpoint_blocks=(mode=="zero"),
                    checkpoint_decoder=(mode=="zero"),decoder_context=decoder_context)
                mode_patcher=getattr(model,"current_patcher",None)
                mode_shapes=(hasattr(model,"latent_shapes"),getattr(model,"latent_shapes",None))
                try:
                    with (nullcontext() if mode=="zero" else torch.no_grad()),session:
                        math_expected["DiT_active"]=J.actual_math_state(torch)
                        evidence.data.setdefault("session_math",{})[mode]=math_expected["DiT_active"]
                        trace=evidence.data.setdefault("traces",{}).setdefault(mode,[])
                        def callback(event):
                            trace.append({"step":int(event["i"]),"sigma":float(event["sigma"]),
                                "x":tensor_record(event["x"],torch),"denoised_x0":tensor_record(event["denoised"],torch)})
                            evidence.save()
                        evidence.phase(mode,torch)
                        current_candidate=session.sample(guide,raw,seed=SEED,audio_init="zeros",callback=callback)
                        evidence.data.setdefault("trajectory_contracts",{})[mode]={**current_candidate["contract"],
                            "adapter_arithmetic": "merged BF16 weights; zero independent adapters" if mode=="merged" else "independent FP32 adapter branch",
                            "merged":mode=="merged","actual_live_adapter_tensors":0 if mode=="merged" else len(expected),
                            "adapter_B_all_zero_at_session_entry":None if mode=="merged" else current_candidate["contract"]["adapter_B_all_zero_at_session_entry"],
                            "B_zero_boundary":"not applicable after merge" if mode=="merged" else "actual independent adapter parameters"}
                        expected_noise={k:v for k,v in prior["noise_observations"][0].items() if k!="phase"}
                        actual_noise={k:v for k,v in evidence.data["noise_observations"][-1].items() if k!="phase"}
                        require(expected_noise==actual_noise and len(trace)==3,"fixed noise/input/trace differs",evidence)
                        for stream in ("video","audio"): J.save_array(evidence,mode+"_"+stream,current_candidate[stream].float(),torch)
                        if mode=="zero":
                            ref={s:torch.from_numpy(prior_arrays["reference_"+s]) for s in ("video","audio")}
                            J.save_latent_parity(ref,current_candidate,evidence,torch)
                            current_candidate["video"].retain_grad()
                        audit_state(mode+"_before_decode")
                        evidence.phase(mode+"_decode",torch)
                        rgb=session.decode(current_candidate["video"])
                        save_mode(mode,current_candidate,rgb,evidence,torch)
                        audit_state(mode+"_after_decode")
                        if mode=="zero":
                            require(evidence.data["arrays"]["zero_decoder_input"]["array_sha256"]==
                                evidence.data["arrays"]["reference_decoder_input"]["array_sha256"],"zero FP16 decoder input differs",evidence)
                            J.rgb_backward(torch.from_numpy(prior_arrays["reference_rgb"]),rgb,current_candidate["video"],evidence,torch)
                            J.save_gradients(dit,current_candidate["video"],evidence,torch)
                            require(evidence.data["adapter_gradients"]["passed"] and evidence.data["retained_video_gradient"]["passed"],
                                "zero-B RGB gradient gate failed",evidence)
                            audit_state("after_RGB_backward",full=True)
                finally:
                    restoration=J.restore_patcher_state(model,session,mode_patcher,mode_shapes)
                    restoration["mode"]=mode
                    restoration["inactive"]=not session.active
                    evidence.data["session_restorations"].append(restoration)
                    evidence.save()
                require(all(restoration.get(k) is True for k in ("session_state_initialized","session_shapes_restored",
                    "session_current_patcher_restored","whole_run_shapes_restored","whole_run_current_patcher_restored","inactive")),
                    "session restoration failed",evidence)
                try: session._guard_backward(torch.zeros(()),session.generation)
                except RuntimeError: evidence.data.setdefault("stale_backward_guard_rejected",{})[mode]=True
                else: raise RuntimeError("stale backward guard accepted")
                if mode=="reloaded": compare_modes("updated","reloaded",evidence,torch,strict=True)
                if mode!="zero": compare_modes("zero",mode,evidence,torch)
                if mode=="merged": compare_modes("reloaded","merged",evidence,torch)
                audit_state(mode+"_complete",full=(mode!="zero"))
                del rgb,current_candidate
                current_candidate=None
    finally:
        if sys.exc_info()[0] is not None: evidence.data["failure_origin_phase"]=evidence.data["phase"]
        hook.remove()
        evidence.data["wrapper_attributes_restored"]=J.wrappers_restored(wrappers)
        evidence.data["math_after"]=J.actual_math_state(torch)
        evidence.data["math_restored"]=evidence.data["math_after"]==initial_math
        evidence.data["patcher_restoration"]=J.restore_patcher_state(model,session,original_patcher,original_shapes)
        try:
            final={"DiT":J.frozen_inventory(dit,torch,names),"VAE":J.frozen_inventory(stage,torch)}
            atomic_json(evidence.path/"frozen_final.json",final)
            evidence.data["final_frozen"]={"path":"frozen_final.json","sha256":sha_file(evidence.path/"frozen_final.json"),
                "scope":"original DiT plus exactly declared merge target writes; VAE and all registered buffers unchanged"}
            if snapshots is not None: audit_state("finally",full=True)
        except BaseException as error: evidence.data["finally_state_error"]=repr(error)
        try: evidence.data["actual_imports"]=runtime_imports(evidence.data["verified_runtime_files"])
        except BaseException as error: evidence.data["actual_imports_error"]=repr(error)
        if evidence.data["actual"]["backward_attempts"] and "adapter_gradients" not in evidence.data:
            try: J.save_gradients(dit,None if current_candidate is None else current_candidate["video"],evidence,torch)
            except BaseException as error: evidence.data["gradient_save_error"]=repr(error)
        evidence.phase("finally_saved",torch)
    a=evidence.data["actual"]
    evidence.data["budget_passed"]=(a["NFE"]=={m:3 for m in MODES} and a["VAE_decode_calls"]=={m:1 for m in MODES}
        and a["stage_decode_calls"]=={m+"_decode":1 for m in MODES} and not a["forbidden_calls"] and
        all(a[k]==1 for k in ("RGB_loss_expressions","backward_attempts","completed_backward_calls","optimizer_steps",
            "adapter_updates","checkpoint_saves","clear_reload_cycles","merge_calls","unscale_calls")) and a["VAE_encode_calls"]==0)
    counts=evidence.data["block_forward_calls"]
    evidence.data["block_budget_passed"]=(all(counts["DiT"].get(m)=={str(i):3 for i in range(50)} for m in (*MODES,"backward"))
        and all(counts["VAE"].get(m)=={str(i):1 for i in range(36)} for m in (*(m+"_decode" for m in MODES),"backward")))
    require(evidence.data["budget_passed"] and evidence.data["block_budget_passed"],"execution budget mismatch",evidence)
    require(evidence.data["wrapper_attributes_restored"] and evidence.data["math_restored"] and
        evidence.data["patcher_restoration"]["whole_run_shapes_restored"] and
        evidence.data["patcher_restoration"]["whole_run_current_patcher_restored"] and
        all(r["restored"] for r in evidence.data["decoder_context_restorations"]),"final restoration failed",evidence)
    require(evidence.data.get("actual_imports",{}).get("passed") and "finally_state_error" not in evidence.data,
        "terminal frozen/import gate failed",evidence)
    evidence.data["complete"]=True
    evidence.data["scope"]="small synthetic one-SGD persistence/arithmetic diagnostic only; no restoration quality or full-resolution acceptance"
    evidence.phase("completed",torch)
    atomic_json(evidence.path/"report.json",evidence.data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-manifest",type=Path,required=True)
    parser.add_argument("--execution-contract",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--execute",action="store_true")
    args=parser.parse_args()
    contract=json.loads(args.execution_contract.read_text())
    if contract["schema"]!="openh3.nonzero_adapter_probe_execution.v1": raise ValueError("wrong schema")
    if sha_file(__file__)!=contract["probe_sha256"]: raise ValueError("new entrypoint not pinned")
    if sha_file(args.pretrained_manifest)!=MANIFEST_SHA or contract["manifest_sha256"]!=MANIFEST_SHA:
        raise ValueError("wrong pretrained manifest")
    root=Path(contract["root"])
    if contract["source_contract_sha256"]!=SOURCE_CONTRACT_SHA or sha_file(root/contract["source_contract"])!=SOURCE_CONTRACT_SHA:
        raise ValueError("wrong frozen nonzero contract")
    verified=verify_pins(root,contract["implementation_sha256"])
    from openh3_assets import read_manifest
    manifest=read_manifest(args.pretrained_manifest,verify=False)
    if sha_file(manifest["_paths"]["context"])!=contract["context_sha256"]: raise ValueError("context differs")
    prior,arrays,inventory=inspect_prerequisite(contract)
    inspection={"inspection_only":not args.execute,"source_contract_sha256":SOURCE_CONTRACT_SHA,
        "contract_sha256":sha_file(args.execution_contract),"runtime_files_verified":len(verified),
        "joint_prerequisite":contract["joint_prerequisite"],"reference_arrays_verified":list(arrays),
        "assets_size_only_verified":True,"asset_bytes_sha_verified":False,"pretrained_fingerprint":manifest["_fingerprint"],
        "planned":{"NFE":12,"VAE_decode":4,"VAE_encode":0,"RGB_backward":1,"SGD_steps":1,"lr":LR,
            "loss_scale":LOSS_SCALE,"checkpoint_saves":1,"clear_reload_cycles":1,"merge_calls":1}}
    if not args.execute:
        args.out.mkdir(parents=True,exist_ok=False)
        atomic_json(args.out/"inspection.json",inspection)
        print(json.dumps(inspection),flush=True)
        return
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES","").isdecimal(): raise ValueError("one explicitly admitted card required")
    evidence=Evidence(args.out,{**contract,"execution_contract_sha256":sha_file(args.execution_contract)})
    evidence.data["inspection"]=inspection
    evidence.data["verified_runtime_files"]=verified
    try: execute(args,contract,evidence,prior,arrays,inventory)
    except BaseException as error:
        evidence.data["complete"]=False
        evidence.data["failure"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc(),
            "phase":evidence.data.get("failure_origin_phase",evidence.data["phase"])}
        evidence.save()
        atomic_json(evidence.path/"failure.json",evidence.data)
        raise


if __name__=="__main__": main()
