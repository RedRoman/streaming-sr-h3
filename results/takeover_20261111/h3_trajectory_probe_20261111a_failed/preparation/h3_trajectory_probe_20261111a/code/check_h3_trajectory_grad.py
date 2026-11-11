"""Finite, latent-only real H3 probe; inspection is stdlib-only, execution opt-in.

Exactly one official reference and one retained candidate (3 NFE each), then
exact video/audio parity before a single backward. No VAE/RGB/optimizer path.
An external admitted queue supplies the device, lock and owned-process timeout.
This is numerical engineering evidence at 5-frame/128px-equivalent geometry.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback


SHAPE = (1, 24, 2, 8, 8)
SEED = 101
TARGETS = ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2")
MANIFEST_SHA = "1023bd6e1df1552240cc7ff4bfac960836d2eeda05e680ecd2eb4b752bf7d975"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temp, path)


def tensor_sha(value, torch):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy().tobytes()).hexdigest()


def tensor_record(value, torch):
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "sha256_bytes": tensor_sha(value, torch), "device": str(value.device)}


def canonical_base_items(model, adapted_names=()):
    """Use exact injected module prefixes; do not strip unrelated '.base.' names."""
    result = []
    for kind, iterator in (("parameter", model.named_parameters()), ("buffer", model.named_buffers())):
        for name, value in iterator:
            if kind == "parameter" and name.endswith((".lora_a", ".lora_b")):
                continue
            canonical = name
            for module_name in adapted_names:
                prefix = module_name + ".base."
                if name.startswith(prefix):
                    canonical = module_name + "." + name[len(prefix):]
                    break
            result.append((kind + ":" + canonical, value))
    result.sort(key=lambda row: row[0])
    if not result or len({n for n, _ in result}) != len(result):
        raise ValueError("empty or ambiguous base tensor identity")
    return result


def base_fingerprint(model, torch, adapted_names=()):
    rows = []
    for name, value in canonical_base_items(model, adapted_names):
        if value.requires_grad or value.grad is not None:
            raise ValueError("base is not frozen: " + name)
        rows.append({"name": name, "shape": list(value.shape), "dtype": str(value.dtype),
                     "sha256_bytes": tensor_sha(value, torch)})
    digest = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "tensors": rows,
            "parameter_tensors": sum(r["name"].startswith("parameter:") for r in rows),
            "buffer_tensors": sum(r["name"].startswith("buffer:") for r in rows)}


def snapshot_versions(model, adapted_names=()):
    return {name: (value, value._version) for name, value in canonical_base_items(model, adapted_names)}


def check_versions(model, snapshot, adapted_names=()):
    current = dict(canonical_base_items(model, adapted_names))
    failures = []
    if current.keys() != snapshot.keys():
        failures.append("base tensor names changed")
    for name, (value, version) in snapshot.items():
        if current.get(name) is not value:
            failures.append(name + ": object replaced")
        if value._version != version or value.requires_grad or value.grad is not None:
            failures.append(name + ": version/grad/frozen mismatch")
    return {"passed": not failures, "failures": failures,
            "tensors_checked": len(snapshot), "scope": "after residency restore and injection through backward"}


def parity_metrics(reference, candidate, torch):
    a, b = reference.detach().float().cpu(), candidate.detach().float().cpu()
    same_shape = a.shape == b.shape
    finite = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
    delta = a - b if same_shape else None
    return {"torch_equal": same_shape and finite and bool(torch.equal(a, b)),
            "same_shape": same_shape, "finite": finite,
            "reference_shape": list(a.shape), "candidate_shape": list(b.shape),
            "max_abs": float(delta.abs().max()) if same_shape and finite else None,
            "relative_l2": float(delta.double().norm() / a.double().norm().clamp_min(1e-30)) if same_shape and finite else None,
            "reference_sha256_float32": tensor_sha(a, torch),
            "candidate_sha256_float32": tensor_sha(b, torch),
            "acceptance": "strict torch.equal; no nonzero tolerance"}


def adapter_audit(model, module_names, torch):
    expected = {f"blocks.{i}.{target}" for i in range(50) for target in TARGETS}
    if set(module_names) != expected or len(module_names) != 200 or len(model.blocks) != 50:
        raise ValueError("requires exactly 50 blocks and the 200 declared LoRA modules")
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected_params = {name + suffix for name in expected for suffix in (".lora_a", ".lora_b")}
    if params.keys() != expected_params:
        raise ValueError("requires exactly 400 declared trainable LoRA tensors")
    rows = {}
    for name, p in params.items():
        if p.dtype != torch.float32 or p.grad is not None:
            raise ValueError("adapter must be fresh FP32: " + name)
        if p.ndim != 2 or (p.shape[0] if name.endswith(".lora_a") else p.shape[1]) != 8:
            raise ValueError("adapter rank must be 8: " + name)
        zeros = int(torch.count_nonzero(p)) == 0
        if name.endswith(".lora_b") and not zeros:
            raise ValueError("nonzero B before candidate: " + name)
        rows[name] = {**tensor_record(p, torch), "all_zero": zeros}
    return {"modules": 200, "trainable_tensors": 400, "rank": 8, "tensors": rows}


def gradient_audit(model, torch, directory=None):
    """Save each actual gradient before deciding validity, including missing/NaN."""
    rows, b_square, b_nonzero, all_finite = {}, 0.0, 0, True
    for name, p in model.named_parameters():
        if not name.endswith((".lora_a", ".lora_b")):
            continue
        if p.grad is None:
            rows[name] = {"present": False}
            continue
        g = p.grad.detach().float().cpu().contiguous()
        finite = torch.isfinite(g)
        valid = g[finite]
        norm = float(valid.double().norm())
        nonzero = int(torch.count_nonzero(valid))
        row = {"present": True, **tensor_record(g, torch), "elements": g.numel(),
               "finite_count": int(finite.sum()), "nonzero_finite_count": nonzero,
               "nan_count": int(torch.isnan(g).sum()), "inf_count": int(torch.isinf(g).sum()),
               "finite_l2": norm, "all_finite": bool(finite.all())}
        if directory is not None:
            import numpy as np
            path = Path(directory) / (name + ".npy")
            np.save(path, g.numpy(), allow_pickle=False)
            row["array_file"] = str(path)
            row["array_file_sha256"] = sha_file(path)
        all_finite = all_finite and row["all_finite"]
        if name.endswith(".lora_b"):
            b_square += norm * norm
            b_nonzero += nonzero > 0
        rows[name] = row
    missing = [n for n, r in rows.items() if not r["present"]]
    return {"tensors": rows, "missing": missing,
            "present_tensors": sum(r["present"] for r in rows.values()),
            "all_present_finite": all_finite, "B_l2": math.sqrt(b_square),
            "B_nonzero_tensors": b_nonzero,
            "passed": not missing and all_finite and b_nonzero > 0 and b_square > 0,
            "A_zero_at_zero_B_is_legitimate": True}


@contextmanager
def count_block_calls(blocks, phase, counts):
    """Install before checkpoint wrappers so their captured originals count recompute."""
    saved = []
    try:
        for index, block in enumerate(blocks):
            original = block.forward
            saved.append((block, original, "forward" in block.__dict__))
            def wrapped(*args, _original=original, _index=index, **kwargs):
                label = phase()
                bucket = counts.setdefault(label, {})
                key = str(_index)
                bucket[key] = bucket.get(key, 0) + 1
                return _original(*args, **kwargs)
            block.forward = wrapped
        yield
    finally:
        for block, original, overridden in reversed(saved):
            if overridden:
                block.forward = original
            elif "forward" in block.__dict__:
                del block.forward


@contextmanager
def forbid_vae(h3, calls):
    saved = []
    try:
        for owner, names in ((h3, ("encode", "decode", "encode_guide", "encode_guide_pixels", "ref_block")),
                             (h3.vae, ("encode", "decode"))):
            for name in names:
                if not hasattr(owner, name):
                    continue
                original = getattr(owner, name)
                saved.append((owner, name, original, name in owner.__dict__))
                def prohibited(*args, _name=name, **kwargs):
                    calls[_name] = calls.get(_name, 0) + 1
                    raise RuntimeError("VAE/RGB path forbidden in latent-only probe: " + _name)
                setattr(owner, name, prohibited)
        yield
    finally:
        for owner, name, original, overridden in reversed(saved):
            if overridden:
                setattr(owner, name, original)
            else:
                delattr(owner, name)


@contextmanager
def observe_noise(sample_module, pack, phase, records, torch):
    original = sample_module.prepare_noise
    def observed(latent, seed, *args, **kwargs):
        noise = original(latent, seed, *args, **kwargs)
        initial, shapes = pack(list(latent.unbind()))
        packed_noise, noise_shapes = pack(list(noise.unbind()))
        if shapes != noise_shapes:
            raise ValueError("noise/initial shapes differ")
        records.append({"phase": phase(), "seed": seed,
                        "packed_initial_sha256": tensor_sha(initial, torch),
                        "packed_noise_sha256": tensor_sha(packed_noise, torch),
                        "joint_shapes": [list(s) for s in shapes]})
        return noise
    sample_module.prepare_noise = observed
    try:
        yield
    finally:
        sample_module.prepare_noise = original


def memory_record(torch):
    try:
        torch.npu.synchronize()
        return {"synchronized": True, "allocated": torch.npu.memory_allocated(),
                "reserved": torch.npu.memory_reserved(),
                "peak_allocated": torch.npu.max_memory_allocated(),
                "peak_reserved": torch.npu.max_memory_reserved()}
    except Exception as error:
        return {"synchronized": False, "error": repr(error)}


class Evidence:
    def __init__(self, path, contract):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=False)
        self.started = time.monotonic()
        self.data = {"schema": "openh3.latent_trajectory_probe.v1", "complete": False,
                     "contract": contract, "phase": "created", "events": [],
                     "actual": {"reference_NFE": 0, "candidate_NFE": 0, "backward_attempts": 0,
                                "completed_backward_calls": 0, "VAE_calls": {}, "optimizer_steps": 0,
                                "RGB_loss_calls": 0}, "block_forward_calls": {}, "noise_observations": []}
        self.save()

    def save(self):
        self.data["elapsed_seconds"] = time.monotonic() - self.started
        atomic_json(self.path / "partial.json", self.data)

    def phase(self, name, torch=None):
        self.data["phase"] = name
        event = {"phase": name, "elapsed_seconds": time.monotonic() - self.started}
        if torch is not None:
            event["memory"] = memory_record(torch)
        self.data["events"].append(event)
        self.save()
        print(json.dumps(event), flush=True)


def require(gate, message, evidence):
    evidence.save()  # evidence survives a verdict failure
    if not gate:
        raise RuntimeError(message)


def parity_then_backward(candidate, reference, evidence, torch):
    """No extra forward. Called while the retained trajectory context is active."""
    import numpy as np
    output = evidence.path / "latent_outputs.npz"
    np.savez(output, **{arm + "_" + stream: values[stream].detach().float().cpu().numpy()
                       for arm, values in (("reference", reference), ("candidate", candidate))
                       for stream in ("video", "audio")})
    evidence.data["latent_outputs"] = {"path": str(output), "sha256": sha_file(output)}
    evidence.data["parity"] = {s: parity_metrics(reference[s], candidate[s], torch)
                               for s in ("video", "audio")}
    require(all(v["torch_equal"] for v in evidence.data["parity"].values()),
            "strict final video/audio parity failed; backward forbidden", evidence)
    loss = candidate["video"].float().square().mean() + 0.01 * candidate["audio"].float().square().mean()
    evidence.data["loss"] = float(loss.detach()) if bool(torch.isfinite(loss)) else None
    require(bool(torch.isfinite(loss)) and loss.requires_grad, "loss nonfinite or no retained graph", evidence)
    evidence.phase("backward", torch if hasattr(torch, "npu") else None)
    evidence.data["actual"]["backward_attempts"] += 1
    evidence.save()
    loss.backward()
    evidence.data["actual"]["completed_backward_calls"] += 1
    evidence.save()


def verify_pins(root, pins):
    verified = {}
    for relative, expected in pins.items():
        path = (root / relative).resolve()
        actual = sha_file(path)
        if actual != expected:
            raise ValueError("implementation pin mismatch: " + str(path))
        verified[str(path)] = actual
    return verified


def runtime_imports(verified):
    """Bind actual project/Comfy/kitchen Python imports, not only reading-basis files."""
    records, missing = {}, []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file:
            continue
        path = Path(file).resolve()
        if path.suffix == ".pyc":
            continue
        relevant = (name == "comfy" or name.startswith(("comfy.", "comfy_kitchen", "comfy_extras."))
                    or (str(path).startswith(os.environ.get("SSR0", "/nonexistent") + "/")
                        and "site-packages" not in str(path)))
        if relevant and path.suffix == ".py":
            value = sha_file(path)
            records[name] = {"path": str(path), "sha256": value}
            if verified.get(str(path)) != value:
                missing.append(str(path))
    return {"modules": records, "unbound": sorted(set(missing)), "passed": not missing}


def execute(args, contract, evidence):
    import torch
    import torch_npu  # initialize the supported backend before H3's JIT assertion
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    from openh3_assets import NATIVE_SIGMAS
    from h3_comfy import H3
    from openh3_runtime import inject_adaptation
    from openh3_train_trajectory import ResidentH3Trajectory
    evidence.data["environment"] = {"torch": torch.__version__, "torch_npu": torch_npu.__version__, "python": sys.version,
        "visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "synthetic_no_dataset": True}
    evidence.phase("load_and_verify_assets", torch)
    h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS,
            adaptation_path=None, lora_path=None)
    dit = h3.patcher.model.diffusion_model.eval().requires_grad_(False)
    evidence.data["pretrained"] = h3.record
    evidence.data["asset_bytes_sha_verified"] = True  # H3 constructor read_manifest(verify=True)
    evidence.data["precision"] = {"logical_device": 0,
        "jit_compile_false": bool(torch.npu.is_jit_compile_false()),
        "matmul_allow_hf32": bool(torch.npu.matmul.allow_hf32),
        "conv_allow_hf32": bool(torch.npu.conv.allow_hf32)}
    evidence.data["vae_devices_initial"] = sorted({str(v.device) for v in h3.vae.first_stage_model.parameters()})
    dit.to(h3.dev)
    # Check the actual static patcher/residency before either model trajectory.
    preflight_session = ResidentH3Trajectory(h3, checkpoint_blocks=True)
    preflight_session._validate_resident()
    evidence.data["actual_imports_before_sampling"] = runtime_imports(evidence.data["verified_runtime_files"])
    require(evidence.data["actual_imports_before_sampling"]["passed"], "unbound Python import before sampling", evidence)
    require(not any("npu" in d for d in evidence.data["vae_devices_initial"]), "VAE already on NPU", evidence)
    require(not torch.npu.matmul.allow_hf32 and not torch.npu.conv.allow_hf32,
            "HF32 must remain disabled", evidence)
    torch.npu.reset_peak_memory_stats()
    evidence.phase("initial_resident_fingerprint", torch)
    initial = base_fingerprint(dit, torch)
    atomic_json(evidence.path / "base_before_reference.json", initial)
    evidence.data["base_sha256_before_reference"] = initial["sha256"]
    evidence.data["base_versions_before_reference"] = {n: p._version for n, p in canonical_base_items(dit)}
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    guide = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    raw = torch.randn(SHAPE, generator=generator, dtype=torch.float32)
    evidence.data["inputs"] = {"guide": tensor_record(guide, torch), "raw_draft": tensor_record(raw, torch),
        "distinct": not bool(torch.equal(guide, raw)), "seed": SEED, "audio_init": "zeros",
        "context_sha256": sha_file(h3.pretrained_manifest["_paths"]["context"]) if isinstance(h3.pretrained_manifest, dict)
                              else contract["context_sha256"]}
    require(len(dit.blocks) == 50 and evidence.data["inputs"]["distinct"], "wrong blocks or identical guide/raw", evidence)
    import comfy.sample
    import comfy.utils
    original_forwards = [(b.forward, "forward" in b.__dict__) for b in dit.blocks]
    stage = lambda: evidence.data["phase"]
    counts = evidence.data["block_forward_calls"]
    nfe = evidence.data["actual"]
    def dit_observer(module, args):
        phase = stage()
        key = "reference_NFE" if phase == "reference" else "candidate_NFE" if phase == "candidate" else None
        if key is None:
            raise RuntimeError("unexpected whole DiT invocation in phase " + phase)
        nfe[key] += 1
        if nfe[key] > 3:
            raise RuntimeError("explicit NFE budget exceeded")
    observer_handle = dit.register_forward_pre_hook(dit_observer)
    trajectory = None
    names, versions = [], None
    try:
        with ExitStack() as scope:
            scope.enter_context(forbid_vae(h3, nfe["VAE_calls"]))
            scope.enter_context(observe_noise(comfy.sample, comfy.utils.pack_latents, stage,
                                             evidence.data["noise_observations"], torch))
            scope.enter_context(count_block_calls(dit.blocks, stage, counts))
            evidence.phase("reference", torch)
            reference, reference_record = h3.sample(guide, mode="base", hw=(128, 128), seed=SEED,
                                                   refs=[], audio_init="zeros", warm_start=raw)
            evidence.data["reference_record"] = reference_record
            # Save reference even if restore/injection/candidate later fails.
            import numpy as np
            reference_path = evidence.path / "reference_latents.npz"
            np.savez(reference_path, **{k: reference[k].detach().float().cpu().numpy() for k in ("video", "audio")})
            evidence.data["reference_latents"] = {"path": str(reference_path), "sha256": sha_file(reference_path)}
            require(nfe["reference_NFE"] == 3, "reference NFE mismatch", evidence)
            evidence.phase("restore_full_residency", torch)
            needs_restore = (getattr(h3.patcher.model, "model_lowvram", False)
                             or any(p.device != torch.device(h3.dev) for p in dit.parameters())
                             or any(p.device != torch.device(h3.dev) for p in dit.buffers()))
            evidence.data["full_residency_restore_called"] = needs_restore
            if needs_restore:
                h3.patcher.patch_model(device_to=h3.dev, lowvram_model_memory=0, load_weights=True)
            restored = base_fingerprint(dit, torch)
            atomic_json(evidence.path / "base_after_restore.json", restored)
            evidence.data["base_sha256_after_restore"] = restored["sha256"]
            require(initial == restored, "base bytes changed in reference/residency restore", evidence)
            evidence.phase("inject_zero_B", torch)
            # Full restore may legitimately replace parameters. Freeze the actual
            # post-restore objects here, then require injection only renames them.
            versions = snapshot_versions(dit)
            names = inject_adaptation(dit, rank=8, seed=SEED)
            evidence.data["base_versions_after_injection"] = check_versions(dit, versions, names)
            require(evidence.data["base_versions_after_injection"]["passed"],
                    "adapter injection replaced or mutated pretrained tensors", evidence)
            evidence.data["adapter_initial"] = adapter_audit(dit, names, torch)
            injected = base_fingerprint(dit, torch, names)
            require(injected == initial, "base bytes changed during adapter injection", evidence)
            evidence.data["base_versions_candidate_entry"] = {n: v for n, (_, v) in versions.items()}
            adapters = {n: p for n, p in dit.named_parameters() if p.requires_grad}
            adapter_versions = {n: p._version for n, p in adapters.items()}
            trace = evidence.data["candidate_euler_trace"] = []
            def trace_step(event):
                trace.append({"step": int(event["i"]), "sigma": float(event["sigma"]),
                              "x": tensor_record(event["x"], torch),
                              "denoised_x0": tensor_record(event["denoised"], torch)})
                evidence.save()
            trajectory = ResidentH3Trajectory(h3, checkpoint_blocks=True)
            with trajectory:
                evidence.phase("candidate", torch)
                candidate = trajectory.sample(guide, raw, seed=SEED, audio_init="zeros", callback=trace_step)
                evidence.data["candidate_contract"] = candidate["contract"]
                observations = evidence.data["noise_observations"]
                evidence.data["noise_and_initial_match"] = (len(observations) == 2 and
                    {k: v for k, v in observations[0].items() if k != "phase"} ==
                    {k: v for k, v in observations[1].items() if k != "phase"})
                require(nfe["candidate_NFE"] == 3 and len(trace) == 3 and evidence.data["noise_and_initial_match"],
                        "candidate NFE/trace/noise identity mismatch", evidence)
                require(all(counts.get(phase, {}) == {str(i): 3 for i in range(50)}
                            for phase in ("reference", "candidate")),
                        "expected 50 blocks times 3 calls in each forward", evidence)
                parity_then_backward(candidate, reference, evidence, torch)
                evidence.phase("gradient_and_frozen_audit", torch)
                gradient_dir = evidence.path / "adapter_gradients"
                gradient_dir.mkdir(exist_ok=False)
                evidence.data["adapter_gradients"] = gradient_audit(dit, torch, gradient_dir)
                evidence.data["base_versions_after_backward"] = check_versions(dit, versions, names)
                final = base_fingerprint(dit, torch, names)
                atomic_json(evidence.path / "base_after_backward.json", final)
                evidence.data["base_sha256_after_backward"] = final["sha256"]
                evidence.data["adapter_values_unchanged"] = all(
                    p._version == adapter_versions[n] and tensor_sha(p, torch) ==
                    evidence.data["adapter_initial"]["tensors"][n]["sha256_bytes"] for n, p in adapters.items())
                require(evidence.data["adapter_gradients"]["passed"], "adapter gradient gate failed", evidence)
                require(evidence.data["base_versions_after_backward"]["passed"] and final == initial,
                        "frozen base changed", evidence)
                require(evidence.data["adapter_values_unchanged"], "adapter parameters changed without optimizer", evidence)
            # Lifecycle check invokes only the guard; never a second backward.
            try:
                trajectory._guard_backward(torch.zeros(()), trajectory.generation)
            except RuntimeError:
                evidence.data["stale_backward_guard_rejected"] = True
            else:
                evidence.data["stale_backward_guard_rejected"] = False
            evidence.data["stale_check_scope"] = "direct guard call after exit; no second autograd backward"
    finally:
        observer_handle.remove()
        evidence.data["wrapper_attributes_restored"] = all(
            b.forward == old and ("forward" in b.__dict__) == overridden
            for b, (old, overridden) in zip(dit.blocks, original_forwards))
        evidence.data["vae_devices_final"] = sorted({str(v.device) for v in h3.vae.first_stage_model.parameters()})
        evidence.data["session_inactive"] = trajectory is None or not trajectory.active
        evidence.data["checkpoint_recomputed_block_calls"] = sum(counts.get("backward", {}).values())
        evidence.data["checkpoint_recompute_is_not_zero_extra_compute"] = True
        # Best-effort post-state even after parity/backward failure; never another forward.
        try:
            if versions is not None and "base_versions_after_backward" not in evidence.data:
                evidence.data["base_versions_on_exit"] = check_versions(dit, versions, names)
            if "base_sha256_after_backward" not in evidence.data:
                exit_fingerprint = base_fingerprint(dit, torch, names)
                atomic_json(evidence.path / "base_on_exit.json", exit_fingerprint)
                evidence.data["base_sha256_on_exit"] = exit_fingerprint["sha256"]
            if nfe["backward_attempts"] and "adapter_gradients" not in evidence.data:
                gradient_dir = evidence.path / "adapter_gradients_on_failure"
                gradient_dir.mkdir(exist_ok=False)
                evidence.data["adapter_gradients_on_failure"] = gradient_audit(dit, torch, gradient_dir)
        except Exception as audit_error:
            evidence.data["post_failure_audit_error"] = repr(audit_error)
        evidence.save()
    evidence.phase("final_import_and_budget_audit", torch)
    evidence.data["actual_imports"] = runtime_imports(evidence.data["verified_runtime_files"])
    require(evidence.data["actual_imports"]["passed"], "unbound actual Python runtime import", evidence)
    require(evidence.data["wrapper_attributes_restored"] and evidence.data["stale_backward_guard_rejected"],
            "checkpoint/lifecycle restoration failed", evidence)
    require(counts.get("backward", {}) == {str(i): 3 for i in range(50)},
            "expected checkpoint backward to recompute all 50 blocks in all 3 steps", evidence)
    require(nfe["reference_NFE"] == nfe["candidate_NFE"] == 3 and nfe["completed_backward_calls"] == 1
            and nfe["backward_attempts"] == 1 and not nfe["VAE_calls"], "execution budget mismatch", evidence)
    require(evidence.data["vae_devices_initial"] == evidence.data["vae_devices_final"]
            and not any("npu" in d for d in evidence.data["vae_devices_final"]), "VAE moved to NPU", evidence)
    evidence.data["complete"] = True
    evidence.data["acceptance_scope"] = "small synthetic complete H3 latent trajectory only; no RGB, joint VAE, quality or full22 claim"
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
    if contract["schema"] != "openh3.latent_probe_execution.v1":
        raise ValueError("unknown execution contract")
    if sha_file(args.pretrained_manifest) != MANIFEST_SHA or contract["manifest_sha256"] != MANIFEST_SHA:
        raise ValueError("wrong pretrained manifest")
    if sha_file(__file__) != contract["probe_sha256"]:
        raise ValueError("entrypoint differs from reviewed execution contract")
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    root = Path(contract["root"])
    verified = verify_pins(root, contract["implementation_sha256"])
    inspection = {"inspection_only": not args.execute, "contract_sha256": sha_file(args.execution_contract),
                  "assets_size_only_verified": True, "asset_bytes_sha_verified": False,
                  "runtime_files_verified": len(verified), "pretrained_fingerprint": manifest["_fingerprint"],
                  "planned": {"reference_NFE": 3, "candidate_NFE": 3, "backward_calls": 1,
                              "VAE_calls": 0, "optimizer_steps": 0, "RGB_loss_calls": 0}}
    if not args.execute:
        if args.out.exists():
            raise FileExistsError(args.out)
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "inspection.json", inspection)
        print(json.dumps(inspection), flush=True)
        return
    if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal():
        raise ValueError("explicit single admitted physical NPU required")
    evidence = Evidence(args.out, {**contract, "execution_contract_sha256": sha_file(args.execution_contract)})
    evidence.data["inspection"] = inspection
    evidence.data["verified_runtime_files"] = verified
    try:
        execute(args, contract, evidence)
    except BaseException as error:
        evidence.data["failure"] = {"type": type(error).__name__, "message": str(error),
                                    "traceback": traceback.format_exc(), "phase": evidence.data["phase"]}
        if "torch" in sys.modules and hasattr(sys.modules["torch"], "npu"):
            evidence.data["failure"]["memory"] = memory_record(sys.modules["torch"])
        evidence.save()
        atomic_json(evidence.path / "failure.json", evidence.data)
        raise


if __name__ == "__main__":
    main()
