"""One full22 real TRAIN-cache H3 -> VAE RGB backward engineering probe.

Default inspection uses NumPy/stdlib, no Torch/Comfy import or model call.
Execution is separately admitted: 6 NFE, 2 decodes, one scaled RGB backward,
zero encode/Rphi/optimizer/update. Existing pinned sources remain unchanged.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

import check_h3_joint_rgb_grad as J
from check_h3_joint_rgb_grad import (atomic_json, sha_file, tensor_record,
    canonical_base_items, parity_metrics, adapter_audit, observe_noise,
    runtime_imports, verify_pins, require, save_array, frozen_inventory,
    freeze_snapshot, check_snapshot, adapter_snapshot, check_adapters,
    save_gradients, save_latent_parity, guarded_vae, function_id,
    actual_math_state, wrapper_snapshot, wrappers_restored, restore_patcher_state,
    override_method)

SOURCE_CONTRACT_SHA = "0b1b34b1fbedb26cf18b8d9d636e820185fe6ff2e701822b9661a627172d8df3"
ROOT_REVIEW_SHA = "53906f6d8af614c87cbeb1603af17684f16b3aff983ebcd726fa92da6540db9a"
INVENTORY_SHA = "c1a5e5339e335c1e536474da5712dbd04429753743803c51e6f3cfde8c880367"
NONZERO_AUDIT_SHA = "8d113ad97d2d207e820102879efb9451a05cf46cf7280239e768e9fc84d9441a"
MANIFEST_SHA = J.MANIFEST_SHA
SEED, LOSS_SCALE, ATOL, RTOL = 101, 1024.0, .0002, .0001
SHAPE, AUDIO_SHAPE, RGB_SHAPE = (1,24,7,30,52), (1,32,2,37), (22,480,832,3)
VAE_SOURCE_SHA = "1fd97eb3f756eb97bc8060c0cde5e31a9013a2bbd93b18d263801919980207fe"
TARGET_SHA = "e15655f1da67e8000ba2fa78e7a3fcbf4407b758afc59cbf3c0760ed3d6def5d"


class Evidence(J.Evidence):
    def __init__(self, path, contract):
        super().__init__(path, contract)
        self.data.update(schema="openh3.full22_joint_rgb_probe.v1", scope="single TRAIN engineering workload, not quality or streaming",
                         native_tile_rows=[], stage_decoder_inputs={}, input_inspection=None)
        self.data["actual"].update(VAE_encode_calls=0, Rphi_calls=0)
        self.save()


def insist(value, message):
    if not value: raise ValueError(message)


def array_sha(value):
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def checked_json(root, record, expected=None):
    p = Path(record["path"])
    if not p.is_absolute(): p = Path(root) / p
    expected = expected or record["sha256"]
    insist(record["sha256"] == expected and sha_file(p) == expected, "JSON pin mismatch: " + str(p))
    return json.loads(p.read_text())


def checked_array(record):
    import numpy as np
    p = Path(record["path"])
    insist(p.stat().st_size == record["bytes"] and sha_file(p) == record["file_sha256"], "array file pin mismatch: " + str(p))
    a = np.load(p, allow_pickle=False, mmap_mode="r")
    insist(list(a.shape) == record["shape"] and str(a.dtype) == record["dtype"], "array geometry/dtype mismatch")
    insist(array_sha(a) == record["array_sha256"] and bool(np.isfinite(a).all()), "array pixel hash/nonfinite")
    return a


def clean_target(npz_path, spec):
    import numpy as np
    insist(sha_file(npz_path) == spec["npz_file_sha256"], "target file differs")
    with np.load(npz_path, allow_pickle=False) as z:
        a = z[spec["key"]]
        insist(a.dtype == np.uint8 and list(a.shape) == spec["shape"], "target must be exact uint8 THWC")
        insist(array_sha(a) == spec["array_sha256"], "target pixel hash differs")
        target = a.astype(np.float32) / np.float32(255.)
    insist(array_sha(target) == spec["loss_target_float32_sha256"], "target float32 conversion differs")
    return target


def validate_identity(inspection, expected):
    for k in ("source_id", "source_sha256", "sample_id", "record_sha256", "role", "task", "geometry"):
        insist(inspection[k] == expected[k], "cache identity differs: " + k)
    insist(inspection["role"] == "train" and inspection["task"] == "compression", "only ordinary compression TRAIN is admitted")
    isolation = inspection["source_isolation"]
    insist(isolation["source_id_overlap"] == isolation["source_sha256_overlap"] == 0, "train/dev overlap")
    insist(inspection["source_id"] in isolation["train_source_ids"] and inspection["source_id"] not in isolation["dev_source_ids"], "wrong train source")
    insist(inspection["source_sha256"] in isolation["train_source_sha256"] and inspection["source_sha256"] not in isolation["dev_source_sha256"], "wrong train source bytes")


def load_inputs(spec, inventory):
    """Return only guide/raw as model inputs; the clean target stays separate."""
    import numpy as np
    cache = checked_json("/", spec["inputs"]["cache_manifest"])
    insist(cache["complete"] is True and cache["phase"] == "completed", "cache e is incomplete")
    i = cache["inspection"]
    validate_identity(i, spec["inputs"]["identity"])
    insist(i["record"] == inventory["source_record"], "source record changed")
    insist(i["record_sha256"] == hashlib.sha256(json.dumps(i["record"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "record canonical hash mismatch")
    fresh = {}
    for p, expected in inventory["selected_files"].items():
        insist(Path(p).stat().st_size == expected["bytes"] and sha_file(p) == expected["sha256"], "selected file pin differs: " + p)
        fresh[p] = expected["sha256"]
    for p, expected in {**i["input_files"], **i["selected_asset_files"]}.items():
        insist(fresh.get(p) == expected, "cache input coverage differs: " + p)
    context = spec["inputs"]["context"]
    insist(sha_file(context["path"]) == context["sha256"], "context changed")
    # Independent source split check from the exact contracted manifests.
    rows = {}
    for split in ("train", "dev"):
        path = next(p for p in fresh if p.endswith("/" + split + "_materialized.jsonl"))
        rows[split] = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        insist(all(x["split"] == split for x in rows[split]), "manifest role mismatch")
    for key in ("source_id", "source_sha256"):
        insist(not ({x[key] for x in rows["train"]} & {x[key] for x in rows["dev"]}), "fresh train/dev source overlap")
    insist(sum(x == i["record"] for x in rows["train"]) == 1, "selected exact record missing/duplicate")
    guide = checked_array(spec["inputs"]["guide"])
    raw = checked_array(spec["inputs"]["warm_start"])
    vae_r = checked_array(spec["inputs"]["cached_VAE_R_provenance_only"])
    insist(tuple(guide.shape) == tuple(raw.shape) == SHAPE and tuple(vae_r.shape) == RGB_SHAPE, "full22 cache geometry changed")
    insist(not np.array_equal(guide, raw), "guide/raw unexpectedly identical")
    for role, key in (("guide", "rphi_guide_latent"), ("warm_start", "raw_warm_start_latent"), ("cached_VAE_R_provenance_only", "vae_rphi_as_returned_internal_clipped")):
        expected = spec["inputs"][role]
        actual = cache["arrays"][key]
        insist(all(actual[k] == expected[k] for k in ("file_sha256", "array_sha256", "shape", "dtype")), "cache artifact/receipt mismatch")
    row = i["export_row"]
    ordinary = next(p for p in fresh if p.endswith("/guides/manifest.json"))
    guide_npz = Path(ordinary).parent / row["guide_npz"]
    with np.load(i["record"]["observation_npz"], allow_pickle=False) as z:
        observation = z["frames"]
        insist(array_sha(observation) == row["observation_pixel_sha256"], "observation pixel mismatch")
        observed_float = observation.astype(np.float32) / np.float32(255.)
    with np.load(guide_npz, allow_pickle=False) as z:
        for key, meta in row["arrays"].items():
            a = z[key]
            insist(str(a.dtype) == meta["dtype"] and list(a.shape) == meta["shape"] and array_sha(a) == meta["array_sha256"], "RGB guide array mismatch: " + key)
        insist(np.array_equal(observed_float, z["raw_draft"]), "warm-start RGB is not actual observed input")
        insist(np.array_equal(np.clip(z["rphi_raw_rgb"], 0, 1), z["rphi_rgb"]), "clipped Rphi cache relation mismatch")
    target_spec = spec["inputs"]["target"]
    insist(target_spec["npz"] == i["record"]["target_npz"] and target_spec["array_sha256"] == i["record"]["target_pixel_sha256"], "target source binding mismatch")
    target = clean_target(target_spec["npz"], target_spec)
    insist(tuple(target.shape) == RGB_SHAPE, "target geometry changed")
    inspection = {"complete": True, "identity": spec["inputs"]["identity"], "cache_manifest": spec["inputs"]["cache_manifest"],
        "verified_selected_files": fresh, "guide": spec["inputs"]["guide"], "warm_start": spec["inputs"]["warm_start"],
        "target": target_spec, "target_float32_sha256": array_sha(target),
        "cached_VAE_R_provenance_only": spec["inputs"]["cached_VAE_R_provenance_only"],
        "context": context, "model_input_keys": ["guide", "raw_warm_start"], "target_model_input": False,
        "target_usage": "loss only, original clean RGB proxy; no resize/clamp/frame selection/encode",
        "Rphi_calls": 0, "VAE_calls": 0, "H3_calls": 0, "torch_imported": "torch" in sys.modules,
        "original_source_verification": inventory["original_source_scope"], "native_geometry_plan": cache["native_geometry_plan"]}
    return {"guide": np.array(guide, copy=True), "raw_warm_start": np.array(raw, copy=True)}, target, inspection


def validate_native_plan(stage):
    plan = {"tiling": stage.tiling, "tile_size": stage.tile_size, "vae_ratio": stage.vae_ratio, "vae_ratio_t": stage.vae_ratio_t,
        "temporal_padding_and_chunks": list(stage._decode_temporal_chunks(7)),
        "height_tile_plan": stage.split_tiles(480), "width_tile_plan": stage.split_tiles(832),
        "output_shape": list(stage.decode_output_shape(SHAPE)), "changed": False}
    insist(plan["tiling"] and plan["tile_size"] == 256 and plan["vae_ratio"] == 16 and plan["vae_ratio_t"] == 4, "native VAE settings changed")
    insist(plan["temporal_padding_and_chunks"] == [0,1] and plan["output_shape"] == [1,3,22,480,832], "native temporal geometry changed")
    insist(json.loads(json.dumps(plan["height_tile_plan"])) == [[0,112,224],[256,256,256],[144,144]], "height tile plan changed")
    insist(json.loads(json.dumps(plan["width_tile_plan"])) == [[0,192,384,576],[256,256,256,256],[64,64,64]], "width tile plan changed")
    return plan


@contextmanager
def native_tile_guard(stage, evidence, torch):
    """Pass through native generators; observe actual grouped input, never choose batch.

    A row is counted before executing its generator. Each yielded tile is counted
    before yielding, so native consumption of exactly four next() calls does not
    depend on a fifth call/GC to complete the observation.
    """
    active = {"row": None}
    original_row, original_pixels, original_decode = stage._decode_tile_row, stage._decode_pixels, stage.decode
    def decode(z, *args, **kwargs):
        phase = evidence.data["phase"]
        require(phase in ("reference_decode", "candidate_decode"), "unexpected stage input phase", evidence)
        arm = phase.split("_")[0]
        meta = save_array(evidence, arm + "_stage_decoder_input", z, torch)
        expected = evidence.data["arrays"][arm + "_decoder_input"]
        require(tuple(z.shape) == SHAPE and z.dtype == torch.float16 and meta["array_sha256"] == expected["array_sha256"], "actual stage decoder input differs", evidence)
        evidence.data["stage_decoder_inputs"][arm] = meta
        return original_decode(z, *args, **kwargs)
    def row(z, x_idx, x_len):
        phase = evidence.data["phase"]
        require(phase in ("reference_decode", "candidate_decode"), "tile row outside two decodes", evidence)
        prior = [r for r in evidence.data["native_tile_rows"] if r["phase"] == phase]
        record = {"phase": phase, "row_index": len(prior), "input_shape": list(z.shape), "x_idx": list(x_idx), "x_len": list(x_len), "pixel_calls": 0, "yielded_tiles": 0, "groups": []}
        evidence.data["native_tile_rows"].append(record)
        require(len(prior) < 3 and tuple(z.shape) == (1,24,7,16,52) and list(x_idx) == [0,192,384,576] and list(x_len) == [256]*4, "native tile row geometry/count differs", evidence)
        active["row"] = record
        try:
            for tile in original_row(z, x_idx, x_len):
                record["yielded_tiles"] += 1
                require(record["yielded_tiles"] <= 4, "native extra tile", evidence)
                yield tile
        finally:
            if active["row"] is record: active["row"] = None
    def pixels(z):
        record = active["row"]
        require(record is not None and record["phase"] == evidence.data["phase"], "decode pixels without admitted row", evidence)
        record["pixel_calls"] += 1
        record["groups"].append(tensor_record(z, torch))
        require(record["pixel_calls"] == 1 and tuple(z.shape) == (4,24,7,16,16) and z.dtype == torch.float16,
                "native tile batch must be4; changed memory schedule/retry forbidden", evidence)
        return original_pixels(z)
    with override_method(stage, "decode", decode), override_method(stage, "_decode_tile_row", row), override_method(stage, "_decode_pixels", pixels):
        yield


def tile_budget_passed(rows):
    return all(len(selected := [r for r in rows if r["phase"] == phase]) == 3 and
        [r["row_index"] for r in selected] == [0,1,2] and
        all(r["pixel_calls"] == 1 and r["yielded_tiles"] == 4 and len(r["groups"]) == 1 and
            r["groups"][0]["shape"] == [4,24,7,16,16] for r in selected)
        for phase in ("reference_decode", "candidate_decode")) and len(rows) == 6


@contextmanager
def bounded_blocks(blocks, group, evidence, observe_math):
    phases = ("reference", "candidate", "backward") if group == "DiT" else ("reference_decode", "candidate_decode", "backward")
    with ExitStack() as scope:
        for i, block in enumerate(blocks):
            original = block.forward
            def wrapped(*args, _i=i, _original=original, **kwargs):
                phase = evidence.data["phase"]
                counts = evidence.data["block_forward_calls"][group].setdefault(phase, {})
                counts[str(_i)] = counts.get(str(_i), 0) + 1
                require(phase in phases and counts[str(_i)] <= 3, "block count/phase exceeded", evidence)
                observe_math(group, phase, _i)
                return _original(*args, **kwargs)
            scope.enter_context(override_method(block, "forward", wrapped))
        yield


def full_rgb_backward(reference_rgb, rgb, video, target, evidence, torch, *, expected_shape=RGB_SHAPE, target_sha256=TARGET_SHA):
    save_array(evidence, "reference_rgb", reference_rgb.float(), torch)
    save_array(evidence, "candidate_rgb", rgb.float(), torch)
    metrics = parity_metrics(reference_rgb, rgb, torch)
    metrics.update(atol=ATOL, rtol=RTOL, allclose=metrics["same_shape"] and metrics["finite"] and bool(torch.allclose(rgb.detach().float().cpu(), reference_rgb.detach().float().cpu(), atol=ATOL, rtol=RTOL)))
    evidence.data["rgb_parity"] = metrics
    evidence.data["graph"] = {"video_requires_grad": video.requires_grad, "video_has_grad_fn": video.grad_fn is not None,
        "rgb_requires_grad": rgb.requires_grad, "rgb_has_grad_fn": rgb.grad_fn is not None}
    require(tuple(rgb.shape) == tuple(expected_shape) and metrics["allclose"], "RGB geometry/parity failed; backward forbidden", evidence)
    require(all(evidence.data["graph"].values()), "retained joint graph absent", evidence)
    require(target.device.type == "cpu" and target.dtype == torch.float32 and tuple(target.shape) == tuple(expected_shape)
        and not target.requires_grad and bool(torch.isfinite(target).all()), "target role/shape/dtype invalid", evidence)
    meta = save_array(evidence, "target", target, torch)
    require(meta["array_sha256"] == target_sha256, "actual clean target bytes differ", evidence)
    require(evidence.data["actual"]["RGB_loss_expressions"] == 0, "second RGB loss forbidden", evidence)
    pixels = rgb.float().cpu()
    evidence.data["actual"]["RGB_loss_expressions"] += 1
    loss = (pixels - target).square().mean()
    import numpy as np
    evidence.data["loss"] = {"torch_float32_mse": float(loss.detach()) if bool(torch.isfinite(loss)) else None,
        "numpy_float64_mse": float(np.mean((pixels.detach().numpy().astype(np.float64)-target.numpy().astype(np.float64))**2)),
        "scale": LOSS_SCALE, "recipe": "mean((candidate_RGB.float().cpu()-clean_target_float32)**2)",
        "target_recipe": "exact TRAIN target uint8.astype(float32)/float32(255); no resize or P",
        "RGB_semantics": "as_returned_internal_clipped; no extra clipping/quantization", "auxiliary_latent_audio_loss": False}
    require(bool(torch.isfinite(loss)) and loss.requires_grad, "RGB loss disconnected/nonfinite", evidence)
    require(evidence.data["actual"]["backward_attempts"] == 0, "second backward forbidden", evidence)
    evidence.phase("backward", torch if hasattr(torch, "npu") else None)
    evidence.data["actual"]["backward_attempts"] += 1
    (loss * LOSS_SCALE).backward()
    evidence.data["actual"]["completed_backward_calls"] += 1
    evidence.save()


def execute(args, contract, evidence, inputs, target_array):
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
        "synthetic_no_dataset": False, "role": "train", "quality_claim": False}
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
    evidence.data["vae_native_plan"] = validate_native_plan(stage)
    torch.npu.reset_peak_memory_stats()
    evidence.phase("initial_resident_fingerprints", torch)
    baseline = {"DiT": frozen_inventory(dit, torch), "VAE": frozen_inventory(stage, torch)}
    atomic_json(evidence.path / "frozen_initial.json", baseline)
    evidence.data["frozen_initial"] = {"path": "frozen_initial.json", "sha256": sha_file(evidence.path / "frozen_initial.json")}
    guide = torch.from_numpy(inputs["guide"])
    raw = torch.from_numpy(inputs["raw_warm_start"])
    target = torch.from_numpy(target_array)
    save_array(evidence, "guide", guide, torch)
    save_array(evidence, "raw_warm_start", raw, torch)
    save_array(evidence, "target", target, torch)
    evidence.data["inputs"] = {"seed": SEED, "audio_init": "zeros", "context_sha256": contract["context_sha256"],
        "distinct": not bool(torch.equal(guide, raw)), "target_is_never_model_input": True,
        "model_input_keys": ["guide", "raw_warm_start"], "source_id": "2BdYaGuVAKo", "role": "train"}
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
            active_phase = evidence.data["phase"]
            maximum = {"candidate_decode": 109, "backward": 108}.get(active_phase)
            current_count = sum(r["phase"] == active_phase for r in evidence.data["decoder_context_restorations"])
            require(maximum is not None and current_count < maximum, "decoder context budget/phase exceeded", evidence)
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
    original_methods += [(stage, n) for n in ("decode", "_decode_tile_row", "_decode_pixels")]
    original_methods += [(h3, n) for n in ("encode", "encode_guide", "encode_guide_pixels", "ref_block") if hasattr(h3, n)]
    original_methods += [(comfy.sample, "prepare_noise")]
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
            scope.enter_context(native_tile_guard(stage, evidence, torch))
            scope.enter_context(observe_noise(comfy.sample, comfy.utils.pack_latents, phase, evidence.data["noise_observations"], torch))
            scope.enter_context(bounded_blocks(dit.blocks, "DiT", evidence, observe_math))
            scope.enter_context(bounded_blocks(stage.decoder.transformer_blocks, "VAE", evidence, observe_math))
            evidence.phase("reference", torch)
            reference, evidence.data["reference_record"] = h3.sample(guide, mode="base", hw=(480, 832), seed=SEED,
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
                require(tuple(candidate["video"].shape) == tuple(reference["video"].shape) == SHAPE and
                    tuple(candidate["audio"].shape) == tuple(reference["audio"].shape) == AUDIO_SHAPE,
                    "actual final joint geometry differs", evidence)
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
                require(tile_budget_passed(evidence.data["native_tile_rows"]), "two-path native tile group budget differs", evidence)
                full_rgb_backward(reference_rgb, rgb, candidate["video"], target, evidence, torch)
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
        for p in ("reference", "candidate", "backward")) and all(counts["VAE"].get(p) == {str(i): 3 for i in range(36)}
        for p in ("reference_decode", "candidate_decode", "backward")))
    evidence.data["tile_budget_passed"] = tile_budget_passed(evidence.data["native_tile_rows"])
    contexts = evidence.data["decoder_context_restorations"]
    evidence.data["decoder_context_budget_passed"] = (len(contexts) == 217 and
        sum(r["phase"] == "candidate_decode" for r in contexts) == 109 and
        sum(r["phase"] == "backward" for r in contexts) == 108)
    actual = evidence.data["actual"]
    evidence.data["budget_passed"] = (actual["reference_NFE"] == actual["candidate_NFE"] == 3 and
        actual["reference_VAE_decode_calls"] == actual["candidate_VAE_decode_calls"] == 1 and
        actual["stage_decode_calls"] == {"reference_decode": 1, "candidate_decode": 1} and
        actual["backward_attempts"] == actual["completed_backward_calls"] == actual["RGB_loss_expressions"] == 1
        and actual["optimizer_steps"] == actual["adapter_updates"] == 0 and not actual["forbidden_calls"])
    require(evidence.data["block_budget_passed"] and evidence.data["budget_passed"] and
        evidence.data["tile_budget_passed"] and evidence.data["decoder_context_budget_passed"], "execution budget mismatch", evidence)
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
    evidence.data["scope"] = "full22 TRAIN same-graph RGB engineering only; no optimizer, quality, causal/streaming or full32 training acceptance"
    evidence.phase("completed", torch)
    atomic_json(evidence.path / "report.json", evidence.data)


def inspect(args):
    contract = json.loads(args.execution_contract.read_text())
    insist(contract["schema"] == "openh3.full22_joint_rgb_probe_execution.v1", "wrong execution schema")
    insist(contract["probe_sha256"] == sha_file(__file__), "entry source not pinned")
    root = Path(contract["root"])
    spec = checked_json(root, contract["source_contract"], SOURCE_CONTRACT_SHA)
    review = checked_json(root, contract["root_preparation_review"], ROOT_REVIEW_SHA)
    insist(review["status"] == "approved_for_new_implementation_only_not_executable", "wrong preparation review")
    accepted = checked_json(root, contract["nonzero_terminal_audit"], NONZERO_AUDIT_SHA)
    insist(accepted["complete"] is True and accepted["numerical_acceptance"] == "passed_small_single_SGD_strict_reload_merge_descriptive_only", "nonzero b prerequisite not accepted")
    inventory = checked_json(root, contract["input_inventory"], INVENTORY_SHA)
    insist(sha_file(args.pretrained_manifest) == contract["manifest_sha256"] == MANIFEST_SHA, "pretrained manifest differs")
    verified = verify_pins(root, contract["implementation_sha256"])
    native_vae_source = (Path(contract["comfy_root"]) / "comfy/ldm/minimax/vae.py").resolve()
    insist(verified.get(str(native_vae_source)) == VAE_SOURCE_SHA, "native VAE source differs from reviewed tile plan")
    # The loaded Python helpers must be bound, not merely an unrelated copy.
    for module in (J, sys.modules["check_h3_trajectory_grad"]):
        source = str(Path(module.__file__).resolve())
        insist(verified.get(source) == sha_file(source), "actual helper import unbound: " + source)
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    insist(sha_file(manifest["_paths"]["context"]) == contract["context_sha256"] == spec["inputs"]["context"]["sha256"], "context differs")
    insist(manifest["_fingerprint"] == inventory["identity"]["pretrained_fingerprint"], "recipe fingerprint changed")
    inputs, target, inspection = load_inputs(spec, inventory)
    inspection.update(schema="openh3.full22_joint_rgb_input_inspection.v1", inspection_only=not args.execute,
        execution_contract_sha256=sha_file(args.execution_contract), source_contract_sha256=SOURCE_CONTRACT_SHA,
        root_preparation_review_sha256=ROOT_REVIEW_SHA, nonzero_terminal_audit_sha256=NONZERO_AUDIT_SHA,
        runtime_files_verified=len(verified), model_assets_size_only_verified=True,
        model_asset_bytes_SHA_verified=False, nonzero_adapter_weights_loaded=False,
        planned={"NFE": 6, "VAE_decodes": 2, "VAE_encodes": 0, "Rphi_calls": 0, "RGB_backward": 1,
                 "loss_scale": LOSS_SCALE, "optimizer_steps": 0, "adapter_updates": 0})
    insist("torch" not in sys.modules and "torch_npu" not in sys.modules and "comfy" not in sys.modules, "CPU inspection imported model runtime")
    return contract, verified, inputs, target, inspection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrained-manifest", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    insist(not args.out.exists(), "output exists; overwrite forbidden")
    contract, verified, inputs, target, inspection = inspect(args)
    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=False)
        atomic_json(args.out / "inspection.json", inspection)
        print(json.dumps({"complete": True, "inspection_only": True, "model_calls": 0, "source_id": inspection["identity"]["source_id"]}), flush=True)
        return
    insist(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "").isdecimal(), "one explicitly admitted physical card required")
    admission = checked_json(Path(contract["root"]), contract["root_admission"])
    insist(admission["approved"] is True and admission["scope"] == "full22_single_joint_engineering_probe" and
        admission["probe_sha256"] == contract["probe_sha256"] and admission["execution_payload_sha256"] == execution_payload_sha256(contract), "wrong fresh root admission")
    evidence = Evidence(args.out, {**contract, "execution_contract_sha256": sha_file(args.execution_contract)})
    evidence.data["input_inspection"] = inspection
    evidence.data["verified_runtime_files"] = verified
    try:
        execute(args, contract, evidence, inputs, target)
    except BaseException as error:
        evidence.data["complete"] = False
        evidence.data["failure"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(),
            "phase": evidence.data.get("failure_origin_phase", evidence.data["phase"])}
        evidence.save()
        atomic_json(evidence.path / "failure.json", evidence.data)
        raise


def execution_payload_sha256(contract):
    payload = {k:v for k,v in contract.items() if k != "root_admission"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


if __name__ == "__main__":
    main()
