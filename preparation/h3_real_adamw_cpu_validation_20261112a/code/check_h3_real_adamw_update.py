"""One real-TRAIN full22 AdamW update, exact reload, descriptive BF16 merge.

Preparation/CPU inspection is default. This is not a trainer or NPU admission.
The accepted full22 reference is reused: exactly12NFE/4decode/1RGBbackward/
1AdamW, no encode/Rphi or extra reference. Old pinned sources are unchanged.
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
import check_h3_full22_joint_rgb_grad as F
import check_h3_nonzero_adapter as N
from check_h3_trajectory_grad import (MANIFEST_SHA, TARGETS, atomic_json, sha_file,
    tensor_sha, tensor_record, canonical_base_items, parity_metrics, observe_noise,
    runtime_imports, verify_pins)

PROTOCOL_SHA = 'e407367aa14f9039da7e14cc0b78d652e30b7b76ca4c21a146ef9ccbc19bd413'
REFERENCE_REPORT_SHA = 'bf3db9f4eea9f9d2e29ea8c014e15cdd08e725712fe6f2e33cca07c6b51bae0f'
REFERENCE_AUDIT_SHA = 'fd6df63d3121474d9884e91d90514f273a15041ef5b8c6686d22a65e3c72e1f7'
REFERENCE_REVIEW_SHA = 'aabfc788ab872f63e99af10b4bfaf677d03cebe2787aa91ace3fd6f820b15821'
SEED, LOSS_SCALE = 101, 1024.
BASE_LR, WARMUP, GLOBAL_STEP = 1e-4, 100, 1
LR = BASE_LR * min(1., GLOBAL_STEP / WARMUP)
MODES = ('zero', 'updated', 'reloaded', 'merged')
SHAPE, AUDIO_SHAPE, RGB_SHAPE = F.SHAPE, F.AUDIO_SHAPE, F.RGB_SHAPE
require = J.require


class Evidence(N.Evidence):
    def __init__(self, path, contract):
        super().__init__(path, contract)
        self.data.update(schema='openh3.real_train_adamw_single_update.v1',
            scope='one TRAIN compression engineering probe, not quality/generalization',
            native_tile_rows=[], stage_decoder_inputs={})
        self.data['actual'].update(clip_calls=0, optimizer_state_saves=0,
            optimizer_state_clear_reload=0, Rphi_calls=0)
        self.save()


def optimizer_recipe():
    return {'class': 'torch.optim.AdamW', 'base_lr': BASE_LR, 'warmup_steps': WARMUP,
        'global_step': GLOBAL_STEP, 'actual_lr': LR,
        'lr_rule': 'base_lr * min(1.0, global_step / warmup_steps)',
        'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0., 'amsgrad': False,
        'maximize': False, 'capturable': False, 'differentiable': False, 'foreach': False, 'fused': False,
        'clip_grad_norm': {'max_norm': 1., 'norm_type': 2., 'error_if_nonfinite': True, 'foreach': False},
        'loss_scale': LOSS_SCALE}


def adapter_parameters(model, expected, torch):
    import lora as L
    params = dict(L.lora_parameters(model)); N.validate_state(params, expected, torch)
    if len({id(p) for p in params.values()}) != len(params): raise ValueError('aliased adapter parameters')
    return params


def save_tensor_set(evidence, label, values, torch):
    """Incremental real-array evidence; flush metadata every16 arrays and on error."""
    import numpy as np
    rows = {}; evidence.data[label] = rows
    try:
        for i, (name, value) in enumerate(values.items()):
            key = label + '__' + name
            path = evidence.path / 'arrays' / (key + '.npy')
            if path.exists(): raise FileExistsError('array evidence must not overwrite: ' + key)
            array = value.detach().cpu().contiguous().numpy()
            np.save(path, array, allow_pickle=False)
            row = {'path': str(path.relative_to(evidence.path)), 'file_sha256': sha_file(path),
                'array_sha256': hashlib.sha256(array.tobytes()).hexdigest(), 'dtype': str(array.dtype),
                'shape': list(array.shape), 'role': key}
            rows[name] = row; evidence.data['arrays'][key] = row
            if i % 16 == 0: evidence.save()
    finally: evidence.save()
    return rows


def optimizer_inventory(optimizer, params, expected, evidence, torch, label):
    require(set(optimizer.state) == set(params.values()), 'optimizer state not exact adapter objects', evidence)
    require(len(optimizer.param_groups) == 1 and
            [id(p) for p in optimizer.param_groups[0]['params']] == [id(p) for p in params.values()],
            'optimizer parameter identity/order changed', evidence)
    group = optimizer.param_groups[0]
    wanted = dict(lr=LR, betas=(.9, .999), eps=1e-8, weight_decay=0., amsgrad=False,
        maximize=False, capturable=False, differentiable=False, foreach=False, fused=False)
    require(all(group[k] == v for k, v in wanted.items()), 'optimizer actual hyperparameters changed', evidence)
    tensors, inventory = {}, {}
    for name, param in params.items():
        state = optimizer.state[param]
        require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'}, 'unexpected AdamW state keys: ' + name, evidence)
        for key, value in state.items():
            shape = () if key == 'step' else tuple(expected[name])
            require(value.dtype == torch.float32 and tuple(value.shape) == shape
                    and bool(torch.isfinite(value).all()), 'invalid FP32 optimizer state: ' + name + '/' + key, evidence)
            if key == 'step': require(float(value.detach().cpu()) == 1., 'optimizer step must equal1', evidence)
            else: require(value.device == param.device, 'optimizer moment device differs', evidence)
            if key == 'exp_avg_sq': require(bool((value >= 0).all()), 'negative second moment', evidence)
            tensors[name + '::' + key] = value
        inventory[name] = {key: {'device': str(value.device), 'shape': list(value.shape),
            'dtype': str(value.dtype), 'sha256': tensor_sha(value, torch)} for key, value in state.items()}
    arrays = save_tensor_set(evidence, label, tensors, torch)
    return {'named_entries': len(inventory), 'state_tensors': len(tensors), 'inventory': inventory,
        'arrays': arrays, 'parameter_order': list(params),
        'parameter_group': {k: v for k, v in optimizer.state_dict()['param_groups'][0].items() if k != 'params'}}


def one_adamw_update(model, expected, evidence, torch):
    params = adapter_parameters(model, expected, torch)
    a = evidence.data['actual']
    require(all(a[k] == 0 for k in ('optimizer_steps', 'adapter_updates', 'unscale_calls', 'clip_calls')),
            'second update/unscale/clip forbidden', evidence)
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and bool(torch.isfinite(p.grad).all())
                for p in params.values()), 'missing/nonfinite FP32 gradient; cannot fill None', evidence)
    before = {n: p.detach().cpu().clone() for n, p in params.items()}
    save_tensor_set(evidence, 'parameters_before_update', before, torch)
    a['unscale_calls'] += 1
    with torch.no_grad():
        for name, p in params.items():
            p.grad.div_(LOSS_SCALE)
            require(tensor_sha(p.grad, torch) == evidence.data['adapter_gradients']['tensors'][name]['unscaled']['array_sha256'],
                    'actual single unscale differs from saved FP32 gradient: ' + name, evidence)
    require(all(bool(torch.isfinite(p.grad).all()) for p in params.values()), 'unscaled gradients nonfinite', evidence)
    a['clip_calls'] += 1
    norm = torch.nn.utils.clip_grad_norm_(list(params.values()), max_norm=1., norm_type=2.,
        error_if_nonfinite=True, foreach=False)
    evidence.data['gradient_clip'] = {'pre_clip_total_norm': float(norm.detach().cpu()),
        'recipe': optimizer_recipe()['clip_grad_norm'], 'applied_once': True}
    require(bool(torch.isfinite(norm)) and all(bool(torch.isfinite(p.grad).all()) for p in params.values()),
            'clipped gradients nonfinite', evidence)
    save_tensor_set(evidence, 'gradient_clipped', {n: p.grad for n, p in params.items()}, torch)
    optimizer = torch.optim.AdamW(list(params.values()), lr=LR, betas=(.9, .999), eps=1e-8,
        weight_decay=0., amsgrad=False, maximize=False, capturable=False, differentiable=False,
        foreach=False, fused=False)
    require(not optimizer.state, 'AdamW must start without inherited state', evidence)
    evidence.data['optimizer'] = optimizer_recipe()
    a['optimizer_steps'] += 1; optimizer.step(); a['adapter_updates'] += 1
    rows, b_changed = {}, 0
    for name, p in params.items():
        after = p.detach().cpu(); delta = after.double() - before[name].double()
        count = int(torch.count_nonzero(delta)); all_finite = bool(torch.isfinite(after).all())
        rows[name] = {'changed_coordinates': count, 'before_sha256': tensor_sha(before[name], torch),
            'after_sha256': tensor_sha(after, torch), 'update_l2': N.finite_stat(delta.norm()),
            'update_rms': N.finite_stat(delta.square().mean().sqrt()), 'all_finite': all_finite,
            'zero_gradient_coordinates_unchanged': bool(torch.equal(after[p.grad.detach().cpu() == 0],
                                                                   before[name][p.grad.detach().cpu() == 0]))}
        if name.endswith('.lora_b'): b_changed += count
    save_tensor_set(evidence, 'parameters_after_update', params, torch)
    evidence.data['update'] = {'tensors': rows, 'B_changed_coordinates': b_changed,
        'A_zero_gradient_is_legal': True, 'parameter_change_does_not_prove_functional_effect': True,
        'backend_AdamW_formula_reexecution_claim': False}
    evidence.data['optimizer_after_update'] = optimizer_inventory(optimizer, params, expected, evidence, torch, 'optimizer_state_after_update')
    require(b_changed > 0 and all(r['all_finite'] and r['zero_gradient_coordinates_unchanged'] for r in rows.values()),
            'nonfinite/update/zero-coordinate identity failure', evidence)
    return optimizer


def save_checkpoint(model, optimizer, expected, pretrained, diagnostic, evidence, torch):
    import lora as L
    from safetensors.torch import save_file, load_file
    from openh3_runtime import adapter_metadata
    params = adapter_parameters(model, expected, torch)
    require(evidence.data['actual']['checkpoint_saves'] == evidence.data['actual']['optimizer_state_saves'] == 0,
            'second checkpoint save forbidden', evidence)
    adapter_path = evidence.path / 'real_train_AdamW_step1.safetensors'
    optimizer_path = evidence.path / 'real_train_AdamW_step1_optimizer.safetensors'
    require(not adapter_path.exists() and not optimizer_path.exists(), 'checkpoint overwrite forbidden', evidence)
    state = L.state_dict(model); N.validate_state(state, expected, torch)
    save_file(state, str(adapter_path), metadata=adapter_metadata(pretrained, diagnostic, rank=8))
    evidence.data['actual']['checkpoint_saves'] += 1
    saved, meta = N.strict_saved_state(adapter_path, expected, pretrained, diagnostic, torch)
    require(all(torch.equal(state[n], saved[n]) for n in params), 'adapter checkpoint differs from live update', evidence)
    optim_state = {n + '::' + k: v.detach().cpu().contiguous() for n, p in params.items() for k, v in optimizer.state[p].items()}
    metadata = {'schema': 'openh3.named_AdamW_state.v1', 'adapter_sha256': sha_file(adapter_path),
        'parameter_order': json.dumps(list(params)), 'optimizer_recipe': json.dumps(optimizer_recipe(), sort_keys=True),
        'training_contract': json.dumps(diagnostic, sort_keys=True)}
    save_file(optim_state, str(optimizer_path), metadata=metadata)
    evidence.data['actual']['optimizer_state_saves'] += 1
    reloaded = load_file(str(optimizer_path), device='cpu')
    require(set(reloaded) == set(optim_state) and all(torch.equal(optim_state[k], reloaded[k]) for k in optim_state),
            'optimizer checkpoint bytes differ from live state', evidence)
    evidence.data['checkpoint'] = {'path': adapter_path.name, 'sha256': sha_file(adapter_path),
        'bytes': adapter_path.stat().st_size, 'metadata': meta, 'expected_shapes': {n: list(s) for n, s in expected.items()},
        'exact_saved_live_tensors': {n: bool(torch.equal(state[n], saved[n])) for n in params},
        'role': 'one_real_TRAIN_AdamW_engineering_step_not_quality_checkpoint',
        'optimizer': {'path': optimizer_path.name, 'sha256': sha_file(optimizer_path),
            'bytes': optimizer_path.stat().st_size, 'metadata': metadata,
            'parameter_order': list(params), 'state_tensors': len(optim_state)}}
    evidence.save()
    return adapter_path, optimizer_path


def clear_reload(model, optimizer, expected, paths, pretrained, diagnostic, evidence, torch):
    import lora as L
    from safetensors import safe_open
    from safetensors.torch import load_file
    adapter_path, optim_path = paths
    params = adapter_parameters(model, expected, torch)
    require(evidence.data['actual']['clear_reload_cycles'] == evidence.data['actual']['optimizer_state_clear_reload'] == 0,
            'second clear/reload forbidden', evidence)
    ckpt = evidence.data['checkpoint']; optim_meta = ckpt['optimizer']
    require(sha_file(adapter_path) == ckpt['sha256'] and sha_file(optim_path) == optim_meta['sha256'],
            'checkpoint changed before reload', evidence)
    state, meta = N.strict_saved_state(adapter_path, expected, pretrained, diagnostic, torch)
    with safe_open(optim_path, framework='pt', device='cpu') as f:
        require(f.metadata() == optim_meta['metadata'], 'optimizer metadata identity differs', evidence)
    saved_optim = load_file(str(optim_path), device='cpu')
    expected_keys = {n + '::' + k for n in params for k in ('step', 'exp_avg', 'exp_avg_sq')}
    require(set(saved_optim) == expected_keys and optim_meta['parameter_order'] == list(params),
            'optimizer key/parameter order differs', evidence)
    for name in params:
        for key in ('step', 'exp_avg', 'exp_avg_sq'):
            value = saved_optim[name + '::' + key]
            require(value.dtype == torch.float32 and tuple(value.shape) == (() if key == 'step' else tuple(expected[name]))
                    and bool(torch.isfinite(value).all()), 'saved optimizer state dtype/shape/finite differs', evidence)
            require(tensor_sha(value, torch) == evidence.data['optimizer_after_update']['inventory'][name][key]['sha256'],
                    'saved optimizer state differs from actual update', evidence)
    with torch.no_grad():
        for p in params.values(): p.zero_()
        for entry in optimizer.state.values():
            for value in entry.values(): value.zero_()
    cleared = {'parameters_all_zero': all(not bool(torch.count_nonzero(p)) for p in params.values()),
        'optimizer_tensors_all_zero': all(not bool(torch.count_nonzero(v)) for entry in optimizer.state.values() for v in entry.values())}
    optimizer.state.clear(); cleared['optimizer_mapping_empty'] = not optimizer.state
    evidence.data['reload'] = {'cleared': cleared}; evidence.save()
    require(all(cleared.values()), 'clear did not actually clear parameters/state', evidence)
    count = L.load(model, state, strict=True, strict_unused=True)
    groups = optimizer.state_dict()['param_groups']
    restored = {i: {k: saved_optim[n + '::' + k].clone() for k in ('step', 'exp_avg', 'exp_avg_sq')}
                for i, n in enumerate(params)}
    optimizer.load_state_dict({'state': restored, 'param_groups': groups})
    evidence.data['actual']['clear_reload_cycles'] += 1
    evidence.data['actual']['optimizer_state_clear_reload'] += 1
    inventory = optimizer_inventory(optimizer, params, expected, evidence, torch, 'optimizer_state_after_reload')
    exact = {n: bool(torch.equal(p.detach().cpu(), state[n])) for n, p in params.items()}
    state_exact = {n: all(inventory['inventory'][n][k]['sha256'] == evidence.data['optimizer_after_update']['inventory'][n][k]['sha256']
                          for k in ('step', 'exp_avg', 'exp_avg_sq')) for n in params}
    evidence.data['reload'].update(loaded_count=count, exact_adapter=exact, exact_optimizer=state_exact,
        optimizer_inventory=inventory, metadata=meta, resume_next_step_trajectory_equivalence_claim=False)
    require(count == len(expected) and all(exact.values()) and all(state_exact.values())
            and sha_file(adapter_path) == ckpt['sha256'] and sha_file(optim_path) == optim_meta['sha256'],
            'strict adapter/optimizer reload failed', evidence)


def save_mode(mode, output, rgb, target, evidence, torch):
    import numpy as np
    for stream in ('video', 'audio'): J.save_array(evidence, mode + '_' + stream, output[stream].float(), torch)
    J.save_array(evidence, mode + '_rgb', rgb.float(), torch)
    pixels = rgb.detach().float().cpu()
    valid = tuple(output['video'].shape) == SHAPE and tuple(output['audio'].shape) == AUDIO_SHAPE and tuple(pixels.shape) == RGB_SHAPE
    valid = valid and all(bool(torch.isfinite(x).all()) for x in (output['video'], output['audio'], pixels))
    evidence.data['mode_outputs'][mode] = {'geometry_and_finite': valid,
        'torch_float32_mse': float((pixels - target).square().mean()) if valid else None,
        'numpy_float64_mse': float(np.mean((pixels.numpy().astype(np.float64) - target.numpy().astype(np.float64)) ** 2)) if valid else None,
        'clipping': {'at_zero': int((pixels == 0).sum()), 'at_one': int((pixels == 1).sum()),
            'outside_unit_interval': int(((pixels < 0) | (pixels > 1)).sum()), 'semantics': 'native internally clipped; no extra clamp'}}
    require(valid, 'full22 mode output geometry/finite failed', evidence)


def compare_modes(left, right, evidence, torch, strict=False):
    rows = {}
    for stream in ('video', 'audio', 'decoder_input', 'rgb'):
        a, b = (torch.from_numpy(N.load_array(evidence.path, evidence.data['arrays'][m + '_' + stream])) for m in (left, right))
        rows[stream] = {**parity_metrics(a, b, torch), 'changed_coordinates': int(torch.count_nonzero(a != b)),
                       'acceptance': 'strict torch.equal' if strict else 'descriptive; no equivalence threshold'}
    evidence.data.setdefault('comparisons', {})[left + '__' + right] = rows
    require(not strict or all(r['torch_equal'] for r in rows.values()), 'same-mode strict reload output differs', evidence)
    return rows


def functional_result(comparisons):
    return {'resident_unmerged_RGB_changed': comparisons['zero__updated']['rgb']['changed_coordinates'] > 0,
            'deployed_merged_RGB_changed': comparisons['zero__merged']['rgb']['changed_coordinates'] > 0,
            'same_sample_loss_descent_required': False, 'quality_or_generalization_claim': False,
            'no_LR_rank_step_adjustment': True}


save_array = J.save_array
override_method = J.override_method


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
        require(phase in tuple(m + "_decode" for m in MODES), "unexpected stage input phase", evidence)
        arm = phase.split("_")[0]
        meta = save_array(evidence, arm + "_stage_decoder_input", z, torch)
        expected = evidence.data["arrays"][arm + "_decoder_input"]
        require(tuple(z.shape) == SHAPE and z.dtype == torch.float16 and meta["array_sha256"] == expected["array_sha256"], "actual stage decoder input differs", evidence)
        evidence.data["stage_decoder_inputs"][arm] = meta
        return original_decode(z, *args, **kwargs)
    def row(z, x_idx, x_len):
        phase = evidence.data["phase"]
        require(phase in tuple(m + "_decode" for m in MODES), "tile row outside four decodes", evidence)
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
        for phase in tuple(m + "_decode" for m in MODES)) and len(rows) == 12


@contextmanager
def bounded_blocks(blocks, group, evidence, observe_math):
    phases = (*MODES, "backward") if group == "DiT" else (*(m + "_decode" for m in MODES), "backward")
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


def execute(args, contract, evidence, prior, prior_arrays, prior_inventory, inputs, target_array):
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
        "python": sys.version, "visible_device": os.environ["ASCEND_RT_VISIBLE_DEVICES"], "synthetic_no_dataset": False, "role": "train", "quality_claim": False}
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
    require(N.stable_pretrained(h3.record) == N.stable_pretrained(prior["pretrained"]) and vae.vae_dtype == torch.float16,
            "joint recipe/VAE precision differs", evidence)
    preflight = ResidentH3Trajectory(h3, checkpoint_blocks=True, checkpoint_decoder=True)
    preflight._validate_resident(); preflight._validate_vae()
    require(len(dit.blocks) == 50 and len(stage.decoder.transformer_blocks) == 36, "wrong block inventory", evidence)
    evidence.data["vae_native_plan"] = F.validate_native_plan(stage)
    require(json.loads(json.dumps(evidence.data["vae_native_plan"])) == prior["vae_native_plan"],
            "native full22 VAE geometry/padding differs", evidence)
    initial_math = J.actual_math_state(torch)
    evidence.data["math_before"] = initial_math
    flags = ("in_training", "jit_compile_false", "matmul_allow_hf32", "conv_allow_hf32",
             "deterministic_algorithms", "deterministic_warn_only")
    require(all(initial_math[k] == prior["math_before"][k] for k in flags), "joint runtime math flags differ", evidence)
    evidence.data["actual_imports_before_sampling"] = runtime_imports(evidence.data["verified_runtime_files"])
    require(evidence.data["actual_imports_before_sampling"]["passed"], "unbound runtime import", evidence)
    torch.npu.reset_peak_memory_stats()
    names, session, snapshots, adapters, optimizer = [], None, None, None, None
    original_patcher = getattr(model, "current_patcher", None)
    original_shapes = (hasattr(model, "latent_shapes"), getattr(model, "latent_shapes", None))
    methods = [(b,"forward") for b in list(dit.blocks)+list(stage.decoder.transformer_blocks)]
    methods += [(owner,n) for owner, ns in ((h3,("encode","encode_guide","encode_guide_pixels","ref_block","sample")),
        (vae,("decode","encode","decode_tiled_1d","_decode_tiled_owned","decode_tiled_","decode_tiled_3d")),
        (stage,("decode","_decode_tile_row","_decode_pixels")), (comfy.sample,("prepare_noise",))) for n in ns if hasattr(owner,n)]
    wrappers = J.wrapper_snapshot(methods)
    guide, raw = (torch.from_numpy(inputs[n].copy()) for n in ("guide","raw_warm_start"))
    target = torch.from_numpy(target_array)
    require(all(torch.equal(torch.from_numpy(prior_arrays[n]), value) for n, value in
                (("guide",guide),("raw_warm_start",raw),("target",target))),
            "accepted full22 inputs/target do not match actual cache e", evidence)
    for n,v in (("guide",guide),("raw_warm_start",raw),("target",target)): J.save_array(evidence,n,v,torch)
    for n in ("reference_video","reference_audio","reference_rgb","reference_decoder_input"):
        J.save_array(evidence,n,torch.from_numpy(prior_arrays[n]),torch)
    evidence.data["inputs"] = {"seed": SEED, "audio_init":"zeros", "target_used_as_model_input":False,
        "source":"exact accepted full22 reference plus independently verified cachee TRAIN compression",
        "model_input_keys":["guide","raw_warm_start"],"source_id":"2BdYaGuVAKo","role":"train"}
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
            phase = evidence.data["phase"]
            maximum = {"zero_decode":109,"backward":108,"updated_decode":1,"reloaded_decode":1,"merged_decode":1}.get(phase)
            count = sum(r["phase"] == phase for r in evidence.data["decoder_context_restorations"])
            require(maximum is not None and count < maximum,"decoder context phase/budget exceeded",evidence)
            with decoder_math(native_training_math,"ascend_native"): yield
        finally:
            after = J.actual_math_state(torch)
            evidence.data["decoder_context_restorations"].append({"phase":evidence.data["phase"],
                "before":before,"after":after,"restored":before==after})
            if before != after: raise RuntimeError("native decoder context did not restore")
    def audit_state(label, full=False):
        evidence.phase(("full_state_hash_" if full else "lifecycle_")+label, torch)
        check = J.check_snapshot if full else N.check_lifecycle
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
        expected=N.expected_adapters(dit,torch)
        evidence.data["adapter_initial"]=J.adapter_audit(dit,names,torch)
        require(evidence.data["adapter_initial"] == prior["adapter_initial"],
                "fresh seed101 adapter bytes differ from accepted full22 initialization", evidence)
        adapters=J.adapter_snapshot(dit,torch)
        audit_state("after_injection")
        with ExitStack() as scope:
            scope.enter_context(N.guarded_vae(h3,evidence,torch))
            scope.enter_context(native_tile_guard(stage,evidence,torch))
            scope.enter_context(observe_noise(comfy.sample,comfy.utils.pack_latents,lambda:evidence.data["phase"],evidence.data["noise_observations"],torch))
            scope.enter_context(bounded_blocks(dit.blocks,"DiT",evidence,observe_math))
            scope.enter_context(bounded_blocks(stage.decoder.transformer_blocks,"VAE",evidence,observe_math))
            for mode in MODES:
                if mode=="updated":
                    evidence.phase("single_real_AdamW_update",torch)
                    require(current_candidate is None,"prior graph still retained before optimizer",evidence)
                    optimizer=one_adamw_update(dit,expected,evidence,torch)
                    diagnostic={"role":"one_real_TRAIN_AdamW_engineering_step_not_quality_checkpoint",
                        "execution_contract_sha256":evidence.data["contract"]["execution_contract_sha256"],
                        "protocol_sha256":PROTOCOL_SHA,"manifest_sha256":MANIFEST_SHA,
                        "implementation_sha256":contract["implementation_sha256"],
                        "input_identity":evidence.data["input_inspection"]["identity"],
                        "input_arrays":{n:evidence.data["arrays"][n] for n in ("guide","raw_warm_start","target")},
                        "noise":evidence.data["noise_observations"][0],"sigmas":list(NATIVE_SIGMAS),
                        "sigmas_float32_sha256":tensor_sha(torch.tensor(NATIVE_SIGMAS,dtype=torch.float32),torch),
                        "rank":8,"alpha":8,"steps":1,"loss_scale":LOSS_SCALE,
                        "objective":"original clean proxy RGB MSE only", "optimizer":evidence.data["optimizer"]}
                    checkpoint_paths=save_checkpoint(dit,optimizer,expected,h3.record["openh3_pretrained"],diagnostic,evidence,torch)
                    adapters=J.adapter_snapshot(dit,torch)
                    audit_state("after_update_and_save",full=True)
                elif mode=="reloaded":
                    evidence.phase("clear_and_strict_reload",torch)
                    clear_reload(dit,optimizer,expected,checkpoint_paths,h3.record["openh3_pretrained"],diagnostic,evidence,torch)
                    adapters=J.adapter_snapshot(dit,torch)
                    audit_state("after_reload")
                elif mode=="merged":
                    evidence.phase("descriptive_BF16_merge",torch)
                    # Release optimizer moments and grads before weight merging.
                    optimizer.state.clear(); del optimizer; optimizer=None
                    for _, p in __import__("lora").lora_parameters(dit): p.grad=None
                    evidence.data["optimizer_released_before_merge"]=True
                    snapshots["DiT"]=N.merge_once(dit,names,snapshots["DiT"],evidence,torch)
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
                        save_mode(mode,current_candidate,rgb,target,evidence,torch)
                        mode_rows=[r for r in evidence.data["native_tile_rows"] if r["phase"]==mode+"_decode"]
                        require(len(mode_rows)==3 and all(r["pixel_calls"]==1 and r["yielded_tiles"]==4 for r in mode_rows),
                                "native three-row four-tile decode incomplete", evidence)
                        audit_state(mode+"_after_decode")
                        if mode=="zero":
                            require(evidence.data["arrays"]["zero_decoder_input"]["array_sha256"]==
                                evidence.data["arrays"]["reference_decoder_input"]["array_sha256"],"zero FP16 decoder input differs",evidence)
                            F.full_rgb_backward(torch.from_numpy(prior_arrays["reference_rgb"]),rgb,current_candidate["video"],target,evidence,torch)
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
                # Backward has finished; no graph/output survives into AdamW.
                del rgb,current_candidate
                current_candidate=None
    finally:
        optimizer=None
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
            "adapter_updates","checkpoint_saves","clear_reload_cycles","merge_calls","unscale_calls","clip_calls","optimizer_state_saves","optimizer_state_clear_reload")) and a["VAE_encode_calls"]==a["Rphi_calls"]==0)
    counts=evidence.data["block_forward_calls"]
    evidence.data["block_budget_passed"]=(all(counts["DiT"].get(m)=={str(i):3 for i in range(50)} for m in (*MODES,"backward"))
        and all(counts["VAE"].get(m)=={str(i):3 for i in range(36)} for m in (*(m+"_decode" for m in MODES),"backward")))
    evidence.data["tile_budget_passed"]=tile_budget_passed(evidence.data["native_tile_rows"])
    contexts=evidence.data["decoder_context_restorations"]
    evidence.data["decoder_context_budget_passed"]=(len(contexts)==220 and
        all(sum(r["phase"]==phase for r in contexts)==count for phase,count in
            {"zero_decode":109,"backward":108,"updated_decode":1,"reloaded_decode":1,"merged_decode":1}.items()))
    require(evidence.data["budget_passed"] and evidence.data["block_budget_passed"] and
            evidence.data["tile_budget_passed"] and evidence.data["decoder_context_budget_passed"],"execution budget mismatch",evidence)
    require(evidence.data["wrapper_attributes_restored"] and evidence.data["math_restored"] and
        evidence.data["patcher_restoration"]["whole_run_shapes_restored"] and
        evidence.data["patcher_restoration"]["whole_run_current_patcher_restored"] and
        all(r["restored"] for r in evidence.data["decoder_context_restorations"]),"final restoration failed",evidence)
    require(evidence.data.get("actual_imports",{}).get("passed") and "finally_state_error" not in evidence.data,
        "terminal frozen/import gate failed",evidence)
    evidence.data["functional_effect"]=functional_result(evidence.data['comparisons'])
    evidence.data["functional_deployment_ready"]=evidence.data["functional_effect"]["deployed_merged_RGB_changed"]
    evidence.data["arithmetic_persistence_passed"]=True
    evidence.data["formal_training_runs"]=0
    evidence.data["complete"]=True
    evidence.data["scope"]="one real TRAIN AdamW arithmetic/persistence/functional sensitivity measurement; no quality/generalization/formaltraining acceptance"
    evidence.phase("completed",torch)
    atomic_json(evidence.path/"report.json",evidence.data)


def inspect(args):
    """Hash actual cache and accepted reference; no Torch/Comfy/model construction."""
    import numpy as np
    contract = json.loads(args.execution_contract.read_text()); root = Path(contract['root'])
    F.insist(contract['schema'] == 'openh3.real_train_adamw_probe_execution.v1'
             and contract['probe_sha256'] == sha_file(__file__), 'entry/schema not pinned')
    protocol = F.checked_json(root, contract['protocol'], PROTOCOL_SHA)
    F.insist(protocol['optimizer']['actual_lr'] == LR and protocol['optimizer']['loss_scale'] == LOSS_SCALE,
             'frozen actual first-step LR/scale differs')
    spec = F.checked_json(root, contract['full22_source_contract'], F.SOURCE_CONTRACT_SHA)
    inventory = F.checked_json(root, contract['input_inventory'], F.INVENTORY_SHA)
    prior_ref = contract['accepted_full22']
    prior = F.checked_json(root, prior_ref['report'], REFERENCE_REPORT_SHA)
    audit = F.checked_json(root, prior_ref['audit'], REFERENCE_AUDIT_SHA)
    review = F.checked_json(root, prior_ref['independent_review'], REFERENCE_REVIEW_SHA)
    F.insist(prior['complete'] is True and audit['complete'] is True and audit['audit_complete'] is True
             and not audit.get('errors') and audit['report_sha256'] == REFERENCE_REPORT_SHA
             and audit['numerical_acceptance'] == 'passed_full22_real_train_joint_RGB_engineering_only'
             and review['complete'] is True and review['verdict'] == 'accepted_full22_joint_RGB_backward_engineering_only',
             'full22 prerequisite not accepted')
    F.insist(prior['actual']['optimizer_steps'] == prior['actual']['adapter_updates'] == 0
             and prior['input_inspection']['identity'] == spec['inputs']['identity'], 'reference initial mode/input identity changed')
    report_path = Path(prior_ref['report']['path'])
    if not report_path.is_absolute(): report_path = root / report_path
    prior_arrays = {name: N.load_array(report_path.parent, prior['arrays'][name]) for name in
        ('guide', 'raw_warm_start', 'target', 'reference_video', 'reference_audio', 'reference_rgb', 'reference_decoder_input')}
    invref = prior['frozen_initial']; invpath = (report_path.parent / invref['path']).resolve()
    F.insist(invpath.is_relative_to(report_path.parent.resolve()) and sha_file(invpath) == invref['sha256'],
             'accepted initial fingerprint file differs/escapes')
    prior_inventory = json.loads(invpath.read_text())
    inputs, target, inspection = F.load_inputs(spec, inventory)
    F.insist(all(np.array_equal(prior_arrays[k], value) for k, value in
                (('guide', inputs['guide']), ('raw_warm_start', inputs['raw_warm_start']), ('target', target))),
             'cache e differs from accepted full22 reference inputs/target')
    F.insist(sha_file(args.pretrained_manifest) == contract['manifest_sha256'] == MANIFEST_SHA,
             'pretrained manifest differs')
    verified = verify_pins(root, contract['implementation_sha256'])
    vae_source = (Path(contract['comfy_root']) / 'comfy/ldm/minimax/vae.py').resolve()
    F.insist(verified.get(str(vae_source)) == F.VAE_SOURCE_SHA, 'native VAE tile-plan source changed')
    for module in (J, F, N, sys.modules['check_h3_trajectory_grad']):
        source = str(Path(module.__file__).resolve())
        F.insist(verified.get(source) == sha_file(source), 'actual helper import unbound: ' + source)
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    F.insist(sha_file(manifest['_paths']['context']) == contract['context_sha256'] == spec['inputs']['context']['sha256']
             and manifest['_fingerprint'] == inventory['identity']['pretrained_fingerprint'], 'pretrained recipe/context differs')
    inspection.update(schema='openh3.real_train_adamw_probe_input_inspection.v1', inspection_only=not args.execute,
        execution_contract_sha256=sha_file(args.execution_contract), protocol_sha256=PROTOCOL_SHA,
        accepted_full22=prior_ref, accepted_reference_arrays={n: prior['arrays'][n] for n in prior_arrays},
        runtime_files_verified=len(verified), model_assets_size_only_verified=True,
        model_asset_bytes_SHA_verified=False, nonzero_b_weights_loaded=False,
        planned=protocol['budget'], optimizer=optimizer_recipe(), formal_training_runs=0)
    F.insist(not any(n.split('.')[0] in {'torch', 'torch_npu', 'comfy'} for n in sys.modules),
             'CPU inspection imported model runtime')
    return contract, verified, inputs, target, inspection, prior, prior_arrays, prior_inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-manifest', type=Path, required=True)
    parser.add_argument('--execution-contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(); F.insist(not args.out.exists(), 'output exists; no overwrite/retry')
    contract, verified, inputs, target, inspection, prior, prior_arrays, prior_inventory = inspect(args)
    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=False); atomic_json(args.out / 'inspection.json', inspection)
        print(json.dumps({'complete': True, 'inspection_only': True, 'model_calls': 0})); return
    F.insist(os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').isdecimal(), 'one explicitly admitted physical card required')
    admission = F.checked_json(Path(contract['root']), contract['root_admission'])
    F.insist(admission['approved'] is True and admission['scope'] == 'real_TRAIN_single_AdamW_engineering_probe'
             and admission['probe_sha256'] == contract['probe_sha256']
             and admission['execution_payload_sha256'] == F.execution_payload_sha256(contract), 'wrong fresh root admission')
    evidence = Evidence(args.out, {**contract, 'execution_contract_sha256': sha_file(args.execution_contract)})
    evidence.data['input_inspection'] = inspection; evidence.data['verified_runtime_files'] = verified
    try: execute(args, contract, evidence, prior, prior_arrays, prior_inventory, inputs, target)
    except BaseException as exc:
        evidence.data['complete'] = False
        evidence.data['failure'] = {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc(),
            'phase': evidence.data.get('failure_origin_phase', evidence.data['phase'])}
        evidence.save(); atomic_json(evidence.path / 'failure.json', evidence.data); raise


if __name__ == '__main__': main()
