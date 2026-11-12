"""Fresh, separately admitted inference continuation of one saved TRAIN update.

The old run remains failed. Only an independently accepted saved subset permits
execution: exact saved adapter/AdamW restoration, 3 NFE + native decode versus
the old updated output, then one declared BF16 merge and 3 NFE + decode. No
backward, optimizer step, unscale, clipping, encoding, Rphi or checkpoint save.
Default inspection imports no Torch/Comfy and performs no model calls.
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
import check_h3_full22_joint_rgb_grad as F
import check_h3_nonzero_adapter as N
import check_h3_real_adamw_update as A
import h3_evidence_checks as C
from h3_evidence_checks import require
from check_h3_trajectory_grad import (MANIFEST_SHA, TARGETS, atomic_json, sha_file,
    tensor_sha, tensor_record, canonical_base_items, parity_metrics, observe_noise,
    runtime_imports, verify_pins)

SEED, MODES = 101, ('reloaded', 'merged')
SHAPE, AUDIO_SHAPE, RGB_SHAPE = F.SHAPE, F.AUDIO_SHAPE, F.RGB_SHAPE
PARTIAL_SHA = 'ff278a84665a7cae78246dc017cbbfe0b0ddb2e93ac9f7d08c6fa31c0309b646'
QUEUE_SHA = '28299a42e3e76ce00b5e8fe7dae9bd471ebfa7fc7950051199857280de9d9edb'
PRODUCER_SHA = '27fb9145790635a34398f52ae437f6a3f3f698f097c9a22ccc74ebcd4c6827b3'
ADAPTER_SHA = 'b805168bc6d1af3506db5788b9c22a6eec1c30f804dba34268013001558932be'
OPTIMIZER_SHA = '569eeaf6d470036cb75b216bfd1dcf0fcd824aadbfe283a0f779657e4055664e'
PLANNED = {'NFE': 6, 'VAE_decodes': 2, 'stage_decodes': 2, 'merge_calls': 1,
    'saved_adapter_loads': 1, 'saved_optimizer_loads': 1,
    'backward_attempts': 0, 'completed_backward_calls': 0, 'optimizer_steps': 0,
    'adapter_updates': 0, 'checkpoint_saves': 0, 'VAE_encode_calls': 0,
    'Rphi_calls': 0, 'unscale_calls': 0, 'clip_calls': 0}


class Evidence(J.Evidence):
    def __init__(self, path, contract):
        super().__init__(path, contract)
        self.data.update(schema='openh3.saved_adamw_continuation.v1',
            original_run_accepted=False, original_failure_preserved=True,
            formal_training_runs=0, native_tile_rows=[], stage_decoder_inputs={},
            mode_outputs={}, session_restorations=[], planned=PLANNED,
            successful_checks='in memory; explicit phase/progress/failure persistence')
        self.data['actual'] = {k: 0 for k in PLANNED}
        self.data['actual'].update(NFE={m: 0 for m in MODES},
            VAE_decode_calls={m: 0 for m in MODES}, stage_decode_calls={}, forbidden_calls=[])
        del self.data['actual']['VAE_decodes'], self.data['actual']['stage_decodes']
        self.save()

    def progress(self, label, completed, total):
        self.data.setdefault('progress', {})[label] = {'completed': completed, 'total': total}
        self.save()


def bound_path(root, record):
    path = Path(record['path'])
    if not path.is_absolute(): path = Path(root) / path
    if not path.is_file() or sha_file(path) != record['sha256']:
        raise ValueError('bound file differs: ' + str(path))
    return path


def child_path(directory, name):
    path = (Path(directory) / name).resolve()
    F.insist(path.is_relative_to(Path(directory).resolve()), 'artifact escapes original output')
    return path


def validate_subset_audit(audit, contract, prior):
    """Keep original failure separate from the strictly named accepted subset."""
    F.insist(audit['schema'] == 'openh3.saved_adamw_partial_audit.v1'
        and audit['audit_complete'] is True and audit['complete'] is True
        and audit['partial_artifact_acceptance'] == 'verified_saved_one_update_and_zero_updated_outputs_only'
        and audit['saved_subset_ready_for_continuation'] is True
        and all(audit[k] is False for k in ('original_run_accepted', 'strict_reload_accepted',
            'merge_accepted', 'functional_deployment_ready')), 'saved subset is not independently accepted')
    for key in ('saved_update', 'outputs'):
        component = audit['components'][key]
        F.insist(component['status'] == 'verified' and component['facts']['passed'] is True
            and component['facts']['actual_arrays_verified'] is True, 'partial component unverified: ' + key)
    b = audit['binding']; refs = contract['failed_run']
    wanted = {'partial_sha256': refs['partial']['sha256'], 'queue_sha256': refs['queue']['sha256'],
        'root_review_sha256': refs['root_review']['sha256'],
        'original_execution_contract_sha256': prior['contract']['execution_contract_sha256'],
        'producer_sha256': PRODUCER_SHA, 'protocol_sha256': A.PROTOCOL_SHA}
    F.insist(all(b.get(k) == v for k, v in wanted.items()), 'subset audit identity differs')
    F.insist(b['checkpoint']['sha256'] == ADAPTER_SHA and b['optimizer_checkpoint']['sha256'] == OPTIMIZER_SHA
        and b['checkpoint']['bytes'] == prior['checkpoint']['bytes']
        and b['accepted_full22'] == prior['contract']['accepted_full22'], 'subset checkpoint/reference binding differs')


def byte_parity(left, right):
    """Strict bytes also reject signed-zero differences hidden by array_equal."""
    import numpy as np
    same = left.shape == right.shape and left.dtype == right.dtype
    finite = bool(np.isfinite(left).all() and np.isfinite(right).all())
    a, b = F.array_sha(left), F.array_sha(right)
    delta = left.astype(np.float64) - right.astype(np.float64) if same else None
    return {'same_shape_and_dtype': same, 'finite': finite, 'byte_exact': same and a == b,
        'array_equal': bool(np.array_equal(left, right)) if same else False,
        'reference_sha256': a, 'candidate_sha256': b,
        'changed_coordinates': int(np.count_nonzero(left != right)) if same else None,
        'max_abs': float(np.max(np.abs(delta))) if same else None,
        'mean_abs': float(np.mean(np.abs(delta))) if same else None}


def inspect(args):
    import numpy as np
    contract = json.loads(args.execution_contract.read_text()); root = Path(contract['root'])
    F.insist(contract['schema'] == 'openh3.saved_adamw_continuation_execution.v1'
        and contract['probe_sha256'] == sha_file(__file__) and contract['planned'] == PLANNED,
        'continuation source/schema/budget differs')
    refs = contract['failed_run']
    prior = F.checked_json(root, refs['partial'], PARTIAL_SHA)
    queue = F.checked_json(root, refs['queue'], QUEUE_SHA)
    review = F.checked_json(root, refs['root_review'])
    F.insist(prior['complete'] is False and queue['complete'] is False and queue['state'] == 'failed',
        'original failure must remain failed')
    F.insist(review['status'] == 'failed_wall_budget_not_accepted'
        and review['acceptance_boundaries']['original_failure_preserved'] is True,
        'wrong original failure review')
    audit = F.checked_json(root, contract['partial_cpu_audit'])
    validate_subset_audit(audit, contract, prior)
    protocol = F.checked_json(root, contract['protocol'], A.PROTOCOL_SHA)
    F.insist(protocol['optimizer']['actual_lr'] == A.LR and protocol['optimizer']['loss_scale'] == 1024.,
        'original optimizer protocol changed')
    prior_dir = bound_path(root, refs['partial']).parent
    cp = prior['checkpoint']
    adapter = child_path(prior_dir, cp['path']); optimizer = child_path(prior_dir, cp['optimizer']['path'])
    F.insist(cp['sha256'] == ADAPTER_SHA and cp['optimizer']['sha256'] == OPTIMIZER_SHA
        and sha_file(adapter) == ADAPTER_SHA and sha_file(optimizer) == OPTIMIZER_SHA
        and adapter.stat().st_size == cp['bytes'], 'saved checkpoint bytes differ')
    for key, path in (('checkpoint', adapter), ('optimizer_checkpoint', optimizer)):
        ref = audit['binding'][key]
        F.insist(child_path(prior_dir, ref['path']) == path and sha_file(path) == ref['sha256']
            and path.stat().st_size == ref['bytes'],
            'subset actual checkpoint path/size differs')
    original_exec = F.checked_json(root, contract['original_execution_contract'],
        prior['contract']['execution_contract_sha256'])
    F.insist(original_exec['probe_sha256'] == PRODUCER_SHA
        and {k: v for k, v in prior['contract'].items() if k != 'execution_contract_sha256'} == original_exec,
        'original execution contract differs')
    # Re-read the exact accepted TRAIN cache, never substitute a new window/target.
    spec = F.checked_json(root, contract['full22_source_contract'], F.SOURCE_CONTRACT_SHA)
    inventory = F.checked_json(root, contract['input_inventory'], F.INVENTORY_SHA)
    inputs, target, inspection = F.load_inputs(spec, inventory)
    F.insist(prior['input_inspection']['identity'] == inspection['identity']
        and prior['inputs']['role'] == 'train' and prior['inputs']['source_id'] == '2BdYaGuVAKo'
        and prior['inputs']['target_used_as_model_input'] is False, 'original TRAIN identity differs')
    for name, arr in (('guide', inputs['guide']), ('raw_warm_start', inputs['raw_warm_start']), ('target', target)):
        old = N.load_array(prior_dir, prior['arrays'][name])
        F.insist(byte_parity(old, arr)['byte_exact'], 'same-cache actual array differs: ' + name)
    # Only those old output arrays used in comparisons are read. They remain old evidence.
    for mode in ('updated', 'zero', 'reference'):
        for stream in ('video', 'audio', 'decoder_input', 'rgb'):
            N.load_array(prior_dir, prior['arrays'][mode + '_' + stream])
    inv = prior['frozen_initial']; invpath = child_path(prior_dir, inv['path'])
    F.insist(sha_file(invpath) == inv['sha256'], 'original frozen inventory changed')
    prior_inventory = json.loads(invpath.read_text())
    verified = verify_pins(root, contract['implementation_sha256'])
    for module in (J, F, N, A, C, sys.modules['check_h3_trajectory_grad']):
        source = str(Path(module.__file__).resolve())
        F.insist(verified.get(source) == sha_file(source), 'actual helper source not bound: ' + source)
    F.insist(sha_file(A.__file__) == PRODUCER_SHA, 'original AdamW numerical helper changed')
    F.insist(sha_file(args.pretrained_manifest) == contract['manifest_sha256'] == MANIFEST_SHA,
        'pretrained manifest differs')
    from openh3_assets import read_manifest
    manifest = read_manifest(args.pretrained_manifest, verify=False)
    F.insist(sha_file(manifest['_paths']['context']) == contract['context_sha256'] == original_exec['context_sha256']
        and manifest['_fingerprint'] == inventory['identity']['pretrained_fingerprint'], 'recipe/context differs')
    vae_path = (Path(contract['comfy_root']) / 'comfy/ldm/minimax/vae.py').resolve()
    F.insist(verified.get(str(vae_path)) == F.VAE_SOURCE_SHA, 'native VAE source differs')
    inspection.update(schema='openh3.saved_adamw_continuation_input_inspection.v1',
        original_run_accepted=False, original_failure_preserved=True,
        original_active_worker_seconds=queue['active_worker_seconds'],
        original_last_partial_elapsed_seconds=prior['elapsed_seconds'],
        partial_cpu_audit=contract['partial_cpu_audit'], checkpoints={
            'adapter': {'path': str(adapter), 'sha256': ADAPTER_SHA},
            'optimizer': {'path': str(optimizer), 'sha256': OPTIMIZER_SHA}},
        planned=PLANNED, inspection_only=not args.execute, runtime_files_verified=len(verified),
        model_assets_size_only_verified=True, model_asset_bytes_SHA_verified=False)
    F.insist(not any(n.split('.')[0] in {'torch', 'torch_npu', 'comfy'} for n in sys.modules),
        'inspection imported model runtime')
    return contract, verified, prior, prior_dir, prior_inventory, inputs, target, inspection


def save_actual(evidence, name, value, torch, *, flush=True):
    """One CPU transfer reused for finite, SHA and actual NPY; never overwrite."""
    import numpy as np
    array = value.detach().cpu().contiguous().numpy()
    require(bool(np.isfinite(array).all()), 'nonfinite saved tensor: ' + name, evidence)
    path = evidence.path / 'arrays' / (name + '.npy')
    require(not path.exists() and not path.is_symlink(), 'array overwrite forbidden', evidence)
    with path.open('xb') as handle: np.save(handle, array, allow_pickle=False)
    row = {'path': str(path.relative_to(evidence.path)), 'file_sha256': sha_file(path),
        'array_sha256': F.array_sha(array), 'dtype': str(array.dtype), 'shape': list(array.shape), 'role': name}
    evidence.data['arrays'][name] = row
    if flush: evidence.save()
    return row, array


def optimizer_group_recipe():
    """Map the frozen probe recipe into Torch's parameter-group field names."""
    recipe = A.optimizer_recipe()
    return {'lr': recipe['actual_lr'], 'betas': tuple(recipe['betas']),
        **{key: recipe[key] for key in ('eps', 'weight_decay', 'amsgrad', 'maximize',
            'foreach', 'capturable', 'differentiable', 'fused')}}


def restore_optimizer(params, expected, saved, inventory, parameter_order, torch, on_restored=None, on_loaded=None):
    """Fresh optimizer load only. No step; every saved field is strictly checked."""
    N.validate_state(params, expected, torch)
    if list(params) != parameter_order or len(set(parameter_order)) != len(params):
        raise ValueError('optimizer canonical parameter order differs')
    fields = ('step', 'exp_avg', 'exp_avg_sq')
    if set(saved) != {n + '::' + k for n in params for k in fields} or set(inventory) != set(params):
        raise ValueError('optimizer saved key coverage differs')
    for name, parameter in params.items():
        if set(inventory[name]) != set(fields): raise ValueError('optimizer inventory fields differ')
        for key in fields:
            value = saved[name + '::' + key]; shape = () if key == 'step' else tuple(expected[name])
            if value.dtype != torch.float32 or tuple(value.shape) != shape or not bool(torch.isfinite(value).all()):
                raise ValueError('optimizer tensor shape/dtype/finite differs')
            record = inventory[name][key]
            if record['dtype'] != 'torch.float32' or record['shape'] != list(shape) or tensor_sha(value, torch) != record['sha256']:
                raise ValueError('optimizer saved state SHA/metadata differs')
            if key == 'step' and float(value) != 1.: raise ValueError('optimizer step must be one')
            if key == 'exp_avg_sq' and not bool((value >= 0).all()): raise ValueError('negative second moment')
    optimizer = torch.optim.AdamW(list(params.values()), lr=A.LR, betas=(.9, .999), eps=1e-8,
        weight_decay=0., amsgrad=False, maximize=False, foreach=False, capturable=False,
        differentiable=False, fused=False)
    if optimizer.state: raise ValueError('fresh optimizer has state')
    sd = optimizer.state_dict()
    sd['state'] = {i: {k: saved[n + '::' + k].clone() for k in fields} for i, n in enumerate(params)}
    optimizer.load_state_dict(sd)
    if on_loaded is not None: on_loaded()
    group = optimizer.param_groups[0]
    for key, wanted in optimizer_group_recipe().items():
        if group[key] != wanted: raise ValueError('loaded optimizer group differs: ' + key)
    if len(optimizer.param_groups) != 1 or set(optimizer.state) != set(params.values()) or any(
        a is not b for a, b in zip(group['params'], params.values())): raise ValueError('loaded optimizer parameter identity differs')
    for name, parameter in params.items():
        if set(optimizer.state[parameter]) != set(fields): raise ValueError('loaded optimizer state fields differ')
        for key, value in optimizer.state[parameter].items():
            cpu = value.detach().cpu()
            if value.dtype != torch.float32 or not torch.equal(cpu, saved[name + '::' + key]):
                raise ValueError('loaded optimizer differs from checkpoint')
            expected_device = 'cpu' if key == 'step' else str(parameter.device)
            if str(value.device) != expected_device or str(value.device) != inventory[name][key]['device']:
                raise ValueError('loaded optimizer state device differs')
            if on_restored is not None: on_restored(name, key, value, cpu)
    return optimizer


def strict_load_saved(dit, expected, prior, prior_dir, evidence, torch):
    import lora as L
    from safetensors import safe_open
    from safetensors.torch import load_file
    cp = prior['checkpoint']; diagnostic = cp['metadata']['training']
    ap, op = child_path(prior_dir, cp['path']), child_path(prior_dir, cp['optimizer']['path'])
    require(sha_file(ap) == ADAPTER_SHA and sha_file(op) == OPTIMIZER_SHA, 'checkpoint changed before load', evidence)
    state, metadata = N.strict_saved_state(ap, expected, prior['pretrained']['openh3_pretrained'], diagnostic, torch)
    require(metadata == cp['metadata'] and len(state) == 400 and
        {n: list(s) for n, s in expected.items()} == cp['expected_shapes'], 'saved adapter metadata/geometry differs', evidence)
    with safe_open(str(op), framework='pt', device='cpu') as handle: ometa = handle.metadata()
    require(ometa == cp['optimizer']['metadata'], 'optimizer safetensors metadata differs', evidence)
    require(json.loads(ometa['optimizer_recipe']) == A.optimizer_recipe()
        and ometa['adapter_sha256'] == ADAPTER_SHA and json.loads(ometa['training_contract']) == diagnostic
        and ometa['schema'] == 'openh3.named_AdamW_state.v1', 'optimizer provenance differs', evidence)
    loaded = L.load(dit, state, strict=True, strict_unused=True)
    evidence.data['actual']['saved_adapter_loads'] += 1
    params = dict(L.lora_parameters(dit)); rows = {}
    for i, (name, parameter) in enumerate(params.items()):
        row, array = save_actual(evidence, 'loaded_adapter__' + name, parameter, torch, flush=False)
        wanted = prior['parameters_after_update'][name]
        require(row['array_sha256'] == tensor_sha(state[name], torch) == wanted['array_sha256']
            and parameter.requires_grad and parameter.grad is None, 'loaded adapter bytes/lifecycle differs: ' + name, evidence)
        rows[name] = {'actual': row, 'device': str(parameter.device), 'version': parameter._version,
            'requires_grad': parameter.requires_grad, 'grad_is_none': parameter.grad is None}
        if (i + 1) % 32 == 0: evidence.progress('adapter_load', i + 1, len(params))
    evidence.data['adapter_reload'] = {'loaded_count': loaded, 'tensors': rows, 'metadata': metadata}
    require(loaded == 400 and len(rows) == 400, 'adapter count differs', evidence)
    evidence.phase('restore_saved_optimizer', torch)
    saved = load_file(str(op), device='cpu')
    order = cp['optimizer']['parameter_order']
    require(json.loads(ometa['parameter_order']) == order == prior['optimizer_after_update']['parameter_order'],
        'saved optimizer order differs', evidence)
    records = {}
    def record_restored(name, key, value, cpu):
        # Reuse restore_optimizer's one actual device->CPU copy for equality,
        # finite validation, content hash and NPY evidence.
        row, array = save_actual(evidence, 'restored_optimizer__' + name + '::' + key, cpu, torch, flush=False)
        require(row['array_sha256'] == prior['optimizer_after_update']['inventory'][name][key]['sha256'],
            'actual restored state SHA differs', evidence)
        records.setdefault(name, {})[key] = {'actual': row, 'device': str(value.device),
            'dtype': str(value.dtype), 'shape': list(value.shape)}
        if len(records[name]) == 3 and len(records) % 32 == 0:
            evidence.progress('optimizer_load', len(records), len(params))
    def optimizer_loaded():
        evidence.data['actual']['saved_optimizer_loads'] += 1
        evidence.data['optimizer_load_state_dict_returned'] = True
        evidence.data['optimizer_strict_inventory_verified'] = False
        evidence.save()
    optimizer = restore_optimizer(params, expected, saved, prior['optimizer_after_update']['inventory'], order, torch,
        on_restored=record_restored, on_loaded=optimizer_loaded)
    require(json.loads(json.dumps({k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}))
        == prior['optimizer_after_update']['parameter_group'], 'complete optimizer parameter group differs', evidence)
    evidence.data['optimizer_strict_inventory_verified'] = True
    evidence.data['optimizer_reload'] = {'named_entries': len(records), 'state_tensors': sum(map(len, records.values())),
        'inventory': records, 'parameter_order': order, 'metadata': ometa,
        'parameter_group': {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'},
        'fresh_process_load_not_old_clear_reload_completion': True, 'optimizer_steps': 0}
    require(sha_file(ap) == ADAPTER_SHA and sha_file(op) == OPTIMIZER_SHA, 'checkpoint changed during load', evidence)
    # Preserve actual restored state evidence, then release all moments before inference.
    optimizer.state.clear()
    del optimizer, saved, state, params
    evidence.data['optimizer_released_before_inference'] = True
    evidence.progress('optimizer_load', 400, 400)


def snapshot_from_inventory(model, inventory, torch, names=()):
    """Reuse the just-read initial bytes; retain actual parameter objects across injection."""
    rows = {prefix + r['name']: r for prefix, group in
        (('parameter:', 'parameters'), ('buffer:', 'registered_buffers')) for r in inventory[group]}
    actual = dict(canonical_base_items(model, names))
    if actual.keys() != rows.keys(): raise ValueError('initial inventory canonical names differ')
    return {n: {'object': p, 'version': p._version, 'device': str(p.device),
        'sha256_bytes': rows[n]['sha256_bytes']} for n, p in actual.items()}


override_method = J.override_method
def save_array(evidence, name, value, torch): return save_actual(evidence, name, value, torch)[0]
finite_stat = N.finite_stat


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
        require(phase in tuple(m + "_decode" for m in MODES), "tile row outside two decodes", evidence)
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
        for phase in tuple(m + "_decode" for m in MODES)) and len(rows) == 6


@contextmanager
def bounded_blocks(blocks, group, evidence, observe_math):
    phases = MODES if group == "DiT" else tuple(m + "_decode" for m in MODES)
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
            save_array(evidence, mode + "_decoder_input", value, torch)
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


def merge_once(model, names, frozen_snapshot, evidence, torch):
    """Declared 200 base writes, not an erroneous all-base-unchanged gate.

    Record the intended same-device addmm + BF16 cast hash before calling the
    existing merge. Keep only hashes/statistics, never duplicate the 20B base.
    """
    from openh3_runtime import merge_adaptation_for_inference
    import lora as L
    require(evidence.data["actual"]["merge_calls"] == 0, "second merge forbidden", evidence)
    require(len(names) == 200 and len(set(names)) == 200, "merge target count differs", evidence)
    rows = {}
    with torch.no_grad():
        for index, name in enumerate(names):
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
            if (index + 1) % 16 == 0: evidence.progress("merge_expected", index + 1, len(names))
    evidence.data["merge"] = {"layers": rows, "expected_writes": len(names),
        "rounding_scope": "ideal scale*(B@A); storage from runtime-identical FP32 addmm then one BF16 cast",
        "equivalence_claim": False}
    evidence.save()
    require(all(r["all_finite"] for r in rows.values()), "nonfinite expected merged weights", evidence)
    evidence.phase("apply_declared_BF16_merge", torch)
    evidence.data["actual"]["merge_calls"] += 1
    evidence.save()
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


def compare_outputs(left_dir, left_arrays, left_mode, right_dir, right_arrays, right_mode, evidence, strict=False):
    rows = {}
    for stream in ('video', 'audio', 'decoder_input', 'rgb'):
        left = N.load_array(left_dir, left_arrays[left_mode + '_' + stream])
        right = N.load_array(right_dir, right_arrays[right_mode + '_' + stream])
        rows[stream] = byte_parity(left, right)
        del left, right
    evidence.data.setdefault('comparisons', {})[left_mode + '__' + right_mode] = rows
    evidence.save()
    require(all(r['finite'] and r['same_shape_and_dtype'] and (not strict or r['byte_exact']) for r in rows.values()),
        'strict persisted output parity failed' if strict else 'output geometry/nonfinite differs', evidence)
    return rows


def functional_result(comparisons):
    baseline_exact = comparisons['reference__zero']['rgb']['byte_exact']
    changed = comparisons['reference__merged']['rgb']['changed_coordinates'] > 0
    return {'zero_vs_accepted_reference_RGB_byte_exact': baseline_exact,
        'merged_vs_accepted_reference_RGB_changed': changed,
        'readiness_gate_passed': baseline_exact and changed,
        'attribution': 'numerical_mode_confounded' if not baseline_exact else
            'observed_deployed_RGB_change' if changed else 'no_deployed_RGB_change',
        'strict_reload_is_not_a_resumed_optimizer_trajectory_test': True,
        'same_sample_loss_descent_required': False, 'quality_or_generalization_claim': False}


def budget_passed(data):
    a = data['actual']; counts = data['block_forward_calls']; contexts = data['decoder_context_restorations']
    scalar = {k: v for k, v in PLANNED.items() if k not in ('NFE', 'VAE_decodes', 'stage_decodes')}
    return (all(a[k] == v for k, v in scalar.items()) and a['NFE'] == {m: 3 for m in MODES}
        and a['VAE_decode_calls'] == {m: 1 for m in MODES}
        and a['stage_decode_calls'] == {m + '_decode': 1 for m in MODES} and not a['forbidden_calls']
        and set(counts['DiT']) == set(MODES) and set(counts['VAE']) == {m + '_decode' for m in MODES}
        and all(counts['DiT'][m] == {str(i): 3 for i in range(50)} for m in MODES)
        and all(counts['VAE'][m + '_decode'] == {str(i): 3 for i in range(36)} for m in MODES)
        and len(data['noise_observations']) == 2
        and [r['phase'] for r in data['noise_observations']] == list(MODES)
        and len(data['math_observations']) == 516
        and len(contexts) == 2 and {r['phase'] for r in contexts} == {m + '_decode' for m in MODES}
        and all(r['restored'] for r in contexts) and tile_budget_passed(data['native_tile_rows']))


def execute(args, contract, evidence, prior, prior_dir, prior_inventory, inputs, target_array):
    import torch
    import torch_npu
    torch.npu.set_device(0); torch.npu.set_compile_mode(jit_compile=False)
    from h3_comfy import H3, native_training_math
    from check_vae_decode_grad import decoder_math
    from openh3_assets import NATIVE_SIGMAS
    from openh3_runtime import inject_adaptation
    from openh3_train_trajectory import ResidentH3Trajectory
    import comfy.sample
    import comfy.utils
    from comfy_kitchen.backends.ascend.rope import rms_rope_split_half as native_rope
    evidence.data['environment'] = {'torch': torch.__version__, 'torch_npu': torch_npu.__version__,
        'python': sys.version, 'visible_device': os.environ['ASCEND_RT_VISIBLE_DEVICES'],
        'synthetic_no_dataset': False, 'role': 'train', 'quality_claim': False}
    require(all(evidence.data['environment'][k] == prior['environment'][k] for k in ('torch', 'torch_npu', 'python')),
        'original runtime versions differ', evidence)
    evidence.phase('load_and_verify_assets', torch)
    h3 = H3(pretrained_manifest=args.pretrained_manifest, sigma_schedule=NATIVE_SIGMAS,
        adaptation_path=None, lora_path=None)
    dit, vae = h3.patcher.model.diffusion_model, h3.vae
    stage, model = vae.first_stage_model, h3.patcher.model
    dit.eval().requires_grad_(False).to(h3.dev); stage.eval().requires_grad_(False).to(h3.dev)
    vae.device = vae.offload_device = h3.dev; vae.disable_offload = True
    evidence.data['pretrained'] = h3.record; evidence.data['asset_bytes_sha_verified'] = True
    require(N.stable_pretrained(h3.record) == N.stable_pretrained(prior['pretrained']) and vae.vae_dtype == torch.float16,
        'original pretrained/VAE precision differs', evidence)
    preflight = ResidentH3Trajectory(h3, checkpoint_blocks=False, checkpoint_decoder=False)
    preflight._validate_resident(); preflight._validate_vae()
    require(len(dit.blocks) == 50 and len(stage.decoder.transformer_blocks) == 36, 'wrong block inventory', evidence)
    evidence.data['vae_native_plan'] = F.validate_native_plan(stage)
    require(json.loads(json.dumps(evidence.data['vae_native_plan'])) == prior['vae_native_plan'], 'native decode plan differs', evidence)
    initial_math = J.actual_math_state(torch); evidence.data['math_before'] = initial_math
    flags = ('in_training', 'jit_compile_false', 'matmul_allow_hf32', 'conv_allow_hf32',
        'deterministic_algorithms', 'deterministic_warn_only')
    require(all(initial_math[k] == prior['math_before'][k] for k in flags), 'original math flags differ', evidence)
    evidence.data['actual_imports_before_sampling'] = runtime_imports(evidence.data['verified_runtime_files'])
    require(evidence.data['actual_imports_before_sampling']['passed'], 'unbound runtime imports', evidence)
    torch.npu.reset_peak_memory_stats()
    evidence.data['memory_scope'] = 'peak reset after loading; not optimizer throughput or total end-to-end peak'
    names, session, snapshots, adapters = [], None, None, None
    original_patcher = getattr(model, 'current_patcher', None)
    original_shapes = (hasattr(model, 'latent_shapes'), getattr(model, 'latent_shapes', None))
    methods = [(b, 'forward') for b in list(dit.blocks) + list(stage.decoder.transformer_blocks)]
    methods += [(owner, name) for owner, ns in ((h3, ('encode', 'encode_guide', 'encode_guide_pixels', 'ref_block', 'sample')),
        (vae, ('decode', 'encode', 'decode_tiled_1d', '_decode_tiled_owned', 'decode_tiled_', 'decode_tiled_3d')),
        (stage, ('decode', '_decode_tile_row', '_decode_pixels')), (comfy.sample, ('prepare_noise',))) for name in ns if hasattr(owner, name)]
    wrappers = J.wrapper_snapshot(methods)
    guide, raw = (torch.from_numpy(inputs[n].copy()) for n in ('guide', 'raw_warm_start'))
    target = torch.from_numpy(target_array)
    for n, value in (('guide', guide), ('raw_warm_start', raw), ('target', target)):
        save_array(evidence, n, value, torch)
    evidence.data['inputs'] = {'seed': SEED, 'audio_init': 'zeros', 'target_used_as_model_input': False,
        'model_input_keys': ['guide', 'raw_warm_start'], 'source_id': '2BdYaGuVAKo', 'role': 'train',
        'raw_warm_start_is_original_raw_encoding': True}
    evidence.data['original_artifacts'] = {'directory': str(prior_dir), 'partial_sha256': PARTIAL_SHA,
        'arrays': {k: prior['arrays'][k] for k in prior['arrays'] if k in
            {m + '_' + s for m in ('zero', 'updated', 'reference') for s in ('video', 'audio', 'decoder_input', 'rgb')}},
        'checkpoint': contract['failed_run']['partial'], 'actual_checkpoint': prior['checkpoint']}
    math_expected = {}
    def observe_math(group, phase, index):
        current = J.actual_math_state(torch)
        evidence.data['math_observations'].append({'group': group, 'phase': phase, 'block': index, 'state': current})
        require(not torch.is_grad_enabled(), 'continuation unexpectedly enabled gradients', evidence)
        require(current['in_training'] and all(current[k] == initial_math[k] for k in flags if k != 'in_training'),
            'block math flags changed', evidence)
        require(current['rope'] == J.function_id(native_rope) if group == 'VAE' else current == math_expected['DiT_active'],
            'wrong DiT/VAE math context', evidence)
    @contextmanager
    def decoder_context():
        before = J.actual_math_state(torch); phase = evidence.data['phase']
        require(phase in {m + '_decode' for m in MODES}
            and not any(r['phase'] == phase for r in evidence.data['decoder_context_restorations']),
            'unexpected/repeated decoder context', evidence)
        try:
            with decoder_math(native_training_math, 'ascend_native'): yield
        finally:
            after = J.actual_math_state(torch)
            evidence.data['decoder_context_restorations'].append({'phase': phase, 'before': before, 'after': after, 'restored': before == after})
            require(before == after, 'decoder math restoration failed', evidence)
    def audit_state(label, full=False):
        evidence.phase(('full_state_hash_' if full else 'lifecycle_') + label, torch)
        check = J.check_snapshot if full else N.check_lifecycle
        result = {'DiT': check(dit, snapshots['DiT'], torch, names, h3.dev),
            'VAE': check(stage, snapshots['VAE'], torch, device=h3.dev)}
        if adapters is not None: result['adapters'] = J.check_adapters(dit, adapters, torch, h3.dev)
        evidence.data['state_checks'][label] = result
        require(all(row['passed'] for row in result.values()), 'frozen state failed: ' + label, evidence)
        evidence.save()
    def nfe_hook(module, values):
        mode = evidence.data['phase']; require(mode in MODES, 'unexpected whole DiT phase', evidence)
        evidence.data['actual']['NFE'][mode] += 1
        require(evidence.data['actual']['NFE'][mode] <= 3, 'NFE budget exceeded', evidence)
    hook = dit.register_forward_pre_hook(nfe_hook)
    try:
        evidence.phase('initial_resident_fingerprints', torch)
        initial = {'DiT': J.frozen_inventory(dit, torch), 'VAE': J.frozen_inventory(stage, torch)}
        atomic_json(evidence.path / 'frozen_initial.json', initial)
        evidence.data['frozen_initial'] = {'path': 'frozen_initial.json', 'sha256': sha_file(evidence.path / 'frozen_initial.json')}
        require(all(initial[g][k] == prior_inventory[g][k] for g in initial for k in
            ('parameters_sha256', 'registered_buffers_sha256')), 'initial base/VAE bytes differ from failed run', evidence)
        snapshots = {'DiT': snapshot_from_inventory(dit, initial['DiT'], torch),
            'VAE': snapshot_from_inventory(stage, initial['VAE'], torch)}
        evidence.phase('inject_then_strict_load_saved_adapter', torch)
        names = inject_adaptation(dit, rank=8, seed=SEED)
        expected = N.expected_adapters(dit, torch)
        require(len(names) == 200 and len(expected) == 400, 'wrong adapter scope', evidence)
        evidence.data['adapter_initial'] = J.adapter_audit(dit, names, torch)
        require(evidence.data['adapter_initial'] == prior['adapter_initial'], 'fresh seed101 adapter initialization differs', evidence)
        strict_load_saved(dit, expected, prior, prior_dir, evidence, torch)
        adapters = J.adapter_snapshot(dit, torch)
        audit_state('after_saved_load', full=True)
        with ExitStack() as scope:
            scope.enter_context(guarded_vae(h3, evidence, torch))
            scope.enter_context(native_tile_guard(stage, evidence, torch))
            scope.enter_context(observe_noise(comfy.sample, comfy.utils.pack_latents,
                lambda: evidence.data['phase'], evidence.data['noise_observations'], torch))
            scope.enter_context(bounded_blocks(dit.blocks, 'DiT', evidence, observe_math))
            scope.enter_context(bounded_blocks(stage.decoder.transformer_blocks, 'VAE', evidence, observe_math))
            for mode in MODES:
                if mode == 'merged':
                    audit_state('premerge', full=True)
                    evidence.phase('prepare_declared_BF16_merge', torch)
                    snapshots['DiT'] = merge_once(dit, names, snapshots['DiT'], evidence, torch)
                    adapters = None
                    audit_state('after_declared_merge')
                session = ResidentH3Trajectory(h3, checkpoint_blocks=False, checkpoint_decoder=False,
                    decoder_context=decoder_context)
                mode_patcher = getattr(model, 'current_patcher', None)
                mode_shapes = (hasattr(model, 'latent_shapes'), getattr(model, 'latent_shapes', None))
                try:
                    with torch.no_grad(), session:
                        math_expected['DiT_active'] = J.actual_math_state(torch)
                        evidence.data.setdefault('session_math', {})[mode] = math_expected['DiT_active']
                        trace = evidence.data.setdefault('traces', {}).setdefault(mode, [])
                        def callback(event):
                            trace.append({'step': int(event['i']), 'sigma': float(event['sigma']),
                                'x': tensor_record(event['x'], torch), 'denoised_x0': tensor_record(event['denoised'], torch)})
                            evidence.save()
                        evidence.phase(mode, torch)
                        output = session.sample(guide, raw, seed=SEED, audio_init='zeros', callback=callback)
                        evidence.data.setdefault('trajectory_contracts', {})[mode] = {**output['contract'],
                            'adapter_arithmetic': 'merged BF16 weights' if mode == 'merged' else 'independent FP32 adapter branch',
                            'merged': mode == 'merged', 'actual_live_adapter_tensors': 0 if mode == 'merged' else 400,
                            'adapter_B_all_zero_at_session_entry': None if mode == 'merged' else
                                output['contract']['adapter_B_all_zero_at_session_entry'],
                            'B_zero_boundary': 'not applicable after merge' if mode == 'merged' else 'actual independent adapter parameters'}
                        expected_noise = {k: v for k, v in prior['noise_observations'][1].items() if k != 'phase'}
                        actual_noise = {k: v for k, v in evidence.data['noise_observations'][-1].items() if k != 'phase'}
                        require(expected_noise == actual_noise and len(trace) == 3, 'same seed/noise/input trace differs', evidence)
                        for stream in ('video', 'audio'): save_array(evidence, mode + '_' + stream, output[stream].float(), torch)
                        audit_state(mode + '_before_decode')
                        evidence.phase(mode + '_decode', torch)
                        rgb = session.decode(output['video'])
                        require(not rgb.requires_grad and not output['video'].requires_grad
                            and tuple(rgb.shape) == RGB_SHAPE and tuple(output['video'].shape) == SHAPE
                            and tuple(output['audio'].shape) == AUDIO_SHAPE, 'no-grad output geometry differs', evidence)
                        row, array = save_actual(evidence, mode + '_rgb', rgb.float(), torch)
                        import numpy as np
                        evidence.data['mode_outputs'][mode] = {'geometry_and_finite': True,
                            'numpy_float64_mse': float(np.mean((array.astype(np.float64) - target_array.astype(np.float64)) ** 2)),
                            'clipping': {'at_zero': int((array == 0).sum()), 'at_one': int((array == 1).sum()),
                                'outside_unit_interval': int(((array < 0) | (array > 1)).sum()),
                                'semantics': 'as-returned internally clipped VAE RGB; no extra clamp'},
                            'target_used_only_for_offline_descriptive_scoring': True}
                        del array, rgb, output
                        audit_state(mode + '_after_decode')
                finally:
                    if sys.exc_info()[0] is not None: evidence.data['failure_origin_phase'] = evidence.data['phase']
                    restoration = J.restore_patcher_state(model, session, mode_patcher, mode_shapes)
                    restoration.update(mode=mode, inactive=not session.active)
                    evidence.data['session_restorations'].append(restoration); evidence.save()
                require(all(restoration.get(k) is True for k in ('session_state_initialized', 'session_shapes_restored',
                    'session_current_patcher_restored', 'whole_run_shapes_restored', 'whole_run_current_patcher_restored', 'inactive')),
                    'session restoration failed', evidence)
                if mode == 'reloaded':
                    compare_outputs(prior_dir, prior['arrays'], 'updated', evidence.path, evidence.data['arrays'], mode, evidence, strict=True)
                    evidence.data['strict_reload_passed'] = True
                else:
                    compare_outputs(evidence.path, evidence.data['arrays'], 'reloaded', evidence.path, evidence.data['arrays'], mode, evidence)
                    for left in ('reference', 'zero'):
                        compare_outputs(prior_dir, prior['arrays'], left, evidence.path, evidence.data['arrays'], mode, evidence)
                audit_state(mode + '_complete')
    finally:
        if sys.exc_info()[0] is not None: evidence.data.setdefault('failure_origin_phase', evidence.data['phase'])
        hook.remove()
        evidence.data['wrapper_attributes_restored'] = J.wrappers_restored(wrappers)
        evidence.data['math_after'] = J.actual_math_state(torch)
        evidence.data['math_restored'] = evidence.data['math_after'] == initial_math
        evidence.data['patcher_restoration'] = J.restore_patcher_state(model, session, original_patcher, original_shapes)
        try:
            evidence.phase('final_resident_fingerprints', torch)
            final = {'DiT': J.frozen_inventory(dit, torch, names), 'VAE': J.frozen_inventory(stage, torch)}
            atomic_json(evidence.path / 'frozen_final.json', final)
            evidence.data['final_frozen'] = {'path': 'frozen_final.json', 'sha256': sha_file(evidence.path / 'frozen_final.json'),
                'scope': 'canonical named parameters and all registered buffers; declared200 merge writes only'}
            if snapshots is not None:
                audit_state('finally')
                for group in ('DiT', 'VAE'):
                    actual = {prefix + r['name']: r['sha256_bytes'] for prefix, key in
                        (('parameter:', 'parameters'), ('buffer:', 'registered_buffers')) for r in final[group][key]}
                    require(actual == {n: r['sha256_bytes'] for n, r in snapshots[group].items()},
                        'final bytes differ from declared frozen inventory: ' + group, evidence)
            evidence.data['final_frozen']['bytes_match_declared_snapshot'] = snapshots is not None
        except BaseException as error: evidence.data['finally_state_error'] = repr(error)
        try: evidence.data['actual_imports'] = runtime_imports(evidence.data['verified_runtime_files'])
        except BaseException as error: evidence.data['actual_imports_error'] = repr(error)
        evidence.phase('finally_saved', torch)
    evidence.data['budget_passed'] = budget_passed(evidence.data)
    require(evidence.data['budget_passed'], 'continuation budget differs', evidence)
    require(evidence.data['wrapper_attributes_restored'] and evidence.data['math_restored']
        and evidence.data['patcher_restoration']['whole_run_shapes_restored']
        and evidence.data['patcher_restoration']['whole_run_current_patcher_restored']
        and evidence.data.get('actual_imports', {}).get('passed') and 'finally_state_error' not in evidence.data,
        'final frozen/import/restoration gate failed', evidence)
    compare_outputs(prior_dir, prior['arrays'], 'reference', prior_dir, prior['arrays'], 'zero', evidence)
    evidence.data['functional_effect'] = functional_result(evidence.data['comparisons'])
    evidence.data['functional_deployment_ready'] = evidence.data['functional_effect']['readiness_gate_passed']
    evidence.data['combined_work_across_failed_and_continuation'] = {'NFE': 12, 'VAE_decodes': 4,
        'backwards': 1, 'AdamW_updates': 1, 'asset_loads': 2,
        'original_active_worker_seconds': evidence.data['input_inspection']['original_active_worker_seconds'],
        'original_last_partial_elapsed_seconds': prior['elapsed_seconds'], 'not_a_monolithic_budget_pass': True}
    evidence.data.update(complete=True, arithmetic_persistence_passed=True,
        scope='saved one-TRAIN-update continuation engineering only; old failed run unchanged; no quality/formal-training acceptance')
    evidence.phase('completed', torch); atomic_json(evidence.path / 'report.json', evidence.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pretrained-manifest', type=Path, required=True)
    parser.add_argument('--execution-contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    F.insist(not args.out.exists() and not args.out.is_symlink(), 'fresh output required; no overwrite/retry')
    contract, verified, prior, prior_dir, prior_inventory, inputs, target, inspection = inspect(args)
    if not args.execute:
        args.out.mkdir(parents=True, exist_ok=False); atomic_json(args.out / 'inspection.json', inspection)
        print(json.dumps({'complete': True, 'inspection_only': True, 'model_calls': 0})); return
    F.insist(os.environ.get('ASCEND_RT_VISIBLE_DEVICES', '').isdecimal(), 'one explicit physical card required')
    admission = F.checked_json(Path(contract['root']), contract['root_admission'])
    F.insist(admission['approved'] is True and admission['scope'] == 'saved_TRAIN_AdamW_reload_merge_continuation'
        and admission['probe_sha256'] == contract['probe_sha256']
        and admission['execution_payload_sha256'] == F.execution_payload_sha256(contract)
        and admission['partial_cpu_audit_sha256'] == contract['partial_cpu_audit']['sha256'],
        'fresh continuation admission differs')
    evidence = Evidence(args.out, {**contract, 'execution_contract_sha256': sha_file(args.execution_contract)})
    evidence.data['input_inspection'] = inspection; evidence.data['verified_runtime_files'] = verified
    try: execute(args, contract, evidence, prior, prior_dir, prior_inventory, inputs, target)
    except BaseException as exc:
        evidence.data['complete'] = False
        evidence.data['failure'] = {'type': type(exc).__name__, 'message': str(exc),
            'traceback': traceback.format_exc(), 'phase': evidence.data.get('failure_origin_phase', evidence.data['phase'])}
        evidence.save(); atomic_json(evidence.path / 'failure.json', evidence.data); raise


if __name__ == '__main__': main()
