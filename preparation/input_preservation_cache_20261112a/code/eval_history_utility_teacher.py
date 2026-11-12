"""Frozen final1000 teacher headroom evaluation on the revised six-source dev set.

No optimizer, training, checkpoint selection or student admission. Headroom uses
unclamped RGB MAE; deployment-clipped PSNR is a separate descriptive measure.
LPIPS and motion-compensated temporal quality remain separate later evaluations.
Default inspection reads actual dev pixels/rows but constructs no model.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import train_history_utility as T

SCHEMA = 'rphi.history_utility_teacher_headroom_execution.v1'
HEADROOM_SHA = 'd529c075ba5499d512b3728ff689cc104f1a80e9b985d5bb10bda7eaca690b03'
MAX_CROP_FORWARDS = 11520
EXPECTED_CROP_FORWARDS = 11430
MAX_OUTPUT_BYTES = 3*(1 << 30)
ARRAY_NAMES = ('current', 'target', 'all_history', 'current_only', 'remove_0', 'remove_1')
DEPENDENCIES = (*T.DEPENDENCIES, 'eval_history_utility_teacher.py', 'history_utility_headroom.py')


def verify_contract(path, *, execute=False, out=None):
    c = T.read(path); root = c['evidence_root']
    T.require(c['schema'] == SCHEMA and c['batch_size'] == 16
              and c['max_crop_forwards'] == MAX_CROP_FORWARDS
              and c['max_output_bytes'] == MAX_OUTPUT_BYTES
              and type(c['data_cache_bytes']) is int and 0 <= c['data_cache_bytes'] <= 4*(1 << 30),
              'fixed dev evaluation geometry/budget/cache differs')
    T.require(c['headroom_contract']['sha256'] == HEADROOM_SHA, 'headroom rule must precede outcomes')
    paths = {k: T.bound_path(c[k], root) for k in
             ('teacher_contract', 'teacher_report', 'teacher_checkpoint', 'dataset_audit', 'headroom_contract')}
    teacher, report, audit, protocol = (T.read(paths[k]) for k in
                                      ('teacher_contract', 'teacher_report', 'dataset_audit', 'headroom_contract'))
    T.require(teacher['schema'] == T.SCHEMA and teacher['phase'] == 'teacher'
              and teacher['steps'] == 1000 and teacher['batch_size'] == 16 and teacher['seed'] == T.SEED
              and teacher['optimizer'] == T.OPTIMIZER and teacher['loss'] == T.LOSS
              and teacher['model_config'] == T.CONFIG and teacher['baseline_checkpoint']['sha256'] == T.BASE_SHA
              and teacher['cohort']['sha256'] == T.COHORT_SHA
              and teacher['headroom_contract']['sha256'] == HEADROOM_SHA,
              'not the preregistered final1000 teacher')
    T.require(report['schema'] == 'rphi.history_utility_teacher_run.v1' and report['complete'] is True
              and report['status'] == 'teacher1000_complete_pending_headroom'
              and report['completed_step'] == 1000 and report['final_frozen'] is True
              and report['baseline_file_unchanged'] is True
              and report['execution_contract_file_sha256'] == c['teacher_contract']['sha256']
              and report['execution_contract'] == teacher
              and report['binding']['training_identity_sha256'] == T.training_identity(teacher)
              and report['binding']['baseline_checkpoint_sha256'] == T.BASE_SHA
              and report['binding']['cohort_sha256'] == T.COHORT_SHA
              and report['final_model_sha256'] == report['final_observed_model_sha256'],
              'teacher terminal report is not a frozen successful final1000')
    finals = [x for x in report['checkpoints'] if x['final']]
    T.require(len(finals) == 1 and finals[0]['step'] == 1000
              and finals[0]['sha256'] == c['teacher_checkpoint']['sha256']
              and finals[0]['model_sha256'] == report['final_model_sha256'], 'checkpoint is not the unique final teacher')
    T.require(audit['schema'] == 'openh3.history_utility_revised_actual_terminal_audit.v1'
              and audit['complete'] is True and audit['status'] == 'passed_revised_dataset_and_schedule_only'
              and audit['revised_pair_ready'] is True and audit['actual_npz_files_pixel_metadata_verified'] == 100
              and audit['pair'] == report['data']['pair']
              and c['dataset_audit']['sha256'] == teacher['files']['dataset_audit']['sha256']
              and protocol['data_audit']['sha256'] == c['dataset_audit']['sha256']
              and report['data']['actual_data_audit_sha256'] == c['dataset_audit']['sha256'],
              'actual dataset/teacher/headroom audit identity differs')
    expected_data = {'pair': audit['pair'],
        'dataset_sha256': {s: audit['datasets'][s]['dataset_sha256'] for s in ('train', 'dev')},
        'schedule_sha256': {s: audit['schedules'][s]['schedule_sha256'] for s in ('teacher', 'student')},
        'actual_data_audit_sha256': c['dataset_audit']['sha256'], 'legacy_research_ready': False,
        'headroom_accepted': False, 'student_training_admitted': False}
    T.require(report['data'] == expected_data and all(report['binding'].get(k) == v for k, v in expected_data.items()),
              'actual teacher training data proof differs')
    for binding in [teacher['cohort'], teacher['source_contract'], teacher['baseline_checkpoint'], *teacher['files'].values()]:
        T.bound_path(binding, teacher['evidence_root'])
    pins = {str((Path(p) if Path(p).is_absolute() else Path(root)/p).resolve()): s
            for p, s in c['implementation_sha256'].items()}
    T.require(all(T.file_sha(p) == s for p, s in pins.items()), 'evaluation code pin changed')
    for name in DEPENDENCIES:
        p = Path(__file__).resolve().parent/name
        T.require(pins.get(str(p)) == T.file_sha(p), 'loaded evaluator dependency not pinned: ' + name)
    T.require(T.file_sha(Path(__file__).parent/'history_utility_headroom.py') == protocol['implementation']['sha256'],
              'reducer differs from the preregistered headroom implementation')
    # Original training paths remain immutable; copying identical source to this
    # new evaluation snapshot does not rewrite the original training identity.
    original_by_name = {}
    for p, digest in teacher['implementation_sha256'].items():
        resolved = Path(p) if Path(p).is_absolute() else Path(teacher['evidence_root'])/p
        T.require(T.file_sha(resolved) == digest, 'original teacher code changed')
        original_by_name[resolved.name] = digest
    for name in T.DEPENDENCIES:
        T.require(original_by_name.get(name) == T.file_sha(Path(__file__).parent/name), 'evaluation uses different teacher math')
    rt = c['runtime']
    T.require(rt['device'] in ('cpu', 'npu:0') and rt['dtype'] == 'float32' and rt['autocast'] is False
              and rt['version_match'] == 'exact' and rt['determinism'] in ('inherit', 'warn')
              and type(rt['cpu_threads']) is int and 1 <= rt['cpu_threads'] <= 8
              and set(rt['expected_versions']) == {'python', 'numpy', 'torch', 'torch_npu'}, 'invalid evaluation runtime')
    if execute:
        a = T.read(T.bound_path(c['admission'], root))
        T.require(a['schema'] == 'rphi.history_utility_teacher_headroom_admission.v1' and a['approved'] is True
                  and a['execution_sha256'] == T.execution_sha(c) and a['entry_sha256'] == T.file_sha(__file__)
                  and a['out'] == str(Path(out).resolve()) and a['device'] == rt['device'], 'separate headroom admission required')
    return c, teacher, report, audit, protocol, paths


def load_dev(c, teacher, audit, protocol):
    from history_utility_revised_data import RevisedHistoryUtilityDataset
    paths = {k: T.bound_path(teacher['files'][k], teacher['evidence_root'])
             for k in ('dev_receipt_index', 'plan', 'train_manifest', 'dev_manifest')}
    index = T.read(paths['dev_receipt_index'])
    T.require(index['schema'] == 'openh3.history_utility_revised_receipt_index.v1'
              and index['cohort'] == teacher['cohort'] and len(index['receipts']) == 6
              and T.file_sha(paths['dev_receipt_index']) == audit['datasets']['dev']['receipt_index']['sha256'],
              'dev index differs from actual audited six-source set')
    data = RevisedHistoryUtilityDataset(index['receipts'], cohort=teacher['cohort'], plan=paths['plan'],
        train_manifest=paths['train_manifest'], dev_manifest=paths['dev_manifest'], split='dev',
        evidence_root=teacher['evidence_root'], cache_bytes=c['data_cache_bytes'])
    T.require(data.dataset_sha256 == audit['datasets']['dev']['dataset_sha256']
              and data.source_ids == protocol['dev_source_ids'], 'actual dev pixels/source set differs')
    rows = data.evaluation_rows()
    validate_rows(rows, protocol)
    return data, rows


def validate_rows(rows, protocol):
    from history_utility_data import PROFILES, history_indices, true_availability
    T.require(protocol['profile_ids'] == [*PROFILES, 'identity', 'near_clean'], 'profile order differs')
    expected = [(s, p, t) for s in protocol['dev_source_ids'] for p in protocol['profile_ids'] for t in range(96)]
    T.require(len(rows) == 2880 and len(expected) == len(rows), 'exact all96-frame dev coverage required')
    for i, (row, key) in enumerate(zip(rows, expected)):
        T.require((row['source_id'], row['profile_id'], row['frame_index']) == key and row['split'] == 'dev'
                  and row['row_index'] == i and row['crop_tlhw'] == [192, 368, 96, 96]
                  and row['history_indices'] == history_indices(key[2])
                  and row['true_availability'] == true_availability(key[2])
                  and row['external_mask'] == [1, 1] and row['teacher_mask'] is None
                  and T.value_sha({k: v for k, v in row.items() if k != 'row_sha256'}) == row['row_sha256'],
                  'dev row chronology/profile/provenance differs')
    T.require(sum(2+sum(r['true_availability']) for r in rows) == EXPECTED_CROP_FORWARDS,
              'crop-forward accounting differs')


def load_teacher(path, teacher_report, runtime):
    import torch
    from rphi_history_utility import MaskAwareRestorer
    from train_rphi import setup_device, configure_determinism, determinism_snapshot
    torch.set_num_threads(runtime['cpu_threads'])
    device = setup_device(runtime['device']); configure_determinism(runtime['determinism'])
    versions = {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': str(np.__version__),
                'torch': str(torch.__version__), 'torch_npu': str(sys.modules['torch_npu'].__version__) if 'torch_npu' in sys.modules else None}
    T.require(versions == runtime['expected_versions'], 'actual runtime versions differ')
    state = torch.load(path, map_location='cpu', weights_only=True)
    T.require(state['schema'] == T.CHECKPOINT_SCHEMA and state['phase'] == 'teacher'
              and state['step'] == 1000 and state['final'] is True and state['next_row_index'] == 16000
              and state['binding'] == teacher_report['binding'] and state['runtime'] == teacher_report['runtime']
              and state['optimizer_specification'] == T.OPTIMIZER and state['headroom_accepted'] is False
              and state['student_training_admitted'] is False,
              'only the authenticated final1000 teacher can be evaluated')
    model = MaskAwareRestorer(T.CONFIG)
    T.require(state['model_config'] == model.model_config
              and state['parameter_names'] == [n for n, _ in model.named_parameters()]
              and set(state['model_state']) == set(model.state_dict()), 'strict final model config/keys differ')
    for name, target in model.state_dict().items():
        value = state['model_state'][name]
        T.require(value.dtype == torch.float32 and value.shape == target.shape and torch.isfinite(value).all(),
                  'final model shape/dtype/finite mismatch')
    T.require(T.state_sha(state['model_state']) == state['model_sha256'] == teacher_report['final_model_sha256'],
              'final checkpoint model fingerprint differs')
    model.load_state_dict(state['model_state'], strict=True)
    model.to(device).eval().requires_grad_(False)
    T.require(T.state_sha(model.state_dict()) == state['model_sha256'], 'device copy changed teacher state')
    del state
    return model, device, {**determinism_snapshot(device, runtime['determinism']), 'versions': versions}


def metrics(prediction, target, current):
    """Float64 reductions of actual float32 predictions; no epsilon PSNR floor."""
    T.require(prediction.dtype == np.float32 and target.dtype == np.float32
              and prediction.shape == target.shape and np.isfinite(prediction).all(), 'invalid RGB metric array')
    raw, y = prediction.astype(np.float64), target.astype(np.float64)
    clipped = np.clip(raw, 0., 1.)
    mse = float(np.mean((clipped-y)**2))
    return {'unclamped_mae': float(np.mean(np.abs(raw-y))),
            'unclamped_mse': float(np.mean((raw-y)**2)),
            'clipped_mae': float(np.mean(np.abs(clipped-y))), 'clipped_mse': mse,
            'clipped_psnr': None if mse == 0 else float(-10*np.log10(mse)),
            'clipped_exact_match': mse == 0, 'clip_fraction': float(np.mean((raw < 0) | (raw > 1))),
            'changed_pixel_fraction': float(np.mean(np.any(prediction != current, axis=0))),
            'prediction_float32_sha256': __import__('hashlib').sha256(prediction.tobytes()).hexdigest()}


def _tile_errors(prediction, target):
    error = np.abs(prediction.astype(np.float64)-target.astype(np.float64)).mean(axis=0)
    h, w = error.shape
    T.require(h % 16 == 0 and w % 16 == 0, 'tile16 geometry differs')
    return error.reshape(h//16, 16, w//16, 16).mean(axis=(1, 3))


def evaluate_batch(model, batch, device, counts, *, return_arrays=False):
    """Only observed+weights enter the restorer; target is read after all calls."""
    import torch
    from rphi_history_utility import remove_history
    x = torch.from_numpy(batch['model_inputs']['observed']).to(device)
    a = torch.from_numpy(batch['model_inputs']['availability']).to(device)
    T.require(set(batch['model_inputs']) == {'observed', 'availability'} and x.ndim == 5
              and x.shape[1:3] == (3, 3) and a.shape == (len(x), 2)
              and x.dtype == a.dtype == torch.float32 and torch.isfinite(x).all()
              and ((x >= 0) & (x <= 1)).all() and ((a == 0) | (a == 1)).all(), 'invalid raw-only model inputs')
    weights = a[:, :, None, None].expand(-1, -1, x.shape[-2], x.shape[-1]).contiguous()
    def predict(obs, w):
        counts['model_invocation_attempts'] += 1; counts['crop_forward_attempts'] += len(obs)
        T.require(counts['crop_forward_attempts'] <= MAX_CROP_FORWARDS, 'forward budget exceeded')
        result = model(obs, w)['rgb']
        counts['model_invocations_completed'] += 1; counts['crop_forwards_completed'] += len(obs)
        T.require(result.dtype == torch.float32 and result.shape == obs[:, -1].shape and torch.isfinite(result).all(),
                  'invalid frozen teacher prediction')
        return result.detach().cpu().numpy().copy()
    with torch.inference_mode():
        all_history = predict(x, weights)
        current_only = predict(x, torch.zeros_like(weights))
        without = np.repeat(all_history[:, None], 2, axis=1)
        for slot in (0, 1):
            valid = a[:, slot].bool()
            n = int(valid.sum())
            counts['unavailable_LOO_crops_skipped'] += len(x)-n
            if n:
                ids = valid.detach().cpu().numpy()
                without[ids, slot] = predict(x[valid], remove_history(weights[valid], slot))
    target = batch['target']
    T.require(target.dtype == np.float32 and target.shape == all_history.shape and np.isfinite(target).all()
              and ((target >= 0) & (target <= 1)).all(), 'invalid target used only for scoring')
    rows = []
    for i, meta in enumerate(batch['metadata']):
        current = batch['model_inputs']['observed'][i, -1]
        modes = {'input': metrics(current, target[i], current),
                 'all_history': metrics(all_history[i], target[i], current),
                 'current_only': metrics(current_only[i], target[i], current)}
        base = _tile_errors(all_history[i], target[i]); raw = np.zeros((2, *base.shape), np.float64)
        available = batch['model_inputs']['availability'][i].astype(int).tolist()
        for slot in (0, 1):
            modes['remove_' + str(slot)] = metrics(without[i, slot], target[i], current)
            if available[slot]: raw[slot] = _tile_errors(without[i, slot], target[i])-base
        rows.append({**meta, 'availability': available,
            'all_history_mae': modes['all_history']['unclamped_mae'],
            'current_only_mae': modes['current_only']['unclamped_mae'],
            'raw_utility': raw.tolist(), 'metrics': modes,
            'all_history_mae_harm_vs_input': modes['all_history']['unclamped_mae']-modes['input']['unclamped_mae'],
            'current_only_mae_harm_vs_input': modes['current_only']['unclamped_mae']-modes['input']['unclamped_mae']})
    arrays = {'current': batch['model_inputs']['observed'][:, -1], 'target': target,
              'all_history': all_history, 'current_only': current_only,
              'remove_0': without[:, 0], 'remove_1': without[:, 1]}
    return (rows, arrays) if return_arrays else rows


def save_arrays(out, index, arrays, rows):
    """Persist existing CPU outputs, with no second model call or quantization."""
    T.require(set(arrays) == set(ARRAY_NAMES), 'six actual RGB arrays required')
    arrays = {k: np.ascontiguousarray(arrays[k]) for k in ARRAY_NAMES}
    shape = arrays['target'].shape
    T.require(len(shape) == 4 and shape[:2] == (len(rows), 3)
              and all(a.dtype == np.float32 and a.shape == shape and np.isfinite(a).all() for a in arrays.values()),
              'actual saved RGB shape/dtype/finite differs')
    before = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
    # Fixed geometry has1.911GB array payload for all2880rows; leave16MiB for
    # NPZ headers, the finite metric JSON and final receipt before each write.
    T.require(before+sum(a.nbytes for a in arrays.values())+(16 << 20) <= MAX_OUTPUT_BYTES,
              'declared3GiB output budget would be exceeded')
    folder = out/'arrays'; folder.mkdir(exist_ok=True)
    path = folder/f'batch_{index:04d}.npz'
    T.require(not path.exists(), 'actual RGB cache exists; no overwrite')
    temporary = path.with_suffix('.npz.tmp')
    with temporary.open('xb') as f:
        np.savez(f, **arrays)  # ZIP_STORED, not a lossy/quantized export.
    temporary.replace(path)
    sha = lambda a: __import__('hashlib').sha256(a.tobytes()).hexdigest()
    return {'path': str(path), 'sha256': T.file_sha(path), 'bytes': path.stat().st_size,
            'arrays': {k: {'dtype': str(a.dtype), 'shape': list(a.shape), 'sha256': sha(a)} for k, a in arrays.items()},
            'row_indices': [r['row_index'] for r in rows], 'row_sha256': [r['row_sha256'] for r in rows],
            'semantics': 'actual float32 unclamped predictions and input/target96x96crops; zero extra forwards'}


def summarize(records):
    grouped = defaultdict(list)
    for r in records:
        grouped[(r['source_id'], r['profile_id'], r['severity_phase'])].append(r)
        grouped[(r['source_id'], r['profile_id'], 'all_frames')].append(r)
    result = []
    for (source, profile, phase), rows in sorted(grouped.items()):
        modes = {}
        for mode in ('input', 'all_history', 'current_only', 'remove_0', 'remove_1'):
            entries = [r['metrics'][mode] for r in rows]
            values = {k: float(np.mean([e[k] for e in entries])) for k in
                      ('unclamped_mae', 'unclamped_mse', 'clipped_mae', 'clipped_mse', 'clip_fraction', 'changed_pixel_fraction')}
            mse = values['clipped_mse']; values['clipped_psnr_from_mean_mse'] = None if mse == 0 else float(-10*np.log10(mse))
            values['clipped_exact_match_frames'] = sum(e['clipped_exact_match'] for e in entries)
            modes[mode] = values
        result.append({'source_id': source, 'profile_id': profile, 'severity_phase': phase, 'frames': len(rows),
                       'metrics': modes, 'all_history_mae_harm_vs_input': float(np.mean([r['all_history_mae_harm_vs_input'] for r in rows])),
                       'current_only_mae_harm_vs_input': float(np.mean([r['current_only_mae_harm_vs_input'] for r in rows]))})
    return result


def run_evaluation(model, data, rows, device, out, report, *, batch_size=16, protocol=None):
    """Small fixtures may omit the full-cohort reducer; CLI cannot."""
    import torch
    T.require(not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters()), 'teacher must already be frozen')
    before = T.state_sha(model.state_dict())
    versions = {n: p._version for n, p in model.named_parameters()}
    report.update(frozen_before_sha256=before, counts={'crop_forward_attempts': 0, 'crop_forwards_completed': 0,
        'model_invocation_attempts': 0, 'model_invocations_completed': 0, 'unavailable_LOO_crops_skipped': 0},
        completed_rows=0, actual_rgb_caches=[], array_bytes_on_disk=0)
    records = []
    try:
        for start in range(0, len(rows), batch_size):
            selected = rows[start:start+batch_size]; batch = data.batch(selected)
            T.require(len(batch['metadata']) == len(selected)
                      and all(all(m.get(k) == v for k, v in r.items()) for m, r in zip(batch['metadata'], selected)),
                      'actual batch differs from authenticated rows')
            expected_a = np.asarray([r['true_availability'] for r in selected], np.float32)
            T.require(np.array_equal(batch['model_inputs']['availability'], expected_a), 'actual causal availability differs')
            outputs, arrays = evaluate_batch(model, batch, device, report['counts'], return_arrays=True)
            artifact = save_arrays(out, start//batch_size, arrays, selected)
            report['actual_rgb_caches'].append(artifact)
            report['array_bytes_on_disk'] += artifact['bytes']
            with (out/'records.jsonl').open('a') as f:
                for record in outputs: f.write(json.dumps(record, allow_nan=False)+'\n')
            records.extend(outputs); report['completed_rows'] = len(records)
            report['last_completed_row_sha256'] = selected[-1]['row_sha256']
            report['output_bytes_observed_before_partial_write'] = sum(p.stat().st_size for p in out.rglob('*') if p.is_file())
            T.require(report['output_bytes_observed_before_partial_write'] <= MAX_OUTPUT_BYTES, 'actual3GiB output budget exceeded')
            T.write(out/'partial.json', report)
        expected = sum(2+sum(r['true_availability']) for r in rows)
        T.require(report['counts']['crop_forwards_completed'] == report['counts']['crop_forward_attempts'] == expected,
                  'actual crop-forward count differs')
        if protocol is not None:
            from history_utility_headroom import assess_records
            T.require(len(records) == 2880 and expected == EXPECTED_CROP_FORWARDS, 'partial actual headroom evaluation')
            report['assessment'] = assess_records(records, protocol)
        report['descriptive_by_source_profile_phase'] = summarize(records)
        report['records'] = {'path': str(out/'records.jsonl'), 'sha256': T.file_sha(out/'records.jsonl'), 'rows': len(records)}
    finally:
        report['frozen_after_sha256'] = T.state_sha(model.state_dict())
        report['frozen_flags_preserved'] = not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        report['parameter_versions_preserved'] = versions == {n: p._version for n, p in model.named_parameters()}
        report['state_sha_preserved'] = before == report['frozen_after_sha256']
        if device.type == 'npu':
            report['peak_npu_allocated_bytes'] = torch.npu.max_memory_allocated(device)
            report['peak_npu_reserved_bytes'] = torch.npu.max_memory_reserved(device)
        T.write(out/'partial.json', report)
    T.require(report['frozen_flags_preserved'] and report['parameter_versions_preserved'] and report['state_sha_preserved'],
              'teacher mutated during headroom evaluation')
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-contract', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--execute', action='store_true'); args = p.parse_args()
    c, teacher, terminal, audit, protocol, paths = verify_contract(args.execution_contract, execute=args.execute, out=args.out)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'rphi.history_utility_teacher_headroom_run.v1', 'complete': False,
        'execution_contract': c, 'execution_contract_file_sha256': T.file_sha(args.execution_contract),
        'student_training_admitted': False, 'optimizer_updates': 0, 'backwards': 0,
        'scope': 'Preregistered teacher headroom on revised6dev only; no model selection or final quality acceptance.',
        'metric_precision': 'model outputs FP32; independent MAE/MSE/tile reductions FP64; raw LOO unnormalized',
        'cache_scope': 'six actual uncompressed float32 arrays per batch, only96x96crops; no full-frame RGB device quality claim',
        'output_budget_bytes': MAX_OUTPUT_BYTES,
        'temporal_scope': 'stride1 raw frame/phase diagnostics, no flow/motion-compensated flicker claim',
        'LPIPS': 'not used for this headroom gate; pending later fixed comparison'}
    started = time.monotonic(); T.write(out/'partial.json', report)
    try:
        data, rows = load_dev(c, teacher, audit, protocol)
        report.update(dev_dataset_sha256=data.dataset_sha256, evaluation_rows_sha256=T.value_sha(rows), evaluation_rows=len(rows))
        if args.execute:
            model, device, runtime = load_teacher(paths['teacher_checkpoint'], terminal, c['runtime'])
            report['runtime'] = runtime
            run_evaluation(model, data, rows, device, out, report, batch_size=c['batch_size'], protocol=protocol)
            report['status'] = 'complete_headroom_pass' if report['assessment']['headroom_rule_passed'] else 'complete_headroom_fail'
        else:
            report.update(status='inspected_actual_dev_rows_no_model', model_calls=0)
        verify_contract(args.execution_contract, execute=args.execute, out=args.out)
        T.require(T.file_sha(args.execution_contract) == report['execution_contract_file_sha256'], 'evaluation contract changed')
        report['complete'] = True
    except BaseException as error:
        report.update(status='failed', failure={'type': type(error).__name__, 'message': str(error)})
    report['elapsed_seconds'] = time.monotonic()-started
    T.write(out/('report.json' if report['complete'] else 'failure.json'), report)
    print(json.dumps({k: report.get(k) for k in ('complete', 'status', 'completed_rows', 'failure')}, allow_nan=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
