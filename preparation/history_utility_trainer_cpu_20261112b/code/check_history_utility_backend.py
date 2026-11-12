"""Fresh two-batch teacher backend smoke; never continues research training.

Reuses the trainer's actual initialization, loss, optimizer, data and checkpoint
helpers. Its separately scoped checkpoint cannot satisfy research resume binding.
No headroom decision, teacher selection or student update occurs here.
"""
import argparse
from pathlib import Path
import time

import train_history_utility as T

SCHEMA = 'rphi.history_utility_backend_smoke_execution.v1'
BUDGET = {'teacher_forwards': 2, 'teacher_backwards': 2, 'teacher_optimizer_updates': 2,
          'student_updates': 0, 'quality_evaluation_forwards': 0,
          'train_cache_bytes': 12*(1 << 30), 'owned_process_group_RSS_limit_bytes': 32*(1 << 30)}


def verify_contract(path, *, execute=False, out=None):
    c = T.read(path); root = c['evidence_root']
    T.require(c['schema'] == SCHEMA and c['budget'] == BUDGET, 'backend-only schema/budget differs')
    teacher_path = T.bound_path(c['teacher_contract'], root)
    teacher = T.verify_contract(teacher_path)
    T.require(teacher.get('resume') is None and teacher['data_cache_bytes'] == BUDGET['train_cache_bytes'],
              'backend must start fresh with the declared12GiB cache')
    for name in ('entry', 'trainer'):
        p = T.bound_path(c[name], root)
        expected = Path(__file__).resolve() if name == 'entry' else Path(T.__file__).resolve()
        T.require(p == expected, 'actual loaded backend/trainer path differs')
    if execute:
        a = T.read(T.bound_path(c['admission'], root))
        T.require(a['schema'] == 'rphi.history_utility_backend_smoke_admission.v1'
                  and a['approved'] is True and a['execution_sha256'] == T.execution_sha(c)
                  and a['entry_sha256'] == T.file_sha(__file__)
                  and a['out'] == str(Path(out).resolve())
                  and a['device'] == teacher['runtime']['device'], 'separate backend admission required')
    return c, teacher


def run_core(model, optimizer, batches, binding, runtime, device, out, report):
    """Two actual optimizer updates and clear/reload, with no extra forward."""
    import torch
    T.require(len(batches) == 2, 'exactly two first scheduled batches required')
    scope = {**binding, 'artifact_role': 'backend_two_step_smoke_never_research_resume'}
    report.update(actual={'teacher_forwards': 0, 'teacher_backwards': 0,
        'teacher_optimizer_updates': 0, 'save': 0, 'clear_reload': 0},
        initial_model_sha256=T.state_sha(model.state_dict()), step_metrics=[])
    report['operation_attempts'] = {'forward': 0, 'backward': 0, 'optimizer': 0}
    def observe(event):
        kind, stage = event.split('_')
        if stage == 'attempt':
            report['operation_attempts'][kind] += 1
            T.require(report['operation_attempts'][kind] <= 2, 'backend operation budget exceeded')
        else:
            key = {'forward': 'teacher_forwards', 'backward': 'teacher_backwards',
                   'optimizer': 'teacher_optimizer_updates'}[kind]
            report['actual'][key] += 1
        T.write(out/'partial.json', report)
    for step, batch in enumerate(batches, 1):
        report['phase'] = 'scheduled_batch_' + str(step)
        T.write(out/'partial.json', report)
        if device.type == 'npu': torch.npu.synchronize(device)
        start = time.monotonic()
        metrics = T.train_step(model, optimizer, batch, step, observe=observe)
        if device.type == 'npu': torch.npu.synchronize(device)
        metrics['actual_update_wall_seconds'] = time.monotonic()-start
        report['step_metrics'].append(metrics)
        T.write(out/'partial.json', report)
    report['updated_model_sha256'] = T.state_sha(model.state_dict())
    T.require(report['updated_model_sha256'] != report['initial_model_sha256'], 'two steps did not change teacher weights')
    path = out/'backend_smoke_step2.pt'
    report['checkpoint'] = T.save_checkpoint(path, model, optimizer, 2, scope, runtime, device)
    report['actual']['save'] += 1
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
    optimizer.state.clear()
    step = T.restore_checkpoint(path, model, optimizer, scope, runtime, device)
    report['actual']['clear_reload'] += 1
    T.require(step == 2 and T.state_sha(model.state_dict()) == report['updated_model_sha256'], 'backend strict reload failed')
    report.update(strict_model_optimizer_RNG_reload=True, actual_final_step=step,
                  research_updates=0, checkpoint_eligible_as_research_warmstart=False,
                  headroom_accepted=False, quality_accepted=False)


def main():
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-contract', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    c, teacher = verify_contract(args.execution_contract, execute=args.execute, out=args.out)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'rphi.history_utility_backend_smoke.v1', 'complete': False,
              'execution_contract_sha256': T.file_sha(args.execution_contract), 'execution_contract': c,
              'scope': 'first two actual teacher batches; engineering only, not research warmstart'}
    started = time.monotonic(); T.write(out/'partial.json', report)
    try:
        data, schedules, proof = T.load_inputs(teacher)
        report['data'] = proof
        rows = schedules['teacher']['numeric_schedule']['rows'][:32]
        batches = []
        for start in (0, 16):
            selected = rows[start:start+16]; batch = data.batch(selected); T.validate_batch(batch, selected)
            batches.append(batch)
        report['row_sha256'] = [r['row_sha256'] for r in rows]
        if args.execute:
            import torch
            model, optimizer, device, runtime, initial_sha, baseline = T.initialize_teacher(teacher)
            report['runtime'] = runtime
            tensor_batches = [{k: torch.from_numpy(v).to(device) for k, v in {
                'observed': batch['model_inputs']['observed'], 'teacher_weights': batch['teacher_weights'],
                'target': batch['target']}.items()} for batch in batches]
            binding = {'training_identity_sha256': T.training_identity(teacher), **proof,
                       'initial_teacher_sha256': initial_sha, 'baseline_checkpoint_sha256': T.BASE_SHA}
            try:
                run_core(model, optimizer, tensor_batches, binding, runtime, device, out, report)
                report['status'] = 'passed_two_step_backend_only_not_research_initialization'
            finally:
                report['final_observed_model_sha256'] = T.state_sha(model.state_dict())
                report['baseline_file_unchanged'] = T.file_sha(baseline) == T.BASE_SHA
                if device.type == 'npu':
                    report['peak_npu_allocated_bytes'] = torch.npu.max_memory_allocated(device)
                    report['peak_npu_reserved_bytes'] = torch.npu.max_memory_reserved(device)
                T.write(out/'partial.json', report)
            T.require(report['baseline_file_unchanged'], 'baseline checkpoint changed')
        else:
            report.update(status='inspected_first_two_batches_no_model', model_calls=0, optimizer_steps=0)
        verify_contract(args.execution_contract, execute=args.execute, out=args.out)
        T.require(T.file_sha(args.execution_contract) == report['execution_contract_sha256'], 'contract changed')
        report['complete'] = True
    except BaseException as exc:
        report.update(complete=False, status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
    report['elapsed_seconds'] = time.monotonic()-started
    T.write(out/('report.json' if report['complete'] else 'failure.json'), report)
    print(json.dumps({k: report.get(k) for k in ('complete', 'status', 'actual', 'failure')}, allow_nan=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__': raise SystemExit(main())
