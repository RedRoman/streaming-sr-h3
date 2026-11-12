"""P1 revised-cohort teacher training; separate from the frozen one-step probe.

CLI execution is exactly 1000 updates on the admitted 19/6 cohort, not a model
selection or headroom decision. Student execution is deliberately unsupported
until a frozen teacher, headroom review and label/shuffle contracts exist.
Default inspection validates real data and both immutable schedules, no model.
Resume requires a new output and explicit checkpoint binding; never auto-resume.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

SCHEMA = 'rphi.history_utility_training_execution.v1'
CHECKPOINT_SCHEMA = 'rphi.history_utility_teacher_checkpoint.v1'
BASE_SHA = '61ff3a2f696033791f2168410a15f02fc2f8b25274d8fb0c62a0e8cbf7a3f617'
COHORT_SHA = 'c4c3a1c044fa13e6af93c03259ca0e52ea18531ddb6a2c55bbd19abff81d44e1'
SOURCE_CONTRACT_SHA = 'a2c94df6646936422ae3abde9599323db0eb5181c1a264b97d4593f948f37742'
CONFIG = {'variant': 'blind', 'k': 3, 'ch': 64, 'blocks': 8,
          'condition_dim': 6, 'observation_encoder': 'legacy'}
STEPS, BATCH, SEED = 1000, 16, 20261110
OPTIMIZER = {'type': 'AdamW', 'lr': 1e-4, 'betas': [.9, .999], 'eps': 1e-8,
             'weight_decay': 0., 'amsgrad': False, 'foreach': False, 'fused': False,
             'gradient_clip_norm': 1., 'warmup_steps': 50, 'steps': STEPS,
             'lr_schedule': 'train_rphi.learning_rate_cosine20percent'}
LOSS = {'charbonnier_epsilon': .001, 'edge_weight': .1, 'guard_weight': .5}
FILES = {'materialization_complete', 'dataset_audit', 'train_receipt_index',
         'dev_receipt_index', 'teacher_schedule', 'student_schedule',
         'schedule_manifest', 'plan', 'train_manifest', 'dev_manifest'}
DEPENDENCIES = ('train_history_utility.py', 'rphi_history_utility.py',
                'history_utility_revised_data.py', 'history_utility_data.py',
                'rphi_model.py', 'rphi_condition_v2.py', 'rphi_data.py', 'train_rphi.py')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def value_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bound_path(binding, root):
    require(isinstance(binding, dict) and set(binding) == {'path', 'sha256'}, 'exact path/SHA binding required')
    path = Path(binding['path'])
    path = path if path.is_absolute() else Path(root) / path
    require(file_sha(path) == binding['sha256'], 'bound file changed: ' + str(path))
    return path.resolve()


def execution_sha(contract):
    return value_sha({k: v for k, v in contract.items() if k != 'admission'})


def training_identity(contract):
    # A new resume admission must not turn a previous run's output into an
    # overwrite, but it must retain every data/numerical/runtime requirement.
    return value_sha({k: v for k, v in contract.items() if k not in ('admission', 'resume')})


def verify_contract(path, *, execute=False, out=None):
    c = read(path)
    require(c['schema'] == SCHEMA and c['phase'] == 'teacher',
            'only the independently admitted teacher phase is implemented')
    require(c['steps'] == STEPS and c['batch_size'] == BATCH and c['seed'] == SEED,
            'fixed teacher 1000x16 budget/seed differs')
    require(c['optimizer'] == OPTIMIZER and c['loss'] == LOSS and c['model_config'] == CONFIG,
            'teacher math/config changed')
    require(c['baseline_checkpoint']['sha256'] == BASE_SHA and c['cohort']['sha256'] == COHORT_SHA
            and c['source_contract']['sha256'] == SOURCE_CONTRACT_SHA, 'baseline/cohort/numerical contract differs')
    require(set(c['files']) == FILES and 1 <= c['diagnostic_every'] <= STEPS,
            'missing data bindings or invalid diagnostic frequency')
    require(type(c['data_cache_bytes']) is int and 0 <= c['data_cache_bytes'] <= 12*(1 << 30),
            'preregistered training cache must not exceed12GiB')
    runtime = c['runtime']
    require(runtime['device'] in ('cpu', 'npu:0') and runtime['dtype'] == 'float32'
            and runtime['autocast'] is False and runtime['determinism'] in ('inherit', 'warn')
            and type(runtime['cpu_threads']) is int and 1 <= runtime['cpu_threads'] <= 8,
            'unsupported runtime/precision policy')
    require(isinstance(runtime['expected_versions'], dict)
            and set(runtime['expected_versions']) == {'python', 'numpy', 'torch', 'torch_npu'},
            'exact runtime versions required')
    require(runtime['version_match'] in ('exact', 'ignore_local_build_suffix'), 'unsupported version matcher')
    root = Path(c['evidence_root'])
    for key in ('baseline_checkpoint', 'cohort', 'source_contract'):
        bound_path(c[key], root)
    headroom = read(bound_path(c['headroom_contract'], root))
    require(headroom['schema'] == 'rphi.history_utility_headroom_rule.v1'
            and headroom['cohort_sha256'] == COHORT_SHA
            and headroom['data_audit']['sha256'] == c['files']['dataset_audit']['sha256'],
            'predeclared headroom/data binding differs')
    for binding in c['files'].values():
        bound_path(binding, root)
    pins = c['implementation_sha256']
    actual = {str((Path(p) if Path(p).is_absolute() else root / p).resolve()): digest
              for p, digest in pins.items()}
    require(actual and all(file_sha(p) == digest for p, digest in actual.items()), 'implementation pin mismatch')
    for name in DEPENDENCIES:
        local = str(Path(__file__).resolve().parent / name)
        require(actual.get(local) == file_sha(local), 'loaded dependency is not pinned: ' + name)
    if c.get('resume') is not None:
        bound_path(c['resume'], root)
    if execute:
        admission = read(bound_path(c['admission'], root))
        require(admission['schema'] == 'rphi.history_utility_teacher_admission.v1'
                and admission['approved'] is True and admission['phase'] == 'teacher'
                and admission['entry_sha256'] == file_sha(__file__)
                and admission['execution_sha256'] == execution_sha(c)
                and admission['out'] == str(Path(out).resolve())
                and admission['device'] == runtime['device'], 'exact separate teacher admission required')
    return c


def load_inputs(c):
    """Strict actual 19/6 data verification; neither split is a frozen test."""
    from history_utility_revised_data import RevisedHistoryUtilityDataset, assert_revised_pair
    root = c['evidence_root']
    paths = {k: bound_path(v, root) for k, v in c['files'].items()}
    complete, audit = read(paths['materialization_complete']), read(paths['dataset_audit'])
    require(complete['schema'] == 'openh3.history_utility_revised_materialization.v1'
            and complete['complete'] is True
            and complete['status'] == 'materialized_revised_primary_pending_dataset_validation'
            and complete['new_target_decodes'] == 0 and complete['new_codec_passes_completed'] == 132,
            'revised materialization is not complete')
    require(audit['schema'] == 'openh3.history_utility_revised_actual_terminal_audit.v1'
            and audit['complete'] is True and audit['status'] == 'passed_revised_dataset_and_schedule_only'
            and audit['revised_pair_ready'] is True and audit['actual_npz_files_pixel_metadata_verified'] == 100
            and audit['authority_counts'] == {'new_materialization': 22, 'reused_complete': 3}
            and audit['producer']['complete']['sha256'] == c['files']['materialization_complete']['sha256'],
            'independent actual dataset audit missing or unrelated')
    datasets = {}
    for split, count in (('train', 19), ('dev', 6)):
        index = read(paths[split + '_receipt_index'])
        require(index['schema'] == 'openh3.history_utility_revised_receipt_index.v1'
                and index['cohort'] == c['cohort'] and len(index['receipts']) == count
                and index['receipts'] == complete['receipt_indices'][split]
                and audit['datasets'][split]['receipt_index']['sha256'] == c['files'][split + '_receipt_index']['sha256'],
                'revised index/terminal audit binding differs')
        data = RevisedHistoryUtilityDataset(index['receipts'], cohort=c['cohort'], plan=paths['plan'],
            train_manifest=paths['train_manifest'], dev_manifest=paths['dev_manifest'],
            split=split, evidence_root=root, cache_bytes=c['data_cache_bytes'] if split == 'train' else 0)
        require(data.dataset_sha256 == audit['datasets'][split]['dataset_sha256'], 'actual dataset differs from audit')
        datasets[split] = data
    pair = assert_revised_pair(datasets['train'], datasets['dev'])
    require(pair == audit['pair'] and pair['train_sources'] == 19 and pair['dev_sources'] == 6,
            'independent pair acceptance differs')
    schedules = {}
    for role in ('teacher', 'student'):
        schedule = read(paths[role + '_schedule'])
        datasets['train'].validate_schedule(schedule)
        expected = audit['schedules'][role]
        require(schedule['schedule_sha256'] == expected['schedule_sha256']
                and schedule['numeric_schedule']['schedule_sha256'] == expected['numeric_schedule_sha256']
                and len(schedule['numeric_schedule']['rows']) == 16000, 'frozen audited schedule differs')
        schedules[role] = schedule
    manifest = read(paths['schedule_manifest'])
    require(manifest['schema'] == 'openh3.history_utility_persistent_schedules.v1'
            and manifest['complete'] is True and manifest['cohort_sha256'] == COHORT_SHA
            and manifest['dataset_audit']['sha256'] == c['files']['dataset_audit']['sha256'],
            'persistent schedule manifest differs')
    for role in ('teacher', 'student'):
        entry = manifest[role + '_schedule']
        require(entry['file_sha256'] == c['files'][role + '_schedule']['sha256']
                and entry['schedule_sha256'] == schedules[role]['schedule_sha256']
                and entry['numeric_schedule_sha256'] == schedules[role]['numeric_schedule']['schedule_sha256'],
                'persistent schedule file/value binding differs')
    require(manifest['dataset_sha256'] == {s: d.dataset_sha256 for s, d in datasets.items()},
            'persistent schedule dataset binding differs')
    return datasets['train'], schedules, {
        'pair': pair, 'dataset_sha256': {s: d.dataset_sha256 for s, d in datasets.items()},
        'schedule_sha256': {s: v['schedule_sha256'] for s, v in schedules.items()},
        'actual_data_audit_sha256': c['files']['dataset_audit']['sha256'],
        'legacy_research_ready': False, 'headroom_accepted': False, 'student_training_admitted': False}


def validate_batch(batch, rows):
    import numpy as np
    from history_utility_data import PROFILES
    require(set(batch['model_inputs']) == {'observed', 'availability'}, 'oracle model input keys')
    for name, value, shape in (
        ('observed', batch['model_inputs']['observed'], (16, 3, 3, 96, 96)),
        ('availability', batch['model_inputs']['availability'], (16, 2)),
        ('target', batch['target'], (16, 3, 96, 96)),
        ('teacher_weights', batch['teacher_weights'], (16, 2, 96, 96))):
        require(value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all()
                and ((value >= 0) & (value <= 1)).all(), 'invalid ' + name)
    a = batch['model_inputs']['availability']
    require(((a == 0) | (a == 1)).all() and np.all(batch['teacher_weights'] * (1-a[:, :, None, None]) == 0),
            'mask revives unavailable history')
    require(len(rows) == 16 and Counter(r['profile_id'] for r in rows) ==
            {**{p: 4 for p in PROFILES}, 'identity': 2, 'near_clean': 2}, '12+2+2 composition differs')
    require(len(batch['metadata']) == 16 and all(r['split'] == 'train' for r in rows), 'non-train row')
    for wanted, actual in zip(rows, batch['metadata']):
        require(all(actual.get(k) == v for k, v in wanted.items()), 'batch row differs from frozen schedule')


def tensor_sha(tensor):
    t = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(str((t.dtype, tuple(t.shape))).encode())
    h.update(t.numpy().tobytes())
    return h.hexdigest()


def state_sha(state):
    return value_sha({k: tensor_sha(v) for k, v in sorted(state.items())})


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return copy.deepcopy(value)


def tree_equal(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape) and torch.equal(a.cpu(), b.cpu())
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return set(a) == set(b) and all(tree_equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(tree_equal(x, y) for x, y in zip(a, b))
    return a == b


def capture_rng(device):
    import numpy as np
    import torch
    n = np.random.get_state()
    return {'python': random.getstate(), 'numpy': (n[0], n[1].tolist(), n[2], n[3], n[4]),
            'torch_cpu': torch.get_rng_state(),
            'torch_device': cpu_tree(torch.npu.get_rng_state_all()) if device.type == 'npu' else None}


def restore_rng(state, device):
    import numpy as np
    import torch
    random.setstate(state['python'])
    n = state['numpy']; np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch_cpu'])
    require((state['torch_device'] is not None) == (device.type == 'npu'), 'RNG backend differs')
    if device.type == 'npu':
        require(len(state['torch_device']) == torch.npu.device_count(), 'visible RNG devices differ')
        torch.npu.set_rng_state_all(state['torch_device'])


def make_optimizer(model):
    import torch
    return torch.optim.AdamW(model.parameters(), lr=OPTIMIZER['lr'], betas=tuple(OPTIMIZER['betas']),
        eps=OPTIMIZER['eps'], weight_decay=0., amsgrad=False, foreach=False, fused=False)


def validate_optimizer(model, optimizer, step):
    import torch
    from train_rphi import learning_rate
    parameters = list(model.parameters())
    require(len(optimizer.param_groups) == 1 and len(optimizer.param_groups[0]['params']) == len(parameters)
            and all(p is q for p, q in zip(parameters, optimizer.param_groups[0]['params'])), 'optimizer parameter order differs')
    g = optimizer.param_groups[0]
    for key in ('eps', 'weight_decay', 'amsgrad', 'foreach', 'fused'):
        require(g[key] == OPTIMIZER[key], 'optimizer option differs: ' + key)
    require(tuple(g['betas']) == tuple(OPTIMIZER['betas']) and not g.get('maximize', False)
            and not g.get('capturable', False) and not g.get('differentiable', False), 'optimizer math differs')
    expected_lr = OPTIMIZER['lr'] if step == 0 else learning_rate(step, STEPS, OPTIMIZER['lr'], OPTIMIZER['warmup_steps'])
    require(g['lr'] == expected_lr, 'optimizer LR does not match completed step')
    require((len(optimizer.state) == 0) if step == 0 else (len(optimizer.state) == len(parameters)), 'optimizer state count differs')
    for p in parameters:
        require(p.dtype == torch.float32 and torch.isfinite(p).all(), 'nonfinite/non-FP32 parameter')
        if step:
            state = optimizer.state[p]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and float(state['step']) == step,
                    'optimizer state step/keys differ')
            for key in ('exp_avg', 'exp_avg_sq'):
                v = state[key]
                require(v.dtype == torch.float32 and v.shape == p.shape and torch.isfinite(v).all(), 'invalid optimizer moment')


def train_step(model, optimizer, batch, step, *, loss_function=None, observe=None):
    """One core update; small CPU fixtures can supply tiny batches/models."""
    import torch
    from train_rphi import fidelity_losses, learning_rate
    require(type(step) is int and 1 <= step <= STEPS, 'step outside fixed 1..1000 budget')
    validate_optimizer(model, optimizer, step-1)
    optimizer.zero_grad(set_to_none=True)
    lr = learning_rate(step, STEPS, OPTIMIZER['lr'], OPTIMIZER['warmup_steps'])
    optimizer.param_groups[0]['lr'] = lr
    x, weights, target = batch['observed'], batch['teacher_weights'], batch['target']
    require(all(torch.isfinite(t).all() for t in (x, weights, target)), 'nonfinite input before forward')
    if observe: observe('forward_attempt')
    output = model(x, weights)['rgb']
    if observe: observe('forward_complete')
    require(torch.isfinite(output).all(), 'nonfinite prediction')
    fn = fidelity_losses if loss_function is None else loss_function
    loss, components = fn(output, target, x[:, -1], edge_weight=.1, guard_weight=.5)
    require(loss.numel() == 1 and loss.requires_grad and torch.isfinite(loss), 'invalid training loss')
    if observe: observe('backward_attempt')
    loss.backward()
    if observe: observe('backward_complete')
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    require(not missing, 'missing gradients: ' + repr(missing))
    require(all(torch.isfinite(p.grad).all() for p in model.parameters()), 'nonfinite gradient before optimizer')
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True, foreach=False)
    require(torch.isfinite(norm), 'nonfinite gradient norm')
    if observe: observe('optimizer_attempt')
    optimizer.step()
    if observe: observe('optimizer_complete')
    validate_optimizer(model, optimizer, step)
    result = {'step': step, 'row_end_exclusive': step*BATCH, 'lr': lr, 'lr_hex': lr.hex(),
              'loss': float(loss.detach()), 'gradient_norm_before_clip': float(norm),
              'zero_gradient_norm': float(norm) == 0.,
              **{k: float(v.detach()) for k, v in components.items()}}
    require(all(isinstance(v, (str, bool)) or __import__('math').isfinite(v) for v in result.values()), 'nonfinite diagnostic')
    optimizer.zero_grad(set_to_none=True)
    return result


def save_checkpoint(path, model, optimizer, step, binding, runtime, device, *, final=False):
    import torch
    require(type(step) is int and 1 <= step <= STEPS and final == (step == STEPS), 'only step1000 is final')
    validate_optimizer(model, optimizer, step)
    require(all(p.grad is None for p in model.parameters()), 'checkpoint must be between completed updates')
    payload = {'schema': CHECKPOINT_SCHEMA, 'phase': 'teacher', 'step': step,
        'next_row_index': step*BATCH, 'final': final, 'binding': copy.deepcopy(binding),
        'model_config': copy.deepcopy(model.model_config), 'model_state': cpu_tree(model.state_dict()),
        'parameter_names': [n for n, _ in model.named_parameters()],
        'optimizer_specification': copy.deepcopy(OPTIMIZER), 'optimizer_state': cpu_tree(optimizer.state_dict()),
        'rng_state': capture_rng(device), 'runtime': copy.deepcopy(runtime),
        'headroom_accepted': False, 'student_training_admitted': False}
    payload['model_sha256'] = state_sha(payload['model_state'])
    path = Path(path)
    require(not path.exists(), 'checkpoint already exists')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as f:
        torch.save(payload, f)
    temporary.replace(path)
    # Read actual serialized bytes immediately, without another model forward.
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    require(tree_equal(payload, loaded), 'checkpoint serialization differs')
    return {'path': str(path), 'sha256': file_sha(path), 'model_sha256': payload['model_sha256'], 'step': step, 'final': final}


def restore_checkpoint(path, model, optimizer, binding, runtime, device, *, allow_final=False):
    import torch
    state = torch.load(path, map_location='cpu', weights_only=True)
    step = state['step']
    require(state['schema'] == CHECKPOINT_SCHEMA and state['phase'] == 'teacher'
            and type(step) is int and 1 <= step <= STEPS and (allow_final or step < STEPS)
            and state['final'] == (step == STEPS) and state['next_row_index'] == step*BATCH,
            'invalid/non-resumable checkpoint boundary')
    require(state['binding'] == binding and state['runtime'] == runtime
            and state['model_config'] == model.model_config
            and state['parameter_names'] == [n for n, _ in model.named_parameters()]
            and state['optimizer_specification'] == OPTIMIZER, 'resume changes identity/runtime/optimizer')
    actual = model.state_dict()
    require(set(actual) == set(state['model_state']), 'strict checkpoint key mismatch')
    for name, value in state['model_state'].items():
        require(value.dtype == actual[name].dtype and value.shape == actual[name].shape
                and torch.isfinite(value).all(), 'strict checkpoint dtype/shape/finite mismatch')
    require(state_sha(state['model_state']) == state['model_sha256'], 'saved model fingerprint differs')
    model.load_state_dict(state['model_state'], strict=True)
    optimizer.load_state_dict(state['optimizer_state'])
    validate_optimizer(model, optimizer, step)
    require(state_sha(model.state_dict()) == state['model_sha256']
            and tree_equal(cpu_tree(optimizer.state_dict()), state['optimizer_state']), 'actual model/optimizer reload differs')
    restore_rng(state['rng_state'], device)
    require(tree_equal(capture_rng(device), state['rng_state']), 'actual RNG restore differs')
    return step


def initialize_teacher(c):
    """Shared research/backend initialization; no model forward or update."""
    import numpy as np
    import torch
    from train_rphi import setup_device, configure_determinism, determinism_snapshot
    from rphi_history_utility import MaskAwareRestorer
    rt = c['runtime']
    torch.set_num_threads(rt['cpu_threads'])
    device = setup_device(rt['device'])
    configure_determinism(rt['determinism'])
    versions = {'python': '.'.join(map(str, sys.version_info[:3])), 'numpy': str(np.__version__),
                'torch': str(torch.__version__),
                'torch_npu': str(sys.modules['torch_npu'].__version__) if 'torch_npu' in sys.modules else None}
    normalize = lambda v: v.split('+')[0] if isinstance(v, str) and rt['version_match'] == 'ignore_local_build_suffix' else v
    require({k: normalize(v) for k, v in versions.items()} ==
            {k: normalize(v) for k, v in rt['expected_versions'].items()}, 'actual runtime version differs')
    runtime = determinism_snapshot(device, rt['determinism'])
    runtime.update(versions=versions, version_match=rt['version_match'], dtype='float32', autocast=False,
                   ASCEND_RT_VISIBLE_DEVICES=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'))
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if device.type == 'npu':
        torch.npu.manual_seed_all(SEED)
    checkpoint = bound_path(c['baseline_checkpoint'], c['evidence_root'])
    baseline = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(baseline['schema'] == 'rphi.checkpoint.v1' and baseline['step'] == 3000
            and baseline['model_config'] == CONFIG, 'actual inherited checkpoint differs')
    for p, digest in baseline['implementation_sha256'].items():
        require(file_sha(Path(__file__).parent / Path(p).name) == digest, 'inherited implementation changed')
    model = MaskAwareRestorer.from_rphi(CONFIG, baseline['state_dict']).to(device).train().requires_grad_(True)
    initial_sha = state_sha(model.state_dict())
    optimizer = make_optimizer(model)
    return model, optimizer, device, runtime, initial_sha, checkpoint


def execute(c, data, schedules, proof, out, report):
    import torch
    model, optimizer, device, runtime, initial_sha, checkpoint = initialize_teacher(c)
    binding = {'training_identity_sha256': training_identity(c), 'baseline_checkpoint_sha256': BASE_SHA,
               'cohort_sha256': COHORT_SHA, **proof, 'initial_teacher_sha256': initial_sha}
    step = 0
    if c.get('resume') is not None:
        step = restore_checkpoint(bound_path(c['resume'], c['evidence_root']), model, optimizer, binding, runtime, device)
    report.update(runtime=runtime, binding=binding, start_step=step, completed_step=step,
                  optimizer_updates_this_run=0,
                  operation_attempts={'forward': 0, 'backward': 0, 'optimizer': 0},
                  operation_completions={'forward': 0, 'backward': 0, 'optimizer': 0})
    def observe(event):
        kind, stage = event.split('_')
        key = 'operation_attempts' if stage == 'attempt' else 'operation_completions'
        report[key][kind] += 1
        require(report[key][kind] <= STEPS-step, 'fixed training operation budget exceeded')
        if event == 'optimizer_complete':
            # Count the actual mutation even if post-update validation fails.
            report['optimizer_updates_this_run'] += 1
        write(out/'partial.json', report)
    rows = schedules['teacher']['numeric_schedule']['rows']
    try:
        for current in range(step+1, STEPS+1):
            report['attempted_step'] = current; report['phase'] = 'load_frozen_batch'; write(out/'partial.json', report)
            selected = rows[(current-1)*BATCH:current*BATCH]
            batch = data.batch(selected); validate_batch(batch, selected)
            tensor_batch = {k: torch.from_numpy(v).to(device) for k, v in {
                'observed': batch['model_inputs']['observed'], 'teacher_weights': batch['teacher_weights'], 'target': batch['target']}.items()}
            report['phase'] = 'one_forward_backward_update'; write(out/'partial.json', report)
            metric = train_step(model, optimizer, tensor_batch, current, observe=observe)
            report['completed_step'] = current
            with (out/'metrics.jsonl').open('a') as log:
                log.write(json.dumps(metric, allow_nan=False) + '\n')
            report['last_metric'] = metric
            if current == STEPS:
                model.eval().requires_grad_(False)
            if current % c['diagnostic_every'] == 0 or current == STEPS:
                artifact = save_checkpoint(out/('teacher_final.pt' if current == STEPS else f'teacher_step{current:06d}.pt'),
                    model, optimizer, current, binding, runtime, device, final=current == STEPS)
                report.setdefault('checkpoints', []).append(artifact)
            del batch, tensor_batch
            write(out/'partial.json', report)
        require(report['completed_step'] == STEPS and report['optimizer_updates_this_run'] == STEPS-step,
                'incomplete fixed teacher budget')
        report['final_frozen'] = not model.training and all(not p.requires_grad and p.grad is None for p in model.parameters())
        require(report['final_frozen'], 'teacher final freeze failed')
        report['final_model_sha256'] = state_sha(model.state_dict())
        report['status'] = 'teacher1000_complete_pending_headroom'
        report['headroom_accepted'] = False; report['student_training_admitted'] = False
    finally:
        report['final_observed_model_sha256'] = state_sha(model.state_dict())
        report['baseline_file_unchanged'] = file_sha(checkpoint) == BASE_SHA
        report['actual_completed_optimizer_updates_this_run'] = report['optimizer_updates_this_run']
        if device.type == 'npu':
            report['peak_npu_allocated_bytes'] = torch.npu.max_memory_allocated(device)
            report['peak_npu_reserved_bytes'] = torch.npu.max_memory_reserved(device)
        write(out/'partial.json', report)
    require(report['baseline_file_unchanged'], 'baseline file changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execution-contract', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    c = verify_contract(args.execution_contract, execute=args.execute, out=args.out)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'rphi.history_utility_teacher_run.v1', 'complete': False, 'phase': 'data_validation',
              'execution_contract': c, 'execution_contract_file_sha256': file_sha(args.execution_contract),
              'scope': 'revised19train6dev teacher only; no model selection, headroom or quality acceptance'}
    started = time.monotonic(); write(out/'partial.json', report)
    try:
        data, schedules, proof = load_inputs(c)
        report['data'] = proof
        if args.execute:
            execute(c, data, schedules, proof, out, report)
        else:
            report.update(status='inspected_actual_revised_data_no_model', model_calls=0, optimizer_steps=0)
        # Catch a changed contract, source pin or frozen input before completion.
        verify_contract(args.execution_contract, execute=args.execute, out=args.out)
        require(file_sha(args.execution_contract) == report['execution_contract_file_sha256'], 'execution contract changed')
        report['complete'] = True
    except BaseException as error:
        report.update(complete=False, status='failed', failure={'type': type(error).__name__, 'message': str(error)})
    report['elapsed_seconds'] = time.monotonic()-started
    write(out/('report.json' if report['complete'] else 'failure.json'), report)
    print(json.dumps({k: report.get(k) for k in ('complete', 'status', 'completed_step', 'failure')}, allow_nan=False))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
