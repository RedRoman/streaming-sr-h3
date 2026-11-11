"""Finite standalone R_phi RGB pilot; H3/VAE/LoRA are never loaded.

Each arm shares data, source split, crop sequence, backbone initialization and
training budget. Known conditions are public task/scale; blind arms receive no
condition metadata in their encoder. Confidence calibration has its own optimizer
and detached features/errors, so it cannot reweight or change fidelity gradients.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from rphi_data import RphiDataset, TASKS, assert_disjoint
from rphi_model import RphiModel, uncertainty_nll


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((a.dtype, tuple(a.shape))).encode())
        h.update(a.numpy().tobytes())
    return h.hexdigest()


def setup_device(name):
    """Plain torch modules need no ComfyUI Module.to monkey-patch."""
    if name.startswith('npu'):
        import torch_npu  # noqa: F401
        device = torch.device(name)
        torch.npu.set_device(device)
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.matmul.allow_hf32 = False
        torch.npu.conv.allow_hf32 = False
        torch.npu.reset_peak_memory_stats(device)
        return device
    return torch.device(name)


def sync(device):
    if device.type == 'npu':
        torch.npu.synchronize(device)


def fidelity_losses(pred, target, guide, edge_weight=.1, guard_weight=.5):
    """A matched RGB loss; learned uncertainty never weights this objective."""
    if pred.shape != target.shape or guide.shape != target.shape:
        raise ValueError('fidelity loss requires geometrically aligned HR tensors')
    diff = pred - target
    reconstruction = (diff.square() + 1e-6).sqrt().mean()
    dx = (pred[..., :, 1:] - pred[..., :, :-1]) - (target[..., :, 1:] - target[..., :, :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    edge = .5 * (dx.abs().mean() + dy.abs().mean())
    axes = (1, 2, 3)
    guard = torch.relu(diff.abs().mean(axes) - (guide - target).abs().mean(axes)).mean()
    return reconstruction + edge_weight * edge + guard_weight * guard, {
        'charbonnier': reconstruction, 'edge_l1': edge, 'identity_guard': guard}


def move(batch, device):
    return {key: batch[key].to(device) for key in ('observed', 'target', 'condition')}


def build_contract(args, train, dev):
    return {
        'schema': 'rphi.training_contract.v1',
        'train_manifest_sha256': train.manifest_sha256,
        'dev_manifest_sha256': dev.manifest_sha256,
        'train_source_ids': sorted({r['source_id'] for r in train.records}),
        'train_source_sha256': sorted({r['source_sha256'] for r in train.records}),
        'dev_source_ids': sorted({r['source_id'] for r in dev.records}),
        'task_order': list(TASKS), 'target': 'clean_original_recording',
        'target_limit': 'screened original recording is a clean-reference proxy, not pristine GT',
        'source_verification': 'original source SHA was verified at materialization; this run re-verifies NPZ content/provenance',
        'guide_geometry': 'SR: float Torch bicubic antialias to HR; native tasks: unchanged RGB',
        'anchor': 'Q4(rgb), canonical Torch bicubic antialias, NOT the physical observation operator',
        'k': args.k, 'crop': args.crop, 'seed': args.seed,
        'steps': args.steps, 'batch_size': args.batch,
        'sampling': 'step-addressed PCG64, balanced task slots; identical for every arm',
        'identity_fraction': args.identity_fraction, 'near_clean_fraction': args.near_clean_fraction,
        'near_clean': 'data module fixed low Gaussian sigma recipe; source frames stay in original split',
        'loss': {'rgb': 'Charbonnier eps=0.001', 'edge_l1_weight': args.edge_weight,
                 'identity_guard_weight': args.guard_weight,
                 'confidence': 'detached error Gaussian NLL; separate optimizer; no fidelity reweighting'},
        'evaluation': 'final checkpoint, all dev tasks, original RGB; training crop validation is diagnostic only',
        'rphi_scope': 'standalone canonical RGB prototype; no H3 integration, soft DC or recurrent state yet'}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def log(path, **row):
    row['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with Path(path).open('a') as f:
        f.write(json.dumps(row, allow_nan=False) + '\n')
    print(json.dumps(row, allow_nan=False), flush=True)


@torch.no_grad()
def validate(model, dataset, args, device):
    """Fixed crop proxy, same examples for every checkpoint; never frozen-test evidence."""
    was_training = model.training
    model.eval()
    rng = np.random.default_rng(881031)
    errors, baseline, counts = {}, {}, {}
    for _ in range(args.val_batches):
        b = dataset.sample_batch(rng, args.batch, crop=args.crop, k=args.k,
                                 identity_fraction=args.identity_fraction,
                                 near_clean_fraction=args.near_clean_fraction)
        x = move(b, device)
        out = model(x['observed'], x['condition'])['rgb'].clamp(0, 1)
        pred_mae = (out - x['target']).abs().mean((1, 2, 3)).cpu().tolist()
        base_mae = (x['observed'][:, -1] - x['target']).abs().mean((1, 2, 3)).cpu().tolist()
        for j, (a, z) in enumerate(zip(pred_mae, base_mae)):
            control = int(b['control'][j])
            key = TASKS[int(b['task'][j])] if control == 0 else ('identity' if control == 1 else 'near_clean')
            errors[key] = errors.get(key, 0.) + a
            baseline[key] = baseline.get(key, 0.) + z
            counts[key] = counts.get(key, 0) + 1
    model.train(was_training)
    return {k: {'mae': errors[k] / n, 'guide_mae': baseline[k] / n, 'n_crops': n}
            for k, n in counts.items()}


def save_checkpoint(path, model, optimizer, confidence_optimizer, contract, step, implementation):
    if path.exists():
        raise FileExistsError(path)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    payload = {'schema': 'rphi.checkpoint.v1', 'model_config': model.model_config,
               'state_dict': state, 'training_contract': contract, 'step': step,
               'implementation_sha256': implementation, 'state_sha256': state_digest(state),
               'backbone_sha256': state_digest(model.backbone.state_dict()),
               'fidelity_state_sha256': state_digest({k: v for k, v in state.items() if not k.startswith('confidence_head.')}),
               'optimizer': optimizer.state_dict(),
               'confidence_optimizer': None if confidence_optimizer is None else confidence_optimizer.state_dict()}
    temporary = path.with_suffix('.partial')
    torch.save(payload, temporary)
    temporary.replace(path)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-manifest', type=Path, required=True)
    p.add_argument('--dev-manifest', type=Path, required=True)
    p.add_argument('--root-override', type=Path)
    p.add_argument('--variant', choices=['known', 'blind', 'blind_confidence'], required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--crop', type=int, default=96)
    p.add_argument('--k', type=int, default=3)
    p.add_argument('--ch', type=int, default=64)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--warmup', type=int, default=50)
    p.add_argument('--seed', type=int, default=20261111)
    p.add_argument('--save-every', type=int, default=250)
    p.add_argument('--val-every', type=int, default=250)
    p.add_argument('--val-batches', type=int, default=4)
    p.add_argument('--log-every', type=int, default=25)
    p.add_argument('--identity-fraction', type=float, default=.125)
    p.add_argument('--near-clean-fraction', type=float, default=.125)
    p.add_argument('--edge-weight', type=float, default=.1)
    p.add_argument('--guard-weight', type=float, default=.5)
    p.add_argument('--inspect-only', action='store_true')
    a = p.parse_args()
    if min(a.steps, a.batch, a.save_every, a.val_every, a.val_batches, a.log_every) < 1:
        p.error('budgets and intervals must be positive')
    if a.batch % 4:
        p.error('batch must be divisible by four for equal task slots')
    if a.out.exists():
        p.error('output directory exists; create a new run instead of overwriting')
    torch.set_num_threads(8)
    train = RphiDataset(a.train_manifest, root_override=a.root_override, verify_hashes=True)
    dev = RphiDataset(a.dev_manifest, root_override=a.root_override, verify_hashes=True)
    separation = assert_disjoint(train, dev)
    if set(train.tasks) != set(TASKS) or set(dev.tasks) != set(TASKS):
        raise ValueError('all four tasks must be present in both splits')
    preflight = {'train': train.preflight(), 'dev': dev.preflight(), 'separation': separation}
    contract = build_contract(a, train, dev)
    if a.inspect_only:
        print(json.dumps({'preflight': preflight, 'training_contract': contract}, indent=2))
        return
    a.out.mkdir(parents=True)
    write_json(a.out / 'preflight.json', preflight)
    write_json(a.out / 'training_contract.json', contract)
    write_json(a.out / 'arguments.json', {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()})
    device = setup_device(a.device)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if device.type == 'npu':
        torch.npu.manual_seed_all(a.seed)
    model = RphiModel(variant=a.variant, k=a.k, ch=a.ch, blocks=a.blocks)
    initial_backbone = state_digest(model.backbone.state_dict())
    contract['initial_backbone_sha256'] = initial_backbone
    write_json(a.out / 'training_contract.json', contract)
    model.to(device).train()
    confidence_params = list(model.confidence_head.parameters()) if model.confidence_head is not None else []
    confidence_ids = {id(x) for x in confidence_params}
    fidelity_params = [x for x in model.parameters() if id(x) not in confidence_ids]
    opt = torch.optim.AdamW(fidelity_params, lr=a.lr, betas=(.9, .999), weight_decay=0.)
    confidence_opt = (torch.optim.AdamW(confidence_params, lr=a.lr, weight_decay=0.)
                      if confidence_params else None)
    code = Path(__file__).resolve().parent
    implementation = {str(Path('code') / name): digest(code / name)
                      for name in ['rphi_data.py', 'rphi_model.py', 'train_rphi.py']}
    logfile = a.out / 'train_log.jsonl'
    log(logfile, event='start', model_config=model.model_config, initial_backbone_sha256=initial_backbone,
        parameters=sum(x.numel() for x in model.parameters()),
        confidence_parameters=sum(x.numel() for x in confidence_params),
        torch_version=torch.__version__, device=str(device), implementation_sha256=implementation)
    log(logfile, event='validation_crop_proxy', step=0, metrics=validate(model, dev, a, device))
    start = last = time.perf_counter()
    last_logged_step = 0
    first_finite_gradient = None
    for step in range(1, a.steps + 1):
        rng = np.random.default_rng(a.seed * 1000003 + step)
        batch = train.sample_batch(rng, a.batch, crop=a.crop, k=a.k,
                                   identity_fraction=a.identity_fraction,
                                   near_clean_fraction=a.near_clean_fraction)
        x = move(batch, device)
        warm = min(1., step / max(1, a.warmup))
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * max(0, step - a.warmup) / max(1, a.steps - a.warmup)))
        lr = a.lr * warm * decay
        for optimizer in [opt] + ([confidence_opt] if confidence_opt is not None else []):
            for group in optimizer.param_groups:
                group['lr'] = lr
            optimizer.zero_grad(set_to_none=True)
        outputs = model(x['observed'], x['condition'])
        loss, components = fidelity_losses(outputs['rgb'], x['target'], x['observed'][:, -1], a.edge_weight, a.guard_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite fidelity loss')
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(fidelity_params, 1., error_if_nonfinite=True)
        if first_finite_gradient is None and float(gn) > 0:
            first_finite_gradient = step
        calibration = None
        if confidence_opt is not None:
            calibration = uncertainty_nll(outputs, x['target'])
            if not torch.isfinite(calibration):
                raise FloatingPointError('nonfinite calibration loss')
            calibration.backward()
            torch.nn.utils.clip_grad_norm_(confidence_params, 1., error_if_nonfinite=True)
            confidence_opt.step()
        opt.step()
        if step == 1 or step % a.log_every == 0 or step == a.steps:
            sync(device)
            now = time.perf_counter()
            elapsed_steps = step - last_logged_step
            log(logfile, event='train', step=step, loss=float(loss),
                components={k: float(v) for k, v in components.items()},
                calibration_nll=None if calibration is None else float(calibration), grad_norm=float(gn), lr=lr,
                seconds_since_last_log=now - last, elapsed_steps=elapsed_steps,
                peak_gb=torch.npu.max_memory_allocated(device) / 1e9 if device.type == 'npu' else None,
                control_counts=torch.bincount(batch['control'], minlength=3).tolist())
            last = now
            last_logged_step = step
        if step % a.save_every == 0 or step == a.steps:
            checkpoint = a.out / f'checkpoint_step{step:06d}.pt'
            saved = save_checkpoint(checkpoint, model, opt, confidence_opt, contract, step, implementation)
            log(logfile, event='checkpoint', step=step, checkpoint=str(checkpoint),
                checkpoint_sha256=digest(checkpoint), backbone_sha256=saved['backbone_sha256'])
        if step % a.val_every == 0 or step == a.steps:
            log(logfile, event='validation_crop_proxy', step=step, metrics=validate(model, dev, a, device))
            sync(device)
            last = time.perf_counter()
    if first_finite_gradient is None or state_digest(model.backbone.state_dict()) == initial_backbone:
        raise AssertionError('training did not update the fidelity backbone')
    # Strict reconstruction uses actual checkpoint bytes, including the final
    # confidence head. Verify all output fields on the same device and batch.
    disk = torch.load(checkpoint, map_location='cpu', weights_only=False)
    reloaded = RphiModel(**disk['model_config']).to(device).eval()
    reloaded.load_state_dict(disk['state_dict'], strict=True)
    model.eval()
    with torch.no_grad():
        expected = model(x['observed'], x['condition'])
        actual = reloaded(x['observed'], x['condition'])
        differences = {}
        for key in ('rgb', 'anchor', 'log_variance', 'condition_code'):
            if expected[key] is None:
                if actual[key] is not None:
                    raise AssertionError('reload changed optional output')
                continue
            torch.testing.assert_close(expected[key], actual[key], rtol=1e-6, atol=1e-6)
            differences[key] = float((expected[key] - actual[key]).abs().max())
    log(logfile, event='done', step=a.steps, first_finite_gradient_step=first_finite_gradient,
        checkpoint_reload_max_abs=differences, checkpoint=str(checkpoint),
        seconds_training_validation_and_save=time.perf_counter() - start)


if __name__ == '__main__':
    main()
