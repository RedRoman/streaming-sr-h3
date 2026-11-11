"""Train only an uncertainty head on an existing, strictly frozen blind R_phi.

Only the original TRAIN manifest is read. This is training-set uncertainty fitting,
not independent held-out calibration. The original fidelity training_contract and
step remain unchanged for eval_rphi compatibility; calibration_contract and
calibration_step separately record the added head training and its full budget.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch

from rphi_data import RphiDataset, TASKS
from rphi_model import RphiModel, uncertainty_nll
from train_rphi import digest, log, setup_device, state_digest, sync, write_json


HEAD_PREFIX = "confidence_head."
FIDELITY_OUTPUTS = ("rgb", "anchor", "condition_code")


def fidelity_state(model):
    return {name: value for name, value in model.state_dict().items()
            if not name.startswith(HEAD_PREFIX)}


def build_frozen_models(checkpoint, model_factory=RphiModel):
    """Strictly reconstruct the source and copy every non-head tensor by name.

    Preserve all model_config fields, including future condition-encoder options.
    Construction order/seeds cannot affect copied fidelity parameters or buffers.
    """
    if checkpoint.get("schema") != "rphi.checkpoint.v1":
        raise ValueError("Expected rphi.checkpoint.v1")
    source_config = copy.deepcopy(checkpoint["model_config"])
    if source_config.get("variant") != "blind":
        raise ValueError("Calibration source must be an existing blind checkpoint")
    source = model_factory(**source_config).eval()
    source.load_state_dict(checkpoint["state_dict"], strict=True)
    expected = state_digest(fidelity_state(source))
    if checkpoint.get("fidelity_state_sha256") != expected:
        raise ValueError("Source fidelity state SHA256 is missing or does not match actual tensors")
    config = copy.deepcopy(source_config)
    config["variant"] = "blind_confidence"
    model = model_factory(**config).eval()
    if model.confidence_head is None:
        raise ValueError("blind_confidence must expose a confidence_head")
    source_state, destination = source.state_dict(), model.state_dict()
    fidelity_names = {name for name in destination if not name.startswith(HEAD_PREFIX)}
    if set(source_state) != fidelity_names:
        raise ValueError("Source and calibrated model fidelity state keys differ")
    for name in fidelity_names:
        old, new = source_state[name], destination[name]
        if old.shape != new.shape or old.dtype != new.dtype:
            raise ValueError("Fidelity tensor shape/dtype mismatch: " + name)
        destination[name] = old.detach().clone()
    model.load_state_dict(destination, strict=True)
    source.requires_grad_(False)
    model.requires_grad_(False)
    model.confidence_head.requires_grad_(True)
    # eval() also freezes stateful fidelity buffers if a future model adds them.
    model.eval()
    model.confidence_head.train()
    if state_digest(fidelity_state(model)) != expected:
        raise AssertionError("Head construction changed the source fidelity state")
    return source, model, expected


def validate_train_contract(checkpoint, dataset):
    contract = checkpoint.get("training_contract", {})
    if dataset.manifest_sha256 != contract.get("train_manifest_sha256"):
        raise ValueError("Calibration must use the source checkpoint's exact train manifest")
    if contract.get("target") != "clean_original_recording" or tuple(contract.get("task_order", [])) != TASKS:
        raise ValueError("Source target/task contract is incompatible")
    if not contract.get("dev_manifest_sha256"):
        raise ValueError("Source checkpoint must retain its evaluator dev-manifest binding")
    if checkpoint.get("step") != contract.get("steps"):
        raise ValueError("Use a final source checkpoint at its declared fidelity budget")
    if not dataset.records or any(row.get("split") != "train" for row in dataset.records):
        raise ValueError("Calibration optimization accepts only train records")
    ids = {row["source_id"] for row in dataset.records}
    hashes = {row["source_sha256"] for row in dataset.records}
    expected_hashes = contract.get("train_source_sha256", [])
    expected_hashes = set(expected_hashes.values()) if isinstance(expected_hashes, dict) else set(expected_hashes)
    if ids != set(contract.get("train_source_ids", [])) or hashes != expected_hashes:
        raise ValueError("Calibration train source IDs/hashes differ from the source checkpoint")
    if ids & set(contract.get("dev_source_ids", [])):
        raise ValueError("Calibration train/dev source ID overlap")
    if set(dataset.tasks) != set(TASKS):
        raise ValueError("All four training tasks are required")
    return copy.deepcopy(contract)


@torch.no_grad()
def output_snapshot(model, observed, condition=None, include_confidence=False):
    outputs = model(observed.detach(), None if condition is None else condition.detach())
    keys = (*FIDELITY_OUTPUTS, "log_variance") if include_confidence else FIDELITY_OUTPUTS
    result = {}
    for name in keys:
        value = outputs[name]
        if value is not None:
            if not torch.isfinite(value).all():
                raise FloatingPointError("Nonfinite invariant output: " + name)
            result[name] = value.detach().cpu().clone()
    return result


def assert_exact_outputs(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError("Invariant output fields differ")
    differences = {}
    for name, target in expected.items():
        value = actual[name]
        if not torch.equal(value, target):
            raise AssertionError("Frozen output changed: " + name)
        differences[name] = float((value - target).abs().max())
    return differences


def assert_fidelity_frozen(model, expected_sha256):
    for name, parameter in model.named_parameters():
        if not name.startswith(HEAD_PREFIX) and (parameter.requires_grad or parameter.grad is not None):
            raise AssertionError("Fidelity parameter is trainable or has a gradient: " + name)
    actual = state_digest(fidelity_state(model))
    if actual != expected_sha256:
        raise AssertionError("Frozen fidelity parameters or buffers changed")
    return actual


def calibration_step(model, optimizer, observed, target, condition=None):
    """No RGB training loss, no fidelity optimizer, and detached inputs/errors."""
    allowed = {id(p) for p in model.confidence_head.parameters()}
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimized != allowed:
        raise ValueError("Optimizer must contain exactly the confidence_head parameters")
    for name, parameter in model.named_parameters():
        if not name.startswith(HEAD_PREFIX) and parameter.requires_grad:
            raise AssertionError("Fidelity parameter must be frozen: " + name)
    optimizer.zero_grad(set_to_none=True)
    outputs = model(observed.detach(), None if condition is None else condition.detach())
    if outputs["rgb"].requires_grad or outputs["condition_code"].requires_grad:
        raise AssertionError("Fidelity outputs must be detached from calibration")
    loss = uncertainty_nll(outputs, target.detach())
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite calibration loss")
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.confidence_head.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(grad_norm)


def save_calibrated_checkpoint(path, model, optimizer, source, source_sha256,
                               calibration_contract, calibration_step_number, implementation, invariance):
    expected = source["fidelity_state_sha256"]
    assert_fidelity_frozen(model, expected)
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    payload = {"schema": "rphi.checkpoint.v1", "model_config": copy.deepcopy(model.model_config),
               "state_dict": state, "training_contract": copy.deepcopy(source["training_contract"]),
               "step": source["step"], "source_checkpoint_sha256": source_sha256,
               "initial_fidelity_sha256": source.get("initial_fidelity_sha256"),
               "calibration_contract": copy.deepcopy(calibration_contract),
               "calibration_step": calibration_step_number,
               "implementation_sha256": implementation, "state_sha256": state_digest(state),
               "backbone_sha256": state_digest(model.backbone.state_dict()),
               "fidelity_state_sha256": expected, "fidelity_invariance": invariance,
               "optimizer": None, "confidence_optimizer": optimizer.state_dict()}
    # A fresh output directory plus exclusive creation prevents evidence replacement.
    with Path(path).open("xb") as stream:
        torch.save(payload, stream)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--root-override", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--crop", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261112)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--identity-fraction", type=float, default=.125)
    parser.add_argument("--near-clean-fraction", type=float, default=.125)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    if min(args.steps, args.batch, args.crop, args.save_every, args.log_every) < 1 or args.warmup < 0:
        parser.error("budgets must be positive and warmup nonnegative")
    if args.batch % 4 or args.crop % 4:
        parser.error("batch and crop must be divisible by four")
    fractions = (args.identity_fraction, args.near_clean_fraction)
    if (not math.isfinite(args.lr) or args.lr <= 0
            or not all(math.isfinite(x) and x >= 0 for x in fractions) or sum(fractions) > 1):
        parser.error("invalid learning rate or control fractions")
    if args.device != "cpu" and not args.device.startswith("npu:"):
        parser.error("device must be cpu or npu:<logical-index>")
    if args.out.exists():
        raise FileExistsError(args.out)
    torch.set_num_threads(8)
    source_sha = digest(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if digest(args.checkpoint) != source_sha:
        raise ValueError("Source checkpoint changed while loading")
    dataset = RphiDataset(args.train_manifest, root_override=args.root_override, verify_hashes=True)
    original_contract = validate_train_contract(checkpoint, dataset)
    preflight = dataset.preflight()
    contract = {"schema": "rphi.frozen_confidence_calibration.v1", "source_checkpoint_sha256": source_sha,
                "source_checkpoint": str(args.checkpoint.resolve()), "source_fidelity_step": checkpoint["step"],
                "source_fidelity_state_sha256": checkpoint["fidelity_state_sha256"],
                "train_manifest_sha256": dataset.manifest_sha256,
                "train_source_ids": original_contract["train_source_ids"],
                "train_source_sha256": original_contract["train_source_sha256"],
                "steps": args.steps, "batch_size": args.batch, "crop": args.crop, "seed": args.seed,
                "identity_fraction": args.identity_fraction, "near_clean_fraction": args.near_clean_fraction,
                "optimizer": {"type": "AdamW", "lr": args.lr, "betas": [.9, .999], "weight_decay": 0.,
                              "warmup": args.warmup, "schedule": "linear warmup, cosine decay to 0.2*lr", "gradient_clip_norm": 1.},
                "trainable": "confidence_head only; all other parameters and buffers frozen",
                "loss": "Gaussian NLL without constant, raw RGB and target detached; detached backbone features",
                "data_scope": "original training sources only; no development target optimization or model selection",
                "statistical_scope": "training-set uncertainty fitting, NOT independent held-out calibration",
                "step_semantics": "checkpoint.step/training_contract describe frozen fidelity training; calibration_step describes this additional head training"}
    if args.inspect_only:
        print(json.dumps({"preflight": preflight, "calibration_contract": contract}, indent=2))
        return
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "arguments.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    write_json(args.out / "preflight.json", preflight)
    write_json(args.out / "training_contract.json", original_contract)
    write_json(args.out / "calibration_contract.json", contract)
    device = setup_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "npu":
        torch.npu.manual_seed_all(args.seed)
    source, model, expected_sha = build_frozen_models(checkpoint)
    source.to(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.confidence_head.parameters(), lr=args.lr, betas=(.9, .999), weight_decay=0.)
    k = model.model_config["k"]
    probe_batch = dataset.sample_batch(np.random.default_rng(args.seed * 1000003), args.batch,
                                      crop=args.crop, k=k, identity_fraction=args.identity_fraction,
                                      near_clean_fraction=args.near_clean_fraction)
    observed_probe, condition_probe = probe_batch["observed"].to(device), probe_batch["condition"].to(device)
    expected_output = output_snapshot(source, observed_probe, condition_probe)
    initial_differences = assert_exact_outputs(output_snapshot(model, observed_probe, condition_probe), expected_output)
    del source
    initial_head = state_digest(model.confidence_head.state_dict())
    code = Path(__file__).resolve().parent
    implementation = {"code/" + name: digest(code / name)
                      for name in ("calibrate_rphi.py", "rphi_model.py", "rphi_condition_v2.py", "rphi_data.py", "train_rphi.py")}
    logfile = args.out / "calibration_log.jsonl"
    log(logfile, event="start", source_checkpoint_sha256=source_sha, model_config=model.model_config,
        fidelity_state_sha256=expected_sha, source_rgb_max_abs=initial_differences,
        confidence_parameters=sum(p.numel() for p in model.confidence_head.parameters()),
        optimization_sources="train only", implementation_sha256=implementation)
    start = last = time.perf_counter()
    first_gradient, last_logged = None, 0
    for step in range(1, args.steps + 1):
        batch = dataset.sample_batch(np.random.default_rng(args.seed * 1000003 + step), args.batch,
                                     crop=args.crop, k=k, identity_fraction=args.identity_fraction,
                                     near_clean_fraction=args.near_clean_fraction)
        warm = min(1., step / max(1, args.warmup))
        decay = .2 + .8 * .5 * (1 + math.cos(math.pi * max(0, step - args.warmup) / max(1, args.steps - args.warmup)))
        lr = args.lr * warm * decay
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss, grad_norm = calibration_step(model, optimizer, batch["observed"].to(device),
                                           batch["target"].to(device), batch["condition"].to(device))
        if first_gradient is None and grad_norm > 0:
            first_gradient = step
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            sync(device)
            now = time.perf_counter()
            log(logfile, event="calibration_train", step=step, gaussian_nll_without_constant=loss,
                grad_norm=grad_norm, lr=lr, elapsed_steps=step - last_logged,
                seconds_since_last_log=now - last, control_counts=torch.bincount(batch["control"], minlength=3).tolist())
            last, last_logged = now, step
        if step % args.save_every == 0 or step == args.steps:
            actual_sha = assert_fidelity_frozen(model, expected_sha)
            differences = assert_exact_outputs(output_snapshot(model, observed_probe, condition_probe), expected_output)
            invariance = {"fidelity_before_sha256": expected_sha, "fidelity_after_sha256": actual_sha,
                          "identical": True, "raw_output_max_abs_vs_source": differences,
                          "probe_source": "fixed train-only batch; no dev target read", "probe_samples": probe_batch["sample_metadata"]}
            saved_path = args.out / f"checkpoint_calibration_step{step:06d}.pt"
            save_calibrated_checkpoint(saved_path, model, optimizer, checkpoint, source_sha, contract, step, implementation, invariance)
            log(logfile, event="checkpoint", calibration_step=step, checkpoint=str(saved_path),
                checkpoint_sha256=digest(saved_path), fidelity_invariance=invariance)
    if first_gradient is None or state_digest(model.confidence_head.state_dict()) == initial_head:
        raise AssertionError("Calibration did not update the confidence head")
    disk = torch.load(saved_path, map_location="cpu", weights_only=True)
    reloaded = RphiModel(**disk["model_config"]).to(device).eval()
    reloaded.load_state_dict(disk["state_dict"], strict=True)
    if state_digest(fidelity_state(reloaded)) != expected_sha:
        raise AssertionError("Saved checkpoint changed fidelity state")
    reload_difference = assert_exact_outputs(output_snapshot(reloaded, observed_probe, condition_probe, True),
                                            output_snapshot(model, observed_probe, condition_probe, True))
    report = {"schema": "rphi.frozen_confidence_calibration.v1", "complete": True,
              "quality_verdict": "requires_review", "calibration_step": args.steps,
              "source_checkpoint_sha256": source_sha, "checkpoint": str(saved_path),
              "checkpoint_sha256": digest(saved_path), "first_finite_gradient_step": first_gradient,
              "fidelity_invariance": invariance, "checkpoint_reload_max_abs": reload_difference,
              "confidence_head_changed": True, "seconds": time.perf_counter() - start,
              "statistical_scope": contract["statistical_scope"]}
    write_json(args.out / "completion.json", report)
    log(logfile, event="done", **report)


if __name__ == "__main__":
    main()
