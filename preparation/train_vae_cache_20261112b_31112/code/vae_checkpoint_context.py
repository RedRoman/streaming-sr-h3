"""Scoped activation checkpointing for the 36 frozen video-VAE decoder blocks.

The context MUST enclose both forward and backward and remain inside the same
decoder-math/RoPE context for both. It changes only per-instance ``forward``
attributes, never parameter values, requires_grad, training mode, or buffers.
Disabled mode does not inspect the stage or import torch. No-grad calls bypass
checkpointing. Output gradient guards deliberately survive scope exit so even a
backward needing no recomputation cannot silently run after the scope closes.
"""
from contextlib import contextmanager
from functools import wraps
import threading


_ACTIVE_BLOCK_IDS = set()
_REGISTRY_LOCK = threading.RLock()
_MISSING = object()
SCHEMA = "openh3.vae_decoder_checkpoint_context.v1"


def _require_active(state):
    if not state["active"]:
        raise RuntimeError("VAE activation-checkpoint context exited before backward/recomputation; keep decoder math/RoPE and checkpoint contexts active through backward")


def _guard_outputs(value, state, torch):
    """Attach no-op gradient guards without rebuilding or changing the result."""
    seen = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            if item.requires_grad and id(item) not in seen:
                seen.add(id(item))

                def guard(gradient):
                    _require_active(state)
                    state["backward_guard_calls"] += 1
                    return gradient

                item.register_hook(guard)
                state["output_gradient_guards"] += 1
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)

    visit(value)
    return value


def _wrapper(original, state, block_index, torch, checkpoint):
    # Every invocation receives its own original bound forward and recompute
    # counter. Never capture a mutable loop variable or re-fetch block.forward.
    @wraps(original)
    def forward(*args, **kwargs):
        _require_active(state)
        if not torch.is_grad_enabled():
            state["no_grad_forward_calls"] += 1
            return original(*args, **kwargs)
        state["checkpoint_forward_calls"] += 1
        invocations = 0

        def run(*inner_args, **inner_kwargs):
            nonlocal invocations
            _require_active(state)
            if invocations:
                state["recompute_calls"] += 1
                if block_index not in state["recomputed_block_indices"]:
                    state["recomputed_block_indices"].append(block_index)
            invocations += 1
            return original(*inner_args, **inner_kwargs)

        result = checkpoint(run, *args, use_reentrant=False, preserve_rng_state=True, **kwargs)
        return _guard_outputs(result, state, torch)

    return forward


@contextmanager
def checkpoint_decoder_blocks(stage, enabled=False):
    """Yield a mutable JSON-ready-on-exit audit dictionary; reject nesting.

    Example::

        with decoder_math(...), checkpoint_decoder_blocks(stage, True) as audit:
            prediction = stage.decode(latent)
            loss(prediction).backward()

    A no-grad inference call may leave the context normally. Differentiable
    outputs may not be backpropagated outside it. Reentry after an exited context
    is allowed, but old outputs still retain their old inactive-scope guard.
    """
    state = {"schema": SCHEMA, "enabled": bool(enabled), "active": False,
             "block_count": 0, "expected_block_count": 36, "wrapped": 0,
             "restored": False, "use_reentrant": False, "preserve_rng_state": True,
             "checkpoint_forward_calls": 0, "no_grad_forward_calls": 0,
             "recompute_calls": 0, "recomputed_block_indices": [],
             "output_gradient_guards": 0, "backward_guard_calls": 0,
             "parameter_or_buffer_mutation_performed": False,
             "scope_contract": "Both forward and backward inside unchanged decoder math/RoPE and this context; nested/shared-block scopes rejected"}
    if not enabled:
        state["recomputed_block_indices"] = []
        yield state
        return

    import torch
    from torch.utils.checkpoint import checkpoint

    blocks = tuple(stage.decoder.transformer_blocks)
    if len(blocks) != 36 or len({id(block) for block in blocks}) != 36:
        raise ValueError("VAE checkpoint scope requires exactly 36 distinct decoder transformer_blocks")
    if not all(isinstance(block, torch.nn.Module) and callable(block.forward) for block in blocks):
        raise TypeError("decoder transformer_blocks must be torch Modules with callable forward")
    block_ids = {id(block) for block in blocks}
    with _REGISTRY_LOCK:
        if block_ids & _ACTIVE_BLOCK_IDS:
            raise RuntimeError("nested or overlapping VAE decoder checkpoint scope is not supported")
        _ACTIVE_BLOCK_IDS.update(block_ids)
    originals = []
    state.update(active=True, block_count=36)
    try:
        for index, block in enumerate(blocks):
            prior_attribute = block.__dict__.get("forward", _MISSING)
            original = block.forward
            originals.append((block, prior_attribute))
            block.forward = _wrapper(original, state, index, torch, checkpoint)
            state["wrapped"] += 1
        yield state
    finally:
        # Invalidate first so partially restored models cannot accept a stale
        # backward. Restore original instance attribute presence as well as value.
        state["active"] = False
        errors = []
        for block, prior_attribute in reversed(originals):
            try:
                if prior_attribute is _MISSING:
                    if "forward" in block.__dict__:
                        delattr(block, "forward")
                else:
                    block.forward = prior_attribute
            except BaseException as error:
                errors.append(error)
        with _REGISTRY_LOCK:
            _ACTIVE_BLOCK_IDS.difference_update(block_ids)
        state["restored"] = not errors and all(
            ("forward" not in block.__dict__) if prior is _MISSING else block.__dict__.get("forward") is prior
            for block, prior in originals)
        state["recomputed_block_indices"] = sorted(state["recomputed_block_indices"])
        if errors:
            raise RuntimeError("failed to restore VAE decoder forward attributes") from errors[0]
