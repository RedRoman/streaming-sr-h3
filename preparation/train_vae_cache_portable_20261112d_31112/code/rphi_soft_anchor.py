"""CPU-only post-decode R_phi anchors; no H3, VAE, or confidence integration.

``apply_soft_anchor`` returns ``(rgb, report)``. Native tasks blend full RGB
toward the supplied R_phi prediction. SR corrects only the canonical Q4 anchor
using a numerically checked right inverse. Q4 is imported from rphi_model;
it is NOT the physical Pillow observation operator or a claim of measurement
consistency. Inputs/anchors are never silently pre-clamped. The optional final
RGB clamp and output-dtype rounding can break an attained canonical constraint;
their residuals are reported separately.

alpha=0 returns the original tensor object, without interpolation, casting,
clamping, or any VAE round trip. Other strengths default to final [0,1] clipping.
The scalar strength is explicit; this module does not consume confidence maps.
"""
from functools import lru_cache
import math
import numbers

import torch

from pixel_consistency import BicubicConsistency
from rphi_model import KNOWN_TASKS, canonical_q4


SCHEMA = "rphi.post_decode_soft_anchor.v1"


def _validate(h3_rgb, rphi_rgb, task, alpha, clip):
    if task not in KNOWN_TASKS:
        raise ValueError(f"unsupported task: {task}")
    if isinstance(alpha, bool) or not isinstance(alpha, numbers.Real):
        raise TypeError("alpha must be a real scalar")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be finite and in [0,1]")
    if not isinstance(clip, bool):
        raise TypeError("clip must be boolean")
    for name, value in (("h3_rgb", h3_rgb), ("rphi_rgb", rphi_rgb)):
        if not isinstance(value, torch.Tensor) or not value.is_floating_point():
            raise TypeError(f"{name} must be a floating-point torch tensor")
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU")
        if value.ndim != 4 or value.shape[1] != 3 or min(value.shape) < 1:
            raise ValueError(f"{name} must be nonempty NCHW RGB")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains nonfinite values")
    if h3_rgb.shape != rphi_rgb.shape or h3_rgb.dtype != rphi_rgb.dtype:
        raise ValueError("h3_rgb/rphi_rgb shape and dtype must match")
    if task == "sr" and any(v < 4 or v % 4 for v in h3_rgb.shape[-2:]):
        raise ValueError("SR canonical Q4 requires HR dimensions divisible by four")
    return alpha


@lru_cache(maxsize=4)
def _q4_operator(height, width):
    """Cache CPU float64 factors, never images or predictions."""
    op = BicubicConsistency((height, width), (height // 4, width // 4),
                            dtype=torch.float64, device="cpu")
    generator = torch.Generator(device="cpu").manual_seed(61011)
    probe = torch.rand((1, 3, height, width), generator=generator, dtype=torch.float64)
    probe_error = float((op.down(probe) - canonical_q4(probe)).abs().max())
    if not math.isfinite(probe_error) or probe_error > 1e-12:
        raise ValueError("pixel_consistency.down is not numerically equivalent to canonical Q4; refusing a different operator")
    return op, probe_error


def _residual(value, anchor, task):
    difference = (canonical_q4(value) if task == "sr" else value) - anchor
    return {"rmse": float(difference.square().mean().sqrt()),
            "max_abs": float(difference.abs().max()),
            "per_image_rmse": difference.square().mean((1, 2, 3)).sqrt().tolist()}


@torch.no_grad()
def apply_soft_anchor(h3_rgb, rphi_rgb, task, alpha, *, clip=True):
    """Return CPU RGB in the input dtype and a JSON-serializable audit report.

    Native: x + alpha * (r - x).
    SR:     x + alpha * A_right_inverse(Q4(r) - Q4(x)).

    Both SR residuals use the exact imported canonical_q4 implementation. The
    matrix factors are accepted only after numerical equivalence checks. If
    those checks fail, the call fails rather than substituting an operator.
    Clipping is applied after the update, unless alpha is zero or clip=False.
    """
    alpha = _validate(h3_rgb, rphi_rgb, task, alpha, clip)
    report = {
        "schema": SCHEMA, "task": task, "alpha": alpha,
        "stage": "post_decode_rgb", "physical_observation_consistency": False,
        "constraint_definition": "canonical Q4 of supplied R_phi RGB" if task == "sr" else "full-resolution R_phi RGB fidelity anchor",
        "anchor_input": "supplied R_phi RGB without implicit pre-clamping",
        "confidence_weighting": False, "vae_roundtrip": False,
        "clip_requested": clip, "clip_applied": False,
        "input_dtype": str(h3_rgb.dtype), "output_dtype": str(h3_rgb.dtype),
        "shape": list(h3_rgb.shape), "torch_version": torch.__version__,
        "strict_constraint_claimed": False,
    }
    if alpha == 0:
        report.update(identity_passthrough=True, operator_check=None, residuals=None,
                      clipped_component_fraction=0.0, clipping_increased_residual=False,
                      clipping_broke_attained_anchor=False,
                      note="Original tensor returned unchanged; no constraint/clipping computation performed.")
        return h3_rgb, report

    x, restored = h3_rgb.double(), rphi_rgb.double()
    if task == "sr":
        op, probe_error = _q4_operator(*x.shape[-2:])
        # Compute the correction residual using canonical_q4 itself. The cached
        # separable matrix is only the right-inverse implementation.
        anchor, current = canonical_q4(restored), canonical_q4(x)
        input_error = max(float((op.down(x) - current).abs().max()),
                          float((op.down(restored) - anchor).abs().max()))
        input_tolerance = 1e-12 * max(1.0, float(x.abs().max()), float(restored.abs().max()))
        if not math.isfinite(input_error) or input_error > input_tolerance:
            raise ValueError("pixel_consistency.down differs from canonical Q4 on these inputs; refusing a different operator")
        correction = op.ry @ (anchor - current) @ op.rx.T
        raw = x + alpha * correction
        report["operator_check"] = {
            "definition": "rphi_model.canonical_q4: bicubic, align_corners=False, antialias=True, scale=4",
            "arithmetic": "CPU float64", "comparison": "numerical equivalence, not bitwise equality",
            "probe_max_abs_error": probe_error, "probe_tolerance": 1e-12,
            "input_max_abs_error": input_error, "input_tolerance": input_tolerance,
            "correction_residual_operator": "rphi_model.canonical_q4",
        }
    else:
        anchor = restored
        raw = restored if alpha == 1 else x + alpha * (restored - x)
        report["operator_check"] = None  # No Q4 or spatial projection for native tasks.
    if not torch.isfinite(raw).all():
        raise ValueError("soft anchor produced nonfinite RGB")

    clipped_fraction = float(((raw < 0) | (raw > 1)).double().mean()) if clip else 0.0
    bounded = raw.clamp(0, 1) if clip else raw
    output = bounded.to(h3_rgb.dtype)
    residuals = {"before": _residual(x, anchor, task),
                 "after_soft_update_preclip": _residual(raw, anchor, task),
                 "after_clip_before_dtype_cast": _residual(bounded, anchor, task),
                 "output": _residual(output.double(), anchor, task)}
    preclip = residuals["after_soft_update_preclip"]
    postclip = residuals["after_clip_before_dtype_cast"]
    tolerance = 1e-10 * max(1.0, float(anchor.abs().max()))
    report.update(
        identity_passthrough=False, clip_applied=clip, residuals=residuals,
        clipped_component_fraction=clipped_fraction,
        clipping_increased_residual=postclip["rmse"] > preclip["rmse"] + tolerance,
        clipping_broke_attained_anchor=(preclip["max_abs"] <= tolerance and postclip["max_abs"] > tolerance),
        attained_anchor_tolerance=tolerance,
        output_within_anchor_tolerance=residuals["output"]["max_abs"] <= tolerance,
        output_range=[float(output.min()), float(output.max())],
        note="Soft canonical/fidelity anchoring only. Final clipping and dtype rounding are measured; no exact physical consistency claim.",
    )
    return output, report
