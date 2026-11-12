"""Differentiable zero-churn Euler math for an already noise-scaled joint state.

This is NOT a Comfy sampler, noise initializer, H3 wrapper, or trainer. The caller
supplies the complete packed video/audio tensor and a denoised-x0 callback with
all guide/payload/audio transformations already defined outside this module.
The input is not re-noised, rescaled, split, clipped, moved, or detached.

The recurrence matches the pinned official sample_euler with s_churn=0. It keeps
the graph across every step; it does not override an enclosing disabled gradient
context. Only CPU arithmetic/reference parity is tested here. Actual H3/VAE
backward, NPU kernels, memory, and deployed noise-scaling parity remain untested.
"""
from collections.abc import Callable, Mapping
from typing import Any

import torch


OFFICIAL_REFERENCE = {
    "host_id": "huawei-modelarts-30794",
    "checked_at_utc": "2026-11-11",
    "scope": "CPU math reference; no complete H3/VAE training acceptance",
    "files": {
        "comfy/k_diffusion/sampling.py": {
            "path": "/cache/zihaohe/omni_openh3_20261111/code/ComfyUI/comfy/k_diffusion/sampling.py",
            "sha256": "cfc4221ed0bc1f84a6f5689d3ec440cad3008c2541ab8f5e2599d1a73fac74ce",
            "functions": {
                "to_d": {
                    "sha256": "79b063e3cb9561d840b552ddd71406e9204036cede1eafd025ea0d7b62426a01",
                    "first_line": 63,
                    "def_line": 63,
                    "last_line": 65
                },
                "sample_euler": {
                    "sha256": "d29fcb080c000558a9d94bc545ed64cb71a466ae451d8a1ad83d743e9c5cdff9",
                    "first_line": 189,
                    "def_line": 190,
                    "last_line": 212
                }
            }
        },
        "comfy/k_diffusion/utils.py": {
            "path": "/cache/zihaohe/omni_openh3_20261111/code/ComfyUI/comfy/k_diffusion/utils.py",
            "sha256": "fc30630d8a0e4520b2e864c8f84deea3231c9b04d3f1404fec93fc425f629272",
            "functions": {
                "append_dims": {
                    "sha256": "816349d93f63ced7a88c4fc3967a6c1ba77bffca03536dea4b4e12604b98f5b5",
                    "first_line": 21,
                    "def_line": 21,
                    "last_line": 29
                }
            }
        }
    }
}


def to_d(state: torch.Tensor, sigma: torch.Tensor,
         denoised_x0: torch.Tensor) -> torch.Tensor:
    """Pinned Karras derivative; append singleton axes without a device copy."""
    if sigma.ndim > state.ndim:
        raise ValueError("sigma has more dimensions than the state")
    expanded_sigma = sigma[(...,) + (None,) * (state.ndim - sigma.ndim)]
    return (state - denoised_x0) / expanded_sigma


def differentiable_euler(
    denoised_x0: Callable[..., torch.Tensor],
    packed_state: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    extra_args: Mapping[str, Any] | None = None,
    callback: Callable[[dict[str, Any]], None] | None = None,
    s_churn: float = 0.0,
) -> torch.Tensor:
    """Return the final joint state after exactly len(sigmas)-1 model calls.

    denoised_x0(state, batch_sigma, **extra_args) must return x0, not velocity
    or epsilon, with the complete input shape/device. The state is conventionally
    B x 1 x L from Comfy pack_latents; no modality layout is interpreted here.
    Callbacks see the same pre-update x/i/sigma/sigma_hat/denoised fields as the
    official function and must not mutate them. References retain autograd.

    Only zero-churn, finite nonincreasing schedules are admitted; all evaluated
    sigmas must be positive and the final sigma may be zero. No MPS parity is
    claimed (the official append_dims contains a device-specific detach there).
    """
    if s_churn != 0:
        raise ValueError("Only the pinned zero-churn Euler contract is supported")
    if not isinstance(packed_state, torch.Tensor) or not packed_state.is_floating_point():
        raise TypeError("packed_state must be a floating torch tensor")
    if packed_state.ndim < 2 or packed_state.shape[0] < 1 or packed_state.numel() < 1:
        raise ValueError("packed_state must have a nonempty batch and packed content")
    if not isinstance(sigmas, torch.Tensor) or not sigmas.is_floating_point():
        raise TypeError("sigmas must be a floating torch tensor")
    if sigmas.ndim != 1 or len(sigmas) < 2:
        raise ValueError("sigmas must be one-dimensional with at least two entries")
    if sigmas.device != packed_state.device:
        raise ValueError("sigmas and packed_state must already be on the same device")
    if packed_state.device.type == "mps":
        raise ValueError("MPS is outside the pinned differentiable Euler contract")
    if not bool(torch.isfinite(sigmas).all()):
        raise ValueError("sigmas must be finite")
    if not bool((sigmas[:-1] > 0).all()) or not bool(sigmas[-1] >= 0):
        raise ValueError("evaluated sigmas must be positive; the final sigma may be zero")
    if not bool((sigmas[1:] <= sigmas[:-1]).all()):
        raise ValueError("sigmas must be nonincreasing")
    args = {} if extra_args is None else extra_args
    state = packed_state
    s_in = state.new_ones([state.shape[0]])
    for index in range(len(sigmas) - 1):
        sigma_hat = sigmas[index]
        prediction = denoised_x0(state, sigma_hat * s_in, **args)
        if not isinstance(prediction, torch.Tensor):
            raise TypeError("denoised_x0 must return a torch tensor")
        if prediction.shape != state.shape or prediction.device != state.device:
            raise ValueError("denoised_x0 must preserve the complete joint shape/device")
        derivative = to_d(state, sigma_hat, prediction)
        if callback is not None:
            callback({"x": state, "i": index, "sigma": sigmas[index],
                      "sigma_hat": sigma_hat, "denoised": prediction})
        dt = sigmas[index + 1] - sigma_hat
        state = state + derivative * dt
    return state
