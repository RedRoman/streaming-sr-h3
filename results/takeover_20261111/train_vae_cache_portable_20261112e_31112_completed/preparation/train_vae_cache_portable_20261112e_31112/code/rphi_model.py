"""Matched R_phi controls for canonical RGB restoration.

The loader supplies three causal observations on the desired HR RGB grid. For
SR, resizing the observation is the loader's responsibility. ``rgb`` is an
UNCLAMPED fidelity prediction: train and calibrate against it; clamp a separate
copy only for display or metrics that explicitly require bounded RGB.

``anchor`` is Q4(rgb), a fixed antialiased PyTorch bicubic downsample. Q4 is a
canonical output convention, not an assertion about the physical degradation.
No data-consistency projection or H3 integration is implemented here.

Known conditions contain only four task indicators in KNOWN_TASKS order,
log2(spatial_scale)/3, and a known-condition mask. Neither severity nor target
statistics belong in this vector. Blind variants ignore the condition argument
entirely and infer a 16-dimensional code from the supplied observations.

The observation encoder defaults to ``legacy`` for old checkpoint compatibility.
``native`` extracts fixed-stride features before global pooling, avoiding the
legacy encoder's observation-level resize. This option affects blind variants
only; known conditioning continues to use the same public-condition MLP.
"""
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


KNOWN_TASKS = ("sr", "deblur", "denoise", "compression")
MODEL_SCHEMA = "rphi.canonical_rgb.v1"
CODE_DIM = 16
LOG_VARIANCE_MIN = -12.0
LOG_VARIANCE_MAX = 0.0


def canonical_q4(rgb):
    """Q4 on unclamped RGB; output is [B,3,H/4,W/4], on the input device.

    Pixel geometry must be divisible by four. This fixes resize coordinates
    independently of the original observation's task or physical operator.
    """
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("Q4 expects RGB [B,3,H,W]")
    h, w = rgb.shape[-2:]
    if min(h, w) < 4 or h % 4 or w % 4:
        raise ValueError("Q4 requires H and W >= 4 and divisible by four")
    return F.interpolate(rgb, size=(h // 4, w // 4), mode="bicubic",
                         align_corners=False, antialias=True)


class _FiLMBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.film = nn.Linear(CODE_DIM, 2 * ch)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, features, code):
        gain, bias = self.film(code).chunk(2, dim=1)
        hidden = F.relu(self.conv1(features))
        hidden = hidden * (1 + gain[:, :, None, None]) + bias[:, :, None, None]
        return features + self.conv2(hidden)


class _Backbone(nn.Module):
    """This entire module is constructed before any variant-specific module."""
    def __init__(self, k, ch, blocks):
        super().__init__()
        self.head = nn.Conv2d(3 * k, ch, 3, padding=1)
        self.blocks = nn.ModuleList(_FiLMBlock(ch) for _ in range(blocks))
        self.tail = nn.Conv2d(ch, 3, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, observed, code):
        b, k, c, h, w = observed.shape
        features = self.head(observed.reshape(b, k * c, h, w))
        for block in self.blocks:
            features = block(features, code)
        return observed[:, -1] + self.tail(features), features


class _ObservationEncoder(nn.Module):
    """Learn spatial/temporal evidence, without physical-parameter regression.

    Each sample is pooled separately; there is no BatchNorm, cross-batch state,
    cached future context, target input, or external metadata input. A bounded
    spatial grid keeps conditioning inexpensive at full restoration resolution.
    """
    def __init__(self, k):
        super().__init__()
        # All K RGB frames plus their K-1 adjacent temporal differences.
        channels = 3 * (2 * k - 1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, CODE_DIM), nn.Tanh())

    def forward(self, observed):
        b, k, c, h, w = observed.shape
        small = observed.reshape(b * k, c, h, w)
        if h > 64 or w > 64:
            small = F.adaptive_avg_pool2d(small, (min(64, h), min(64, w)))
        small = small.reshape(b, k, c, *small.shape[-2:])
        changes = small[:, 1:] - small[:, :-1]
        evidence = torch.cat((small.flatten(1, 2), changes.flatten(1, 2)), dim=1)
        return self.net(evidence)


class RphiModel(nn.Module):
    """Three matched controls, with a shared K3 FiLM residual backbone.

    Re-seed torch identically BEFORE constructing each variant to obtain exactly
    identical initial backbone tensors. Blind and blind_confidence also start
    with identical observation-encoder tensors. ``model_config`` can be passed
    verbatim to RphiModel for strict checkpoint reconstruction.
    """
    def __init__(self, variant="known", k=3, ch=64, blocks=8, condition_dim=6,
                 observation_encoder="legacy"):
        super().__init__()
        if variant not in ("known", "blind", "blind_confidence"):
            raise ValueError("variant must be known, blind, or blind_confidence")
        if k != 3:
            raise ValueError("this protocol supports exactly three causal frames")
        if not isinstance(ch, int) or ch < 1 or not isinstance(blocks, int) or blocks < 1:
            raise ValueError("ch and blocks must be positive integers")
        if condition_dim != 6:
            raise ValueError("known condition has exactly six public fields")
        if observation_encoder not in ("legacy", "native"):
            raise ValueError("observation_encoder must be legacy or native")
        self.variant, self.k = variant, k
        self.model_config = {"variant": variant, "k": k, "ch": ch,
                             "blocks": blocks, "condition_dim": condition_dim,
                             "observation_encoder": observation_encoder}

        # Do not move branch initialization above this: the matched experiment
        # relies on the same seed producing the same shared parameter values.
        self.backbone = _Backbone(k, ch, blocks)
        if variant == "known":
            self.condition_encoder = nn.Sequential(
                nn.Linear(condition_dim, 32), nn.SiLU(),
                nn.Linear(32, CODE_DIM), nn.Tanh())
        elif observation_encoder == "legacy":
            self.condition_encoder = _ObservationEncoder(k)
        else:
            from rphi_condition_v2 import NativeObservationEncoder
            self.condition_encoder = NativeObservationEncoder(k=k, code_dim=CODE_DIM)

        self.confidence_head = None
        if variant == "blind_confidence":
            self.confidence_head = nn.Sequential(
                nn.Conv2d(ch, max(1, ch // 2), 1), nn.SiLU(),
                nn.Conv2d(max(1, ch // 2), 1, 1))
            nn.init.zeros_(self.confidence_head[-1].weight)
            nn.init.constant_(self.confidence_head[-1].bias, -6.0)

    def forward(self, observed, condition=None):
        if observed.ndim != 5 or observed.shape[1:3] != (self.k, 3):
            raise ValueError("observed must be [B,3,3,H,W], oldest frame first")
        if observed.shape[0] < 1 or not observed.is_floating_point():
            raise ValueError("observed requires a nonempty floating-point batch")
        h, w = observed.shape[-2:]
        if min(h, w) < 4 or h % 4 or w % 4:
            raise ValueError("HR input geometry must be divisible by four")
        if self.variant == "known":
            if condition is None or condition.shape != (observed.shape[0], 6):
                raise ValueError("known variant requires condition [B,6]")
            code = self.condition_encoder(condition.to(observed))
        else:
            # Deliberately do not inspect condition, even for validation. This
            # permits shared callers while preventing metadata side channels.
            code = self.condition_encoder(observed)
        rgb, features = self.backbone(observed, code)
        log_variance = None
        if self.confidence_head is not None:
            log_variance = self.confidence_head(features.detach()).clamp(
                LOG_VARIANCE_MIN, LOG_VARIANCE_MAX)
        return {"rgb": rgb, "anchor": canonical_q4(rgb),
                "log_variance": log_variance, "condition_code": code}


def uncertainty_nll(outputs: Mapping, target):
    """Calibrate one isotropic RGB variance per pixel; train ONLY the head.

    The squared error uses the *unclamped* prediction and target, both detached.
    Returning 0.5 * (log_variance + squared_error / variance) is Gaussian NLL
    with only the constant log(2*pi)/2 omitted. It may legitimately be negative.
    This loss must never be repurposed as a learned weight on the fidelity loss.
    No confidence-guided DC behavior is implied by this calibration diagnostic.
    """
    rgb, log_variance = outputs["rgb"], outputs["log_variance"]
    if log_variance is None:
        raise ValueError("uncertainty_nll requires blind_confidence outputs")
    if target.shape != rgb.shape:
        raise ValueError("target and predicted RGB must have identical shapes")
    if log_variance.shape != (rgb.shape[0], 1, *rgb.shape[-2:]):
        raise ValueError("log_variance must be [B,1,H,W]")
    error_squared = (rgb.detach().float() - target.detach().float()).square().mean(1, keepdim=True)
    log_variance = log_variance.float()
    return 0.5 * (log_variance + error_squared * torch.exp(-log_variance)).mean()
