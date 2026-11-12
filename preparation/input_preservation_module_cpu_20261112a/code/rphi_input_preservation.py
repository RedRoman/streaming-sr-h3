"""Isolated raw-only input-preservation head; no restorer or training loop.

The frozen K1 prediction is an output candidate, never a head input. Labels are
detached NumPy calculations for a future authenticated train-only cache. Hard
tile selection can create seams; it is deliberately not feathered here.
"""
import numpy as np
import torch
from torch import nn

from rphi_history_utility import HistoryUtilityGate, GATE_PARAMETERS, TILE, UTILITY_SCALE

SCHEMA = 'rphi.input_preservation.v1'


def _tensor(value, name, shape=None, unit=False):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(name + ' must be a float32 tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(name + ' shape differs')
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite')
    if unit and not bool(((value >= 0) & (value <= 1)).all()):
        raise ValueError(name + ' must be unit-range RGB')
    return value


def _rgb(value, name, unit=False):
    _tensor(value, name, unit=unit)
    if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != 3:
        raise ValueError(name + ' must be nonempty B3HW')
    h, w = value.shape[-2:]
    if min(h, w) < TILE or h % TILE or w % TILE:
        raise ValueError(name + ' geometry must be divisible by16; no padding/resize')
    return value.shape[0], h, w


class InputPreservationHead(nn.Module):
    """One signed map from the two shared raw-observation branches.

    Missing history does not disable input-preservation decisions: the feature
    extractor fills it from current RGB and exposes its zero availability.
    All supplied pixels must be finite, including unused historical storage.
    """
    def __init__(self, seed=20261111):
        super().__init__()
        self.gate = HistoryUtilityGate(seed)
        self.model_config = {'schema': SCHEMA, 'seed': seed, 'parameters': GATE_PARAMETERS,
                             'slot_reduction': 'arithmetic_mean', 'tile': TILE,
                             'threshold': 0., 'tie': 'preserve_input'}
        if sum(p.numel() for p in self.parameters()) != 1745:
            raise RuntimeError('input-preservation parameter count changed')

    def features(self, observed, availability):
        """Raw-only cache interface: B,2,11,H/16,W/16; no learned features."""
        _tensor(observed, 'observed', unit=True)
        if observed.ndim != 5 or observed.shape[0] < 1 or tuple(observed.shape[1:3]) != (3, 3):
            raise ValueError('observed must be nonempty B,3,3,H,W, oldest first')
        b, _, _, h, w = observed.shape
        _tensor(availability, 'availability', (b, 2))
        if availability.device != observed.device or not bool(((availability == 0) | (availability == 1)).all()):
            raise ValueError('availability must be binary on the observation device')
        # Original feature implementation enforces exact tile geometry.
        result = self.gate.features(observed, availability)
        return result.reshape(b, 2, 11, h // TILE, w // TILE).contiguous()

    def forward_features(self, features):
        """Use only an authenticated raw-feature cache, never target features.

        File/row/source provenance belongs to the cache reader, not this numeric
        model API. This entry point performs the identical shared-net arithmetic
        used by forward() and permits later head-only cached training.
        """
        _tensor(features, 'raw features')
        if features.ndim != 5 or features.shape[0] < 1 or tuple(features.shape[1:3]) != (2, 11) or min(features.shape[-2:]) < 1:
            raise ValueError('raw features must be nonempty B,2,11,h,w')
        if any(p.dtype != torch.float32 or p.device != features.device for p in self.parameters()):
            raise ValueError('head parameters and raw features must be colocated float32')
        b, _, _, h, w = features.shape
        logits = self.gate.net(features.reshape(b * 2, 11, h, w)).reshape(b, 2, h, w)
        score = logits.mean(dim=1, keepdim=True)
        _tensor(score, 'benefit score', (b, 1, h, w))
        return {'benefit_score': score, 'correction_mask': score > 0}

    def forward(self, observed, availability):
        return self.forward_features(self.features(observed, availability))


def compose_input_preservation(current_rgb, k1_rgb, benefit_score):
    """Select whole16x16 tiles; zero/signed-zero ties preserve raw bytes.

    No RGB loss or threshold gradient is defined here. ``rgb`` stays unclamped;
    ``rgb_clipped`` is a distinct explicit deployment/metric representation.
    """
    b, h, w = _rgb(current_rgb, 'current RGB', unit=True)
    _tensor(k1_rgb, 'K1 RGB', current_rgb.shape)
    _tensor(benefit_score, 'benefit score', (b, 1, h // TILE, w // TILE))
    if k1_rgb.device != current_rgb.device or benefit_score.device != current_rgb.device:
        raise ValueError('composition tensors must share a device')
    mask = benefit_score > 0
    pixels = mask.repeat_interleave(TILE, -2).repeat_interleave(TILE, -1)
    rgb = torch.where(pixels, k1_rgb, current_rgb)
    return {'rgb': rgb, 'rgb_clipped': rgb.clamp(0., 1.),
            'correction_mask': mask, 'pixel_mask': pixels}


def normalize_input_benefit(raw_benefit):
    """Original fixed utility scale/clamp; not a calibrated error estimate."""
    if not isinstance(raw_benefit, np.ndarray) or raw_benefit.dtype != np.float64:
        raise ValueError('raw benefit must be a float64 NumPy array')
    if raw_benefit.ndim != 4 or raw_benefit.shape[0] < 1 or raw_benefit.shape[1] != 1 or min(raw_benefit.shape[-2:]) < 1:
        raise ValueError('raw benefit must be nonempty B1hw')
    if not np.isfinite(raw_benefit).all():
        raise ValueError('raw benefit must be finite')
    return np.ascontiguousarray(np.clip(raw_benefit / UTILITY_SCALE, -4., 4.), dtype=np.float32)


def input_benefit_labels(current_rgb, k1_rgb, target_rgb):
    """Pure CPU FP64 labels from actual FP32 arrays; GT is supervision only.

    Positive benefit means the fixed K1 output has lower unclamped tile MAE.
    Both-absent-history rows remain valid: this is one restoration-benefit map,
    not a per-history utility target. No framework graph or model is invoked.
    """
    values = (current_rgb, k1_rgb, target_rgb)
    for name, value in zip(('current RGB', 'K1 RGB', 'target RGB'), values):
        if not isinstance(value, np.ndarray) or value.dtype != np.float32:
            raise ValueError(name + ' must be a float32 NumPy array')
        if value.ndim != 4 or value.shape[0] < 1 or value.shape[1] != 3 or min(value.shape[-2:]) < TILE or any(s % TILE for s in value.shape[-2:]):
            raise ValueError(name + ' must be nonempty B3HW, divisible by16')
        if value.shape != current_rgb.shape or not np.isfinite(value).all():
            raise ValueError(name + ' geometry/finiteness differs')
    if any(not np.logical_and(v >= 0, v <= 1).all() for v in (current_rgb, target_rgb)):
        raise ValueError('current and target RGB must be unit-range; K1 remains unclamped')
    b, _, h, w = current_rgb.shape
    y = target_rgb.astype(np.float64)

    def error(value):
        pixels = np.abs(value.astype(np.float64) - y).mean(axis=1, keepdims=True)
        return np.ascontiguousarray(pixels.reshape(b, 1, h // TILE, TILE, w // TILE, TILE).mean(axis=(3, 5)))

    input_mae, k1_mae = error(current_rgb), error(k1_rgb)
    raw = input_mae - k1_mae
    return {'input_mae': input_mae, 'k1_mae': k1_mae, 'raw_benefit': raw,
            'normalized_benefit': normalize_input_benefit(raw)}
