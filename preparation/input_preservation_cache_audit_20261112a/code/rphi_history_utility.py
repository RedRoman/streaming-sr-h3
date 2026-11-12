"""Isolated P1 mask-aware restoration and history-utility building blocks.

No checkpoint selection, teacher/student training loop, label cache or scheduler
is implemented here. The caller must bind an accepted blind checkpoint and the
source/crop/mask receipts before training. Pixels cannot establish chronology:
causal_indices supplies the boundary rule, while loaders must enforce timestamps.
Only observed RGB and explicit history availability/weights enter model forward.
GT-derived utility is supervision, never a model input or calibrated reliability.
"""
import copy
import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from rphi_model import RphiModel

SCHEMA = 'rphi.history_utility.v1'
TILE = 16
UTILITY_SCALE = 0.25 / 255
NEUTRAL_MARGIN = 0.05 / 255
GATE_PARAMETERS = 1745
ARMS = {'generic_gate': 0., 'utility_gate': .01, 'shuffled_utility_gate': .01}
_CONFIG_KEYS = {'variant', 'k', 'ch', 'blocks', 'condition_dim', 'observation_encoder'}


def _float32(value, name, shape=None):
    if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
        raise ValueError(name + ' must be a float32 tensor')
    if shape is not None and tuple(value.shape) != tuple(shape):
        raise ValueError(name + ' shape differs')
    return value


def _finite_range(value, name, lo=None, hi=None):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(name + ' must be finite')
    if lo is not None and (bool((value < lo).any()) or bool((value > hi).any())):
        raise ValueError(name + ' outside declared range')
    return value


def _observed(value):
    _float32(value, 'observed')
    if value.ndim != 5 or tuple(value.shape[1:3]) != (3, 3) or value.shape[0] < 1:
        raise ValueError('observed must be nonempty [B,3,3,H,W], oldest first')
    b, _, _, h, w = value.shape
    if min(h, w) < TILE or h % TILE or w % TILE:
        raise ValueError('history utility requires H,W divisible by16; no padding/resize')
    _finite_range(value[:, -1], 'current RGB', 0., 1.)
    return b, h, w


def _availability(value, b, device):
    _float32(value, 'availability', (b, 2))
    if value.device != device or not bool(((value == 0) | (value == 1)).all()):
        raise ValueError('availability must be binary on the observation device')
    return value


def causal_indices(frame_indices):
    """Return [B,3] indices and float32 [B,2] true predecessor availability.

    t=0 -> [0,0,0], [0,0]; t=1 -> [0,0,1], [0,1]. No future slot.
    Metadata may construct this mask, but is not forwarded into either network.
    """
    t = torch.as_tensor(frame_indices)
    if t.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError('frame indices must be integers')
    if t.ndim == 0:
        t = t[None]
    if t.ndim != 1 or t.numel() == 0 or bool((t < 0).any()):
        raise ValueError('frame indices must be a nonempty nonnegative vector')
    t = t.long()
    return torch.stack(((t - 2).clamp_min(0), (t - 1).clamp_min(0), t), 1), torch.stack((t >= 2, t >= 1), 1).float()


def apply_external_mask(true_availability, external_mask):
    """Compose loader-verified causal presence with the frozen binary mask."""
    if not isinstance(true_availability, torch.Tensor) or true_availability.ndim != 2:
        raise ValueError('true availability must be [B,2]')
    b = true_availability.shape[0]
    _availability(true_availability, b, true_availability.device)
    _availability(external_mask, b, true_availability.device)
    return true_availability * external_mask


def fill_history(observed, weights):
    """Fill absent slots from current RGB; never treat zero RGB as absence.

    weights is B,2,H,W. A zero-weight pixel's old content is not evidence; it
    is replaced before interpolation, avoiding even NaN*0 from unused inputs.
    Current RGB remains bit-identical and is the residual skip.
    """
    b, h, w = _observed(observed)
    _float32(weights, 'weights', (b, 2, h, w))
    if weights.device != observed.device:
        raise ValueError('weights and observations must share device')
    _finite_range(weights, 'weights', 0., 1.)
    current = observed[:, -1:]
    a = weights[:, :, None]
    history = torch.where(a == 0, current.expand(-1, 2, -1, -1, -1), observed[:, :2])
    _finite_range(history, 'available historical RGB', 0., 1.)
    filled = a * history + (1 - a) * current
    return torch.cat((filled, current), dim=1)


def _strict_state(module, state):
    if not isinstance(state, Mapping) or set(state) != set(module.state_dict()):
        raise ValueError('warmstart requires the exact state_dict key set')
    for name, expected in module.state_dict().items():
        value = _float32(state[name], 'warmstart ' + name, expected.shape)
        _finite_range(value, 'warmstart ' + name)
    module.load_state_dict(state, strict=True)


class MaskAwareRestorer(nn.Module):
    """Inherited blind fidelity model with an 11-channel explicit-mask head.

    Call from_rphi with a *separately authenticated* accepted checkpoint. This
    constructor neither resolves paths nor guesses a model/checkpoint/config.
    """
    def __init__(self, model_config):
        super().__init__()
        if not isinstance(model_config, Mapping) or set(model_config) != _CONFIG_KEYS:
            raise ValueError('full explicit Rphi model_config is required')
        if model_config['variant'] != 'blind':
            raise ValueError('P1 requires a blind model without a confidence head')
        base = RphiModel(**dict(model_config))
        self.base_model_config = dict(base.model_config)
        self.model_config = {'schema': SCHEMA, 'base_model_config': dict(base.model_config), 'head_channels': 11}
        self.backbone, self.condition_encoder = base.backbone, base.condition_encoder
        old = self.backbone.head
        expanded = nn.Conv2d(11, old.out_channels, 3, padding=1, bias=True)
        with torch.no_grad():
            expanded.weight.zero_()
            expanded.weight[:, :9].copy_(old.weight)
            expanded.bias.copy_(old.bias)
        self.backbone.head = expanded

    @classmethod
    def from_rphi(cls, model_config, state_dict):
        """Strict old9+2zero warmstart, without aliasing or dtype coercion."""
        if not isinstance(model_config, Mapping) or set(model_config) != _CONFIG_KEYS or model_config.get('variant') != 'blind':
            raise ValueError('full explicit blind model_config is required')
        inherited = RphiModel(**dict(model_config))
        _strict_state(inherited, state_dict)
        result = cls(model_config)
        expanded = {name: value.detach().clone() for name, value in inherited.state_dict().items()}
        weight = expanded['backbone.head.weight']
        expanded['backbone.head.weight'] = torch.cat((weight, weight.new_zeros(weight.shape[0], 2, 3, 3)), dim=1)
        _strict_state(result, expanded)
        return result

    def forward(self, observed, weights):
        filled = fill_history(observed, weights)
        code = self.condition_encoder(filled)
        features = self.backbone.head(torch.cat((filled.flatten(1, 2), weights), dim=1))
        for block in self.backbone.blocks:
            features = block(features, code)
        return {'rgb': observed[:, -1] + self.backbone.tail(features), 'condition_code': code}


class HistoryUtilityGate(nn.Module):
    """Shared1745-parameter raw-only gate; sigmoid is an action, not confidence."""
    def __init__(self, seed):
        super().__init__()
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
            raise ValueError('gate initialization requires an explicit nonnegative integer seed')
        # Isolate initialization from the caller's sample schedule RNG.
        with torch.random.fork_rng(devices=[]):
            generator = torch.Generator(device='cpu').manual_seed(seed)
            self.net = nn.Sequential(nn.Conv2d(11, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 1, 3, padding=1))
            nn.init.kaiming_uniform_(self.net[0].weight, a=math.sqrt(5), generator=generator)
            bound = 1 / math.sqrt(11 * 3 * 3)
            nn.init.uniform_(self.net[0].bias, -bound, bound, generator=generator)
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        if sum(p.numel() for p in self.parameters()) != GATE_PARAMETERS:
            raise RuntimeError('shared gate parameter contract changed')

    def features(self, observed, availability):
        b, h, w = _observed(observed)
        _availability(availability, b, observed.device)
        current = observed[:, -1:, :, :, :].expand(-1, 2, -1, -1, -1)
        present = availability[:, :, None, None, None].bool()
        history = torch.where(present, observed[:, :2], current)
        _finite_range(history, 'available gate observations', 0., 1.)
        evidence = torch.cat((current, history, (current - history).abs()), dim=2)
        pooled = F.avg_pool2d(evidence.reshape(b * 2, 9, h, w), TILE, TILE)
        lag = observed.new_tensor([1., .5])[None].expand(b, -1).reshape(b * 2, 1, 1, 1)
        mask = availability.reshape(b * 2, 1, 1, 1)
        return torch.cat((pooled, lag.expand(-1, 1, h // TILE, w // TILE), mask.expand(-1, 1, h // TILE, w // TILE)), 1)

    def forward(self, observed, availability):
        b, h, w = _observed(observed)
        z = self.net(self.features(observed, availability)).reshape(b, 2, h // TILE, w // TILE)
        g = torch.sigmoid(z)
        weights = F.interpolate(g, size=(h, w), mode='bilinear', align_corners=False, antialias=False)
        return {'z': z, 'g': g, 'weights': weights * availability[:, :, None, None]}


class HistoryUtilityStudent(nn.Module):
    """One shared gate invocation and one restorer invocation per frame batch."""
    def __init__(self, teacher, gate_seed):
        super().__init__()
        if not isinstance(teacher, MaskAwareRestorer):
            raise TypeError('students must copy an explicit mask-aware teacher')
        for name, value in teacher.state_dict().items():
            _finite_range(_float32(value, 'teacher ' + name), 'teacher ' + name)
        self.restorer = copy.deepcopy(teacher)
        self.gate = HistoryUtilityGate(gate_seed).to(device=next(teacher.parameters()).device)
        self.requires_grad_(True)
        self.train()
        self.model_config = {'schema': SCHEMA, 'restorer': copy.deepcopy(teacher.model_config), 'gate_parameters': GATE_PARAMETERS}

    def forward(self, observed, availability):
        gate = self.gate(observed, availability)
        return {**self.restorer(observed, gate['weights']), **gate}

    def without_history(self, observed, normal_weights, slot):
        """Own intervention: no gate re-evaluation, other original weights fixed."""
        weights = remove_history(normal_weights, slot)
        return {**self.restorer(observed, weights), 'weights': weights}


def remove_history(weights, slot):
    if isinstance(slot, bool) or not isinstance(slot, int) or slot not in (0, 1):
        raise ValueError('history slot must be0 or1')
    _float32(weights, 'history weights')
    if weights.ndim not in (2, 4) or weights.shape[0] < 1 or weights.shape[1] != 2:
        raise ValueError('history weights must be [B,2] or [B,2,H,W]')
    _finite_range(weights, 'history weights', 0., 1.)
    result = weights.clone()
    result[:, slot] = 0
    return result


def tile_mae(prediction, target):
    _float32(prediction, 'RGB prediction')
    _float32(target, 'RGB target', prediction.shape)
    if prediction.ndim != 4 or prediction.shape[1] != 3 or prediction.shape[0] < 1:
        raise ValueError('tile loss requires nonempty B3HW RGB')
    if prediction.device != target.device:
        raise ValueError('prediction and target devices differ')
    h, w = prediction.shape[-2:]
    if min(h, w) < TILE or h % TILE or w % TILE:
        raise ValueError('tile loss geometry must be divisible by16')
    _finite_range(prediction, 'unclamped prediction')
    _finite_range(target, 'target RGB', 0., 1.)
    return F.avg_pool2d((prediction - target).abs().mean(1, keepdim=True), TILE, TILE)


def utility_targets(with_all, without, target, availability):
    """Pure detached LOO labels; without is B,2,3,H,W, not a model input.

    Nonexistent slots have zero labels and no support. Their unused placeholder
    output is ignored, so callers need not run a teacher for those slots.
    Positive utility means the available history reduced unclamped tile MAE.
    """
    _float32(with_all, 'baseline RGB')
    if with_all.ndim != 4 or with_all.shape[1] != 3:
        raise ValueError('baseline RGB must be B3HW')
    b, _, h, w = with_all.shape
    _availability(availability, b, with_all.device)
    _float32(without, 'leave-one-out RGB', (b, 2, 3, h, w))
    if without.device != with_all.device:
        raise ValueError('leave-one-out devices differ')
    baseline = with_all.detach()
    available = availability[:, :, None, None, None].bool()
    removed = torch.where(available, without.detach(), baseline[:, None])
    base_loss = tile_mae(baseline, target.detach())
    removed_loss = tile_mae(removed.reshape(b * 2, 3, h, w), target.detach()[:, None].expand(-1, 2, -1, -1, -1).reshape(b * 2, 3, h, w))
    raw = removed_loss.reshape(b, 2, h // TILE, w // TILE) - base_loss
    valid = availability.detach().bool()
    raw = torch.where(valid[:, :, None, None], raw, torch.zeros_like(raw))
    return {'raw_utility': raw, 'normalized': (raw / UTILITY_SCALE).clamp(-4., 4.), 'valid': valid}


def utility_loss(z, normalized, valid):
    """Available-slot/tile mean SmoothL1; empty support is differentiable zero."""
    _float32(z, 'utility logits')
    if z.ndim != 4 or z.shape[0] < 1 or z.shape[1] != 2 or min(z.shape[-2:]) < 1:
        raise ValueError('utility logits must be nonempty B2hw')
    _float32(normalized, 'normalized utility', z.shape)
    _finite_range(z, 'utility logits')
    _finite_range(normalized, 'normalized utility', -4., 4.)
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool or tuple(valid.shape) != tuple(z.shape[:2]):
        raise ValueError('utility valid mask must be bool[B,2]')
    if normalized.device != z.device or valid.device != z.device:
        raise ValueError('utility supervision devices differ')
    support = valid[:, :, None, None].expand_as(z)
    losses = F.smooth_l1_loss(z, normalized.detach(), beta=1., reduction='none')
    return torch.where(support, losses, torch.zeros_like(losses)).sum() / support.sum().clamp_min(1)


def matched_losses(outputs, target, current_observed, labels, valid, arm):
    """Fixed original fidelity loss plus predeclared auxiliary coefficient.

    All arms evaluate the same auxiliary graph, including coefficient0. The
    caller must supply the frozen true/shuffled assignment and source receipts.
    No control metadata, target or utility is fed back into model forward.
    """
    if arm not in ARMS:
        raise ValueError('unknown matched history-utility arm')
    from train_rphi import fidelity_losses
    fidelity, components = fidelity_losses(outputs['rgb'], target, current_observed, edge_weight=.1, guard_weight=.5)
    auxiliary = utility_loss(outputs['z'], labels, valid)
    return fidelity + ARMS[arm] * auxiliary, {**components, 'fidelity': fidelity, 'utility': auxiliary}
