"""Untrained native-scale conditioning candidate; not connected to RphiModel.

This module preserves the v1 observation encoder's parameter layout while
removing its observation-level 64-pixel pooling. Fixed-stride convolutions now
extract features at the supplied RGB pixel scale before global pooling. This
addresses that particular scale mismatch; it does not establish better quality
or make the final global code invariant to changes in image content or extent.

Inputs contain exactly three observations, oldest first and current last. The
caller must supply causal histories: pixels alone cannot reveal timestamps or
detect a future frame substituted into one of those three slots. No target,
label, task, severity, or other external conditioning argument is accepted.
"""
import torch
from torch import nn


CANDIDATE_STATUS = "untrained_candidate_not_integrated"


class NativeObservationEncoder(nn.Module):
    """Encode [B,3,3,H,W] observations into a bounded [B,code_dim] code.

    With code_dim=16, ``state_dict`` keys and tensor shapes match the existing
    ``rphi_model._ObservationEncoder`` exactly. Strict weight loading provides
    a controlled implementation comparison, not pretrained v2 validation.
    """
    def __init__(self, k=3, code_dim=16):
        super().__init__()
        if isinstance(k, bool) or not isinstance(k, int) or k != 3:
            raise ValueError("this candidate accepts exactly three causal observations")
        if isinstance(code_dim, bool) or not isinstance(code_dim, int) or code_dim < 1:
            raise ValueError("code_dim must be a positive integer")
        self.k = k
        self.code_dim = code_dim
        channels = 3 * (2 * k - 1)
        # Keep layer order and parameter keys compatible with v1. In
        # particular, global pooling follows both native-scale convolutions.
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, code_dim), nn.Tanh())

    def dense_features(self, observed):
        """Return [B,16,ceil(H/4),ceil(W/4)] before global pooling.

        The two convolutions have stride four and a 7x7 input receptive field.
        Away from boundaries, the same local pixels at offsets divisible by
        four yield the same dense features regardless of surrounding canvas
        size. The pooled code need not be identical when that canvas changes.
        """
        if not isinstance(observed, torch.Tensor):
            raise TypeError("observed must be a floating-point torch.Tensor")
        if observed.ndim != 5 or tuple(observed.shape[1:3]) != (self.k, 3):
            raise ValueError("observed must be [B,3,3,H,W], oldest frame first")
        if min(observed.shape[0], *observed.shape[-2:]) < 1 or not observed.is_floating_point():
            raise ValueError("observed requires nonempty floating-point RGB frames")
        changes = observed[:, 1:] - observed[:, :-1]
        features = torch.cat((observed.flatten(1, 2), changes.flatten(1, 2)), dim=1)
        for index in range(4):
            features = self.net[index](features)
        return features

    def forward(self, observed):
        code = self.dense_features(observed)
        for index in range(4, len(self.net)):
            code = self.net[index](code)
        return code
