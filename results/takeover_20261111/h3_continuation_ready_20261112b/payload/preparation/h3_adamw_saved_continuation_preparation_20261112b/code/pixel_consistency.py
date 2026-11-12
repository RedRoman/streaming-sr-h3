"""Classical output-space consistency control, not a learned SR method.

Project onto a separable floating-point antialiased bicubic observation and the
RGB box. The dataset uses uint8 Pillow resize; its rounding/clipping mismatch is
measured separately. Generated projections never become new evidence packets.
"""
import torch
from torch.nn import functional as F


class BicubicConsistency:
    def __init__(self, hr_size=256, lr_size=64, dtype=torch.float64, device='cpu'):
        high = (hr_size, hr_size) if isinstance(hr_size, int) else tuple(hr_size)
        low = (lr_size, lr_size) if isinstance(lr_size, int) else tuple(lr_size)
        def weights(n, m):
            basis = torch.eye(n, dtype=dtype, device=device)[None, None]
            w = F.interpolate(basis, size=(n, m), mode='bicubic', align_corners=False, antialias=True)[0, 0].T
            return w, torch.linalg.solve(w @ w.T, w).T
        self.wy, self.ry = weights(high[0], low[0])
        self.wx, self.rx = weights(high[1], low[1])
        self.w, self.right_inverse = self.wy, self.ry  # square API compatibility

    def down(self, image):
        return self.wy @ image @ self.wx.T

    def affine_project(self, image, observed):
        return image + self.ry @ (observed - self.down(image)) @ self.rx.T

    def project(self, image, observed, iterations=25):
        """Dykstra projection onto {A x = y} intersect [0,1].

        Fixed iterations are a baseline protocol, not optimized against HR.
        Return the actual remaining affine residual; do not claim exact Pillow
        consistency or convergence when a finite iteration budget was used.
        """
        dtype = image.dtype
        x = image.to(self.w); y = observed.to(self.w)
        p = torch.zeros_like(x); q = torch.zeros_like(x)
        for _ in range(iterations):
            shifted = x + p
            affine = self.affine_project(shifted, y)
            p = shifted - affine
            shifted = affine + q
            x = shifted.clamp(0, 1)
            q = shifted - x
        rms = (self.down(x) - y).square().mean((-1, -2, -3)).sqrt()
        return x.to(dtype), rms


def audit():
    torch.manual_seed(29)
    op = BicubicConsistency(32, 8)
    x = torch.rand(2, 3, 32, 32, dtype=torch.float64)
    exact = F.interpolate(x, size=(8, 8), mode='bicubic', align_corners=False, antialias=True)
    separability = float((op.down(x) - exact).abs().max())
    truth = .2 + .6 * torch.rand_like(x); target = op.down(truth)
    projected = op.affine_project(x, target)
    residual = float((op.down(projected) - target).abs().max())
    v = torch.randn_like(x)
    null_v = v - op.right_inverse @ op.down(v) @ op.right_inverse.T
    null_change = float(((projected - x) * null_v).sum().abs())
    bounded, rms = op.project(x, target)
    rectangular = BicubicConsistency((32, 48), (8, 12))
    rectangle = torch.rand(2, 3, 32, 48, dtype=torch.float64)
    rectangular_error = float((rectangular.down(rectangle)-F.interpolate(rectangle, (8, 12), mode='bicubic', align_corners=False, antialias=True)).abs().max())
    assert separability < 1e-12 and residual < 1e-12 and null_change < 1e-10
    assert rectangular_error < 1e-12
    assert bounded.min() >= 0 and bounded.max() <= 1 and float(rms.max()) < 1e-5
    return {'torch_separable_operator_max_abs': separability,
            'affine_projection_max_lr_residual': residual,
            'affine_update_nullspace_inner_product': null_change,
            'bounded_projection_max_lr_rmse': float(rms.max()),
            'rectangular_operator_max_abs': rectangular_error, 'passed': True}
