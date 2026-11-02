"""The operator as a map, not a label: a radial MTF descriptor of D.

Every all-in-one restoration model this project has compared against conditions on a categorical
degradation identity -- AirNet, PromptIR and MambaOFR all predict or look up a discrete type, and our own
omni arm uses a one-hot over [2, 4, 8, deblur, denoise]. A one-hot cannot represent an operator it has no
slot for, so x3 and x6 are not merely untrained, they are unrepresentable. That is the limitation this
module removes.

D is known at deployment time -- that is the whole non-blind setting, and it is forced on us because
estimating the operator from one LR frame is closed (supervised upper bound AUC 0.61-0.68; Immerkaer's
sigma at 0.490, chance). So instead of naming D we can measure it: push white noise through
U(D(.)) and record how much energy survives in each radial frequency band. The result is a fixed-length
vector that is a continuous function of the operator:

  x2    knee near 0.50 of Nyquist      x3   knee near 0.33
  x4    knee near 0.25                 x6   knee near 0.17
  x8    knee near 0.125
  deblur (Gaussian sigma)              a smooth rolloff with no sharp knee
  denoise (D = identity)               all ones -- which states the degeneracy explicitly: nothing is
                                       removed, so the null space is empty and a decomposition-based
                                       restorer has nothing to synthesise

The descriptor is estimated from the operator, never from the image, so it carries no content and cannot
leak the answer -- the content shortcut that killed the earlier contrastive attempt (InfoNCE fell to
0.278 while the probe stayed at chance) is structurally impossible here.
"""
import numpy as np
import torch
import torch.nn.functional as F

NBINS = 16
PROBE_HW = (256, 256)
PROBE_SEED = 20261101


def _radial_bins(h, w, nbins=NBINS):
    """Index map from rfft2 bin to radial band, normalised so 1.0 is the Nyquist corner."""
    fy = torch.fft.fftfreq(h)[:, None]
    fx = torch.fft.rfftfreq(w)[None, :]
    r = torch.sqrt(fy ** 2 + fx ** 2) / 0.5 ** 0.5 / (2 ** 0.5 / 2)
    r = (r / r.max()).clamp(0, 1 - 1e-6)
    return (r * nbins).long()


def _band_energy(x, idx, nbins=NBINS):
    """Energy per radial band of a [B,C,H,W] batch."""
    F2 = torch.fft.rfft2(x.double())
    p = (F2.real ** 2 + F2.imag ** 2).sum(dim=(0, 1))
    out = torch.zeros(nbins, dtype=torch.float64)
    out.scatter_add_(0, idx.reshape(-1), p.reshape(-1))
    return out


def apply_operator(x, op):
    """U(D(x)) for one operator. x is [B,3,H,W] in [0,1]; the result has the same shape.

    Matches the operators the pair generators use: the scales are the antialiased bicubic chain that
    defines the range/null split everywhere in this project, deblur is a Gaussian at HR, and denoise is
    the identity.
    """
    h, w = x.shape[-2:]
    if isinstance(op, (int, float)) and not isinstance(op, bool):
        s = float(op)
        hh, ww = max(1, int(round(h / s))), max(1, int(round(w / s)))
        d = F.interpolate(x, size=(hh, ww), mode='bicubic', align_corners=False, antialias=True)
        return F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    if isinstance(op, str) and op.startswith('deblur'):
        sigma = float(op.split(':')[1]) if ':' in op else 1.75
        r = max(1, int(round(3 * sigma)))
        g = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
        g = (g / g.sum())
        k = torch.outer(g, g)[None, None].repeat(x.shape[1], 1, 1, 1).to(x.device, x.dtype)
        return F.conv2d(F.pad(x, (r, r, r, r), mode='reflect'), k, groups=x.shape[1])
    if isinstance(op, str) and op.startswith('denoise'):
        return x.clone()
    if isinstance(op, str) and op.startswith('bd'):
        # the fitted Vid4 BD: Gaussian sigma 1.10 then INTER_AREA by the scale, then back up
        s = float(op.split(':')[1]) if ':' in op else 4.0
        sigma = 1.10
        r = max(1, int(round(3 * sigma)))
        g = torch.exp(-torch.arange(-r, r + 1, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
        g = (g / g.sum())
        k = torch.outer(g, g)[None, None].repeat(x.shape[1], 1, 1, 1).to(x.device, x.dtype)
        b = F.conv2d(F.pad(x, (r, r, r, r), mode='reflect'), k, groups=x.shape[1])
        hh, ww = max(1, int(round(h / s))), max(1, int(round(w / s)))
        d = F.interpolate(b, size=(hh, ww), mode='area')
        return F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    raise ValueError(f'unknown operator {op!r}')


@torch.no_grad()
def operator_mtf(op, nbins=NBINS, hw=PROBE_HW, seed=PROBE_SEED):
    """Fraction of energy surviving U(D(.)) in each radial band. Length nbins, values in [0, 1].

    Estimated on a fixed white-noise probe, so it is deterministic and depends only on the operator.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(2, 3, *hw, generator=g)
    idx = _radial_bins(*hw, nbins)
    e_in = _band_energy(x - x.mean(), idx, nbins)
    y = apply_operator(x, op)
    e_out = _band_energy(y - y.mean(), idx, nbins)
    return (e_out / e_in.clamp_min(1e-12)).clamp(0, 1).float()


def descriptor(op, severity=None, nbins=NBINS):
    """The conditioning vector: the MTF, plus severity when an arm is allowed to see it."""
    v = operator_mtf(op, nbins)
    return torch.cat([v, torch.tensor([float(severity or 0.0)])])


DIM = NBINS + 1


if __name__ == '__main__':
    import json
    print('radial MTF, %d bands, 1.0 = fully preserved' % NBINS)
    ops = [2, 3, 4, 6, 8, 'deblur:1.75', 'denoise', 'bd:4']
    head = '%-12s' % 'operator' + ''.join('%6.2f' % ((i + 0.5) / NBINS) for i in range(NBINS))
    print(head)
    print('-' * len(head))
    rows = {}
    for o in ops:
        m = operator_mtf(o)
        rows[str(o)] = [round(float(x), 4) for x in m]
        print('%-12s' % str(o) + ''.join('%6.2f' % float(x) for x in m))
    print()
    # the knee should track 1/s and move continuously, which is the whole claim
    print('%-12s %14s %14s' % ('operator', 'half-power bin', 'expected 1/s'))
    for o in ops:
        m = operator_mtf(o)
        k = next((i for i in range(NBINS) if float(m[i]) < 0.5), NBINS)
        exp = (1.0 / float(o)) if isinstance(o, (int, float)) else (
            1.0 / float(str(o).split(':')[1]) if str(o).startswith('bd:') else float('nan'))
        print('%-12s %14.3f %14s' % (str(o), (k + 0.5) / NBINS,
                                     ('%.3f' % exp) if exp == exp else '-'))
    print()
    print('monotone non-increasing:',
          {str(o): bool(all(float(operator_mtf(o)[i]) >= float(operator_mtf(o)[i + 1]) - 0.02
                            for i in range(NBINS - 1))) for o in ops})
    print()
    print(json.dumps({'dim': DIM, 'nbins': NBINS}, indent=1))
