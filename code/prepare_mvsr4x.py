"""MVSR4x as an eval set, and a measurement of how wrong this project's operator assumption is on it.

MVSR4x is real paired data: a phone captures the same scene through a wide and a tele lens, so the LQ
degradation is genuine optics and sensor, not a script. That makes it the first set here where real
degradation and a reference exist at the same time -- VideoLQ has real degradation but no reference, and
heldout has a reference but synthetic degradation.

Two things had to be checked before using it, both recorded in the package's own README as open:

  the stored LQ is not a native LR.  GT and LQ are both 1024x1024 and the EAVSR reader downsamples at
  read time. Measured here: round-tripping LQ through 256 costs 44.5-56.4 dB while the same round trip
  on GT costs 28.7-38.9 dB, so LQ really does carry ~256 of resolution and the correct LR is LQ
  downsampled by four. That is what this script does.

  the operator is unknown, and R_phi needs one.  Every arm in this project conditions on the MTF of
  U(D(.)), which presumes D. A lens is not a bicubic kernel. Rather than assume x4 silently, this script
  MEASURES the gap: it computes the radial transfer function of the real LQ->GT pair and compares it to
  the x4 bicubic descriptor the model will actually be given, writing both into each meta.json. The
  result is the honest size of the domain shift, available before any card is spent and quotable in the
  paper next to whatever score comes out.

Geometry follows VideoLQ: an 832x480 window of the 1024x1024 frame, chosen by where the GT carries the
most detail the x4 operator would discard, with the LR taken from the same window of LQ. Keeping the
deployed geometry fixed means the only thing that changes against every other number in this project is
where the degradation came from.

90 frames, five blocks of the standard 17/22 plan, same as VideoLQ, so latency and block behaviour stay
comparable across the two real-degradation sets.
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.fft as fft
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SRC = ROOT / 'data/mvsr4x'
DST = ROOT / 'data/real_eval/mvsr4x'
CROP_W, CROP_H = 832, 480
SCALE = 4
FRAMES = 90
NBINS = 16
BLOCKS = [{'k': k, 'range': [17 * k, 17 * k + 22],
           'new': [17 * k + 5 if k else 0, 17 * k + 22]} for k in range(5)]


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def null_energy_window(gt):
    """Top-left of the CROP window with the most detail an x4 operator would discard."""
    t = torch.from_numpy(gt).permute(2, 0, 1)[None].float() / 255
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // SCALE, w // SCALE), mode='bicubic', align_corners=False,
                      antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    n = (t - u).square().mean(1)[0]
    best, by, bx = -1.0, 0, 0
    for y in range(0, h - CROP_H + 1, 16):
        for x in range(0, w - CROP_W + 1, 16):
            e = float(n[y:y + CROP_H, x:x + CROP_W].mean())
            if e > best:
                best, by, bx = e, y, x
    return by, bx


def radial_mtf(lr_up, hr, nbins=NBINS):
    """Measured transfer function of the real degradation: energy(LQ) / energy(GT) per radial band.

    Directly comparable to operator_desc.operator_mtf, which is what the model is conditioned on.
    """
    def bands(x):
        g = torch.from_numpy(x).float().mean(-1).double() / 255
        Fm = fft.rfft2(g)
        p = Fm.real ** 2 + Fm.imag ** 2
        h, w = g.shape
        fy = fft.fftfreq(h)[:, None]
        fx = fft.rfftfreq(w)[None, :]
        r = torch.sqrt(fy ** 2 + fx ** 2)
        r = (r / float(r.max())).clamp(0, 1 - 1e-6)
        idx = (r * nbins).long().clamp(0, nbins - 1)
        out = torch.zeros(nbins, dtype=torch.float64)
        out.scatter_add_(0, idx.reshape(-1), p.reshape(-1))
        return out
    return (bands(lr_up) / bands(hr).clamp_min(1e-20)).sqrt().clamp(0, 1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=0)
    a = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)

    import sys
    sys.path.insert(0, str(ROOT / 'code'))
    from operator_desc import operator_mtf
    assumed = operator_mtf(SCALE)          # what R_phi will be told the operator is

    seqs = sorted(p.name for p in (SRC / 'GT').iterdir() if p.is_dir())
    if a.clips:
        seqs = seqs[:a.clips]
    names, gaps = [], []
    for s in seqs:
        gfs = sorted((SRC / 'GT' / s).glob('*.png'))
        lfs = sorted((SRC / 'LQ' / s).glob('*.png'))
        if len(gfs) < FRAMES or len(lfs) < FRAMES:
            print(json.dumps({'skip': s, 'gt': len(gfs), 'lq': len(lfs)}), flush=True)
            continue
        cy, cx = null_energy_window(load(gfs[0]))
        out = DST / s
        (out / 'lr').mkdir(parents=True, exist_ok=True)
        (out / 'hr').mkdir(parents=True, exist_ok=True)
        recs = {}
        mtf_acc, ctl_acc = [], []
        for t in range(FRAMES):
            hr = load(gfs[t])[cy:cy + CROP_H, cx:cx + CROP_W]
            lq = load(lfs[t])[cy:cy + CROP_H, cx:cx + CROP_W]
            # the real LR: LQ carries ~1/4 of the resolution, so downsample the same window
            lr = cv2.resize(lq, (CROP_W // SCALE, CROP_H // SCALE), interpolation=cv2.INTER_CUBIC)
            ph = out / 'hr' / f'{t:06d}.png'
            pl = out / 'lr' / f'{t:06d}.png'
            Image.fromarray(hr).save(ph)
            Image.fromarray(lr).save(pl)
            recs[str(t)] = {'hr_sha256': hashlib.sha256(ph.read_bytes()).hexdigest(),
                            'lr_sha256': hashlib.sha256(pl.read_bytes()).hexdigest()}
            if t % 30 == 0:
                up = cv2.resize(lr, (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                mtf_acc.append(radial_mtf(up, hr).numpy())
                # Same measurement on a SYNTHETIC x4 pair built from this very frame. operator_mtf uses
                # a white-noise probe while this uses natural images, whose 1/f spectrum makes the
                # high-band ratios noisy -- measured on a true x4 pair the two disagree by L2 1.02, so
                # comparing a natural-image MTF against operator_mtf directly would have reported a gap
                # that is mostly the measurement. This control is the same operator under the same
                # statistics, so the difference against it is the real domain shift.
                syn_lr = cv2.resize(hr, (CROP_W // SCALE, CROP_H // SCALE),
                                    interpolation=cv2.INTER_CUBIC)
                syn_up = cv2.resize(syn_lr, (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                ctl_acc.append(radial_mtf(syn_up, hr).numpy())
        measured = np.mean(mtf_acc, axis=0)
        control = np.mean(ctl_acc, axis=0)
        gap = float(np.linalg.norm(measured - control))
        gap_vs_probe = float(np.linalg.norm(measured - assumed.numpy()))
        ctl_vs_probe = float(np.linalg.norm(control - assumed.numpy()))
        gaps.append(gap)
        meta = {
            'id': s, 'source': 'mvsr4x', 'category': 'real_degradation_with_reference',
            'frames': FRAMES, 'hr_wh': [CROP_W, CROP_H], 'lr_wh': [CROP_W // SCALE, CROP_H // SCALE],
            'orig_wh': [1024, 1024], 'crop_yx': [cy, cx],
            'degradation': {'kind': 'realistic', 'origin': 'real_optics_wide_tele'},
            'assumed_operator': {
                'D': 'bicubic x4',
                'measured_radial_mtf': [round(float(x), 4) for x in measured],
                'synthetic_x4_control_mtf': [round(float(x), 4) for x in control],
                'assumed_radial_mtf_white_noise_probe': [round(float(x), 4) for x in assumed],
                'l2_gap_vs_same_measurement_x4': gap,
                'l2_gap_vs_white_noise_probe': gap_vs_probe,
                'l2_control_vs_white_noise_probe': ctl_vs_probe,
                'note': 'l2_gap_vs_same_measurement_x4 is the usable number: real degradation against '
                        'a synthetic x4 measured identically on the same frames. The probe comparison '
                        'is kept only to show how much of it is measurement rather than domain shift.',
                'why': 'R_phi conditions on the MTF of U(D(.)) and a lens has no analytic D. The gap is '
                       'the size of the domain shift this set imposes on the conditioning, measured '
                       'rather than assumed.'},
            'has_hr': True,
            'frame_records': recs,
        }
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        names.append(s)
        print(json.dumps({'clip': s, 'crop_yx': [cy, cx],
                          'gap_vs_synthetic_x4': round(gap, 4),
                          'gap_vs_probe': round(gap_vs_probe, 4),
                          'control_vs_probe': round(ctl_vs_probe, 4)}), flush=True)

    man = {'clips': names, 'blocks': BLOCKS, 'frames': FRAMES,
           'lr_wh': [CROP_W // SCALE, CROP_H // SCALE], 'hr_wh': [CROP_W, CROP_H],
           'has_hr': True, 'source': 'mvsr4x',
           'note': 'real optical degradation WITH a reference. Reference metrics are defined here, '
                   'unlike videolq. The operator is assumed x4 bicubic and each meta.json records the '
                   'measured gap to that assumption.'}
    (DST / 'manifest.json').write_text(json.dumps(man, indent=1))
    print()
    print(json.dumps({'written': len(names), 'frames': FRAMES, 'blocks': len(BLOCKS),
                      'lr': [CROP_W // SCALE, CROP_H // SCALE], 'hr': [CROP_W, CROP_H],
                      'mtf_gap_mean': round(float(np.mean(gaps)), 4) if gaps else None,
                      'mtf_gap_max': round(float(np.max(gaps)), 4) if gaps else None,
                      'dst': str(DST)}, indent=1))
    print()
    print('For scale: a descriptor L2 of 0.21 (x6 against the trained set) generalised fine, while')
    print('0.32-1.05 (a sigma-0.6 blur) collapsed 2.9 dB. That is the yardstick for the gap above.')


if __name__ == '__main__':
    main()
