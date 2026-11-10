"""RealVSR as a second real-degradation eval set with a reference.

Until now this project had exactly one set where real degradation and a ground truth coexist (mvsr4x,
wide/tele lens pairs). RealVSR (ICCV 2021) is the other published one and it is captured differently --
an iPhone 11 Pro Max multi-camera rig, two focal lengths recorded simultaneously -- so agreement across
the two is evidence about real degradation rather than about one rig.

Three things the package forces a decision on, all recorded in each meta.json rather than assumed:

  the scale is x2, not x4.  The pair comes from two focal lengths, and the distributed LQ is already
  resampled to the GT geometry, so the stored pair is same-size. This project is x4 everywhere and R_phi
  is conditioned on an x4 operator, so the LR is taken as LQ downsampled by four and the chain under test
  is "real phone degradation, then a bicubic x4". The script measures what the x4 downsample costs on
  this LQ (round-trip PSNR through 1/2 and through 1/4) so the amount of real degradation still present
  at the LR is a number and not a hope.

  the geometry is portrait.  Frames are 512x1024 while every number in this project is 832x480. The
  frames are rotated 90 degrees, which is lossless, and the standard 832x480 window is then cut from the
  1024x512 result by the same most-null-energy rule as mvsr4x and videolq.

  50 frames is not a multiple of the 17/22 block plan.  Blocks 0 and 1 cover 39 frames; a third block
  [28,50] with new=[33,50] closes the clip, so there are three blocks and two handovers per clip, and the
  seam metrics stay defined.

The MTF of the real degradation is measured against a synthetic x4 control computed on the same frames
with the same estimator, which is the only comparison that is not dominated by the estimator itself
(see prepare_mvsr4x.py for why the white-noise probe comparison is not usable).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
from prepare_mvsr4x import null_energy_window, radial_mtf   # same window rule, same MTF estimator

SRC = ROOT / 'data/realvsr/RealVSR'
DST = ROOT / 'data/real_eval/realvsr'
CROP_W, CROP_H = 832, 480
SCALE = 4
FRAMES = 50
NBINS = 16
BLOCKS = [{'k': 0, 'range': [0, 22], 'new': [0, 22]},
          {'k': 1, 'range': [17, 39], 'new': [22, 39]},
          {'k': 2, 'range': [28, 50], 'new': [39, 50]}]


def load_rot(p):
    """RealVSR frames are 512x1024 portrait; rotate to 1024x512 landscape (lossless)."""
    return np.rot90(np.asarray(Image.open(p).convert('RGB')), 1).copy()


def rt_psnr(img, factor):
    """What a downsample by `factor` and back costs on this image: how much resolution it really carries."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // factor, w // factor), mode='bicubic', align_corners=False,
                      antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False).clamp(0, 1)
    return float(-10 * torch.log10(((u - t) ** 2).mean() + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=0)
    a = ap.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    from operator_desc import operator_mtf
    assumed = operator_mtf(SCALE)

    seqs = sorted(p.name for p in (SRC / 'GT').iterdir() if p.is_dir())
    if a.clips:
        seqs = seqs[:a.clips]
    names, gaps, rt2, rt4 = [], [], [], []
    for s in seqs:
        gfs = sorted((SRC / 'GT' / s).glob('*.png'))
        lfs = sorted((SRC / 'LQ' / s).glob('*.png'))
        if len(gfs) < FRAMES or len(lfs) < FRAMES:
            print(json.dumps({'skip': s, 'gt': len(gfs), 'lq': len(lfs)}), flush=True)
            continue
        cy, cx = null_energy_window(load_rot(gfs[0]))
        out = DST / s
        (out / 'lr').mkdir(parents=True, exist_ok=True)
        (out / 'hr').mkdir(parents=True, exist_ok=True)
        recs = {}
        mtf_acc, ctl_acc, r2, r4 = [], [], [], []
        for t in range(FRAMES):
            hr = load_rot(gfs[t])[cy:cy + CROP_H, cx:cx + CROP_W]
            lq = load_rot(lfs[t])[cy:cy + CROP_H, cx:cx + CROP_W]
            lr = cv2.resize(lq, (CROP_W // SCALE, CROP_H // SCALE), interpolation=cv2.INTER_CUBIC)
            ph = out / 'hr' / f'{t:06d}.png'
            pl = out / 'lr' / f'{t:06d}.png'
            Image.fromarray(hr).save(ph)
            Image.fromarray(lr).save(pl)
            recs[str(t)] = {'hr_sha256': hashlib.sha256(ph.read_bytes()).hexdigest(),
                            'lr_sha256': hashlib.sha256(pl.read_bytes()).hexdigest()}
            if t % 25 == 0:
                r2.append(rt_psnr(lq, 2)); r4.append(rt_psnr(lq, 4))
                up = cv2.resize(lr, (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                mtf_acc.append(radial_mtf(up, hr).numpy())
                syn_lr = cv2.resize(hr, (CROP_W // SCALE, CROP_H // SCALE),
                                    interpolation=cv2.INTER_CUBIC)
                syn_up = cv2.resize(syn_lr, (CROP_W, CROP_H), interpolation=cv2.INTER_CUBIC)
                ctl_acc.append(radial_mtf(syn_up, hr).numpy())
        measured, control = np.mean(mtf_acc, axis=0), np.mean(ctl_acc, axis=0)
        gap = float(np.linalg.norm(measured - control))
        gaps.append(gap); rt2.append(float(np.mean(r2))); rt4.append(float(np.mean(r4)))
        meta = {
            'id': s, 'source': 'realvsr', 'category': 'real_degradation_with_reference',
            'frames': FRAMES, 'hr_wh': [CROP_W, CROP_H], 'lr_wh': [CROP_W // SCALE, CROP_H // SCALE],
            'orig_wh': [1024, 512], 'rotated_90': True, 'crop_yx': [cy, cx],
            'degradation': {'kind': 'realistic', 'origin': 'real_iphone11promax_dual_focal'},
            'native_scale_of_package': 2,
            'lq_roundtrip_psnr': {'half': round(float(np.mean(r2)), 2), 'quarter': round(float(np.mean(r4)), 2),
                                  'note': 'cost of downsampling the distributed LQ and back. High at 1/2 '
                                          'and low at 1/4 would mean the x4 LR throws real content away; '
                                          'both high would mean the LQ is soft enough that x4 is nearly free.'},
            'assumed_operator': {
                'D': 'bicubic x4',
                'measured_radial_mtf': [round(float(x), 4) for x in measured],
                'synthetic_x4_control_mtf': [round(float(x), 4) for x in control],
                'assumed_radial_mtf_white_noise_probe': [round(float(x), 4) for x in assumed],
                'l2_gap_vs_same_measurement_x4': gap,
                'note': 'same convention as mvsr4x: real degradation against a synthetic x4 measured '
                        'identically on the same frames.'},
            'has_hr': True,
            'frame_records': recs,
        }
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        names.append(s)
        print(json.dumps({'clip': s, 'crop_yx': [cy, cx], 'gap_vs_synthetic_x4': round(gap, 4),
                          'lq_rt_half': round(float(np.mean(r2)), 2),
                          'lq_rt_quarter': round(float(np.mean(r4)), 2)}), flush=True)

    man = {'clips': names, 'blocks': BLOCKS, 'frames': FRAMES,
           'lr_wh': [CROP_W // SCALE, CROP_H // SCALE], 'hr_wh': [CROP_W, CROP_H],
           'has_hr': True, 'source': 'realvsr',
           'note': 'RealVSR ICCV21 test split (50 sequences x 50 frames, 512x1024), rotated to landscape '
                   'and cropped to the project geometry. Real degradation WITH a reference, from a '
                   'different rig than mvsr4x.'}
    (DST / 'manifest.json').write_text(json.dumps(man, indent=1))
    print()
    print(json.dumps({'written': len(names), 'frames': FRAMES, 'blocks': len(BLOCKS),
                      'mtf_gap_mean': round(float(np.mean(gaps)), 4) if gaps else None,
                      'mtf_gap_max': round(float(np.max(gaps)), 4) if gaps else None,
                      'lq_roundtrip_half_mean': round(float(np.mean(rt2)), 2) if rt2 else None,
                      'lq_roundtrip_quarter_mean': round(float(np.mean(rt4)), 2) if rt4 else None,
                      'dst': str(DST)}, indent=1))


if __name__ == '__main__':
    main()
