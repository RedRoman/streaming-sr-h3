"""VideoLQ as an eval set with no ground truth, at the exact geometry the model deploys at.

Every number in this project so far comes from real video with SYNTHETIC degradation: the sources are
AudioSet clips, but the blur, the downsample, the noise and the JPEG are applied by a script. If the
claim is real-world restoration, that observation surface does not cover it. VideoLQ is 50 clips of
genuinely degraded video with no reference, which is the missing axis -- and also the hardest one to
report, because this project has just measured that no-reference metrics cannot be trusted here: a
NIQE-family score fitted on the held-out set's own ground truth cannot separate FlashVSR+ from the
ground truth itself (median 3.01 against 2.98). So the deliverable from this set is side-by-side video
for human inspection, not a leaderboard.

Two geometry decisions, both to keep the comparison honest rather than flattering:

  crop, do not resize.  VideoLQ frames run 320x240 to 720x576. Upscaling 640x480 by four gives
  2560x1920, which is far outside the 480x832 the DiT was trained and benchmarked at, and resizing the
  input first would destroy exactly the real degradation this set exists to test. A 208x120 crop is the
  LR geometry every other number in this project was produced at, so the model sees a tensor it is
  calibrated for and the only thing that changed is where the degradation came from.

  choose the crop by a rule.  The window with the most high-frequency content, measured as the energy
  the x4 operator would discard, so the crop is the part of the frame where restoration is actually
  being asked to do something. Picking by eye would let the result be chosen after the fact.

90 frames, five blocks of the standard 17/22 streaming plan: 46 of the 50 clips have 100 frames and the
remaining four have 34-49, so a uniform 90 keeps one block plan for every clip instead of a per-clip
plan that would make the latency numbers incomparable.

The scale is an assumption, and it is the framework's weak point here. R_phi conditions on the MTF of
U(D(.)), which requires D to be known -- the whole non-blind setting. A real camera's degradation has no
analytic form, so this runs under an assumed x4 bicubic D, and that mismatch is part of what the videos
will show. It is recorded in meta.json as assumed_operator so no later reader mistakes it for a
measurement.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SRC = ROOT / 'data/videolq/Input'
DST = ROOT / 'data/real_eval/videolq'
CROP_W, CROP_H = 208, 120
FRAMES = 90
BLOCKS = [{'k': k, 'range': [17 * k, 17 * k + 22],
           'new': [17 * k + 5 if k else 0, 17 * k + 22]} for k in range(5)]


def frames_of(d):
    fs = sorted(list(d.glob('*.png')) + list(d.glob('*.jpg')))
    return fs


def best_crop(img):
    """Top-left of the CROP_W x CROP_H window holding the most detail the x4 operator would discard."""
    t = torch.from_numpy(np.asarray(img.convert('RGB')).copy()).float().permute(2, 0, 1)[None] / 255
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(max(1, h // 4), max(1, w // 4)), mode='bicubic',
                      align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    n = (t - u).square().mean(1)[0]
    best, by, bx = -1.0, 0, 0
    for y in range(0, max(1, h - CROP_H + 1), 8):
        for x in range(0, max(1, w - CROP_W + 1), 8):
            e = float(n[y:y + CROP_H, x:x + CROP_W].mean())
            if e > best:
                best, by, bx = e, y, x
    return by, bx, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips', type=int, default=0, help='0 = every clip with enough frames')
    ap.add_argument('--min-frames', type=int, default=FRAMES)
    a = ap.parse_args()

    DST.mkdir(parents=True, exist_ok=True)
    names, skipped = [], []
    dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    for d in dirs:
        fs = frames_of(d)
        if len(fs) < a.min_frames:
            skipped.append({'clip': d.name, 'frames': len(fs)})
            continue
        if a.clips and len(names) >= a.clips:
            break
        im0 = Image.open(fs[0])
        W, H = im0.size
        if W < CROP_W or H < CROP_H:
            skipped.append({'clip': d.name, 'size': [W, H]})
            continue
        cy, cx, energy = best_crop(im0)
        out = DST / d.name
        (out / 'lr').mkdir(parents=True, exist_ok=True)
        recs = {}
        for t in range(FRAMES):
            im = Image.open(fs[t]).convert('RGB').crop((cx, cy, cx + CROP_W, cy + CROP_H))
            p = out / 'lr' / f'{t:06d}.png'
            im.save(p)
            recs[str(t)] = {'lr_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
        meta = {
            'id': d.name, 'source': 'videolq', 'category': 'real_degradation_no_reference',
            'frames': FRAMES, 'hr_wh': [CROP_W * 4, CROP_H * 4], 'lr_wh': [CROP_W, CROP_H],
            'orig_wh': [W, H], 'crop_yx': [cy, cx], 'crop_null_energy': energy,
            # there IS no ground truth here; the degradation is whatever the camera and the upload did
            'degradation': {'kind': 'realistic', 'origin': 'real_unknown'},
            'assumed_operator': {'D': 'bicubic x4',
                                 'why': 'R_phi conditions on the MTF of U(D(.)), so D must be known. A '
                                        'real capture has no analytic D and this project measured that '
                                        'estimating it from one frame is closed (supervised AUC '
                                        '0.61-0.68). x4 is therefore ASSUMED, and the mismatch is part '
                                        'of what this set tests.'},
            'has_hr': False,
            'frame_records': recs,
        }
        (out / 'meta.json').write_text(json.dumps(meta, indent=1))
        names.append(d.name)
        if len(names) % 10 == 0:
            print(json.dumps({'done': len(names)}), flush=True)

    man = {'clips': names, 'blocks': BLOCKS, 'frames': FRAMES,
           'lr_wh': [CROP_W, CROP_H], 'hr_wh': [CROP_W * 4, CROP_H * 4],
           'has_hr': False, 'source': 'videolq',
           'note': 'no reference frames exist; reference metrics are undefined on this set and judge '
                   'must not be run against it. Use nr_score.py and side-by-side video.'}
    (DST / 'manifest.json').write_text(json.dumps(man, indent=1))
    print()
    print(json.dumps({'written': len(names), 'skipped': skipped,
                      'blocks': len(BLOCKS), 'frames': FRAMES,
                      'lr': [CROP_W, CROP_H], 'out': [CROP_W * 4, CROP_H * 4],
                      'dst': str(DST)}, indent=1))


if __name__ == '__main__':
    main()
