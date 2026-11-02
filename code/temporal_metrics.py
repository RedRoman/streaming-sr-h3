"""Temporal metrics: the axis on which a video method differs from a per-frame one.

The comparison this project ran against Real-ESRGAN measured only per-frame PSNR and LPIPS, sampled every
eighth frame. Under that protocol temporal modelling is worth exactly nothing, so it could not say anything
about the streaming design -- and Real-ESRGAN, being GAN-trained for single-image perceptual quality, was
being compared on its own strongest axis. These are the measures where the two classes of method actually
differ.

Both are normalised by ground truth, because the degenerate way to win a temporal-stability metric is to
output something blurry and static:

  flicker  mean squared frame-to-frame change in the unobserved band, divided by GT's own. 1.0 means the
           output moves exactly as much as the truth does; above 1 is excess flicker, below 1 is temporal
           over-smoothing, and both are failures.
  warp_err frame t warped to t+1 along GT flow, compared against the output's own frame t+1, again as a
           ratio to GT's. This is the standard tOF-style consistency check.
"""
import argparse, json, os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
OUT = ROOT / 'results/temporal'
PAIR_STRIDE = 8


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def unobserved(x, s=4):
    t = x.permute(0, 3, 1, 2).double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    return (t - u).permute(0, 2, 3, 1).float()


def flow(a, b):
    return cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 3, 7, 1.5, 0)


def warp(img, fl):
    h, w = fl.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img, gx + fl[..., 0], gy + fl[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def ci(v, n=10000, seed=0):
    v = np.asarray(v, float)
    g = np.random.default_rng(seed)
    m = np.array([g.choice(v, v.size, replace=True).mean() for _ in range(n)])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def arm_dir(arm, name):
    return {'raw': ROOT / f'results/stage7_eval/raw/{name}/stream',
            'oracle': ROOT / f'results/stage7_eval/oracle/{name}/stream',
            'realesrgan': ROOT / f'results/baseline_frames/{name}',
            'bicubic': None}.get(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='raw,oracle,realesrgan')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    arms = a.arms.split(',')
    per = {arm: {'flicker': [], 'warp': []} for arm in arms}
    rows = []
    for name in man['clips']:
        T = man['frames']
        pairs = [(t, t + 1) for t in range(0, T - 1, PAIR_STRIDE)]
        g = {t: load(D_ / name / 'hr' / f'{t:06d}.png') for p in pairs for t in p}
        gg = {t: (v.mean(-1).numpy() * 255).astype(np.uint8) for t, v in g.items()}
        r = {'clip': name}
        for arm in arms:
            d = arm_dir(arm, name)
            if d is None or not d.exists():
                continue
            fl_num, fl_den, wp_num, wp_den, ok = [], [], [], [], True
            for (t0, t1) in pairs:
                f0, f1 = d / f'{t0:06d}.png', d / f'{t1:06d}.png'
                if not (f0.exists() and f1.exists()):
                    ok = False
                    break
                y0, y1 = load(f0)[None], load(f1)[None]
                g0, g1 = g[t0][None], g[t1][None]
                ny0, ny1 = unobserved(y0), unobserved(y1)
                ng0, ng1 = unobserved(g0), unobserved(g1)
                fl_num.append(float((ny1 - ny0).square().mean()))
                fl_den.append(float((ng1 - ng0).square().mean()))
                fw = flow(gg[t0], gg[t1])
                wp_num.append(float(np.abs(warp(y0[0].numpy(), fw) - y1[0].numpy()).mean()))
                wp_den.append(float(np.abs(warp(g0[0].numpy(), fw) - g1[0].numpy()).mean()))
            if not ok or not fl_num:
                continue
            # A ratio to ground truth is undefined where the ground truth does not move. Some AudioSet
            # clips are genuinely static (measured: mean|frame0-frame1| exactly 0.0), which drives the
            # denominator to zero and produced ratios of 7544 for one arm and exactly 0.0 for a
            # deterministic per-frame one. Those clips carry no temporal information to compare, so they
            # are excluded, and the count of exclusions is reported rather than hidden.
            if float(np.mean(fl_den)) < 1e-7 or float(np.mean(wp_den)) < 1e-5:
                r[f'{arm}_excluded'] = True
                continue
            fr = float(np.mean(fl_num) / max(np.mean(fl_den), 1e-12))
            wr = float(np.mean(wp_num) / max(np.mean(wp_den), 1e-12))
            r[f'{arm}_flicker'] = fr
            r[f'{arm}_warp'] = wr
            per[arm]['flicker'].append(fr)
            per[arm]['warp'].append(wr)
        rows.append(r)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)

    out = {'n': len(rows), 'per_clip': rows, 'arms': {},
           'static_clips_excluded': sum(1 for r in rows if any(k.endswith('_excluded') for k in r))}
    for arm in arms:
        if per[arm]['flicker']:
            out['arms'][arm] = {k: {'mean': float(np.mean(v)), 'ci95': ci(v)} for k, v in per[arm].items()}
    (OUT / 'metrics.json').write_text(json.dumps(out, indent=1))
    print()
    print('excluded as static: %d of %d clips' % (out['static_clips_excluded'], out['n']))
    print('%-12s %24s %24s' % ('arm', 'flicker (1.0 = like GT)', 'warp err (1.0 = like GT)'))
    for arm, v in out['arms'].items():
        print('%-12s %8.3f %-14s %8.3f %-14s' % (
            arm, v['flicker']['mean'], str([round(x, 2) for x in v['flicker']['ci95']]),
            v['warp']['mean'], str([round(x, 2) for x in v['warp']['ci95']])))


if __name__ == '__main__':
    main()
