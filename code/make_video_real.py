"""Side-by-side video for the real-degradation sets, where the automatic judges are weakest.

Two reasons this exists rather than reusing make_compare_video2.py: that script hard-codes the held-out
arms and requires ground truth, and neither holds here. VideoLQ has no reference at all, and MVSR4x has
one but a different set of arms.

It also exists because of what was measured about no-reference scoring. A NIQE-family metric fitted on
the held-out set's own ground truth cannot separate FlashVSR+ from the ground truth (median 3.01 against
2.98), and OmniVR's own table reports its method above clean ground truth on four of six NR metrics. On
a set with no reference there is therefore no automatic judge worth quoting, and the frames are the
deliverable. On MVSR4x there IS a reference, but the PSNR ordering there is dominated by registration
error -- every generative arm falls below a near-no-op bicubic -- so the frames matter there too.

Crops are chosen by a rule, not by eye: the window with the most energy the assumed x4 operator would
discard, i.e. where restoration is being asked to do something. Clip order is also a rule when a judge
file is supplied -- worst-first by the reference arm's LPIPS, so the cases that argue against us lead.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))


def load(p):
    return np.asarray(Image.open(p).convert('RGB'))


def arm_frame(root, D_, name, t, hw):
    if root == 'bicubic':
        lr = torch.from_numpy(load(D_ / name / 'lr' / f'{t:06d}.png')).float().permute(2, 0, 1)[None] / 255
        return (F.interpolate(lr, size=hw, mode='bicubic', align_corners=False)
                .clamp(0, 1)[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    if root == 'gt':
        p = D_ / name / 'hr' / f'{t:06d}.png'
        return load(p) if p.exists() else None
    for p in (Path(root) / name / 'stream' / f'{t:06d}.png', Path(root) / name / f'{t:06d}.png'):
        if p.exists():
            im = Image.open(p).convert('RGB')
            if im.size != (hw[1], hw[0]):
                im = im.resize((hw[1], hw[0]), Image.BICUBIC)
            return np.asarray(im)
    return None


def detail_window(img, crop):
    """Top-left of the crop holding the most energy an x4 operator would remove."""
    t = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(max(1, h // 4), max(1, w // 4)), mode='bicubic',
                      align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    n = (t - u).square().mean(1)[0]
    best, by, bx = -1.0, 0, 0
    for y in range(0, max(1, h - crop + 1), 16):
        for x in range(0, max(1, w - crop + 1), 16):
            e = float(n[y:y + crop, x:x + crop].mean())
            if e > best:
                best, by, bx = e, y, x
    return by, bx


def label(img, text, sub=None):
    im = Image.fromarray(np.ascontiguousarray(img))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    d.text((8, 4), text, fill=(255, 255, 255))
    if sub:
        d.text((8, 19), sub, fill=(170, 220, 170))
    return np.asarray(im)


def grid(tiles, cols):
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.concatenate([np.concatenate(tiles[r * cols:(r + 1) * cols], axis=1)
                           for r in range(rows)], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-set', required=True)
    ap.add_argument('--arm', action='append', required=True, help='label=path | label=bicubic | label=gt')
    ap.add_argument('--judge', default='', help='optional judge json to order clips and label tiles')
    ap.add_argument('--ref', default='', help='arm key in the judge used to rank clips worst-first')
    ap.add_argument('--clips', type=int, default=3)
    ap.add_argument('--cols', type=int, default=3)
    ap.add_argument('--crop', type=int, default=256)
    ap.add_argument('--fps', type=int, default=24)
    ap.add_argument('--crf', type=int, default=16)
    ap.add_argument('--out', default='results/compare_real')
    a = ap.parse_args()

    D_ = ROOT / a.eval_set
    man = json.loads((D_ / 'manifest.json').read_text())
    names = man['clips']
    arms = [tuple(x.split('=', 1)) for x in a.arm]
    out = ROOT / a.out / Path(a.eval_set).name
    out.mkdir(parents=True, exist_ok=True)

    lp = {}
    if a.judge:
        j = json.loads((ROOT / a.judge).read_text()) if not Path(a.judge).is_absolute() \
            else json.loads(Path(a.judge).read_text())
        for r in j['per_clip']:
            lp[r['clip']] = {k: v.get('lpips') for k, v in r['arms'].items() if v}
        if a.ref:
            names = sorted([n for n in names if n in lp and lp[n].get(a.ref) is not None],
                           key=lambda n: -lp[n][a.ref])
    names = names[:a.clips]
    print(json.dumps({'eval_set': a.eval_set, 'clips': names, 'arms': [k for k, _ in arms],
                      'ordered_by': (a.ref or 'manifest order')}, indent=1), flush=True)

    import av
    for name in names:
        m = json.loads((D_ / name / 'meta.json').read_text())
        T, hw = m['frames'], (m['hr_wh'][1], m['hr_wh'][0])
        ref0 = arm_frame('gt', D_, name, 0, hw)
        if ref0 is None:
            ref0 = arm_frame('bicubic', D_, name, 0, hw)
        cy, cx = detail_window(ref0, a.crop)
        for sheet in ('full', 'zoom'):
            z = sheet == 'zoom'
            th, tw = (a.crop, a.crop) if z else hw
            dst = out / f'{name}_{sheet}.mp4'
            c = av.open(str(dst), mode='w')
            st = c.add_stream('libx264', rate=a.fps)
            rows = (len(arms) + a.cols - 1) // a.cols
            st.width, st.height = a.cols * tw, rows * th
            st.pix_fmt = 'yuv420p'
            st.options = {'crf': str(a.crf)}
            n_ok = 0
            for t in range(T):
                tiles = []
                for key, path in arms:
                    img = arm_frame(path, D_, name, t, hw)
                    if img is None:
                        img = np.zeros((hw[0], hw[1], 3), np.uint8)
                    if z:
                        img = img[cy:cy + a.crop, cx:cx + a.crop]
                    sub = None
                    if name in lp and lp[name].get(key) is not None:
                        sub = 'LPIPS %.4f' % lp[name][key]
                    tiles.append(label(img, key, sub))
                for pkt in st.encode(av.VideoFrame.from_ndarray(grid(tiles, a.cols), format='rgb24')):
                    c.mux(pkt)
                n_ok += 1
            for pkt in st.encode():
                c.mux(pkt)
            c.close()
            print(json.dumps({'clip': name, 'sheet': sheet, 'frames': n_ok,
                              'canvas': [rows * th, a.cols * tw], 'crop_yx': [cy, cx] if z else None,
                              'out': str(dst), 'MB': round(dst.stat().st_size / 1e6, 1)}), flush=True)


if __name__ == '__main__':
    main()
