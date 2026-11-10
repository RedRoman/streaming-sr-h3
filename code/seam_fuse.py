"""Fuse block handovers in the null component only, then restore the range by projection.

The measurement that motivates this: across a block handover our temporal error is 0.2899 in the range
component and 0.9936 in the null, against 0.1281 and 0.3177 inside a block -- 2.3x and 3.1x worse. Both
published streaming baselines are flat across the same boundary (FlashVSR+ is actually better at the
seam, 0.1421/0.2373 against its own interior 0.1983/0.3522), so this is not what streaming costs, it is
what our assembly costs. The cause is that adjacent emitted frames come from independent DiT samples:
the 5-frame overlap is fed to encode_guide as an LR condition, and the previous block's OUTPUT never
enters the next block at all.

The standard remedy in the literature is overlap-and-blend, and the literature also records why it
disappoints. InfVSR reports that blending improves pixel-level smoothness but not semantic consistency;
Morphe reports that naive pixel blending across segment boundaries "often worsens artifacts and
blurring" and moves the fix into training instead.

Blurring is the right thing to worry about, and it is specifically a range-component problem: averaging
two estimates of the OBSERVED content attenuates it, because the two differ by phase and noise rather
than by content. The generated component is different -- there is no single correct null, so averaging
two plausible ones is a legitimate choice of a third.

So this fuses only the null component, and then puts the range back where the observation says it
should be:

    fused = R_x4( w*A + (1-w)*B )  replaced by the projected range,  plus  w*N(A) + (1-w)*N(B)

in effect: take the null components from both blocks, cross-fade them over the overlap, and pair them
with a range component taken from a single block rather than from an average. Nothing is averaged that
the observation determines, which is the part blending normally damages.

Both halves are already on disk. run_deploy writes every block's full 22 frames to block{k:03d}/ and
only the new span into the stream, so the overlap exists twice and no resampling is needed.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
SCALE = 4


def load(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def split(x, s=SCALE):
    """Range and null component of an [H,W,3] frame under the x4 bicubic operator."""
    t = x.permute(2, 0, 1)[None].double()
    h, w = t.shape[-2:]
    d = F.interpolate(t, size=(h // s, w // s), mode='bicubic', align_corners=False, antialias=True)
    u = F.interpolate(d, size=(h, w), mode='bicubic', align_corners=False)
    r = u[0].permute(1, 2, 0).float()
    return r, x - r


def weights(n, kind):
    """Cross-fade profile over n overlap frames, from 'all previous' to 'all current'."""
    if n <= 0:
        return torch.zeros(0)
    t = torch.linspace(0, 1, n + 2)[1:-1]
    if kind == 'linear':
        return t
    if kind == 'cosine':
        return 0.5 - 0.5 * torch.cos(np.pi * t)
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='a deploy_runs tag directory holding block*/ frames')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--eval-set', default='data/real_eval/heldout')
    ap.add_argument('--profile', default='cosine', choices=['linear', 'cosine'])
    ap.add_argument('--range-from', default='current', choices=['current', 'previous', 'average'],
                    help="which block supplies the range component in the overlap. 'average' is the "
                         "behaviour this file argues against and is kept so the claim is testable.")
    ap.add_argument('--clips', type=int, default=0)
    ap.add_argument('--procs', type=int, default=12)
    a = ap.parse_args()

    D_ = ROOT / a.eval_set
    src = ROOT / a.src if not Path(a.src).is_absolute() else Path(a.src)
    dst = ROOT / 'results/deploy' / a.tag
    man = json.loads((D_ / 'manifest.json').read_text())
    blocks = man['blocks']
    names = [c for c in man['clips']
             if json.loads((D_ / c / 'meta.json').read_text())['degradation'].get('kind') == 'realistic']
    if a.clips:
        names = names[:a.clips]

    from multiprocessing import Pool

    def jobs():
        for n in names:
            yield (n, str(src), str(dst), str(D_), blocks, a.profile, a.range_from)

    with Pool(a.procs) as pool:
        for name, nf, ns in pool.imap_unordered(_one, list(jobs())):
            print(json.dumps({'clip': name, 'frames': nf, 'seams_fused': ns}), flush=True)
    print()
    print(json.dumps({'tag': a.tag, 'clips': len(names), 'profile': a.profile,
                      'range_from': a.range_from, 'out': str(dst)}, indent=1))


def _one(job):
    name, src, dst, D_, blocks, profile, range_from = job
    torch.set_num_threads(2)
    src, dst, D_ = Path(src), Path(dst), Path(D_)
    meta = json.loads((D_ / name / 'meta.json').read_text())
    T = meta['frames']
    od = dst / name / 'stream'
    od.mkdir(parents=True, exist_ok=True)

    def blk(k, f):
        p = src / name / f'block{k:03d}' / f'{f:06d}.png'
        return load(p) if p.exists() else None

    out = {}
    seams = 0
    for bi, b in enumerate(blocks):
        k, (s, e) = b['k'], b['range']
        nl, nh = b['new']
        # the overlap this block shares with its predecessor: frames [s, nl)
        ov = list(range(s, min(nl, e)))
        for f in range(nl, min(nh, T)):
            cur = blk(k, f)
            if cur is not None:
                out[f] = cur
        if bi == 0 or not ov:
            continue
        w = weights(len(ov), profile)
        prev_k = blocks[bi - 1]['k']
        for i, f in enumerate(ov):
            A = blk(prev_k, f)      # the predecessor's estimate, which the stream currently keeps
            B = blk(k, f)           # this block's estimate, currently discarded
            if A is None or B is None:
                continue
            ra, na = split(A)
            rb, nb = split(B)
                                    # cross-fade the generated component only
            nf = (1 - w[i]) * na + w[i] * nb
            if range_from == 'current':
                rr = rb
            elif range_from == 'previous':
                rr = ra
            else:
                rr = (1 - w[i]) * ra + w[i] * rb
            out[f] = (rr + nf).clamp(0, 1)
            seams += 1

    n = 0
    for f in sorted(out):
        if f >= T:
            continue
        x = out[f]
        if x.dtype != torch.uint8:
            x = (x.clamp(0, 1) * 255 + 0.5).to(torch.uint8)
        Image.fromarray(x.numpy()).save(od / f'{f:06d}.png')
        n += 1
    (dst / name / 'stream_result.json').write_text(json.dumps(
        {'clip': name, 'op': 'seam_fuse', 'src': str(src), 'profile': profile,
         'range_from': range_from, 'frames': n, 'seam_frames_fused': seams}, indent=1))
    return name, n, seams


if __name__ == '__main__':
    main()
