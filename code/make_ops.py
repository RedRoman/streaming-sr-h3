"""Build range-branch pairs for operators that are not downsampling.

The three scales tested so far are one operator at three strengths: D is always a bicubic downsample, so
the unobserved subspace is always "the high frequencies", and a model could transfer across them by
learning that one structure. The omni claim needs operators whose null space is a different object:

  sr4       D = bicubic downsample x4   null = 15/16 of the spectrum
  deblur    D = Gaussian blur sigma     null = the band the kernel attenuates
  denoise   D = identity                null = EMPTY -- the projection is the whole answer and a
                                        generative prior has nothing to contribute

Denoise is the interesting end: null_volume predicted that what generation can buy falls to ~0.019 LPIPS
as the unobserved volume goes to zero, so a framework built on range/null should degrade gracefully into
"just use the estimate" rather than break.

Everything is synthesised from the existing x4 pairs, which already hold a degraded observation y and the
clean target D(x) at 208x120. Taking that clean target as the image to be corrupted keeps resolution,
clip identity and the train/val split identical across operators, so the only thing that varies between
arms is D itself. No video is re-read.
"""
import os
import argparse, hashlib, json
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

# the x4 pairs these are derived from; the big set has 13672 windows against 782
SRC = os.environ.get('MAKE_OPS_SRC', '/temp/zihaohe_range_pairs')


def jpeg(frames, q):
    out = []
    for f in frames:
        ok, enc = cv2.imencode('.jpg', cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                               [cv2.IMWRITE_JPEG_QUALITY, int(q)])
        out.append(cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return np.stack(out)


def add_noise(frames, sigma, rng):
    x = frames.astype(np.float32)
    x = x + rng.standard_normal(x.shape, dtype=np.float32) * sigma
    return np.clip(x + 0.5, 0, 255).astype(np.uint8)


def blur(frames, sigma):
    return np.stack([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames])


def one(job):
    p, op, out_dir, sig_range = job
    d = np.load(p)
    meta = json.loads(str(d['meta']))
    if meta['degradation'].get('kind') != 'realistic':
        return 0
    x = d['target']                                   # the clean image at 208x120
    # sha256, not hash(): Python salts str hashing per process (PYTHONHASHSEED), so the original
    # expression drew a different kernel for the same window on every regeneration and on every host.
    # The same bug was found and fixed in prepare_audio_pairs.py; this is the other call site.
    _h = hashlib.sha256(('%s|%s|%s' % (Path(p).stem, op, sig_range)).encode()).digest()[:4]
    rng = np.random.default_rng(int.from_bytes(_h, 'little'))
    if op == 'denoise':
        tgt = x                                       # D = identity: nothing is unobserved
        sig = float(rng.uniform(3, 15))
        y = add_noise(x, sig, rng)
        rec = {'op': 'denoise', 'noise_sigma': sig}
    elif op == 'deblur':
        bs = float(rng.uniform(*sig_range))
        tgt = blur(x, bs)                             # D = the kernel; the range is what survives it
        sig = float(rng.uniform(2, 10))
        y = add_noise(tgt, sig, rng)
        rec = {'op': 'deblur', 'blur_sigma': bs, 'noise_sigma': sig}
    else:
        raise ValueError(op)
    if rng.random() < 0.6:
        q = int(rng.integers(40, 95))
        y = jpeg(y, q); rec['jpeg_q'] = q
    o = Path(out_dir) / Path(p).name
    np.savez_compressed(o, y=y, target=tgt,
                        meta=json.dumps({'id': meta['id'],
                                         'degradation': {'kind': 'realistic', **rec}}))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--op', required=True, choices=['denoise', 'deblur'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--procs', type=int, default=32)
    ap.add_argument('--sigma-lo', type=float, default=1.0)
    ap.add_argument('--sigma-hi', type=float, default=2.5,
                    help='Gaussian sigma range for --op deblur. The trained arms saw U(1.0, 2.5); '
                         'generating a set OUTSIDE that range is the test a one-hot structurally cannot '
                         'pass, since its single deblur slot carries the same vector for every kernel '
                         'while an MTF reports the actual rolloff. Unlike the in-vocabulary comparison '
                         'this measures extrapolation, so it is not confounded by training-set size.')
    ap.add_argument('--splits', default='train,val')
    a = ap.parse_args()
    sig_range = (a.sigma_lo, a.sigma_hi)
    total = {}
    for split in [s for s in a.splits.split(',') if s]:
        d = Path(a.out) / split
        d.mkdir(parents=True, exist_ok=True)
        fs = [str(p) for p in sorted((Path(SRC) / split).glob('*.npz'))]
        with Pool(a.procs) as pool:
            n = sum(pool.map(one, [(p, a.op, str(d), sig_range) for p in fs]))
        total[split] = n
        print(json.dumps({'op': a.op, 'split': split, 'written': n, 'from': len(fs),
                          'sigma_range': list(sig_range)}), flush=True)
    print(json.dumps({'op': a.op, 'out': a.out, 'total': total, 'done': True}), flush=True)


if __name__ == '__main__':
    main()
