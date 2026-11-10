"""Is the VAE round-trip's temporal gain a manifold effect or just mild smoothing?

decode(encode(stream)) improved every temporal metric at a near-zero fidelity cost: flicker better on
59/69 clips, warp 68/69, p8.terr_null 64/69, against PSNR -0.017 and LPIPS +0.003 median
(results/deploy/judge_m0.json). That was read as evidence that the sampler latent carries content off the
VAE manifold which the projection removes -- and it is the motivating fact for the projection proposal.

A duller explanation fits the same observation: the round trip is a weak low-pass in time, and ANY weak
temporal smoothing improves those metrics while costing a little LPIPS. The two are only distinguishable
with a matched-cost control, which is what this writes: the shipped stream convolved in time with
[a, 1-2a, a] at three strengths, so the judge can find the one whose LPIPS cost matches the round trip's
and compare the temporal gain at equal cost.

If a 3-tap filter buys the same temporal improvement for the same fidelity, the manifold reading is not
supported by this evidence. If the round trip gets more temporal gain per unit of LPIPS, it is.
"""
import argparse, json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/heldout')
OUT = ROOT / 'results/deploy'


def job(args):
    name, src, strengths = args
    frames = sorted((src / name / 'stream').glob('*.png'))
    if not frames:
        return name, 0
    x = np.stack([np.asarray(Image.open(p).convert('RGB')).astype(np.float32) for p in frames])
    for a in strengths:
        d = OUT / f'tsmooth{int(a*1000):03d}' / name / 'stream'
        d.mkdir(parents=True, exist_ok=True)
        prev = np.concatenate([x[:1], x[:-1]])
        nxt = np.concatenate([x[1:], x[-1:]])
        y = a * prev + (1 - 2 * a) * x + a * nxt
        for p, im in zip(frames, y):
            Image.fromarray(np.uint8(im.clip(0, 255) + .5)).save(d / p.name)
    return name, len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', type=Path, default=ROOT / 'results/deploy_runs/t4_plain2')
    ap.add_argument('--procs', type=int, default=16)
    ap.add_argument('--strengths', default='0.05,0.10,0.15')
    a = ap.parse_args()
    st = [float(x) for x in a.strengths.split(',')]
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    names = [n for n in names if (a.src / n / 'stream').is_dir()]
    with Pool(a.procs) as pool:
        for name, n in pool.imap_unordered(job, [(c, a.src, st) for c in names]):
            print(json.dumps({'clip': name, 'frames': n}), flush=True)
    print(json.dumps({'written': len(names), 'arms': [f'tsmooth{int(x*1000):03d}' for x in st]}))


main()
