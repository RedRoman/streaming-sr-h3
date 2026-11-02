"""Gate B: is the range branch operator-agnostic, or was it fitted to x4?

Everything measured so far was under one operator, D = bicubic x4. The omni plan treats restoration tasks
as a family ordered by how much of the signal the operator leaves unobserved, and null_volume supported
that ordering -- but by re-decomposing x4 outputs at other split scales, never by training under a
different operator. If R_phi only works at x4 then the "framework" is a x4 method with extra steps.

Same architecture, same loss, same schedule as the shipped r2_hs; only the data changes. Three scales,
and the comparison that matters is not the absolute gain (the problems differ in difficulty) but whether
a positive gain over the observation survives the change of operator at all, and whether its size tracks
how much the operator destroys.

The crop has to shrink with the scale: at x8 the whole LR frame is 104x60, so a 112 crop does not exist.
Crop is therefore set to fit the smallest dimension, and reported, because it is a confound between arms.
"""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path('/cache/zihaohe/streaming_sr0')
CROP = {2: 112, 4: 112, 8: 48}          # x8 LR is 104x60; x2 is 416x240
DATA = {2: '/cache/zihaohe/range_pairs_s2', 4: '/temp/zihaohe_range_pairs', 8: '/cache/zihaohe/range_pairs_s8'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scales', default='2,8')
    ap.add_argument('--iters', type=int, default=8000)
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--kinds', default='realistic')
    a = ap.parse_args()
    for s in [int(x) for x in a.scales.split(',')]:
        tag = f'r2_s{s}'
        cmd = [sys.executable, 'train_range2.py', '--tag', tag, '--data', DATA[s],
               '--kinds', a.kinds, '--ch', '96', '--blocks', '12', '--iters', str(a.iters),
               '--batch', '32', '--crop', str(CROP[s]), '--device', a.device]
        print('RUN', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=str(ROOT / 'code'), check=False)
    print('GATE_B_TRAINING_DONE', flush=True)


if __name__ == '__main__':
    main()
