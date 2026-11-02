"""Real-ESRGAN on every frame of the 24-clip set, not only the 32 the fidelity pass sampled.

baseline_realesrgan.py saved frames (u, u+1) for u in range(0, T-1, 8), which is enough for PSNR / LPIPS
and for the stride-8 temporal pairs, but not for the seam pairs (frames 21/22, 38/39, ...). Without those
the baseline cannot be scored on the same temporal table as the streaming arms, and the seam column has to
be left empty for it. Same net, same weights, same preprocessing; frames only, no metrics.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
OUT = ROOT / 'results/baseline_frames_full'
W = ROOT / 'assets/baselines/RealESRGAN_x4.pth'


def work(name):
    torch.set_num_threads(6)
    from baseline_realesrgan import RRDBNet
    sd = torch.load(W, map_location='cpu', weights_only=True)
    sd = sd.get('params_ema', sd.get('params', sd))
    net = RRDBNet().eval()
    miss, unexp = net.load_state_dict(sd, strict=False)
    assert not miss and not unexp
    T = json.loads((D_ / name / 'meta.json').read_text())['frames']
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    n = 0
    with torch.no_grad():
        for t in range(T):
            p = d / f'{t:06d}.png'
            if p.exists():
                n += 1
                continue
            lr = torch.from_numpy(np.asarray(Image.open(D_ / name / 'lr' / f'{t:06d}.png').convert('RGB')).copy())
            y = net(lr.float().div(255).permute(2, 0, 1)[None]).clamp(0, 1)
            Image.fromarray(np.uint8(y[0].permute(1, 2, 0).numpy() * 255 + .5)).save(p)
            n += 1
    return name, n


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    names = json.loads((D_ / 'manifest.json').read_text())['clips']
    with Pool(int(os.environ.get('PROCS', 24))) as pool:
        for name, n in pool.imap_unordered(work, names):
            print(json.dumps({'clip': name, 'frames': n}), flush=True)
    print(json.dumps({'finished': True, 'out': str(OUT)}), flush=True)
