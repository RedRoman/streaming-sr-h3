"""Stage-7 decode-domain evaluation: does the residual parameterisation actually produce better pixels?

The training signal says yes by a wide margin -- validation velocity MSE 0.0830 with a clean range estimate
against 0.1359 with the raw observation, a 39% gap that widened through training. But velocity MSE is a
proxy, and this project's first lesson was that a proxy in the wrong domain can be monotone the wrong way:
the LBH latent upscaler scored a closer latent distance and decoded 8.4 dB worse. So the claim only counts
after decoding.

Three arms, because the oracle-trained LoRA needs a clean guide at inference and we do not have one:

  raw          raw-trained LoRA + U(y)            deployable today, the Stage-1 equivalent
  oracle       oracle-trained LoRA + U(D x_GT)    the bound; not deployable, D(x) is unknown at test time
  rphi         oracle-trained LoRA + U(R_phi(y))  what would actually ship, once R_phi is good enough

The gap between `rphi` and `oracle` is exactly how much the range branch still owes.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from h3_comfy import H3, ROOT, to_u8

D_ = ROOT / os.environ.get('EVAL_SET', 'data/real_eval/const')
OUT = ROOT / 'results/stage7_eval'
H, W = 480, 832


def load_png(p):
    return torch.from_numpy(np.asarray(Image.open(p).convert('RGB')).copy()).float() / 255


def eval_down(hr):
    t = hr.permute(0, 3, 1, 2).double()
    return F.interpolate(t, size=(H // 4, W // 4), mode='bicubic', align_corners=False,
                         antialias=True).clamp(0, 1).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=['raw', 'oracle', 'rphi'])
    ap.add_argument('--lora', type=Path, required=True)
    ap.add_argument('--rphi', type=Path, default=ROOT / 'results/stage7_range/r7sup.pt')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    a = ap.parse_args()
    out_root = OUT / os.environ.get("ARM_TAG", a.arm)
    out_root.mkdir(parents=True, exist_ok=True)
    man = json.loads((D_ / 'manifest.json').read_text())
    blocks = man['blocks']
    h3 = H3(load_dit=True, dit='fl2va', lora_path=a.lora, lora_rank=32)

    rb = None
    if a.arm == 'rphi':
        import sys
        sys.path.insert(0, str(ROOT / 'code'))
        from train_range7 import RangeBranch7, evidence
        st = torch.load(a.rphi, map_location='cpu')
        rb = RangeBranch7().eval(); rb.load_state_dict(st['state'])

    clips = [c for i, c in enumerate(man['clips']) if i % a.nshards == a.shard]
    for name in clips:
        base = out_root / name
        if (base / 'stream_result.json').exists():
            continue
        meta = json.loads((D_ / name / 'meta.json').read_text())
        T = meta['frames']
        lr = torch.stack([load_png(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)]).permute(0, 3, 1, 2)
        if a.arm == 'oracle':
            hr = torch.stack([load_png(D_ / name / 'hr' / f'{t:06d}.png') for t in range(T)])
            src = eval_down(hr)                                   # the clean low-res the bound assumes
        elif a.arm == 'rphi':
            with torch.no_grad():
                pr = torch.stack([lr[t - 1:t + 1] for t in range(1, T)])
                ev = evidence(pr).mean(0, keepdim=True)
                src = torch.cat([rb(lr[t:t + 1], ev)[0] for t in range(T)])
        else:
            src = lr
        stream, recs = {}, []
        for b in blocks:
            k, (s, e) = b['k'], b['range']
            t0 = time.perf_counter()
            guide = h3.encode_guide(src[s:e], [H, W])
            sampled, rec = h3.sample(guide, 'innerdc', low=lr[s:e], hw=[H, W], seed=101)
            px = h3.decode(sampled['video'])
            u8 = to_u8(px)
            lo = b['new'][0]
            for i, f in enumerate(range(s, e)):
                if f >= lo:
                    stream[f] = u8[i]
            recs.append({'block': k, 'sample_s': rec['seconds'], 'total_s': time.perf_counter() - t0})
        sd = base / 'stream'; sd.mkdir(parents=True, exist_ok=True)
        for f, im in sorted(stream.items()):
            Image.fromarray(im).save(sd / f'{f:06d}.png')
        (base / 'stream_result.json').write_text(json.dumps(
            {'clip': name, 'arm': a.arm, 'lora': str(a.lora), 'blocks': recs,
             'frames_emitted': sorted(stream)}, indent=1))
        print(json.dumps({'done': name, 'arm': a.arm,
                          'sec': round(sum(r['total_s'] for r in recs), 1)}), flush=True)
    print(json.dumps({'arm': a.arm, 'shard': a.shard, 'finished': True}), flush=True)


if __name__ == '__main__':
    main()
