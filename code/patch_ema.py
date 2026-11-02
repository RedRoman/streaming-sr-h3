"""Smooth the correction, not the content.

The temporal weakness is in the low band, not the generated detail. Split by band on the 77 held-out
realistic clips, interior frames only, against InfVSR:

  warp_null   ratio 0.88, we are worse on 32/71  -> our generated detail is the MORE stable one
  warp_range  ratio 1.20, we are worse on 49/71  -> the low band is what flickers along motion

The low band is U(R_phi(y_t)) and R_phi is K=1, a per-frame network with no temporal coupling at all. The
K sweep that chose K=1 only ever compared accuracy (+0.938 dB for K=1 against +0.896 for K=5); temporal
stability was never part of that decision, so this is a property we picked without measuring it.

The fix follows from what is and is not constant. The degradation parameters are drawn once per clip, so
the correction R_phi(y) - y is a function of an operator that does not change, while y itself carries all
the real motion. So the correction is smoothed over time and the observation is not:

    est_t = y_t + ema_t,  ema_t = (1-a) * ema_{t-1} + a * (R_phi(y_t) - y_t)

A causal EMA, which a stream can run. alpha = 1 reproduces the current behaviour exactly.
"""
import sys
from pathlib import Path

p = Path('/cache/zihaohe/streaming_sr0/code/deploy_eval.py')
s = p.read_text()
if 'res_ema' in s:
    print('already patched'); sys.exit(0)

old = """    name, src, dst, ckpt, s0, p, sharp, gain = job"""
new = """    name, src, dst, ckpt, s0, p, sharp, gain, ema = job"""
assert old in s
s = s.replace(old, new, 1)

# precompute the estimate for every frame, smooth the correction, and index it in the loop
old2 = """    lr = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
    n = 0
    with torch.no_grad():
        for t in range(T):"""
new2 = """    lr = torch.stack([load(D_ / name / 'lr' / f'{t:06d}.png') for t in range(T)])
    res_ema = None
    if K == 1:
        # one batched call for the whole clip, then a causal EMA on the correction only
        with torch.no_grad():
            est_all = rb(lr.permute(0, 3, 1, 2)[:, None]).clamp(0, 1).permute(0, 2, 3, 1)
        if ema < 1.0:
            res = est_all - lr
            acc = res[0].clone()
            out_res = [acc.clone()]
            for t in range(1, T):
                acc = (1 - ema) * acc + ema * res[t]
                out_res.append(acc.clone())
            est_all = (lr + torch.stack(out_res)).clamp(0, 1)
        res_ema = est_all
    n = 0
    with torch.no_grad():
        for t in range(T):"""
assert old2 in s
s = s.replace(old2, new2, 1)

old3 = """            est = _unsharp(rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0), sharp)"""
new3 = """            est = _unsharp(res_ema[t] if res_ema is not None
                           else rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0), sharp)"""
assert old3 in s
s = s.replace(old3, new3, 1)

s = s.replace("""         'sharp': sharp, 'gain': gain,""",
              """         'sharp': sharp, 'gain': gain, 'res_ema_alpha': ema,""", 1)
s = s.replace("""    p.add_argument('--gain', type=float, default=1.0, help='multiplier on the admitted detail')""",
              """    p.add_argument('--gain', type=float, default=1.0, help='multiplier on the admitted detail')
    p.add_argument('--ema', type=float, default=1.0,
                   help='causal EMA on the correction R_phi(y)-y; 1.0 is the current per-frame behaviour')""", 1)
s = s.replace("""        jobs = [(n, a.src, OUT / a.tag, a.ckpt, a.s0, a.p, a.sharp, a.gain) for n in names]""",
              """        jobs = [(n, a.src, OUT / a.tag, a.ckpt, a.s0, a.p, a.sharp, a.gain, a.ema) for n in names]""", 1)
p.write_text(s)
import ast
ast.parse(s)
print('patched, syntax ok')
