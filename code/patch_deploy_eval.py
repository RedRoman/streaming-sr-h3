import re, sys
from pathlib import Path
p = Path('/cache/zihaohe/streaming_sr0/code/deploy_eval.py')
s = p.read_text()
if 'def _unsharp' in s:
    print('already patched'); sys.exit(0)

s = s.replace('''def _local_std(x, k=8):''', '''def _gauss1d(sigma):
    r = max(1, int(round(3 * sigma)))
    a = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (a / sigma) ** 2)
    return (g / g.sum()).view(1, 1, 1, -1)


def _unsharp(x, alpha, sigma=1.0):
    """x [H,W,3] -> x + alpha * (x - blur(x)), at LR resolution.

    R_phi is trained with L1 against D(x) and an L1-optimal estimate is a conditional median, so its output
    carries 28% less mid-frequency energy than the truth (cross_band.py). guide_sweep.py swept alpha with
    the sampled stream held fixed: 0.2-0.3 buys LPIPS and PSNR together on the realistic clips, and past
    0.6 the halos cost more than the energy is worth.
    """
    if not alpha:
        return x
    t = x.permute(2, 0, 1)[None]
    g = _gauss1d(sigma).expand(3, 1, 1, -1)
    r = g.shape[-1] // 2
    b = F.conv2d(F.pad(t, (r, r, 0, 0), mode='reflect'), g, groups=3)
    b = F.conv2d(F.pad(b, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2), groups=3)
    return (x + alpha * (x - b[0].permute(1, 2, 0))).clamp(0, 1)


def _local_std(x, k=8):''')

s = s.replace("    name, src, dst, ckpt, s0, p = job", "    name, src, dst, ckpt, s0, p, sharp = job")
s = s.replace("""            est = rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0)
            guide = up(est, hw).clamp(0, 1)""",
"""            est = _unsharp(rb(stack.permute(0, 3, 1, 2)[None])[0].permute(1, 2, 0), sharp)
            guide = up(est, hw).clamp(0, 1)""")
s = s.replace("""        {'clip': name, 'op': 'quota', 'src': str(src), 'ckpt': str(ckpt), 's0': s0, 'p': p,""",
"""        {'clip': name, 'op': 'quota', 'src': str(src), 'ckpt': str(ckpt), 's0': s0, 'p': p, 'sharp': sharp,""")
s = s.replace("""    p.add_argument('--p', type=float, default=1.0)""",
"""    p.add_argument('--p', type=float, default=1.0)
    p.add_argument('--sharp', type=float, default=0.0, help='unsharp alpha on R_phi(y); see _unsharp')""")
s = s.replace("""        jobs = [(n, a.src, OUT / a.tag, a.ckpt, a.s0, a.p) for n in names]""",
"""        jobs = [(n, a.src, OUT / a.tag, a.ckpt, a.s0, a.p, a.sharp) for n in names]""")
p.write_text(s)
print('patched')
