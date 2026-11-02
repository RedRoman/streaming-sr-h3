"""Is what is left of the range error noise, or is it structure?

guide_dose priced the remaining gap: on the 19 realistic clips the low band moves LPIPS by about 0.014 per
dB, and reaching InfVSR's 0.2183 needs roughly +0.9 dB over the 30.68 dB we have. Six single-frame training
arms are converging around +2.1 dB on val against the incumbent's +1.92, i.e. +0.2, so single-frame
capacity is not going to find +0.9. Temporal fusion could -- sensor and codec noise are largely independent
across frames while the scene is not -- but a K sweep already measured K=3/5/9 at +1.94/+1.95/+1.92 against
K=1's +1.98, a wash, which is either evidence that the error is not noise or evidence that an unaligned
channel-concat at 0.6M parameters cannot exploit it.

This separates those two readings without training anything. Alignment is only hard where there is motion,
so the test is run where there is none: pixels whose frame-to-frame difference is in the lowest decile.
There a plain box average over K frames IS the aligned average. If averaging the estimate over 5 such
frames buys several dB in those pixels, the residual is noise and the earlier sweep was an optimisation
failure; if it buys nothing, the residual is deterministic and no amount of temporal modelling will help.

Reported for the observation y as well, as the control: y's own error is known to contain the noise, so
y must show the gain if the measurement is sound.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
DATA = Path('/temp/zihaohe_range_pairs/val')
K = 5


def blur(x, sigma=2.0):
    r = max(1, int(round(3 * sigma)))
    a = torch.arange(-r, r + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (a / sigma) ** 2)
    g = (g / g.sum()).view(1, 1, 1, -1)
    y = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), g)
    return F.conv2d(F.pad(y, (0, 0, r, r), mode='reflect'), g.transpose(-1, -2))


def one(p):
    torch.set_num_threads(2)
    from range_branch import RangeBranch
    st = torch.load(ROOT / 'results/range_branch/k1.pt', map_location='cpu')
    rb = RangeBranch(**st['cfg']).eval(); rb.load_state_dict(st['state'])
    d = np.load(p)
    if json.loads(str(d['meta']))['degradation'].get('kind') != 'realistic':
        return None
    y = torch.from_numpy(d['y']).float().permute(0, 3, 1, 2) / 255
    t = torch.from_numpy(d['target']).float().permute(0, 3, 1, 2) / 255
    with torch.no_grad():
        e = rb(y[:, None]).clamp(0, 1)
    T = y.shape[0]
    h = K // 2
    out = {}
    # static mask from the observation's own frame differences, lowest decile over the whole window
    mo = blur(torch.cat([torch.zeros_like(y[:1, :1]),
                         (y[1:] - y[:-1]).abs().mean(1, keepdim=True)]))
    for tag, x in (('y', y), ('rphi', e)):
        s1, sk, n = 0.0, 0.0, 0.0
        for c in range(h, T - h):
            m = mo[c - h:c + h + 1].max(0).values            # static across the whole averaging window
            thr = torch.quantile(m.flatten(), 0.10)
            k = (m <= thr).float()
            avg = x[c - h:c + h + 1].mean(0, keepdim=True)
            e1 = ((x[c:c + 1] - t[c:c + 1]) ** 2).mean(1, keepdim=True)
            ek = ((avg - t[c:c + 1]) ** 2).mean(1, keepdim=True)
            s1 += float((e1 * k).sum()); sk += float((ek * k).sum()); n += float(k.sum())
        out[tag + '_single'] = float(10 * np.log10(1 / max(s1 / n, 1e-12)))
        out[tag + '_avg%d' % K] = float(10 * np.log10(1 / max(sk / n, 1e-12)))
        out[tag + '_gain'] = out[tag + '_avg%d' % K] - out[tag + '_single']
    out['file'] = Path(p).name
    return out


def main():
    fs = [str(x) for x in sorted(DATA.glob('*.npz'))]
    with Pool(24) as pool:
        rows = [r for r in pool.map(one, fs) if r]
    ks = [k for k in rows[0] if k != 'file']
    m = {k: float(np.mean([r[k] for r in rows])) for k in ks}
    (ROOT / 'results/range_branch/temporal_headroom.json').write_text(
        json.dumps({'n': len(rows), 'K': K, 'mean': m, 'per_window': rows}, indent=1))
    print('%d realistic val windows, static decile only, K=%d box average' % (len(rows), K))
    print()
    print('%-6s %10s %10s %8s' % ('arm', 'single', 'avg%d' % K, 'gain dB'))
    for tag in ('y', 'rphi'):
        print('%-6s %10.3f %10.3f %+8.3f'
              % (tag, m[tag + '_single'], m[tag + '_avg%d' % K], m[tag + '_gain']))
    print()
    print('If rphi gains little where y gains a lot, R_phi has already removed the independent part and '
          'the remainder is structural.')


if __name__ == '__main__':
    main()
