"""Per-degradation read-out for the range branch (mech7 judge 1), split the way mech5b was split.

The aggregate validation PSNR mixes two very different regimes. About a quarter of the clips are degraded
with the evaluation protocol's own Pillow-BICUBIC x4, so y is already within ~50 dB of the target and the
identity is nearly optimal there; the rest are blurred, noised and compressed, where the identity is poor.
Averaging the two hides whether the model helps where help is needed and hurts where it is not. mech5b
reported by kind for exactly this reason; this does the same for the trained branch.
"""
import json
from pathlib import Path

import numpy as np
import torch

from range_branch import RangeBranch, DATA, OUT


def psnr(a, b):
    return float(10 * torch.log10(1 / (a - b).square().mean().clamp_min(1e-12)))


@torch.no_grad()
def main():
    files = sorted((DATA / 'val').glob('*.npz'))[:24]
    items = []
    for p in files:
        d = np.load(p)
        kind = json.loads(str(d['meta']))['degradation'].get('kind', '?')
        items.append((kind, d['y'], d['target']))
    print(json.dumps({'val_files': len(items),
                      'kinds': {k: sum(1 for i in items if i[0] == k) for k in {i[0] for i in items}}}),
          flush=True)

    rows = {}
    for tag in ('k1', 'k3', 'k5', 'k9'):
        ck = OUT / f'{tag}.pt'
        if not ck.exists():
            continue
        st = torch.load(ck, map_location='cpu')
        m = RangeBranch(**st['cfg']).eval()
        m.load_state_dict(st['state'])
        k = st['cfg']['k']
        acc = {}
        for kind, y, t in items:
            T = y.shape[0]
            for i in range(k - 1, T, 4):
                s = torch.from_numpy(y[i - k + 1:i + 1]).float().permute(0, 3, 1, 2)[None] / 255
                g = torch.from_numpy(t[i]).float().permute(2, 0, 1)[None] / 255
                a = acc.setdefault(kind, {'model': [], 'base': []})
                a['model'].append(psnr(m(s), g))
                a['base'].append(psnr(s[:, -1], g))
        rows[tag] = {kd: {'model': float(np.mean(v['model'])), 'base': float(np.mean(v['base'])),
                          'gain': float(np.mean(v['model'])) - float(np.mean(v['base'])),
                          'n': len(v['model'])} for kd, v in acc.items()}
        allm = [x for v in acc.values() for x in v['model']]
        allb = [x for v in acc.values() for x in v['base']]
        rows[tag]['ALL'] = {'model': float(np.mean(allm)), 'base': float(np.mean(allb)),
                            'gain': float(np.mean(allm)) - float(np.mean(allb)), 'n': len(allm)}
        print(json.dumps({tag: {kd: {kk: round(vv, 3) for kk, vv in v.items()} for kd, v in rows[tag].items()}}),
              flush=True)
    (OUT / 'per_kind.json').write_text(json.dumps(rows, indent=1))


if __name__ == '__main__':
    main()
