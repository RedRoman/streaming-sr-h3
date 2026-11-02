"""Re-score the oracle-trained arms with the oracle actually supplied.

train_range_sev.evaluate always calls model(x) with s=None, which routes through the severity head. For
the arms trained with --oracle-sev that head is never used in the forward pass during training -- it is
fit only by its own loss and reaches an MAE of 0.43 -- so those arms were trained on the true severity
and evaluated on a bad prediction they had never seen. Their reported pillow numbers (-13.5, -13.3) are
a train/test mismatch, not evidence about conditioning, and in particular they cannot be read as "perfect
information does not help".

This evaluates each checkpoint both ways on the same validation clips:
  predicted  what train_range_sev reported
  oracle     the true severity fed in, which is what the oracle arms were trained for
The oracle column is the upper bound Gate A actually wanted: if conditioning on a perfectly known
severity still cannot keep the clean regime non-negative, the mechanism is dead regardless of estimation.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))


@torch.no_grad()
def evaluate(model, data, dev, use_oracle, stride=4, chunk=8):
    acc = {}
    for yv, tv, k, s in zip(data.y, data.t, data.kind, data.s):
        idx = list(range(0, yv.shape[0], stride))
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = torch.from_numpy(yv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            g = torch.from_numpy(tv[sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            sv = torch.full((len(sel), 1), float(s), device=dev) if use_oracle else None
            o, _ = model(x, sv)
            d = acc.setdefault(k, {'m': [], 'b': []})
            for i in range(len(sel)):
                d['m'].append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                d['b'].append(float(10 * torch.log10(1 / (x[i] - g[i]).square().mean().clamp_min(1e-12))))
    return {k: {'gain_dB': float(np.mean(d['m']) - np.mean(d['b'])), 'n': len(d['m'])}
            for k, d in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tags', nargs='+')
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--device', default='npu:0')
    a = ap.parse_args()
    import torch_npu  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    from train_range_sev import SevRange, Data
    dev = torch.device(a.device)
    va = Data(a.data, 'val')
    print('%-8s %-10s %10s %10s' % ('tag', 'sev fed', 'realistic', 'pillow'))
    out = {}
    for t in a.tags:
        st = torch.load(ROOT / f'results/range_branch/{t}.pt', map_location='cpu')
        m = SevRange(**st['cfg']).eval().to(dev)
        m.load_state_dict(st['state'])
        for mode in ('predicted', 'oracle'):
            ev = evaluate(m, va, dev, use_oracle=(mode == 'oracle'))
            out[f'{t}.{mode}'] = ev
            print('%-8s %-10s %10.3f %10.3f'
                  % (t, mode, ev.get('realistic', {}).get('gain_dB', float('nan')),
                     ev.get('pillow_bicubic', {}).get('gain_dB', float('nan'))))
    (ROOT / 'results/range_branch/sev_oracle_eval.json').write_text(json.dumps(out, indent=1))
    print()
    print('Gate A wants both columns >= 0. The oracle row is the information upper bound.')


if __name__ == '__main__':
    main()
