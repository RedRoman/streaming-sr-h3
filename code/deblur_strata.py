"""Does the deblur result survive being split by blur sigma, and was MTF ever given the sigma at all?

The omni arms train on deblur pairs whose kernel is drawn per sample, sigma ~ U(1.0, 2.5). The two
conditioning signals should not be equivalent there: a one-hot has a single 'deblur' slot and cannot say
which kernel, while an MTF is a measurement of U(D(.)) and changes with sigma -- ||MTF(1.0) - MTF(2.5)||
is 0.8851, with band 1 reading 0.83 against 0.33. Deblur should be the cleanest demonstration in the
project, cleaner than x6: on x6 the one-hot raises SystemExit and is visibly undefined, while on a
different sigma it silently proceeds with the wrong condition.

It is instead the operator where MTF trails one-hot the most (-0.144 dB at x2/x4/x8/deblur/denoise of
-0.030/-0.032/-0.072/-0.144/-0.014). This script exists because that inversion has an obvious candidate
cause: mtf_cond caches on the operator STRING, every deblur sample passes the bare 'deblur', and
operator_desc falls back to sigma=1.75. So every deblur sample in training, whatever its kernel, was
handed one identical vector -- MTF degenerated to a higher-dimensional one-hot carrying a prior that is
wrong for all but one sigma.

Three arms, no retraining, so this separates "the idea does not work" from "the idea was never run":

  trained      the condition the checkpoint was trained with, bare 'deblur' -> sigma 1.75 always
  persample    the SAME weights, but conditioned at inference on each sample's true sigma
  onehot       the one-hot checkpoint, which has no sigma to give either way

If persample beats trained, the mapping from MTF to behaviour did survive and only the conditioning was
degenerate, which makes a retrain worth the cards. If persample is worse, the network bound that one
vector as the identity of 'deblur' and reads anything else as a different operator -- also an answer, and
it would mean the retrain has to happen before any deblur claim can be made at all.

Per sigma bin, because the pooled mean is what hid this.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path('/cache/zihaohe/streaming_sr0')
OUT = ROOT / 'results/omni'


def load_pairs(data_dir, split, limit):
    """Pairs plus the degradation record, which Multi drops but which is the whole question here."""
    items = []
    for p in sorted((Path(data_dir) / split).glob('*.npz')):
        if limit and len(items) >= limit:
            break
        d = np.load(p)
        meta = json.loads(str(d['meta']))
        deg = meta['degradation']
        if deg.get('kind') != 'realistic' or deg.get('blur_sigma') is None:
            continue
        items.append({'y': d['y'], 't': d['target'], 'sigma': float(deg['blur_sigma'])})
    return items


@torch.no_grad()
def score(model, items, cond_fn, dev, k, stride=8, chunk=4):
    """PSNR of the model and of the identity, per item."""
    out = []
    for it in items:
        idx = list(range(0, it['y'].shape[0], stride))
        c = torch.tensor([cond_fn(it)], dtype=torch.float32, device=dev)
        ms, bs = [], []
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            st = np.stack([[max(0, t - j) for j in range(k - 1, -1, -1)] for t in sel])
            x = torch.from_numpy(it['y'][st]).float().permute(0, 1, 4, 2, 3).to(dev) / 255
            g = torch.from_numpy(it['t'][sel]).float().permute(0, 3, 1, 2).to(dev) / 255
            o = model(x, c.expand(len(sel), -1))
            for i in range(len(sel)):
                ms.append(float(10 * torch.log10(1 / (o[i] - g[i]).square().mean().clamp_min(1e-12))))
                bs.append(float(10 * torch.log10(1 / (x[i, -1] - g[i]).square().mean().clamp_min(1e-12))))
        out.append({'sigma': it['sigma'], 'psnr': float(np.mean(ms)), 'base': float(np.mean(bs)),
                    'gain_dB': float(np.mean(ms) - np.mean(bs))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mtf-ckpt', default='omni_mtf_k3')
    ap.add_argument('--onehot-ckpt', default='omni_onehot_k3')
    ap.add_argument('--data', default='/cache/zihaohe/range_pairs_deblurbig')
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--bins', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='results/omni/deblur_strata.json')
    a = ap.parse_args()

    dev = torch.device(a.device)
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
    from omni_stage1 import COND_DIM, OPS, OmniRange, mtf_cond, severity_of
    from operator_desc import operator_mtf

    items = load_pairs(a.data, a.split, a.limit)
    for it in items:
        it['sev'] = severity_of(it['y'][0], it['t'][0])
    sig = np.array([it['sigma'] for it in items])
    print(json.dumps({'items': len(items), 'sigma_min': float(sig.min()),
                      'sigma_max': float(sig.max())}), flush=True)

    def load(spec):
        p = Path(spec) if '/' in spec else OUT / f'{spec}.pt'
        st = torch.load(p, map_location='cpu')
        cfg = dict(st['cfg'])
        cfg.pop('omni', None); cfg.pop('ops', None)
        mode = cfg.pop('cond', 'onehot')
        k = int(cfg.get('k', 1))
        cfg.setdefault('cdim', COND_DIM[mode])
        m = OmniRange(**cfg).eval()
        m.load_state_dict(st['state'])
        return m.to(dev), mode, k

    m_mtf, mode_mtf, k_mtf = load(a.mtf_ckpt)
    m_oh, mode_oh, k_oh = load(a.onehot_ckpt)
    assert mode_mtf == 'mtf' and mode_oh == 'onehot', (mode_mtf, mode_oh)

    # the three conditionings
    def c_trained(it):
        return mtf_cond('deblur', it['sev'], False)              # bare -> sigma 1.75, always

    def c_persample(it):
        v = [float(x) for x in operator_mtf('deblur:%.4f' % it['sigma'])]
        return v + [0.0]                                          # use_sev=False, as the arms were eval'd

    def c_onehot(it):
        v = [0.0] * (len(OPS) + 1)
        v[OPS.index('deblur')] = 1.0
        return v

    arms = {'trained': (m_mtf, c_trained, k_mtf),
            'persample': (m_mtf, c_persample, k_mtf),
            'onehot': (m_oh, c_onehot, k_oh)}
    res = {}
    for tag, (m, fn, k) in arms.items():
        res[tag] = score(m, items, fn, dev, k)
        print(json.dumps({'arm': tag, 'mean_gain_dB': round(
            float(np.mean([r['gain_dB'] for r in res[tag]])), 4)}), flush=True)

    # bins of equal count, so each row has the same weight
    order = np.argsort(sig)
    edges = [sig[order[int(i * len(order) / a.bins)]] for i in range(a.bins)] + [float(sig.max()) + 1e-6]
    out = {'data': a.data, 'split': a.split, 'items': len(items), 'bins': a.bins,
           'edges': [float(e) for e in edges], 'arms': {}}
    for tag in arms:
        rows = res[tag]
        out['arms'][tag] = {'overall_gain_dB': float(np.mean([r['gain_dB'] for r in rows])),
                            'per_item': rows, 'bins': []}
        for b in range(a.bins):
            lo, hi = edges[b], edges[b + 1]
            g = [r['gain_dB'] for r in rows if lo <= r['sigma'] < hi]
            out['arms'][tag]['bins'].append(
                {'lo': float(lo), 'hi': float(hi), 'n': len(g),
                 'gain_dB': float(np.mean(g)) if g else None})
    (ROOT / a.out).write_text(json.dumps(out, indent=1))

    print()
    print('gain over the degraded input, dB. Bins hold equal counts of blur sigma.')
    hdr = '%-12s %10s' % ('arm', 'overall')
    for b in range(a.bins):
        hdr += '  %12s' % ('s %.2f-%.2f' % (edges[b], edges[b + 1]))
    print(hdr)
    for tag in ('trained', 'persample', 'onehot'):
        line = '%-12s %10.4f' % (tag, out['arms'][tag]['overall_gain_dB'])
        for bb in out['arms'][tag]['bins']:
            line += '  %12s' % (('%.4f' % bb['gain_dB']) if bb['gain_dB'] is not None else '-')
        print(line)
    print()
    t = out['arms']['trained']['overall_gain_dB']
    p = out['arms']['persample']['overall_gain_dB']
    o = out['arms']['onehot']['overall_gain_dB']
    print('persample - trained = %+.4f dB   (>0: the MTF->behaviour mapping survived a degenerate'
          ' training condition)' % (p - t))
    print('trained   - onehot  = %+.4f dB   (the published deblur delta, reproduced here)' % (t - o))
    print('wrote', ROOT / a.out)


if __name__ == '__main__':
    main()
