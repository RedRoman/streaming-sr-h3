"""Is the severity signal absent, or just miscalibrated?

Four estimator variants all land on a near-constant prediction: the per-kind means separate by only
0.05-0.13 where the truth separates by 0.79, and adding temporal frames makes it worse, not better. But
the training loss does fall (L1 0.41 -> 0.18, BCE 0.68 -> 0.50), so the models are fitting something.
Mean absolute error conflates two failures that need different fixes:

  no information   the prediction does not order the clips at all -> AUC ~ 0.5, and the whole
                   severity-conditioning route is dead for the clean regime
  miscalibration   the prediction orders clips correctly but on the wrong scale -> AUC high, and an
                   affine or isotonic rescale recovers it for free

So this reports, per checkpoint: AUC separating realistic from clean, Spearman against the true severity
within the realistic family, and the MAE achievable after the best affine rescale fitted on validation
(an optimistic bound, stated as such, because the point is to upper-bound what calibration could buy).
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (pos.size * neg.size))


@torch.no_grad()
def predict(model, data, dev, frames, stride=8, chunk=8):
    out = []
    for yv in data.y:
        idx = list(range(frames - 1, yv.shape[0], stride))
        ps = []
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = torch.from_numpy(np.stack([yv[t - frames + 1:t + 1] for t in sel])).float()
            ps.append(model(x.permute(0, 1, 4, 2, 3).to(dev) / 255).sigmoid())
        out.append(float(torch.cat(ps).mean()))
    return np.array(out)


def main():
    import torch_npu  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    from sev_est import Est, Data
    dev = torch.device('npu:0')
    va = Data('/temp/zihaohe_range_pairs', 'val')
    s = np.array(va.s)
    kind = np.array(va.kind)
    real, pill = kind == 'realistic', kind != 'realistic'
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    print('val: %d realistic, %d pillow; true severity %.3f vs %.3f'
          % (real.sum(), pill.sum(), s[real].mean(), s[pill].mean()))
    print()
    print('%-11s %7s %9s %11s %11s' % ('ckpt', 'AUC', 'rho(real)', 'MAE raw', 'MAE affine'))
    res = {}
    for tag in sys.argv[1:]:
        st = torch.load(ROOT / f'results/range_branch/{tag}.pt', map_location='cpu')
        m = Est(**st['cfg']).eval().to(dev)
        m.load_state_dict(st['state'])
        p = predict(m, va, dev, st['cfg']['frames'])
        a = auc(p[real], p[pill])
        rho = float(np.corrcoef(rk(p[real]), rk(s[real]))[0, 1])
        A = np.stack([p, np.ones_like(p)], 1)
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        res[tag] = {'auc': a, 'rho_realistic': rho, 'mae_raw': float(np.abs(p - s).mean()),
                    'mae_affine': float(np.abs(A @ coef - s).mean()),
                    'pred_real': float(p[real].mean()), 'pred_pill': float(p[pill].mean())}
        print('%-11s %7.3f %9.3f %11.3f %11.3f'
              % (tag, a, rho, res[tag]['mae_raw'], res[tag]['mae_affine']))
    (ROOT / 'results/range_branch/sev_auc.json').write_text(json.dumps(res, indent=1))
    print()
    print('AUC 1.0 = clean clips always score below degraded ones. The affine column is fitted ON the')
    print('validation set, so it is an optimistic bound on what recalibration alone could buy.')


if __name__ == '__main__':
    main()
