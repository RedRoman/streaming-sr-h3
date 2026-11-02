"""Is the operator partly identifiable, even though the whole code is not?

A CNN reading raw frames converges to a val MAE of 0.254 against the per-dimension median's 0.252, and
accumulating over 21 frames instead of 1 changes nothing (0.256 -> 0.255). As a single number that reads
as a flat no. But the code mixes quantities of very different natures -- two blur widths, an anisotropy
flag, a noise flag, a second-order flag, an H.264 CRF, a JPEG quality and a codec indicator -- and some of
them are conditionally defined, so a flat average can hide that the additive parts are recoverable while
the convolutional ones are not.

That distinction is what the omni plan actually needs: condition on what is identifiable, and treat the
rest as unknown. So the comparison is made per dimension, against the same trivial predictor, on the same
validation clips.
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))
NAMES = ['blur_sigma_x', 'blur_sigma_y', 'anisotropic', 'noise', 'second_order',
         'h264_crf', 'jpeg_q', 'codec_kind']


def main():
    import torch_npu  # noqa: F401
    from h3_comfy import patch_module_to, set_precision
    patch_module_to(); set_precision()
    from op_cnn import OpCNN, load
    dev = torch.device('npu:0')
    ys, ps, ks = load('val')
    st = torch.load(ROOT / 'results/operator_estimator/opcnn.pt', map_location='cpu')
    m = OpCNN(**st['cfg']).eval().to(dev)
    m.load_state_dict(st['state'])
    th = np.stack(ps)
    med = np.median(th, 0)

    pred = []
    with torch.no_grad():
        for yv in ys:
            n = yv.shape[0] - 1
            outs = []
            for s0 in range(0, n, 16):
                idx = range(s0, min(s0 + 16, n))
                x = torch.from_numpy(np.stack([yv[t:t + 2] for t in idx])).float()
                outs.append(m(x.permute(0, 1, 4, 2, 3).to(dev) / 255))
            pred.append(torch.cat(outs).mean(0).cpu().numpy())
    pred = np.stack(pred)

    print('%-14s %7s %9s %9s %8s %9s' % ('dimension', 'std', 'median MAE', 'CNN MAE', 'gain', 'spearman'))
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    for i, nm in enumerate(NAMES):
        mb = float(np.abs(th[:, i] - med[i]).mean())
        mc = float(np.abs(th[:, i] - pred[:, i]).mean())
        sp = float(np.corrcoef(rk(th[:, i]), rk(pred[:, i]))[0, 1]) if th[:, i].std() > 1e-6 else float('nan')
        print('%-14s %7.3f %9.4f %9.4f %+8.1f%% %9.3f'
              % (nm, th[:, i].std(), mb, mc, 100 * (mb - mc) / max(mb, 1e-9), sp))
    print()
    print('%-14s %7s %9.4f %9.4f' % ('ALL', '', float(np.abs(th - med).mean()),
                                     float(np.abs(th - pred).mean())))
    (ROOT / 'results/operator_estimator/op_dims.json').write_text(json.dumps(
        {'names': NAMES, 'median_mae': [float(np.abs(th[:, i] - med[i]).mean()) for i in range(8)],
         'cnn_mae': [float(np.abs(th[:, i] - pred[:, i]).mean()) for i in range(8)]}, indent=1))


if __name__ == '__main__':
    main()
