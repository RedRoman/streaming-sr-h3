"""Score any range-branch checkpoint on the shared validation split with train_range2's own evaluate(),
so a new arm and the shipped k1 are read off the same ruler. Prints per-degradation-kind PSNR, the
identity's PSNR, and the HR-space mid-frequency ratio that cross_band showed to be the actual defect."""
import argparse, json, sys
from pathlib import Path

import torch

ROOT = Path('/cache/zihaohe/streaming_sr0')
sys.path.insert(0, str(ROOT / 'code'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt', nargs='+')
    ap.add_argument('--data', default='/temp/zihaohe_range_pairs')
    ap.add_argument('--device', default='npu:0')
    a = ap.parse_args()
    if a.device.startswith('npu'):
        import torch_npu  # noqa: F401
        from h3_comfy import patch_module_to, set_precision
        patch_module_to(); set_precision()
    from train_range2 import Data, evaluate
    from range_branch import RangeBranch
    dev = torch.device(a.device)
    va = Data(a.data, 'val', 'all')
    print('%-14s %-16s %7s %7s %7s %9s' % ('ckpt', 'kind', 'PSNR', 'base', 'gain', 'mid_ratio'))
    for c in a.ckpt:
        st = torch.load(c, map_location='cpu')
        m = RangeBranch(**st['cfg']).eval().to(dev)
        m.load_state_dict(st['state'])
        ev = evaluate(m, va, dev, k=st['cfg']['k'])
        for k, v in sorted(ev.items()):
            print('%-14s %-16s %7.3f %7.3f %+7.3f %9.4f'
                  % (Path(c).stem, k, v['psnr'], v['base'], v['gain_dB'], v['mid_ratio']))
        print('%-14s %-16s %7s %7s %7s %9.4f'
              % ('', '(identity)', '', '', '', ev['realistic']['mid_ratio_y']))


if __name__ == '__main__':
    main()
