"""Common float-output perceptual evaluation of fixed connection pilot readouts."""
import argparse
import hashlib
import json
from pathlib import Path

import lpips
import numpy as np
import torch

from evaluate import ssim_rgb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/h3_core/connection_perceptual'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(device):
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval().requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)
    rows, inputs = [], []
    with torch.inference_mode():
        for source in ['101085', '101087']:
            path = ROOT / f'results/h3_core/connection_pilot_v1/val_{source}/eval_readouts.pt'
            data = torch.load(path, map_location='cpu', weights_only=True)
            inputs.append({'source': source, 'path': str(path.relative_to(ROOT)), 'sha256': sha(path)})
            target = data['target_evaluation_only']
            candidates = [(k, 'AB', data[k]) for k in ['reference', 'classical_AB', 'H3_response_mean_AB']]
            candidates += [(method, order, values[order]) for method, values in data['outputs'].items()
                           for order in ['AB', 'BA']]
            for method, order, prediction in candidates:
                assert target.shape == prediction.shape == (1, 3, 256, 256)
                assert float(prediction.min()) >= 0 and float(prediction.max()) <= 1
                rows.append({'source': source, 'method': method, 'order': order,
                    'psnr_full': float(-10 * torch.log10((prediction - target).square().mean())),
                    'ssim_full': ssim_rgb(prediction[0].numpy(), target[0].numpy()),
                    'lpips_full': float(model(2 * prediction.to(device) - 1, 2 * target.to(device) - 1).flatten()[0])})
    by_source = []
    for source in ['101085', '101087']:
        for method in sorted({r['method'] for r in rows}):
            selected = [r for r in rows if r['source'] == source and r['method'] == method]
            by_source.append({'source': source, 'method': method,
                              **{key: float(np.mean([r[key] for r in selected]))
                                 for key in ['psnr_full', 'ssim_full', 'lpips_full']}})
    means = {method: {key: float(np.mean([r[key] for r in by_source if r['method'] == method]))
                      for key in ['psnr_full', 'ssim_full', 'lpips_full']}
             for method in sorted({r['method'] for r in rows})}
    result = {'status': 'complete', 'source_sha256': sha(Path(__file__)), 'inputs': inputs,
              'protocol': 'Full RGB float outputs after identical existing bounded64-far/box DC. Each source has equal weight; learned AB/BA orders each half. No model or protocol selection.',
              'lpips': 'alex v0.1; same validated evaluation package/weights as all real-video runs',
              'scope': '2 development validation source images, repeated static native22 context. No motion, no project-new confirmation. Perceptual score does not establish identity preservation.',
              'rows': rows, 'by_source': by_source, 'source_means': means}
    (OUT / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(means, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cpu'); a = p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main(a.device)
