"""Common perceptual evaluation of both fixed operator-family pilots."""
import argparse
import hashlib
import json
from pathlib import Path

import lpips
import numpy as np
import torch

from evaluate import ssim_rgb

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/h3_core/operator_perceptual'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(device):
    OUT.mkdir(parents=True, exist_ok=True)
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval().requires_grad_(False)
    rows, inputs = [], []
    with torch.inference_mode():
        for version in ['operator_function_v1', 'operator_function_dc_v1']:
            for source in ['101085', '101087']:
                path = ROOT / f'results/h3_core/{version}/fixed_seed17/val_{source}_readouts.pt'
                data = torch.load(path, map_location='cpu', weights_only=True)
                inputs.append({'version':version,'source':source,'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
                target = data['target_evaluation_only']
                for key, prediction in data['outputs'].items():
                    method, carrier = key.rsplit('_u', 1)
                    assert target.shape == prediction.shape == (1,3,256,256)
                    assert float(prediction.min()) >= 0 and float(prediction.max()) <= 1
                    assert torch.isfinite(prediction).all()
                    rows.append({'version':version, 'source':source, 'method':method, 'carrier':int(carrier),
                        'psnr':float(-10*torch.log10((prediction-target).square().mean())),
                        'ssim':ssim_rgb(prediction[0].numpy(),target[0].numpy()),
                        'stored_dtype':str(prediction.dtype),
                        'lpips':float(model(2*prediction.to(device=device,dtype=torch.float32)-1,
                                            2*target.to(device=device,dtype=torch.float32)-1).flatten()[0])})
    source_means, aggregate = [], []
    for version in sorted({r['version'] for r in rows}):
        for method in sorted({r['method'] for r in rows}):
            for source in ['101085','101087']:
                selected = [r for r in rows if r['version']==version and r['method']==method and r['source']==source]
                assert len(selected)==2
                source_means.append({'version':version,'method':method,'source':source,
                    **{k:float(np.mean([r[k] for r in selected])) for k in ['psnr','ssim','lpips']}})
            selected = [r for r in source_means if r['version']==version and r['method']==method]
            aggregate.append({'version':version,'method':method,'sources':len(selected),
                **{k:float(np.mean([r[k] for r in selected])) for k in ['psnr','ssim','lpips']}})
    weights = [Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',Path(lpips.__file__).parent/'weights/v0.1/alex.pth']
    result = {'status':'complete','code_sha256':sha(Path(__file__)),'inputs':inputs,
              'weights_sha256':{str(p):sha(p) for p in weights},'device':device,
              'torch':torch.__version__,'tf32_matmul':False,'tf32_cudnn':False,
              'protocol':'Full float RGB after the same pre-existing bounded64 far/box DC. PSNR/SSIM use stored precision; LPIPS consistently casts all inputs to float32, matching the fixed pretrained network. Average two carrier readouts within source, then equal average of two development validation sources. No fitting or choice between raw/DC versions.',
              'scope':'Conditional-response representation pilot after A+B arrival; not a causal compressed-before-cue memory experiment. Static repeated22 codec context. No motion or sealed confirmation.',
              'rows':rows,'source_means':source_means,'aggregate':aggregate}
    assert len(rows)==96
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps([r for r in aggregate if r['method'] in ['mixed','additive','joint_set_cnp','joint_robust_inverse','dense_conditional_H3_pushforward_not_posterior','dense_VAE_family_pushforward']],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');a=p.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    main(a.device)
