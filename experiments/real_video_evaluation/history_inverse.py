"""Fixed causal LR multi-frame inverse baseline around frozen H3 VAE output.

Known numerical operators use estimated LR-only DIS geometry. Classical ridge
inversion and output DC are controls, not a new memory architecture.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/real_video'
OUT = ROOT / 'results/real_video_history_inverse'
sys.path.insert(0, str(ROOT / 'src'))
from pixel_consistency import BicubicConsistency


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def cases():
    return [c for c in json.loads((ROOT / 'results/real_video_vae/input_manifest.json').read_text())['cases']
            if c['resolution'] == '256' and c['split'] in ['train', 'val']]


def prepare():
    import cv2
    from video_correspondence import pair_correspondence, grid
    cv2.setNumThreads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'history': 4, 'cg_iterations': 24, 'ridge': .01, 'dc_iterations': 25,
                'flow_photo_sigma': .05, 'flow_cycle_sigma': 1., 'flow': 'LR-only DIS medium direct pair',
                'splits': ['train', 'val'], 'frame_indices': list(range(22)), 'spatial_context': '256 fixed ROI',
                'fit_or_search': False, 'evaluation_unit': 'six distinct source videos, not132 independent frames',
                'waiting': 'Same pretrained VAE block0..21 for all outputs; history residual uses <=current index only',
                'scope': 'Classical multi-frame inverse control; no learned memory or long-gap retrieval',
                'cases': cases()}
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    for case in protocol['cases']:
        folder = OUT / case['id']; folder.mkdir(exist_ok=True)
        observed = [np.asarray(Image.open(DATA / p).convert('RGB')) for p in case['lr_paths']]
        hashes = [sha(DATA / p) for p in case['lr_paths']]
        assert hashes == case['observed_lr_sha256']
        maps, confidence, pairs = [], [], []
        for current in range(22):
            for past in range(max(0, current - protocol['history']), current):
                # Target grid is the PAST observation, sampling the CURRENT candidate.
                coord, weight, _ = pair_correspondence(observed[current], observed[past], .05, 1.)
                maps.append(coord - grid(*coord.shape[:2])); confidence.append(weight)
                pairs.append([current, past])
        np.savez_compressed(folder / 'geometry.npz', displacement=np.stack(maps),
                            confidence=np.stack(confidence), pairs=np.array(pairs, dtype=np.int64))
        (folder / 'geometry_provenance.json').write_text(json.dumps({
            'lr_sha256': hashes, 'geometry_sha256': sha(folder / 'geometry.npz'),
            'no_hr_or_annotation_input': True, 'coordinates': 'past LR pixel centers -> current LR pixel centers',
            'opencv': cv2.__version__, 'pair_count': len(pairs)}, indent=2))
        print(json.dumps({'prepared': case['id'], 'pairs': len(pairs)}), flush=True)


class Observation:
    def __init__(self, displacement, weight, operator):
        self.operator = operator
        h, w = operator.wy.shape[1], operator.wx.shape[1]
        lh, lw = displacement.shape[:2]
        dis = torch.as_tensor(displacement, device=operator.w.device, dtype=operator.w.dtype).permute(2, 0, 1)[None]
        dis = F.interpolate(dis, size=(h, w), mode='bilinear', align_corners=False)
        yy, xx = torch.meshgrid(torch.arange(h, device=dis.device, dtype=dis.dtype),
                               torch.arange(w, device=dis.device, dtype=dis.dtype), indexing='ij')
        gx = 2 * (xx + dis[0, 0] * w/lw + .5) / w - 1
        gy = 2 * (yy + dis[0, 1] * h/lh + .5) / h - 1
        self.grid = torch.stack([gx, gy], -1)[None]
        self.weight = torch.as_tensor(weight, device=dis.device, dtype=dis.dtype)[None]

    def forward(self, value):
        warped = F.grid_sample(value[None], self.grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]
        return self.operator.down(warped)

    def transpose(self, value):
        with torch.enable_grad():
            zero = torch.zeros(3, self.operator.wy.shape[1], self.operator.wx.shape[1],
                               device=value.device, dtype=value.dtype, requires_grad=True)
            return torch.autograd.grad(self.forward(zero), zero, value, create_graph=False)[0]


def null(value, operator):
    return value - operator.ry @ operator.down(value) @ operator.rx.T


def solve(base, observed, observations, operator, iterations=24, ridge=.01):
    if not observations:
        return base.clone(), {'iterations': 0, 'relative_normal_residual': 0.}
    rhs = torch.zeros_like(base)
    for obs, target in zip(observations, observed):
        rhs += obs.transpose(obs.weight * (target - obs.forward(base)))
    rhs = null(rhs, operator)

    def normal(value):
        constrained = null(value, operator)
        out = ridge * value
        for obs in observations:
            out += null(obs.transpose(obs.weight * obs.forward(constrained)), operator)
        return out

    x = torch.zeros_like(rhs); residual = rhs.clone(); direction = residual.clone()
    old = residual.square().sum(); initial = old.sqrt().item()
    used = 0
    for used in range(iterations):
        if old.item() < 1e-18:
            break
        product = normal(direction)
        denom = (direction * product).sum()
        assert denom.item() > 0 and torch.isfinite(denom)
        alpha = old / denom
        x = x + alpha * direction; residual = residual - alpha * product
        new = residual.square().sum()
        direction = residual + (new / old) * direction; old = new
    result = base + null(x, operator)
    return result, {'iterations': used + 1, 'relative_normal_residual': old.sqrt().item() / max(initial, 1e-12),
                    'current_lr_change_rmse': (operator.down(result-base).square().mean().sqrt().item())}


def audit(device):
    torch.manual_seed(19)
    op = BicubicConsistency(32, 8, dtype=torch.float64, device=device)
    displacement = np.zeros((8, 8, 2)); displacement[..., 0] = .35
    obs = Observation(displacement, np.ones((8, 8)), op)
    x = torch.randn(3, 32, 32, dtype=torch.float64, device=device)
    y = torch.randn(3, 8, 8, dtype=torch.float64, device=device)
    lhs = (obs.forward(x) * y).sum(); rhs = (x * obs.transpose(y)).sum()
    error = float((lhs-rhs).abs())
    truth = .2 + .6 * torch.rand_like(x); current = op.down(truth)
    base = op.affine_project(.5 * torch.ones_like(truth), current)
    measured = obs.forward(truth)
    estimate, info = solve(base, [measured], [obs], op)
    before = float((obs.forward(base)-measured).square().mean())
    after = float((obs.forward(estimate)-measured).square().mean())
    unchanged, _ = solve(base, [], [], op)
    assert error < 1e-10 and after < before and torch.equal(unchanged, base)
    assert info['current_lr_change_rmse'] < 1e-12
    return {'adjoint_inner_product_error': error, 'new_observation_mse_before': before,
            'new_observation_mse_after': after, 'no_history_exact_noop': True, **info, 'passed': True}


def inference(device):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    (OUT / 'operator_audit.json').write_text(json.dumps(audit(device), indent=2))
    op = BicubicConsistency(256, 64, dtype=torch.float32, device=device)
    final_op = BicubicConsistency(256, 64, device=device)
    for case in protocol['cases']:
        folder = OUT / case['id']; prov = json.loads((folder/'geometry_provenance.json').read_text())
        assert sha(folder/'geometry.npz') == prov['geometry_sha256']
        assert [sha(DATA/p) for p in case['lr_paths']] == prov['lr_sha256']
        lr = torch.stack([load(DATA/p) for p in case['lr_paths']]).to(device)
        vae_folder = ROOT / 'results/real_video_vae' / case['id']
        native = json.loads((vae_folder/'inference.json').read_text())
        assert [sha(vae_folder/f'{i:04d}.png') for i in range(22)] == native['output_sha256']
        vae = torch.stack([load(vae_folder/f'{i:04d}.png') for i in range(22)]).to(device)
        cubic = (F.interpolate(lr, size=(256, 256), mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)*255).round()/255
        arrays = np.load(folder/'geometry.npz'); rows=[]; outputs={k:[] for k in ['bicubic_dc', 'vae_dc', 'inverse_bicubic_dc', 'inverse_vae_dc']}
        for t in range(22):
            selected = np.flatnonzero(arrays['pairs'][:, 0] == t)
            operators = [Observation(arrays['displacement'][i], arrays['confidence'][i], op) for i in selected]
            targets = [lr[int(arrays['pairs'][i, 1])] for i in selected]
            for label, initial in [('bicubic', cubic[t]), ('vae', vae[t])]:
                base, _ = final_op.project(initial, lr[t], iterations=25)
                outputs[label+'_dc'].append(base.cpu())
                start = time.perf_counter()
                candidate, info = solve(base, targets, operators, op, 24, .01)
                projected, residual = final_op.project(candidate, lr[t], iterations=25)
                if device.startswith('cuda'): torch.cuda.synchronize()
                outputs['inverse_'+label+'_dc'].append(projected.cpu())
                rows.append({'frame':t,'prior':label,'past_observations':len(targets),
                             'mean_confidence':float(np.mean(arrays['confidence'][selected])) if len(selected) else 0.,
                             'seconds':time.perf_counter()-start,'final_lr_rmse':float(residual),**info})
        torch.save({k:torch.stack(v) for k,v in outputs.items()},folder/'outputs.pt')
        (folder/'inference.json').write_text(json.dumps({'case':case['id'],'source':case['sequence'],
            'status':'complete','hr_read':False,'source_sha256':sha(Path(__file__)),
            'output_sha256':sha(folder/'outputs.pt'),'geometry_sha256':prov['geometry_sha256'],
            'rows':rows},indent=2))
        print(json.dumps({'inference_complete':case['id'],'frames':22,'methods':4}),flush=True)


def evaluate(device):
    import lpips
    from evaluate import ssim_rgb, comparison_panel
    model = lpips.LPIPS(net='alex',version='0.1',verbose=False).to(device).eval().requires_grad_(False)
    summaries=[]
    for case in cases():
        folder=OUT/case['id']; proof=json.loads((folder/'inference.json').read_text())
        assert sha(folder/'outputs.pt')==proof['output_sha256']
        methods=torch.load(folder/'outputs.pt',map_location='cpu',weights_only=True)
        hr=torch.stack([load(DATA/p) for p in case['hr_paths']]);rows=[]
        with torch.inference_mode():
            for name,pred in methods.items():
                for start in range(0,22,2):
                    distances=model(2*pred[start:start+2].to(device)-1,2*hr[start:start+2].to(device)-1).flatten().cpu().tolist()
                    for offset,distance in enumerate(distances):
                        i=start+offset
                        rows.append({'frame':i,'method':name,'psnr':float(-10*torch.log10((pred[i]-hr[i]).square().mean())),
                                     'ssim':ssim_rgb(pred[i].numpy(),hr[i].numpy()),'lpips':distance})
        summary={'case':case['id'],'split':case['split'],'source':case['sequence'],
                 'methods':{name:{key:float(np.mean([r[key] for r in rows if r['method']==name])) for key in ['psnr','ssim','lpips']} for name in methods}}
        (folder/'evaluation.json').write_text(json.dumps(summary,indent=2));(folder/'per_frame.json').write_text(json.dumps(rows,indent=2))
        comparison_panel(hr,methods,list(methods),[0,11,21],folder/'comparison.png')
        summaries.append(summary);print(json.dumps(summary),flush=True)
    (OUT/'summary.json').write_text(json.dumps({'sources':summaries,'protocol':json.loads((OUT/'protocol.json').read_text()),
        'limitations':['Classical estimated-geometry inverse, no novel learned core','Past4 LR frames only; not persistent long-gap retrieval','VAE still usesblock0..21; not zero-lookahead','Sixsource development evaluation; no confirmation-cohort access']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['prepare','audit','infer','evaluate']);parser.add_argument('--device',default='cpu');args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.phase=='prepare':prepare()
    elif args.phase=='audit':print(json.dumps(audit(args.device),indent=2))
    elif args.phase=='infer':inference(args.device)
    else:evaluate(args.device)
