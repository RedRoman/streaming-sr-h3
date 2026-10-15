"""Source-aligned H3 video evaluation plus a classical LR projection control."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import importlib.metadata
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
import torch
from torch.nn import functional as F
import lpips

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pixel_consistency import BicubicConsistency, audit


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2, 0, 1).float()/255


def save(tensor, path):
    Image.fromarray(np.uint8(tensor.permute(1, 2, 0).numpy().clip(0, 1)*255+.5)).save(path)


def ssim_rgb(x, y):
    # Standard 11x11/sigma1.5 population-moment SSIM, valid interior, RGB mean.
    def smooth(a): return gaussian_filter(a, sigma=(0, 1.5, 1.5), truncate=3.5)
    ux, uy = smooth(x), smooth(y)
    vx, vy = smooth(x*x)-ux*ux, smooth(y*y)-uy*uy
    cov = smooth(x*y)-ux*uy
    value = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    return float(value[:, 5:-5, 5:-5].mean())


def comparison_panel(hr, methods, chosen, indices, path):
    """Show identical frames at native size without selectively cropping outputs."""
    h, w = hr.shape[-2:]
    panel = Image.new('RGB', (w*(len(chosen)+1), (h+26)*len(indices)), 'white')
    draw = ImageDraw.Draw(panel)
    for row, i in enumerate(indices):
        for col, (name, value) in enumerate([('HR evaluation', hr[i])]+[(n, methods[n][i]) for n in chosen]):
            draw.text((col*w+4, row*(h+26)+5), f'{name} / t={i}', fill='black')
            panel.paste(Image.fromarray(np.uint8(value.permute(1, 2, 0).numpy().clip(0, 1)*255+.5)), (col*w, row*(h+26)+26))
    panel.save(path)


def warp(previous, absolute_lr):
    h, w = previous.shape[-2:]; lh, lw = absolute_lr.shape[:2]
    yy, xx = torch.meshgrid(torch.arange(lh), torch.arange(lw), indexing='ij')
    base = torch.stack((xx, yy), -1).float()
    flow = (torch.from_numpy(absolute_lr).float()-base).permute(2, 0, 1)[None]
    flow = F.interpolate(flow, (h, w), mode='bilinear', align_corners=False)[0]
    flow *= torch.tensor([w/lw, h/lh])[:, None, None]
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    gx = (xx+flow[0]+.5)*2/w-1; gy = (yy+flow[1]+.5)*2/h-1
    return F.grid_sample(previous[None], torch.stack((gx, gy), -1)[None],
                         mode='bilinear', padding_mode='zeros', align_corners=False)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='bmx-trees'); p.add_argument('--split', default='test')
    p.add_argument('--frames', type=int, default=22)
    p.add_argument('--run', type=Path, default=ROOT/'results/h3_setup/dit_bf16_neutral_davis22_normal49')
    p.add_argument('--out', type=Path, default=ROOT/'results/real_video_evaluation/bmx_trees22')
    p.add_argument('--input-manifest', type=Path)
    p.add_argument('--case-id')
    p.add_argument('--flow', type=Path)
    args = p.parse_args(); args.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    audits = audit(); (args.out/'projection_audit.json').write_text(json.dumps(audits, indent=2))
    dataset = ROOT/'data/real_video'
    if args.input_manifest:
        manifest = json.loads(args.input_manifest.read_text())
        window = next(v for v in manifest['cases'] if v['id']==args.case_id)
        window = dict(window, source_video_id='DAVIS/'+window['sequence'])
        assert window['split']==args.split and window['sequence']==args.source
    else:
        manifest = json.loads((dataset/'h3_windows_39.json').read_text())
        window = next(v for v in manifest['windows'] if v['sequence']==args.source and v['split']==args.split)
    source_lr = [dataset/path for path in window['lr_paths'][:args.frames]]
    source_hr = [dataset/path for path in window['hr_paths'][:args.frames]]
    lr = torch.stack([load(path) for path in source_lr]); hr = torch.stack([load(path) for path in source_hr])
    # Verify exact observed inputs of the existing H3 execution before pairing GT.
    encoded = json.loads((args.run/'encode_result.json').read_text())
    hashes = list(encoded['observed_input_sha256'].values())
    assert hashes == [hashlib.sha256(path.read_bytes()).hexdigest() for path in source_lr]
    assert encoded['actual_latest_distinct_observation_index'] == args.frames-1
    original = {'bicubic': F.interpolate(lr, size=hr.shape[-2:], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1),
                'h3_vae_only': torch.stack([load(args.run/f'guide_roundtrip_{i:02d}.png') for i in range(args.frames)]),
                'h3_bf16': torch.stack([load(args.run/f'output_{i:02d}.png') for i in range(args.frames)])}
    op = BicubicConsistency(tuple(hr.shape[-2:]), tuple(lr.shape[-2:])); projections = {}; runtime = {}
    for name, value in original.items():
        begin = time.perf_counter(); result, residual = op.project(value, lr)
        runtime[name] = {'seconds': time.perf_counter()-begin, 'max_float_operator_lr_rmse': float(residual.max())}
        projections[name+'_dc'] = result
    methods = {**original, **projections}
    flow_path = args.flow or ROOT/f'results/video_correspondence/{args.split}/{args.source}/lr_correspondence.npz'
    flow = np.load(flow_path)
    flow_provenance = json.loads((flow_path.parent/'provenance.json').read_text())
    assert flow_provenance['lr_sha256'][:args.frames]==hashes
    assert tuple(flow['pair_confidence'].shape[1:])==tuple(lr.shape[-2:])
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().requires_grad_(False)
    # Record pretrained network files and hashes; no evaluation weight fitting.
    netpath = Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth'
    calpath = Path(lpips.__file__).parent/'weights/v0.1/alex.pth'
    weights = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (netpath, calpath)}
    rows=[]; summary={}
    with torch.no_grad():
        for name, output in methods.items():
            distances=[]
            for start in range(0, len(output), 4):
                distances += model(2*output[start:start+4]-1, 2*hr[start:start+4]-1).flatten().tolist()
            folder=args.out/name; folder.mkdir(exist_ok=True)
            for i, prediction in enumerate(output):
                x, y = prediction.numpy(), hr[i].numpy()
                pillow = Image.fromarray(np.uint8(prediction.permute(1,2,0).numpy().clip(0,1)*255+.5)).resize((lr.shape[-1],lr.shape[-2]), Image.Resampling.BICUBIC)
                low = torch.from_numpy(np.asarray(pillow).copy()).permute(2,0,1).float()/255
                high_x = x-gaussian_filter(x,sigma=(0,1.2,1.2))
                high_y = y-gaussian_filter(y,sigma=(0,1.2,1.2))
                row={'method':name, 'frame':i, 'psnr_rgb':float(-10*torch.log10((prediction-hr[i]).square().mean().clamp_min(1e-12))),
                     'ssim_rgb':ssim_rgb(x,y), 'lpips_alex_v01':float(distances[i]),
                     'lr_pillow_rmse':float((low-lr[i]).square().mean().sqrt()),
                     'highpass_error_rmse':float(np.sqrt(np.mean((high_x-high_y)**2))),
                     'highpass_energy_ratio':float(np.mean(high_x**2)/max(np.mean(high_y**2),1e-12))}
                if i:
                    coordinates=flow['current_to_previous'][i]
                    predicted_delta=prediction-warp(output[i-1],coordinates)
                    true_delta=hr[i]-warp(hr[i-1],coordinates)
                    confidence=F.interpolate(torch.from_numpy(flow['pair_confidence'][i])[None,None],prediction.shape[-2:],mode='bilinear',align_corners=False)[0,0]
                    mask=confidence>=.5
                    row['temporal_error_excess_rmse']=float((predicted_delta-true_delta)[:,mask].square().mean().sqrt()) if mask.any() else None
                    row['temporal_mask_coverage']=float(mask.float().mean())
                rows.append(row); save(prediction,folder/f'{i:04d}.png')
            keys=['psnr_rgb','ssim_rgb','lpips_alex_v01','lr_pillow_rmse','highpass_error_rmse','highpass_energy_ratio','temporal_error_excess_rmse','temporal_mask_coverage']
            summary[name]={key:float(np.mean([r[key] for r in rows if r['method']==name and r.get(key) is not None])) for key in keys}
            print(json.dumps({'method':name,**summary[name]}),flush=True)
    metadata={'source_video':window['source_video_id'],'split':args.split,'frames':args.frames,
              'methods':summary,'projection':runtime,'projection_iterations':25,'projection_type':'Classical separable floating-AA bicubic affine + RGB box Dykstra; no HR fitting',
              'input_sha_verified':True,'perceptual_weights_sha256':weights,'lpips_package':importlib.metadata.version('lpips'),
              'shape_hw':list(hr.shape[-2:]),'flow_input_sha_verified':True,'flow_path':str(flow_path),
              'torch':torch.__version__,'lpips_reference':'https://github.com/richzhang/PerceptualSimilarity',
              'limits':['One clip, no sequence-level significance inference','LR is synthetically degraded and motion is real','Pillow integer forward differs from floating Torch operator; actual Pillow residual reported',
                        'Temporal residual subtracts HR motion/appearance residual; flow is from LR only','Highpass ratio is diagnostic, not a perceptual score','H3 baseline is frozen foundation inference, no learned memory yet']}
    (args.out/'summary.json').write_text(json.dumps(metadata,indent=2));(args.out/'per_frame.json').write_text(json.dumps(rows,indent=2))
    indices = [0, args.frames//2, args.frames-1]
    comparison_panel(hr, methods, ['bicubic','h3_vae_only','h3_bf16','h3_bf16_dc'], indices, args.out/'comparison.png')
    comparison_panel(hr, methods, ['bicubic_dc','h3_vae_only_dc','h3_bf16_dc'], indices, args.out/'comparison_projected.png')


if __name__=='__main__':main()
