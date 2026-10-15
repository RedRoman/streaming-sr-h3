"""Fixed source-paired evaluation of the completed Relay/H3 video cohort."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import lpips
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiments/real_video_evaluation'))
from evaluate import load, ssim_rgb, comparison_panel, warp, BicubicConsistency, audit

COHORT = ROOT / 'results/relay_baseline/cohort'
OUT = COHORT / 'evaluation'
METRICS = ['psnr', 'ssim', 'lpips', 'lr_pillow_rmse',
           'temporal_error_excess_rmse', 'temporal_mask_coverage']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_metrics(rows):
    return {k: float(np.mean([r[k] for r in rows if r.get(k) is not None]))
            for k in METRICS if any(r.get(k) is not None for r in rows)}


def evaluate_case(case, model, device):
    folder = OUT / case['id']; folder.mkdir(parents=True, exist_ok=True)
    lr_paths = [ROOT / 'data/real_video' / p for p in case['lr_paths']]
    hr_paths = [ROOT / 'data/real_video' / p for p in case['hr_paths']]
    assert [sha(p) for p in lr_paths] == case['observed_lr_sha256']
    lr = torch.stack([load(p) for p in lr_paths]); hr = torch.stack([load(p) for p in hr_paths])
    vae_dir = ROOT / 'results/real_video_vae' / case['id']
    vae_meta = json.loads((vae_dir / 'inference.json').read_text())
    vae_paths = [vae_dir / f'{i:04d}.png' for i in range(22)]
    assert vae_meta['observed_lr_sha256'] == case['observed_lr_sha256']
    assert [sha(p) for p in vae_paths] == vae_meta['output_sha256']
    bicubic = F.interpolate(lr, size=case['shape_hw'], mode='bicubic', align_corners=False,
                            antialias=True).clamp(0, 1)
    methods = {'bicubic': (bicubic * 255).round() / 255,
               'h3_vae': torch.stack([load(p) for p in vae_paths])}
    measurements, provenance = {}, {'h3_vae': vae_meta['output_sha256']}
    for protocol in ['official', 'causal_repeat_start']:
        directory = COHORT / protocol / case['id']
        record = json.loads((directory / 'verification.json').read_text())
        paths = [directory / 'png' / f'{i:05d}.png' for i in range(22)]
        assert record['verified'] and record['observed_lr_sha256'] == case['observed_lr_sha256']
        assert [sha(p) for p in paths] == record['output_sha256']
        name = 'relay_' + protocol
        methods[name] = torch.stack([load(p) for p in paths])
        provenance[name] = record['output_sha256']
        measurements[name] = json.loads((directory / 'measurement.json').read_text())
    flow_dir = ROOT / 'results/video_correspondence'
    if case['resolution'] == 'native480': flow_dir /= 'native480'
    flow_dir = flow_dir / case['split'] / case['sequence']
    flow_meta = json.loads((flow_dir / 'provenance.json').read_text())
    assert flow_meta['lr_sha256'][:22] == case['observed_lr_sha256']
    flow = np.load(flow_dir / 'lr_correspondence.npz')
    assert tuple(flow['pair_confidence'].shape[1:]) == tuple(lr.shape[-2:])
    true_delta, masks = {}, {}
    for i in range(1, 22):
        true_delta[i] = hr[i] - warp(hr[i-1], flow['current_to_previous'][i])
        masks[i] = F.interpolate(torch.from_numpy(flow['pair_confidence'][i])[None, None],
                                 hr.shape[-2:], mode='bilinear', align_corners=False)[0, 0] >= .5
    op = BicubicConsistency(tuple(hr.shape[-2:]), tuple(lr.shape[-2:]), device=device)
    projections = {}
    for name, pred in list(methods.items()):
        pred_gpu, lr_gpu = pred.to(device), lr.to(device)
        if device.startswith('cuda'): torch.cuda.synchronize()
        start = time.perf_counter()
        projected, residual = op.project(pred_gpu, lr_gpu, iterations=25)
        if device.startswith('cuda'): torch.cuda.synchronize()
        projections[name] = {'seconds': time.perf_counter()-start,
                             'max_float_lr_rmse': float(residual.max())}
        methods[name+'_dc'] = projected.cpu()
        del pred_gpu, lr_gpu, projected
    rows = []
    for name, pred in methods.items():
        assert pred.shape == hr.shape and torch.isfinite(pred).all()
        assert float(pred.min()) >= 0 and float(pred.max()) <= 1
        distances = []
        for start in range(0, 22, 2):
            distances.extend(model(2*pred[start:start+2].to(device)-1,
                                   2*hr[start:start+2].to(device)-1).flatten().tolist())
        for i, value in enumerate(pred):
            pillow = Image.fromarray(np.uint8(value.permute(1, 2, 0).numpy().clip(0, 1)*255+.5))
            down = pillow.resize((lr.shape[-1], lr.shape[-2]), Image.Resampling.BICUBIC)
            low = torch.from_numpy(np.asarray(down).copy()).permute(2, 0, 1).float()/255
            row = {'case': case['id'], 'source': case['sequence'], 'split': case['split'],
                   'resolution': case['resolution'], 'method': name, 'frame': i,
                   'psnr': float(-10*torch.log10((value-hr[i]).square().mean().clamp_min(1e-12))),
                   'ssim': ssim_rgb(value.numpy(), hr[i].numpy()), 'lpips': distances[i],
                   'lr_pillow_rmse': float((low-lr[i]).square().mean().sqrt())}
            if i:
                difference = value - warp(pred[i-1], flow['current_to_previous'][i]) - true_delta[i]
                row['temporal_error_excess_rmse'] = float(difference[:, masks[i]].square().mean().sqrt()) if masks[i].any() else None
                row['temporal_mask_coverage'] = float(masks[i].float().mean())
            rows.append(row)
    summary = {name: mean_metrics([r for r in rows if r['method'] == name]) for name in methods}
    result = {'case': case['id'], 'source': case['sequence'], 'split': case['split'],
              'resolution': case['resolution'], 'methods': summary, 'projection': projections,
              'output_sha256': provenance, 'lr_sha256': case['observed_lr_sha256'],
              'hr_evaluation_only_sha256': [sha(p) for p in hr_paths],
              'flow_sha256': sha(flow_dir / 'lr_correspondence.npz'),
              'measurements': measurements, 'status': 'complete'}
    (folder / 'summary.json').write_text(json.dumps(result, indent=2))
    (folder / 'per_frame.json').write_text(json.dumps(rows, indent=2))
    for suffix in ['', '_dc']:
        comparison_panel(hr, methods, [n+suffix for n in ['bicubic', 'h3_vae', 'relay_official', 'relay_causal_repeat_start']],
                         [0, 11, 21], folder / ('comparison' + suffix + '.png'))
    print(json.dumps({'case': case['id'], 'methods': summary}), flush=True)
    return result, rows


def main(device):
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((COHORT / 'protocol.json').read_text())
    completed = json.loads((COHORT / 'all_outputs_verified.json').read_text())
    assert completed['passed'] and completed['case_count'] == 24 and completed['png_count'] == 528
    metadata = {'status': 'running', 'pid': os.getpid(), 'started_unix': time.time(),
                'code_sha256': sha(Path(__file__)), 'device': device,
                'protocol': '12 cases from 6 source videos: 4train/2val x 2 spatial contexts. No fitting, tuning or selection. Each method gets exactly one final P25, including bicubic and H3 VAE.',
                'image_precision': '8-bit raw PNG/readouts for all methods; bicubic rounded to 8-bit; final P25 evaluated in float32 without re-quantization.',
                'dc': 'float64 AA-bicubic affine / RGB box Dykstra,25 iterations; actual residual reported; not exact Pillow operator.',
                'temporal': 'Fixed LR-only DIS, paired confidence>=.5; subtract warped HR residual. Does not measure motion endpoint error.',
                'lpips': 'alex v0.1, CUDA and cuDNN TF32 both disabled',
                'comparison_limit': 'Different pretrained models, data, budgets and waiting. H3 VAE sees all22; official Relay waits up to15 frames; causal Relay has0 future frames but changes reference conditioning and3->2 Wan calls.',
                'h3_vae_preprocess': 'Existing Torch AA-bicubic guide. Distinct from official H3 AddGuide uint8/Lanczos variant in native SR pilot.',
                'scope': 'Development6source cohort; no project-new confirmation data opened.',
                'torch': torch.__version__, 'lpips_version': importlib.metadata.version('lpips'),
                'projection_audit': audit()}
    (OUT / 'protocol.json').write_text(json.dumps(metadata, indent=2))
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(device).eval().requires_grad_(False)
    weights = [Path(torch.hub.get_dir())/'checkpoints/alexnet-owt-7be5be79.pth',
               Path(lpips.__file__).parent/'weights/v0.1/alex.pth']
    metadata['perceptual_weights_sha256'] = {str(p): sha(p) for p in weights}
    assert list(metadata['perceptual_weights_sha256'].values()) == [
        '7be5be791159472b1fbf3c69796f7cb30dca7ad8466c2df70058c37116cdee02',
        'df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0']
    summaries, rows = [], []
    with torch.inference_mode():
        same = torch.zeros(1, 3, 64, 64, device=device)
        assert float(model(same, same)) == 0
        for case in protocol['cases']:
            result, local = evaluate_case(case, model, device)
            summaries.append(result); rows.extend(local)
            (OUT / 'LIVE_STATE.json').write_text(json.dumps({'pid': os.getpid(),
                'phase': 'evaluating', 'complete_cases': len(summaries), 'total_cases': 12}))
    source_rows = [{'source': s['source'], 'split': s['split'], 'resolution': s['resolution'],
                    'method': name, **values} for s in summaries for name, values in s['methods'].items()]
    aggregate = []
    for resolution in ['256', 'native480']:
        for split in ['train', 'val', 'all']:
            for method in summaries[0]['methods']:
                selected = [r for r in source_rows if r['resolution'] == resolution
                            and (split == 'all' or r['split'] == split) and r['method'] == method]
                aggregate.append({'resolution': resolution, 'split': split, 'method': method,
                                  'source_count': len(selected), **mean_metrics(selected)})
    for filename, table in [('source_metrics.csv', source_rows), ('aggregate.csv', aggregate)]:
        with (OUT / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    metadata.update(status='complete', finished_unix=time.time(), case_count=12,
                    frame_metric_rows=len(rows), cases=summaries, source_metrics=source_rows,
                    aggregate=aggregate)
    assert len(rows) == 2112
    (OUT / 'evaluation.json').write_text(json.dumps(metadata, indent=2))
    (OUT / 'LIVE_STATE.json').write_text(json.dumps({'status': 'complete', 'cases': 12, 'metric_rows': 2112}))
    print(json.dumps({'complete': True, 'aggregate': aggregate}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--device', default='cpu'); args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    main(args.device)
