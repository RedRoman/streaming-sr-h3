"""Native three-step Open-H3 pilot, scored against original RGB NPZ targets.

CPU contract check: --inspect-only. Only an actual inference run imports H3.
Each source contributes its existing w0 window. This is a development comparison,
not the four-task omni pilot or an unseen generalization test.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from openh3_assets import NATIVE_SIGMAS, read_manifest, safetensors_header, select_samples, sha256
from restoration_metrics import METRIC_SCHEMA_VERSION, assert_compatible_output, metric_metadata


def select_dev_windows(manifest, data_root, rgb_root, expected_sources=8):
    rows = select_samples(manifest, data_root, 'val')
    sources = {r['source_id'] for r in rows}
    if None in sources or '' in sources or len(sources) != expected_sources:
        raise ValueError(f'Expected {expected_sources} explicit development sources, got {sources}')
    result = []
    for source in sorted(sources):
        candidates = [r for r in rows if r['source_id'] == source and r['path'].stem == f'{source}_w0']
        if len(candidates) != 1:
            raise ValueError(f'Expected exactly one w0 latent for {source}')
        latent = candidates[0]['path']
        rgb = Path(rgb_root) / f'{latent.stem}.npz'
        header = safetensors_header(latent)
        if 'guide_raw' not in header or 'target' not in header:
            raise ValueError(f'Missing raw guide/target header in {latent}')
        shape = header['guide_raw']['shape']
        if shape != header['target']['shape'] or len(shape) != 5 or shape[:2] != [1, 24]:
            raise ValueError(f'Invalid guide/target geometry in {latent}: {shape}')
        latent_meta = json.loads(header.get('__metadata__', {}).get('meta', '{}'))
        with np.load(rgb, allow_pickle=False) as pair:
            meta = json.loads(str(pair['meta'].item()))
            if meta.get('id') != source or latent_meta.get('id') != source:
                raise ValueError(f'Source mismatch between manifest, RGB and latent: {source}')
            for key in ('start_s', 'degradation'):
                if key not in meta or key not in latent_meta or meta[key] != latent_meta[key]:
                    raise ValueError(f'RGB/latent {key} mismatch for {source}')
            hr, lr, dx = pair['hr'], pair['lr'], pair['dx']
            if hr.dtype != np.uint8 or lr.dtype != np.uint8 or dx.dtype != np.uint8:
                raise ValueError(f'Expected uint8 original RGB data for {source}')
            if hr.ndim != 4 or hr.shape[-1] != 3 or lr.shape != dx.shape or lr.shape[0] != hr.shape[0]:
                raise ValueError(f'RGB geometry mismatch for {source}')
            t, h, w, _ = hr.shape
            if t != 22 or shape != [1, 24, 7, h // 16, w // 16] or lr.shape != (t, h // 4, w // 4, 3):
                raise ValueError(f'Expected 22-frame x4 H3 pilot geometry for {source}')
        result.append({'source_id': source, 'clip': latent.stem, 'latent_path': str(latent),
                       'rgb_path': str(rgb), 'latent_sha256': sha256(latent), 'rgb_sha256': sha256(rgb),
                       'frames': t, 'height': h, 'width': w, 'meta': meta})
    return result


def compare_results(current, baseline):
    from deploy_eval import paired
    if not current.get('complete') or not baseline.get('complete'):
        raise ValueError('Both arms must complete all development sources before comparison')
    if baseline.get('metric_schema_version') != METRIC_SCHEMA_VERSION:
        raise ValueError('Baseline must be recomputed under the same v2 metric schema')
    for key in ('contract', 'samples'):
        if current[key] != baseline.get(key):
            raise ValueError(f'Baseline comparison contract mismatch: {key}')
    if baseline.get('adaptation') is not None:
        raise ValueError('The comparison reference must be the unadapted pinned baseline')
    a = {row['clip']: row for row in current['per_clip']}
    b = {row['clip']: row for row in baseline['per_clip']}
    if set(a) != set(b):
        raise ValueError('Baseline/current evaluated clip sets differ')
    rows = [{'arms': {'adapted': a[name]['arms']['model'], 'baseline': b[name]['arms']['model']}}
            for name in sorted(a)]
    keys = ['psnr', 'lpips', 'psnr_gt_lowband_substitution', 'lpips_gt_lowband_substitution',
            'detail_cosine', 'detail_energy_ratio', 'p8.temporal_lowband_error_x1000',
            'p8.temporal_detail_error_x1000', 'p8.warp', 'p8.flicker']
    return paired(rows, 'adapted', 'baseline', keys)


def main():
    root = Path(os.environ.get('SSR0', str(Path(__file__).resolve().parents[1])))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pretrained-manifest', type=Path, required=True)
    ap.add_argument('--dev-manifest', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, default=root / 'data/stage7_latents')
    ap.add_argument('--rgb-root', type=Path, default=root / 'data/stage7_train/val')
    ap.add_argument('--adaptation', type=Path)
    ap.add_argument('--baseline-json', type=Path)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--lora-rank', type=int, default=8)
    ap.add_argument('--expected-sources', type=int, default=8)
    ap.add_argument('--frame-stride', type=int, default=4)
    ap.add_argument('--inspect-only', action='store_true')
    a = ap.parse_args()
    if a.frame_stride <= 0:
        ap.error('--frame-stride must be positive')
    if a.adaptation and not a.baseline_json:
        ap.error('--adaptation requires --baseline-json for a matched development comparison')
    samples = select_dev_windows(a.dev_manifest, a.data_root, a.rgb_root, a.expected_sources)
    pretrained = read_manifest(a.pretrained_manifest, verify=False)
    contract = {'dev_manifest_sha256': sha256(a.dev_manifest), 'pretrained_fingerprint': pretrained['_fingerprint'],
                'sigma_schedule': list(NATIVE_SIGMAS), 'seed': a.seed, 'frame_stride': a.frame_stride,
                'guide_key': 'guide_raw', 'draft_key': 'guide_raw', 'draft_source': 'pixel_upsample_guide_fallback',
                'pretrained_latent_upsampler_used': False, 'mode': 'base',
                'target': 'original uint8 hr from matching RGB NPZ; no decoded latent GT',
                'prediction_quantization': 'round(clamp(output,0,1)*255)/255',
                'metric_schema_version': METRIC_SCHEMA_VERSION}
    baseline = json.loads(a.baseline_json.read_text()) if a.baseline_json else None
    if baseline and (not baseline.get('complete') or baseline.get('adaptation') is not None
                     or baseline.get('metric_schema_version') != METRIC_SCHEMA_VERSION):
        raise ValueError('Expected a complete unadapted baseline scored under the current metric schema')
    if baseline and (baseline.get('contract') != contract or baseline.get('samples') != samples):
        raise ValueError('Baseline contract/source hashes differ; refusing unmatched inference')
    inspection = {'contract': contract, 'samples': samples, 'inspect_only': a.inspect_only,
                  'asset_hashes_verified': False, 'source_count': len(samples)}
    if a.inspect_only:
        print(json.dumps(inspection, indent=2))
        return

    output = a.output_dir / 'metrics_v2.json'
    assert_compatible_output(output)
    if output.exists():
        raise ValueError(f'Result exists: {output}; use a fresh output directory')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from deploy_eval import _eval, summarise
    from h3_comfy import H3
    # Import h3_comfy first so torch_npu has registered the extension. This is
    # reached only by the explicitly scheduled inference path, never inspection.
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)
    # Exactly one base load per arm. H3 verifies asset hashes before loading.
    h3 = H3(pretrained_manifest=a.pretrained_manifest, adaptation_path=a.adaptation,
            lora_rank=a.lora_rank, sigma_schedule=NATIVE_SIGMAS)
    import lpips
    import revisit_metrics as rgb_metrics
    with torch.device('cpu'):
        rgb_metrics._LPIPS = lpips.LPIPS(net='alex', version='0.1', verbose=False).cpu().eval()
    if any(p.device.type != 'cpu' for p in rgb_metrics._LPIPS.parameters()):
        raise RuntimeError('RGB evaluation LPIPS must stay on CPU')
    rows = []
    for sample in samples:
        start = time.perf_counter()
        with np.load(sample['rgb_path'], allow_pickle=False) as pair:
            gt = torch.from_numpy(pair['hr'].copy()).float() / 255
            lr = torch.from_numpy(pair['lr'].copy()).float() / 255
        guide = load_file(sample['latent_path'], device='cpu')['guide_raw'].float()
        generated, timing = h3.sample(guide, 'base', hw=[sample['height'], sample['width']],
                                      seed=a.seed, warm_start=guide)
        pixels = h3.decode(generated['video'])
        if pixels.shape != gt.shape or not torch.isfinite(pixels).all():
            raise ValueError(f'Invalid decoded RGB output for {sample["clip"]}: {pixels.shape}')
        pred = (pixels.clamp(0, 1).mul(255).round() / 255).cpu()
        bicubic = torch.nn.functional.interpolate(lr.permute(0, 3, 1, 2),
                    size=(sample['height'], sample['width']), mode='bicubic', align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
        arrays = {'model': pred, 'bicubic': bicubic}
        with torch.device('cpu'):
            row = _eval((sample['clip'], {'model': 'model', 'bicubic': 'bicubic'}),
                        frame_reader=lambda arm, name, t: arrays[arm][t], gt_reader=lambda t: gt[t],
                        clip_meta={'frames': sample['frames'], 'degradation': sample['meta']['degradation']},
                        frame_stride=a.frame_stride)
        row.update({'source_id': sample['source_id'], 'wall_seconds_with_cpu_metrics': time.perf_counter() - start,
                    'sampling_record': timing})
        rows.append(row)
        qa = a.output_dir / 'frames' / sample['clip']
        qa.mkdir(parents=True, exist_ok=True)
        for t in (0, sample['frames'] // 2, sample['frames'] - 1):
            for label, sequence in {'gt': gt, 'bicubic': bicubic, 'output': pred}.items():
                rgb = sequence[t].mul(255).round().clamp(0, 255).to(torch.uint8).numpy()
                Image.fromarray(rgb).save(qa / f'{t:03d}_{label}.png')
        (a.output_dir / 'partial_metrics_v2.json').write_text(json.dumps(
            {'metric_schema_version': METRIC_SCHEMA_VERSION, 'contract': contract, 'samples': samples,
             'per_clip': rows, 'complete': False}, indent=1, allow_nan=False))
        print(json.dumps({'done': sample['clip'], 'psnr': row['arms']['model']['psnr'],
                          'lpips': row['arms']['model']['lpips']}), flush=True)
    result = {'metric_schema_version': METRIC_SCHEMA_VERSION, 'metric_schema': metric_metadata(),
              'contract': contract, 'samples': samples, 'per_clip': rows,
              'summary': summarise(rows, ['model', 'bicubic']), 'model_record': h3.record,
              'adaptation': None if not a.adaptation else {'path': str(a.adaptation.resolve()), 'sha256': sha256(a.adaptation), 'rank': a.lora_rank},
              'complete': True, 'scope': '8-source existing x4 development windows; not native multi-task omni or frozen test evidence'}
    if baseline:
        result['paired_adapted_minus_baseline'] = compare_results(result, baseline)
    output.write_text(json.dumps(result, indent=1, allow_nan=False))
    print(json.dumps({'written': str(output), 'sources': len(rows), 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
