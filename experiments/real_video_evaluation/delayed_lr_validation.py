"""Frozen prefix-21 update, validated only by genuinely later LR observations.

This is a classical signal diagnostic, not a new learned memory method. The
score phase never opens HR images. Its artifact and fixed decisions must exist
before a separate evaluation command opens GT. No threshold is fitted.
"""
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from pixel_consistency import BicubicConsistency
from video_correspondence import pair_correspondence
from evaluate import warp, ssim_rgb

OUT = ROOT / 'results/real_video_history_inverse/delayed_lr_validation'
DATA = ROOT / 'data/real_video'
SOURCES = ['bear', 'bus', 'car-turn', 'flamingo', 'cows', 'dance-twirl']
FRAME = 21
FUTURE = list(range(22, 38))
PRIMARY_HORIZON = 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB')).copy()


def tensor(array):
    return torch.from_numpy(array).permute(2, 0, 1).float() / 255


def windows():
    all_windows = json.loads((DATA / 'h3_windows_39.json').read_text())['windows']
    return [next(w for w in all_windows if w['sequence'] == source) for source in SOURCES]


def pair(source):
    directory = ROOT / 'results/real_video_history_inverse/matched_final_projection' / f'{source}_256'
    meta = json.loads((directory / 'inference.json').read_text())
    assert not meta['hr_read'] and meta['repair_no_hr_input']
    assert sha(directory / 'outputs.pt') == meta['output_sha256']
    value = torch.load(directory / 'outputs.pt', map_location='cpu', weights_only=True)
    assert value['vae_dc'].shape == (22, 3, 256, 256)
    return value['vae_dc'][FRAME], value['inverse_vae_dc'][FRAME], meta['output_sha256']


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'LOCKED_PROTOCOL.json'
    if path.exists():
        raise FileExistsError('Protocol already frozen; use a new version to change it')
    protocol = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'sources_in_order': SOURCES, 'reference_frame': FRAME,
        'candidate': 'matched-final-DC inverse_vae_dc', 'base': 'matched-final-DC vae_dc',
        'prefix_observed_indices': list(range(22)), 'future_indices': FUTURE,
        'primary_future_count': PRIMARY_HORIZON,
        'primary_rule': 'At arrival25 retain iff sum(weighted LR SSE(base)-SSE(candidate))>0 and total weighted candidate-difference energy>1e-12; else retract to base.',
        'weight': 'LR-only DIS confidence times indicator(confidence >= 0.5)',
        'flow': {'preset': 'MEDIUM', 'photo_sigma': .05, 'cycle_sigma': 1., 'direct_reference': FRAME},
        'camera': 'Existing separable float64 Torch-AA bicubic after shared bilinear warp; actual future target is uint8 Pillow LR; model discrepancy is retained.',
        'primary_evaluation': 'Reference-frame21 PSNR/SSIM of retrospectively retained state at arrival25. This is not a current-frame25 prediction or zero-wait output.',
        'secondary': 'All cumulative 1..16 horizons and future-frame warp metrics; descriptive, not chosen for best score.',
        'independent_units': 'Six sources. Frames and horizons are not independent replicas.',
        'training': 'none; no HR-based threshold, no new H3 sampling; 0 E/D/DiT/GPU',
        'selection_limits': 'All six existing inverse experiments; harms known at clip-average level before this diagnostic. Not a fresh holdout.',
        'causal_boundary': 'H3 VAE prefix sees all0..21; freeze reference21 and only22..37 are unseen evidence. Earlier candidate frames would leak future within prefix.',
        'files': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), ROOT/'src/video_correspondence.py', ROOT/'src/pixel_consistency.py', Path(__file__).with_name('evaluate.py'), DATA/'h3_windows_39.json']},
    }
    write(path, protocol)
    print(json.dumps({'phase': 'freeze', 'sha256': sha(path)}), flush=True)


def check_protocol():
    path = OUT / 'LOCKED_PROTOCOL.json'
    protocol = json.loads(path.read_text())
    for rel, digest in protocol['files'].items():
        assert sha(ROOT / rel) == digest, rel
    return sha(path)


@torch.no_grad()
def score():
    protocol_sha = check_protocol()
    if (OUT/'scores.json').exists():
        raise FileExistsError('Do not overwrite existing score decisions')
    started = time.perf_counter()
    operator = BicubicConsistency(256, 64)
    result = {'hr_images_opened': 0, 'protocol_sha256': protocol_sha,
              'torch': torch.__version__, 'cv2': cv2.__version__, 'sources': []}
    for w in windows():
        source = w['sequence']
        base, candidate, output_sha = pair(source)
        vae_meta = json.loads((ROOT/'results/real_video_vae'/f'{source}_256'/'inference.json').read_text())
        prefix = [DATA/p for p in w['lr_paths'][:22]]
        assert vae_meta['latest_observed_frame'] == FRAME
        assert vae_meta['lr_paths'] == w['lr_paths'][:22]
        assert [sha(p) for p in prefix] == vae_meta['observed_lr_sha256']
        assert not set(w['lr_paths'][:22]) & set(w['lr_paths'][22:38])
        ref = rgb(prefix[-1])
        assert ref.shape == (64, 64, 3)
        entry = {'source': source, 'split': w['split'], 'output_sha256': output_sha,
                 'prefix_sha_verified': True, 'latest_candidate_observation': FRAME,
                 'future_paths': w['lr_paths'][22:38], 'future_sha256': [], 'rows': []}
        accumulated_gain = accumulated_energy = accumulated_mass = 0.
        maps = {}
        for j in FUTURE:
            future_path = DATA / w['lr_paths'][j]
            current = rgb(future_path)
            coords, confidence, _ = pair_correspondence(ref, current, .05, 1.)
            weight = torch.from_numpy(confidence.astype(np.float64)) * (torch.from_numpy(confidence) >= .5)
            x0 = operator.down(warp(base, coords).double())
            x1 = operator.down(warp(candidate, coords).double())
            target = tensor(current).double()
            delta, residual = x1 - x0, target - x0
            gain = float(((2*residual*delta - delta.square())*weight).sum())
            energy = float((delta.square()*weight).sum())
            mass = float(weight.sum()) * 3
            baseline_sse = float((residual.square()*weight).sum())
            candidate_sse = float(((target-x1).square()*weight).sum())
            assert abs(gain-(baseline_sse-candidate_sse)) < 1e-10
            accumulated_gain += gain; accumulated_energy += energy; accumulated_mass += mass
            keep = accumulated_gain > 0 and accumulated_energy > 1e-12
            row = {'future_frame': j, 'horizon': j-FRAME,
                   'coverage': float((weight>0).float().mean()), 'weighted_rgb_mass': mass,
                   'baseline_lr_mse': baseline_sse/mass if mass else None,
                   'candidate_lr_mse': candidate_sse/mass if mass else None,
                   'gain_sum': gain, 'observable_energy_sum': energy,
                   'cumulative_gain_sum': accumulated_gain,
                   'cumulative_energy_sum': accumulated_energy,
                   'cumulative_weighted_rgb_mass': accumulated_mass,
                   'cumulative_gain_mse': accumulated_gain/accumulated_mass if accumulated_mass else None,
                   'cumulative_observable_energy_mse': accumulated_energy/accumulated_mass if accumulated_mass else None,
                   'retain_candidate': keep}
            entry['rows'].append(row)
            entry['future_sha256'].append(sha(future_path))
            maps[f'coordinates_{j}'] = coords
            maps[f'confidence_{j}'] = confidence
        entry['primary_decision'] = entry['rows'][PRIMARY_HORIZON-1]['retain_candidate']
        maps_path = OUT / f'{source}_lr_geometry.npz'
        np.savez_compressed(maps_path, **maps)
        entry['geometry_sha256'] = sha(maps_path)
        result['sources'].append(entry)
        print(json.dumps({'phase': 'LR-only score', 'source': source,
                          'retain_at25': entry['primary_decision'],
                          'gain_mse_first4': entry['rows'][3]['cumulative_gain_mse']}), flush=True)
    result['cpu_seconds'] = time.perf_counter()-started
    result['completed_utc'] = datetime.now(timezone.utc).isoformat()
    write(OUT/'scores.json', result)
    write(OUT/'SCORES_LOCK.json', {'scores_sha256': sha(OUT/'scores.json'), 'protocol_sha256': protocol_sha,
                                 'hr_images_opened': 0, 'created_utc': datetime.now(timezone.utc).isoformat()})


def metrics(prediction, target):
    return {'psnr': float(-10*torch.log10((prediction-target).square().mean().clamp_min(1e-12))),
            'ssim': ssim_rgb(prediction.numpy(), target.numpy())}


@torch.no_grad()
def evaluate():
    protocol_sha = check_protocol()
    lock = json.loads((OUT/'SCORES_LOCK.json').read_text())
    assert lock['scores_sha256'] == sha(OUT/'scores.json') and lock['protocol_sha256'] == protocol_sha
    if (OUT/'evaluation.json').exists():
        raise FileExistsError('Do not overwrite evaluation')
    scored = json.loads((OUT/'scores.json').read_text())
    started = time.perf_counter(); entries = []
    for w, entry in zip(windows(), scored['sources']):
        source = w['sequence']; assert source == entry['source']
        base, candidate, output_sha = pair(source)
        assert output_sha == entry['output_sha256']
        ground_truth = tensor(rgb(DATA / w['hr_paths'][FRAME]))
        base_m, candidate_m = metrics(base, ground_truth), metrics(candidate, ground_truth)
        retain = entry['primary_decision']
        item = {'source': source, 'split': w['split'], 'retain_at25': retain,
                'reference_metrics': {'never_apply_base': base_m, 'always_apply_candidate': candidate_m,
                                      'delayed_lr_gate': candidate_m if retain else base_m},
                'candidate_psnr_gain': candidate_m['psnr']-base_m['psnr'],
                'candidate_ssim_gain': candidate_m['ssim']-base_m['ssim'],
                'sign_correct': retain == (candidate_m['psnr'] > base_m['psnr']),
                'first4_mean_coverage': float(np.mean([r['coverage'] for r in entry['rows'][:4]])),
                'lr_gain_mse_first4': entry['rows'][3]['cumulative_gain_mse'],
                'lr_observable_energy_mse_first4': entry['rows'][3]['cumulative_observable_energy_mse'],
                'future_warp_metrics': [], 'hr_sha256': {str(FRAME): sha(DATA/w['hr_paths'][FRAME])}}
        geom_path = OUT/f'{source}_lr_geometry.npz'
        assert sha(geom_path) == entry['geometry_sha256']
        geometry = np.load(geom_path)
        for j in FUTURE:
            target_path = DATA/w['hr_paths'][j]
            truth = tensor(rgb(target_path))
            coords, conf = geometry[f'coordinates_{j}'], geometry[f'confidence_{j}']
            mask = F.interpolate(torch.from_numpy(conf)[None,None], (256,256), mode='bilinear', align_corners=False)[0,0] >= .5
            pred0, pred1 = warp(base,coords), warp(candidate,coords)
            if mask.any():
                mse0 = float((pred0-truth)[:,mask].square().mean())
                mse1 = float((pred1-truth)[:,mask].square().mean())
            else: mse0 = mse1 = None
            item['future_warp_metrics'].append({'future_frame': j, 'base_masked_mse': mse0, 'candidate_masked_mse': mse1,
                                                 'masked_mse_gain': mse0-mse1 if mse0 is not None else None})
            item['hr_sha256'][str(j)] = sha(target_path)
        entries.append(item)
        print(json.dumps({'phase': 'GT evaluation', 'source': source, 'retain': retain,
                          'candidate_psnr_gain': item['candidate_psnr_gain'], 'sign_correct': item['sign_correct']}), flush=True)
    means = {method: {metric: float(np.mean([v['reference_metrics'][method][metric] for v in entries]))
                     for metric in ('psnr', 'ssim')}
             for method in ('never_apply_base', 'always_apply_candidate', 'delayed_lr_gate')}
    horizon_results = []
    for k in range(16):
        decisions = [s['rows'][k]['retain_candidate'] for s in scored['sources']]
        p = [v['reference_metrics']['always_apply_candidate' if keep else 'never_apply_base']['psnr'] for v,keep in zip(entries,decisions)]
        horizon_results.append({'horizon': k+1, 'arrival_frame': FRAME+k+1,
                                'retained_sources': [v['source'] for v,keep in zip(entries,decisions) if keep],
                                'source_equal_psnr': float(np.mean(p)),
                                'sign_correct_sources': sum(keep == (v['candidate_psnr_gain'] > 0) for v,keep in zip(entries,decisions))})
    result = {'protocol_sha256': protocol_sha, 'scores_sha256': lock['scores_sha256'], 'sources': entries,
              'source_equal_means': means, 'diagnostic_horizons_not_selected': horizon_results,
              'primary_sign_correct': sum(v['sign_correct'] for v in entries), 'source_count': len(entries),
              'cpu_seconds': time.perf_counter()-started, 'gpu_seconds': 0, 'E': 0, 'D': 0, 'DiT': 0}
    write(OUT/'evaluation.json', result)


if __name__ == '__main__':
    torch.set_num_threads(4); cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(); parser.add_argument('phase', choices=['freeze', 'score', 'evaluate'])
    args = parser.parse_args()
    {'freeze': freeze, 'score': score, 'evaluate': evaluate}[args.phase]()
