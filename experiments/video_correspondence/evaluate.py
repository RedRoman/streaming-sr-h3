"""Audit prefix-only matching on actual DAVIS motion, then cache LR maps.

Interpolation and simple temporal blending are diagnostic baselines, not H3
results. HR access is confined to the evaluation block after model outputs.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from video_correspondence import LRCorrespondence, grid, pair_correspondence, warp_reference


def read(path): return np.asarray(Image.open(path).convert('RGB'))
def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def psnr(a, b): return float(-10 * np.log10(max(np.mean((a - b) ** 2), 1e-12)))


def audit():
    rng = np.random.default_rng(17)
    texture = cv2.GaussianBlur(rng.random((96, 96, 3)).astype(np.float32), (0, 0), .8)
    texture = np.uint8(np.clip(texture * 255, 0, 255))
    moved = cv2.warpAffine(texture, np.array([[1, 0, 3], [0, 1, -2]], np.float32), (96, 96))
    coordinates, confidence, _ = pair_correspondence(texture, moved)
    offset = coordinates - grid(96, 96)
    epe = float(np.linalg.norm(offset[16:-16, 16:-16] - np.array([-3., 2.]), axis=-1).mean())
    tracker = LRCorrespondence()
    tracker.push(texture, 'zero', 0); second = tracker.push(moved, 'one', 1)
    before = tracker.state_digest(); duplicate = tracker.push(moved.copy(), 'one', 1)
    duplicate_unchanged = before == tracker.state_digest() and not duplicate.fresh
    large = cv2.resize(texture.astype(np.float32), (384, 384), interpolation=cv2.INTER_CUBIC)
    ident = float(np.max(np.abs(warp_reference(large, grid(96, 96)) - large)))
    # Evaluate an independently reconstructed prefix, with no later frames.
    clone = LRCorrespondence(); clone.push(texture, 'zero', 0); again = clone.push(moved, 'one', 1)
    prefix_error = float(np.max(np.abs(again.current_to_canonical - second.current_to_canonical)))
    assert epe < .5 and duplicate_unchanged and ident < 1e-5 and prefix_error == 0
    return {'translation_endpoint_error_lr_px': epe, 'duplicate_unchanged': duplicate_unchanged,
            'identity_warp_max_abs': ident, 'independent_prefix_max_abs': prefix_error,
            'passed': True, 'opencv': cv2.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', type=Path, default=ROOT / 'data/real_video')
    ap.add_argument('--out', type=Path, default=ROOT / 'results/video_correspondence')
    ap.add_argument('--split', nargs='+', default=['train', 'val', 'test'])
    ap.add_argument('--audit-only', action='store_true')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(4)
    audited = audit(); (args.out / 'audit.json').write_text(json.dumps(audited, indent=2))
    if args.audit_only: print(json.dumps(audited)); return
    manifest = json.loads((args.data / 'h3_windows_39.json').read_text())
    windows = manifest['windows']
    results = {}; t0 = time.perf_counter()
    config = {'direct_interval': 17, 'photo_sigma': .05, 'cycle_sigma': 1.,
              'fusion_old_weight': .5, 'confidence_reporting_threshold': .5,
              'source_hashes': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__), ROOT / 'src/video_correspondence.py']},
              'frozen_before_heldout': True, 'parameter_selection': 'Fixed defaults; no heldout fitting or selection',
              'opencv': cv2.__version__, 'flow_api': 'https://docs.opencv.org/5.0/main_modules/classcv_1_1DISOpticalFlow.html',
              'scope': 'Classical LR-only motion interface and non-generative baselines; no H3 or learned SR evaluation'}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    for window in windows:
        if window['split'] not in args.split: continue
        tracker = LRCorrespondence(**{k: config[k] for k in ['direct_interval', 'photo_sigma', 'cycle_sigma']})
        rows = []; caches = []; previous_estimate = previous_gt = anchor_up = None
        lr_hashes = []
        for i, (lr_path, gt_path) in enumerate(zip(window['lr_paths'], window['hr_paths'])):
            raw = (args.data / lr_path).read_bytes(); lr_hashes.append(hashlib.sha256(raw).hexdigest())
            lr = read(args.data / lr_path); begin = time.perf_counter()
            c = tracker.push(lr, f"{window['source_video_id']}:{i}", i)
            elapsed = time.perf_counter() - begin
            # Restoration outputs use LR inputs only, before the evaluation HR read.
            up = cv2.resize(lr.astype(np.float32) / 255, (lr.shape[1]*4, lr.shape[0]*4), interpolation=cv2.INTER_CUBIC).clip(0, 1)
            if anchor_up is None: anchor_up = up.copy()
            pair_conf = cv2.resize(c.pair_confidence, up.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            canonical_conf = cv2.resize(c.canonical_confidence, up.shape[1::-1], interpolation=cv2.INTER_LINEAR)
            if previous_estimate is None: fused = up.copy()
            else:
                warped = warp_reference(previous_estimate, c.current_to_previous)
                weight = config['fusion_old_weight'] * pair_conf[..., None]
                fused = ((1 - weight) * up + weight * warped).clip(0, 1)
            fixed = warp_reference(anchor_up, c.current_to_canonical)
            reference_blend = (canonical_conf[..., None]*fixed + (1-canonical_conf[..., None])*up).clip(0, 1)
            digest = tracker.state_digest()
            # Query calls must not change correspondence/evidence state.
            _ = warp_reference(anchor_up, c.current_to_canonical)
            assert tracker.state_digest() == digest
            # Evaluation only from here: source HR never reaches tracker/baselines.
            gt = read(args.data / gt_path).astype(np.float32) / 255
            row = {'frame': i, 'latest_observed_frame': i, 'input_wait_frames': 0,
                   'flow_seconds': elapsed, 'pair_coverage': float((c.pair_confidence >= .5).mean()),
                   'canonical_coverage': float((c.canonical_confidence >= .5).mean()),
                   'direct_fraction': c.direct_fraction, 'bicubic_psnr': psnr(up, gt),
                   'temporal_blend_psnr': psnr(fused, gt), 'fixed_reference_psnr': psnr(reference_blend, gt)}
            if previous_gt is not None:
                gt_warp = warp_reference(previous_gt, c.current_to_previous)
                error = np.mean((gt_warp - gt)**2, axis=-1)
                trusted = pair_conf >= .5
                row.update(hr_warp_rmse=float(np.sqrt(error.mean())),
                           hr_warp_rmse_trusted=float(np.sqrt(error[trusted].mean())) if trusted.any() else None,
                           hr_warp_rmse_untrusted=float(np.sqrt(error[~trusted].mean())) if (~trusted).any() else None,
                           temporal_error_excess_rmse=rmse(fused-warp_reference(previous_estimate, c.current_to_previous), gt-gt_warp))
            rows.append(row); caches.append(c)
            previous_estimate = fused; previous_gt = gt
        key = f"{window['split']}/{window['sequence']}"
        folder = args.out / window['split'] / window['sequence']; folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'lr_correspondence.npz',
                            current_to_previous=np.stack([c.current_to_previous for c in caches]),
                            current_to_canonical=np.stack([c.current_to_canonical for c in caches]),
                            pair_confidence=np.stack([c.pair_confidence for c in caches]),
                            canonical_confidence=np.stack([c.canonical_confidence for c in caches]))
        (folder / 'per_frame.json').write_text(json.dumps(rows, indent=2))
        (folder / 'provenance.json').write_text(json.dumps({'source_video_id': window['source_video_id'],
           'split': window['split'], 'lr_sha256': lr_hashes, 'inputs': 'Observed LR RGB only, no HR/segmentation',
           'coordinates': 'Absolute LR pixel-center coordinates, current grid -> reference grid',
           'sequence': 'No future beyond frame index; this input interface has zero lookahead, H3 block processing adds its own wait',
           'confidence': 'Local photometric and forward/backward consistency, not calibrated identity probability'}, indent=2))
        columns = [gt, up, fused, reference_blend, np.repeat(canonical_conf[..., None], 3, -1)]
        panel = Image.new('RGB', (5*256, 282), 'white'); draw = ImageDraw.Draw(panel)
        for j, (title, value) in enumerate(zip(['HR evaluation only', 'LR bicubic', 'Temporal blend', 'Anchor reference', 'Canonical confidence'], columns)):
            panel.paste(Image.fromarray(np.uint8(value.clip(0, 1)*255)), (j*256, 26)); draw.text((j*256+5, 6), title, fill='black')
        panel.save(folder / 'final_frame.png')
        metrics = ['bicubic_psnr', 'temporal_blend_psnr', 'fixed_reference_psnr', 'pair_coverage', 'canonical_coverage', 'flow_seconds', 'hr_warp_rmse_trusted', 'hr_warp_rmse_untrusted']
        results[key] = {name: float(np.mean([row[name] for row in rows if row.get(name) is not None])) for name in metrics}
        results[key]['frames'] = len(rows)
        print(json.dumps({'source': key, **results[key]}), flush=True)
    summary = {'scope': config['scope'], 'sequences': results, 'elapsed_seconds': time.perf_counter()-t0,
               'limits': ['Eight short clips do not prove long-gap identity recovery.', 'Confidence is a consistency score, not object recognition.',
                          'Bicubic/temporal blend do not generate H3 details.', 'HR warp differences include true appearance/occlusion changes, not only flow error.'],
               'next': 'Use identical cached LR-only maps for H3 baseline and learned memory comparisons; retain a no-correspondence ablation.'}
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
