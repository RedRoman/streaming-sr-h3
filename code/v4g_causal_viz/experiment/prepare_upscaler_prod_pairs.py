"""Production-shape pairs: LR latent 44x80 (pixels 704x1280) -> HR latent 68x120
(pixels 1920x1088), 120-frame contract unchanged (37 latent tokens).

Reuses the strict reader from prepare_upscaler_5s_pairs.py with overridden
resolution constants. HR is read at 1088x1920 (source 1344x768 is upscaled
1.43x with lanczos; the distillation target is teacher(LR) so reference
softness does not affect the training loss). LR is an area downscale of HR.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_upscaler_5s_pairs as base
from data_utils import encode_pixels, load_video_vae, select_device

LR_W, LR_H = 1280, 704    # production LR pixels (W,H)
HR_W, HR_H = 1920, 1088   # production HR pixels (W,H)
SOURCE_FRAMES, LATENT_FRAMES, FPS = base.SOURCE_FRAMES, base.LATENT_FRAMES, base.FPS


def read_first_120_hr(source, proof_directory, sample_id):
    """Read HR at 1088x1920 (upscale) then area-downscale to 704x1280 for LR."""
    filters = (f'fps={FPS},scale={HR_W}:{HR_H}:force_original_aspect_ratio=increase:flags=lanczos,'
               f'crop={HR_W}:{HR_H},scale={LR_W}:{LR_H}:flags=area')
    hr_filters = (f'fps={FPS},scale={HR_W}:{HR_H}:force_original_aspect_ratio=increase:flags=lanczos,'
                  f'crop={HR_W}:{HR_H}')
    lr_bytes = SOURCE_FRAMES * LR_W * LR_H * 3
    hr_bytes = SOURCE_FRAMES * HR_W * HR_H * 3
    import subprocess
    outs = {}
    for tag, vf, expected in (('lr', filters, lr_bytes), ('hr', hr_filters, hr_bytes)):
        command = [base.ffmpeg_executable(), '-hide_banner', '-nostdin', '-loglevel', 'verbose',
                   '-nostats', '-i', str(source), '-map', '0:v:0', '-an', '-sn', '-dn',
                   '-vf', vf, '-frames:v', str(SOURCE_FRAMES), '-vsync', '0',
                   '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
        result = subprocess.run(command, capture_output=True)
        stderr = result.stderr.decode('utf-8', errors='replace')
        (proof_directory / f'{sample_id}.read_{tag}.log').write_text(stderr)
        stats = base.fps_statistics(stderr)
        if result.returncode or len(result.stdout) != expected:
            raise ValueError(f'{sample_id}: {tag} reader failed '
                             f'({len(result.stdout)}/{expected} bytes)')
        if stats['duplicated_frames'] != 0 or stats['input_frames'] < SOURCE_FRAMES:
            raise ValueError(f'{sample_id}: {tag} duplicated frames')
        outs[tag] = np.frombuffer(result.stdout, dtype=np.uint8).reshape(
            SOURCE_FRAMES, {'lr': LR_H, 'hr': HR_H}[tag], {'lr': LR_W, 'hr': HR_W}[tag], 3).copy()
    lr = torch.from_numpy(outs['lr'])
    hr = torch.from_numpy(outs['hr'])
    return lr, hr


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sources', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base-model', required=True)
    p.add_argument('--holdout-id', default='VIDGEN5S_0380588')
    p.add_argument('--device', default='npu:0')
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.output.exists():
        raise FileExistsError('use a fresh output directory')
    (args.output / 'samples').mkdir(parents=True)
    (args.output / 'source_proof').mkdir()
    device = select_device(args.device)
    vae = load_video_vae(args.base_model, device)
    rows = {'train': [], 'holdout': []}
    for source in args.sources:
        sample_id = Path(source).stem
        split = 'holdout' if sample_id == args.holdout_id else 'train'
        lr, hr = read_first_120_hr(source, args.output / 'source_proof', sample_id)
        draft = encode_pixels(vae, lr, device, args.seed)
        reference = encode_pixels(vae, hr, device, args.seed)
        if tuple(draft.shape) != (24, LATENT_FRAMES, LR_H // 16, LR_W // 16) or \
                tuple(reference.shape) != (24, LATENT_FRAMES, HR_H // 16, HR_W // 16):
            raise ValueError(f'{sample_id}: unexpected latent shapes {draft.shape} {reference.shape}')
        torch.save({'draft': draft, 'reference': reference, 'metadata': {
            'sample_id': sample_id, 'source_video_path': str(source), 'split': split,
            'lr_pixels': [LR_H, LR_W], 'hr_pixels': [HR_H, HR_W],
            'latent_pairing': 'same120frames; HR lanczos-upscale-crop 1088x1920; LR area 704x1280'}},
            args.output / 'samples' / f'{sample_id}.pt')
        rows[split].append({'sample_id': sample_id, 'path': f'samples/{sample_id}.pt',
                            'split': split})
        print(json.dumps({'done': sample_id, 'split': split}), flush=True)
    for split in ('train', 'holdout'):
        (args.output / f'{split}.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in rows[split]) + '\n')
    print(json.dumps({'status': 'completed', 'train': len(rows['train']),
                      'holdout': len(rows['holdout'])}), flush=True)


if __name__ == '__main__':
    main()
