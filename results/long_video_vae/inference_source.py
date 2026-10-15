"""Fixed H3 VAE long-stream control; LR-only inference, separate GT evaluation.

No persistent learned state is claimed here. A 22-frame native block advances
by 17 frames; overlap is context only and previously emitted pixels are immutable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data/long_video'
OUT = ROOT / 'results/long_video_vae'
MANIFEST = DATA / 'derived_h3_manifest.json'
sys.path.insert(0, str(ROOT / 'src'))
from pixel_consistency import BicubicConsistency


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2))


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('RGB')).copy()).permute(2, 0, 1).float() / 255


def schedule(length):
    assert length >= 22 and (length - 22) % 17 == 0
    blocks = []
    for start in range(0, length - 21, 17):
        emit_start = start if start == 0 else start + 5
        blocks.append({'block': len(blocks), 'observed_start': start,
                       'observed_stop_exclusive': start + 22,
                       'latest_observed': start + 21, 'emit_start': emit_start,
                       'emit_stop_exclusive': start + 22})
    emitted = [i for b in blocks for i in range(b['emit_start'], b['emit_stop_exclusive'])]
    assert emitted == list(range(length))
    return blocks


def prepare():
    manifest = json.loads(MANIFEST.read_text())
    cases = []
    for video in manifest['videos']:
        frames = video['frames_metadata']
        cases.append({'sequence': video['sequence'], 'split': video['split'],
                      'shape_hw': list(reversed(video['hr_dimensions_wh'])),
                      'frames': len(frames), 'schedule': schedule(len(frames)),
                      'observed_lr': [{'path': f['lr_x4'], 'sha256': f['lr_sha256'],
                                       'source_frame0': f['source_frame0']} for f in frames]})
    protocol = {
        'dataset_manifest_sha256': sha(MANIFEST), 'cases': cases,
        'method': 'Frozen native H3 FP16 video VAE E(upLR) then D',
        'resize': 'torch AA bicubic align_corners=False, clamp; float input (no AddGuide quantization)',
        'block_frames': 22, 'stride': 17, 'overlap_frames': 5,
        'commit': 'Emit first block22 then each next17 once; no historic rewrite or overlap averaging',
        'wait_frames': 'First22 outputs wait for frame21; subsequent17 outputs wait for each block final frame. Initial max21, steady max16 input intervals.',
        'time': 'Offline compute measured separately; nominal30fps is NOT verified source frame rate',
        'controls': ['bicubic', 'h3_vae', 'bicubic_dc25', 'h3_vae_dc25'],
        'dc': 'Same classical float64 separable bicubic right-inverse + RGB box Dykstra25, no HR fitting',
        'selection': 'Frozen weights, no optimization or test-dependent parameter selection',
        'claim_limit': '4 source pilot; codec/block baseline only, no persistent learned detail state',
        'weights_sha256': '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522',
    }
    write(OUT / 'protocol.json', protocol)
    print(json.dumps({'cases': len(cases), 'blocks': sum(len(c['schedule']) for c in cases),
                      'frames': sum(c['frames'] for c in cases), 'protocol_sha256': sha(OUT / 'protocol.json')}), flush=True)


def infer(args):
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(args.comfy))
    import comfy.sd
    import comfy.utils
    torch.cuda.set_per_process_memory_fraction(.4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((OUT / 'protocol.json').read_text())
    assert sha(MANIFEST) == protocol['dataset_manifest_sha256']
    assert sha(args.weights) == protocol['weights_sha256']
    state, metadata = comfy.utils.load_torch_file(str(args.weights), return_metadata=True)
    vae = comfy.sd.VAE(sd=state, device=torch.device('cuda'), dtype=torch.float16, metadata=metadata)
    del state
    run_start = time.perf_counter()
    for case in protocol['cases']:
        folder = OUT / case['sequence']
        folder.mkdir(parents=True, exist_ok=True)
        for block in case['schedule']:
            done = folder / f"block_{block['block']:03d}.json"
            if done.exists():
                previous = json.loads(done.read_text())
                assert previous['protocol_sha256'] == sha(OUT / 'protocol.json')
                assert all(sha(folder / p) == s for p, s in previous['output_sha256'].items())
                continue
            records = case['observed_lr'][block['observed_start']:block['observed_stop_exclusive']]
            paths = [DATA / r['path'] for r in records]
            assert all(sha(p) == r['sha256'] for p, r in zip(paths, records))
            observed = torch.stack([load(p) for p in paths])
            pixels = F.interpolate(observed, case['shape_hw'], mode='bicubic',
                                   align_corners=False, antialias=True).clamp(0, 1).permute(0, 2, 3, 1).contiguous()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode():
                latent = vae.encode(pixels)
                torch.cuda.synchronize()
                mid = time.perf_counter()
                decoded = vae.decode(latent).float().cpu()
                torch.cuda.synchronize()
            end = time.perf_counter()
            if decoded.ndim == 5:
                decoded = decoded[0]
            assert decoded.shape == pixels.shape and torch.isfinite(decoded).all()
            output_hashes = {}
            for frame in range(block['emit_start'], block['emit_stop_exclusive']):
                value = decoded[frame - block['observed_start']]
                path = folder / f'{frame:04d}.png'
                Image.fromarray(np.uint8(value.numpy().clip(0, 1) * 255 + .5)).save(path, compress_level=3)
                output_hashes[path.name] = sha(path)
            result = {**block, 'sequence': case['sequence'], 'status': 'complete',
                      'input_hr_read': False, 'observed_lr_sha256': [r['sha256'] for r in records],
                      'latent_shape': list(latent.shape), 'encode_seconds': mid - begin,
                      'decode_seconds': end - mid, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'output_sha256': output_hashes, 'protocol_sha256': sha(OUT / 'protocol.json'),
                      'script_sha256': sha(Path(__file__))}
            write(done, result)
            write(OUT / 'LIVE_STATE.json', {'phase': 'inference', 'pid': os.getpid(),
                                           'sequence': case['sequence'], 'block_complete': block['block'],
                                           'elapsed_seconds': time.perf_counter() - run_start})
            print(json.dumps({k: result[k] for k in ['sequence', 'block', 'encode_seconds', 'decode_seconds', 'peak_allocated_bytes']}), flush=True)
            del observed, pixels, latent, decoded
    write(OUT / 'inference_complete.json', {'status': 'complete', 'protocol_sha256': sha(OUT / 'protocol.json'),
                                          'elapsed_seconds': time.perf_counter() - run_start, 'frames': 2400,
                                          'blocks': 140, 'no_persistent_state': True})


def evaluate(args):
    import lpips
    from evaluate import ssim_rgb
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(.4)
    model = lpips.LPIPS(net='alex', version='0.1', verbose=False).eval().requires_grad_(False).to(device)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    manifest = json.loads(MANIFEST.read_text())
    assert sha(MANIFEST) == protocol['dataset_manifest_sha256']
    summaries = []
    with torch.inference_mode():
        for case in protocol['cases']:
            folder = OUT / case['sequence']
            completed = folder / 'evaluation.json'
            if completed.exists():
                cached = json.loads(completed.read_text())
                assert cached['protocol_sha256'] == sha(OUT / 'protocol.json')
                summaries.append(cached)
                continue
            video = next(v for v in manifest['videos'] if v['sequence'] == case['sequence'])
            rows = []
            op = None
            for block in case['schedule']:
                records = video['frames_metadata'][block['emit_start']:block['emit_stop_exclusive']]
                inf = json.loads((folder / f"block_{block['block']:03d}.json").read_text())
                assert inf['protocol_sha256'] == sha(OUT / 'protocol.json')
                for f in records:
                    assert sha(DATA / f['hr']) == f['hr_sha256']
                    assert sha(DATA / f['lr_x4']) == f['lr_sha256']
                    name = f"{f['local_frame0']:04d}.png"
                    assert sha(folder / name) == inf['output_sha256'][name]
                lr = torch.stack([load(DATA / f['lr_x4']) for f in records]).to(device)
                hr = torch.stack([load(DATA / f['hr']) for f in records]).to(device)
                bicubic = F.interpolate(lr, case['shape_hw'], mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
                bicubic = (bicubic * 255).round() / 255
                decoded = torch.stack([load(folder / f"{f['local_frame0']:04d}.png") for f in records]).to(device)
                if op is None:
                    op = BicubicConsistency(tuple(hr.shape[-2:]), tuple(lr.shape[-2:]), device=device)
                methods = {'bicubic': bicubic, 'h3_vae': decoded}
                residuals = {}
                for name, output in list(methods.items()):
                    methods[name + '_dc25'], residuals[name + '_dc25'] = op.project(output, lr, 25)
                for name, prediction in methods.items():
                    distances = []
                    for start in range(0, len(records), 4):
                        distances.extend(model(2 * prediction[start:start + 4] - 1,
                                               2 * hr[start:start + 4] - 1).flatten().tolist())
                    psnr = -10 * torch.log10((prediction - hr).square().mean((1, 2, 3)).clamp_min(1e-12))
                    actual_residual = (op.down(prediction.to(op.w)) - lr.to(op.w)).square().mean((1, 2, 3)).sqrt()
                    for i, f in enumerate(records):
                        x, truth = prediction[i].cpu().numpy(), hr[i].cpu().numpy()
                        pillow = Image.fromarray(np.uint8(x.transpose(1, 2, 0).clip(0, 1) * 255 + .5)).resize(
                            (lr.shape[-1], lr.shape[-2]), Image.Resampling.BICUBIC)
                        low = np.asarray(pillow).astype(np.float32).transpose(2, 0, 1) / 255
                        rows.append({'method': name, 'frame': f['local_frame0'], 'source_frame0': f['source_frame0'],
                                     'block': block['block'], 'latest_observed': block['latest_observed'],
                                     'lookahead_frames': block['latest_observed'] - f['local_frame0'],
                                     'psnr': float(psnr[i]), 'lpips': distances[i], 'ssim': ssim_rgb(x, truth),
                                     'float_lr_rmse': float(actual_residual[i]),
                                     'pillow_lr_rmse': float(np.sqrt(np.mean((low - lr[i].cpu().numpy()) ** 2)))})
                print(json.dumps({'phase': 'evaluate', 'sequence': case['sequence'], 'block': block['block']}), flush=True)
            summary = {'sequence': case['sequence'], 'split': case['split'], 'frames': case['frames'],
                       'protocol_sha256': sha(OUT / 'protocol.json'), 'methods': {}}
            for name in methods:
                selected = [r for r in rows if r['method'] == name]
                summary['methods'][name] = {key: float(np.mean([r[key] for r in selected]))
                                            for key in ['psnr', 'lpips', 'ssim', 'float_lr_rmse', 'pillow_lr_rmse']}
            write(folder / 'per_frame.json', rows)
            write(completed, summary)
            summaries.append(summary)
            print(json.dumps(summary), flush=True)
    write(OUT / 'summary.json', {'cases': summaries, 'protocol_sha256': sha(OUT / 'protocol.json'),
                                'script_sha256': sha(Path(__file__)), 'status': 'complete'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=['prepare', 'infer', 'evaluate'])
    parser.add_argument('--comfy', type=Path)
    parser.add_argument('--weights', type=Path)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'infer':
        infer(args)
    else:
        evaluate(args)
