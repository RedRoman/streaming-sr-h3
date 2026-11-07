#!/usr/bin/env python
"""Causal-streaming demonstration video for v4g at production shape (0086441).

v2 — fixes over v1 (viz_v4g_causal_stream.py), per GLM-V4G-CHECK-20261107:
1. Teacher weights are explicitly RESTORED from the saved original teacher
   state-dict before the teacher forward, and verified by a parameter
   fingerprint (sha256 over all parameter bytes) both at save time and after
   restore. v1 deleted teacher_sd without restoring, so the "teacher" panel
   was produced with v4g weights — invalid as a teacher reference.
2. Time axis: v1 mapped 37 latent frames to 120 pixel frames with repeat(5)
   and marked the future-noise cut at CUT_LATENT*5. The real VAE temporal
   pointwise mapping has NOT been verified. The released decoder produces
   124 frames from 37 latents, then keeps the first 120. No pixel-boundary
   line is drawn. The noised
   panel label states the cut in latent-frame terms, and the error panel is
   nearest-rescaled by the DISPLAY ratio 120/37 with an explicit
   "approximate axis" note. A future attempt may verify the mapping by
   decoding a temporally-sparse probe latent; until then no pixel cut line.
3. Attempt-isolated outputs: pass --tag, outputs go to
   runs/v4g_causal_stream_viz_<id>_<tag>/ with its own log written by the
   launcher (no rm -rf, no log overwrite).
4. Latents are cached before decoding; each of five decoded RGB signals is
   cached atomically before assembly. --from-cache verifies manifest identity,
   checksums and shapes, and writes a NEW attempt. Complete RGB caches skip
   all model loading; partial RGB caches decode only missing signals. Legacy
   caches without a manifest are rejected. VAE identity is config SHA256 plus
   resolved path and weight-file metadata, NOT a whole-model weight hash.

Panels (2x3):
Row 1: v4g STREAMING | v4g FULL | ORIGINAL teacher (bidirectional, noncausal)
Row 2: v4g NOISED-FUTURE (latent frames >= 19 noised) | |stream-full| x50 | reference

STATUS: this v2 is the corrected pipeline. Its teacher panel supersedes v1.
"""
from __future__ import annotations
import argparse, hashlib, json, os, re, shutil, sys, tempfile, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prepare_upscaler_prod_pairs as prep
from experiment.data_utils import (decode_latent, empty_cache, load_video_vae,
                                   select_device, tensor_load, write_mp4)
from experiment.train_upscaler_trial import wrapper_forward
from experiment.prepare_data import load_author_upscaler
from experiment.causal_upscaler import CausalUpscalerAdapter

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get('H3_VIZ_ROOT', PACKAGE_ROOT)).expanduser().resolve()
SAMPLE_ID = 'VIDGEN5S_0086441'
SOURCE = Path(os.environ.get('H3_VIZ_SOURCE', ROOT / 'inputs' / f'{SAMPLE_ID}.mp4')).expanduser()
BASE_MODEL = os.environ.get('H3_VIZ_BASE_MODEL', str(ROOT / 'assets/h3-base'))
UPSCALER = Path(os.environ.get('H3_VIZ_UPSCALER', ROOT / 'assets/latent_upsampler/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors')).expanduser()
CKPT = Path(os.environ.get('H3_VIZ_CHECKPOINT', ROOT / 'assets/v4g_final.pt')).expanduser()
UPSTREAM = Path(os.environ.get('H3_VIZ_UPSTREAM', PACKAGE_ROOT / 'upstream')).expanduser()
PANEL_W, PANEL_H = 960, 544
CUT_LATENT = 19
LATENT_T, DECODED_T, PX_T = 37, 124, 120
CACHE_SCHEMA = 'v4g-causal-stream-v2-reviewed-1'
SIGNALS = ('stream', 'full', 'teacher', 'noised', 'reference')
LATENT_SHAPE = (24, LATENT_T, 68, 120)
RGB_SHAPE = (PX_T, 1088, 1920, 3)

LABELS = {
    'stream': 'v4g STREAMING (past+current only)',
    'full': 'v4g FULL CONTEXT (native_forward)',
    'teacher': 'ORIGINAL teacher (bidirectional, noncausal; weights restored+verified)',
    'noised': 'v4g with FUTURE NOISED from LATENT frame 19 (no pixel line: VAE time map unverified)',
    'error': '|stream-full| x50; display-resampled 37->120 (approximate time axis)',
    'reference': 'reference (VAE of HR)',
}


def fingerprint(model_or_sd):
    """sha256 over parameter bytes — identity check for teacher restore."""
    sd = model_or_sd.state_dict() if hasattr(model_or_sd, 'state_dict') else model_or_sd
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(bytes(t.view(torch.uint8).numpy()) if t.dtype == torch.bfloat16
                 else t.numpy().tobytes())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {'path': str(path), 'size': path.stat().st_size,
            'sha256': file_sha256(path)}


def current_identity():
    """Describe exactly what was checked; do not claim to hash all VAE weights."""
    vae_dir = (Path(BASE_MODEL) / 'vae').resolve(strict=True)
    weights = sorted(p for p in vae_dir.rglob('*')
                     if p.is_file() and p.suffix in ('.safetensors', '.bin'))
    if not weights:
        raise ValueError(f'No VAE weight files found in {vae_dir}')
    return {
        'sample_id': SAMPLE_ID,
        'source': file_identity(SOURCE),
        'checkpoint': file_identity(CKPT),
        'upscaler': file_identity(UPSCALER),
        'vae': {
            'path': str(vae_dir), 'config': file_identity(vae_dir / 'config.json'),
            'weight_files_metadata': [
                {'path': str(p.resolve()), 'size': p.stat().st_size,
                 'mtime_ns': p.stat().st_mtime_ns} for p in weights],
            'verification_scope': 'config SHA256 + resolved paths + weight size/mtime; '
                                  'VAE weight contents are not hashed',
        },
        'processing': {'encode_seed': 42, 'noise_seed': 20261107,
                       'noise_std': 3.0, 'cut_latent': CUT_LATENT,
                       'draft_shape': [24, LATENT_T, 44, 80],
                       'latent_shape': list(LATENT_SHAPE), 'rgb_shape': list(RGB_SHAPE),
                       'decoded_frames_before_crop': DECODED_T,
                       'kept_frames': PX_T, 'fps': 24.0,
                       'error_axis': 'display-only nearest resampling 37->120; '
                                     'not verified VAE pointwise alignment'},
    }


def atomic_write(path, writer):
    """Commit one complete artifact; failed temporary files are never reused."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f'.{path.name}.', suffix='.tmp', dir=path.parent)
    temp = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_write(path, lambda handle: handle.write(
        json.dumps(value, indent=2, allow_nan=False).encode('utf-8')))


def copy_immutable(source, target):
    """Copy into a new attempt, using a hard link for large immutable caches."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    try:
        os.link(source, target)
    except OSError:
        def copy(handle):
            with Path(source).open('rb') as src:
                shutil.copyfileobj(src, handle, length=8 * 1024 * 1024)
        atomic_write(target, copy)


def validate_rgb(value, name):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.dtype != np.uint8 or tuple(value.shape) != RGB_SHAPE:
        raise ValueError(f'{name}: expected uint8 RGB {RGB_SHAPE}, '
                         f'got {value.dtype} {value.shape}')
    return value


def validate_blob(blob, identity):
    if blob.get('schema') != CACHE_SCHEMA or blob.get('identity') != identity:
        raise ValueError('Cache schema/input identity mismatch; refuse cache reuse')
    if blob.get('cut_latent') != CUT_LATENT:
        raise ValueError('Cache cut_latent mismatch')
    latents = blob.get('latents', {})
    if set(latents) != set(SIGNALS[:-1]):
        raise ValueError('Cache latent signals are incomplete or unexpected')
    for name, value in {**latents, 'reference': blob.get('reference')}.items():
        if (not torch.is_tensor(value) or tuple(value.shape) != LATENT_SHAPE
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            raise ValueError(f'Invalid {name} latent: expected finite float32 {LATENT_SHAPE}')
    fp = blob.get('fingerprints', {})
    if fp.get('teacher_restored_verified') is not True:
        raise ValueError('Cache has no verified original-teacher restoration')
    for key in ('teacher_sd_sha256', 'v4g_ckpt_sha256_after_load'):
        if not re.fullmatch(r'[0-9a-f]{64}', fp.get(key, '')):
            raise ValueError(f'Invalid parameter fingerprint: {key}')
    if blob.get('stats', {}).get('fingerprints') != fp:
        raise ValueError('Cache stats/fingerprints disagree')


def load_verified_cache(cache_dir, identity):
    """Fail closed before device initialization or expensive model work."""
    cache_dir = Path(cache_dir)
    manifest = json.loads((cache_dir / 'manifest.json').read_text())
    if (manifest.get('schema') != CACHE_SCHEMA or manifest.get('identity') != identity
            or manifest.get('sample_id') != SAMPLE_ID
            or manifest.get('stages', {}).get('inference') != 'complete'):
        raise ValueError('Manifest schema/input identity/stage mismatch')
    if manifest.get('latent_shapes') != {k: list(LATENT_SHAPE) for k in SIGNALS}:
        raise ValueError('Manifest latent shapes mismatch')
    if file_sha256(cache_dir / 'cache.pt') != manifest.get('cache_sha256'):
        raise ValueError('cache.pt checksum mismatch')
    blob = torch.load(cache_dir / 'cache.pt', map_location='cpu', weights_only=False)
    validate_blob(blob, identity)
    if manifest.get('fingerprints') != blob['fingerprints']:
        raise ValueError('Manifest/cache fingerprints disagree')
    decoded = manifest.get('decoded', {})
    if (not isinstance(decoded, dict) or not set(decoded) <= set(SIGNALS)
            or manifest.get('stages', {}).get('decoded') != sorted(decoded)):
        raise ValueError('Invalid decoded-stage manifest')
    for name, entry in decoded.items():
        relative = f'decoded/{name}.npy'
        if (entry.get('path') != relative or entry.get('dtype') != 'uint8'
                or entry.get('shape') != list(RGB_SHAPE)):
            raise ValueError(f'Invalid RGB manifest for {name}')
        path = cache_dir / relative
        if file_sha256(path) != entry.get('sha256'):
            raise ValueError(f'RGB cache checksum mismatch: {name}')
        validate_rgb(np.load(path, mmap_mode='r', allow_pickle=False), name)
    return blob, manifest


def save_inference_cache(blob, out_dir):
    validate_blob(blob, blob['identity'])
    atomic_write(out_dir / 'cache.pt', lambda handle: torch.save(blob, handle))
    manifest = {
        'schema': CACHE_SCHEMA, 'sample_id': SAMPLE_ID, 'identity': blob['identity'],
        'producer_script': file_identity(__file__),
        'latent_shapes': {k: list(LATENT_SHAPE) for k in SIGNALS},
        'fingerprints': blob['fingerprints'],
        'cache_sha256': file_sha256(out_dir / 'cache.pt'), 'decoded': {},
        'stages': {'inference': 'complete', 'decoded': [], 'assembly': 'pending'},
    }
    atomic_json(out_dir / 'manifest.json', manifest)
    atomic_json(out_dir / 'stats.json', blob['stats'])
    atomic_json(out_dir / 'teacher_v4g_fingerprints.json', blob['fingerprints'])


def recover_cache(source_dir, out_dir, identity):
    blob, manifest = load_verified_cache(source_dir, identity)
    copy_immutable(Path(source_dir) / 'cache.pt', out_dir / 'cache.pt')
    for entry in manifest['decoded'].values():
        copy_immutable(Path(source_dir) / entry['path'], out_dir / entry['path'])
    manifest['recovered_from'] = str(Path(source_dir).resolve())
    manifest['assembly_script'] = file_identity(__file__)
    manifest['stages']['assembly'] = 'pending'
    manifest.pop('outputs', None)
    atomic_json(out_dir / 'manifest.json', manifest)
    atomic_json(out_dir / 'stats.json', blob['stats'])
    atomic_json(out_dir / 'teacher_v4g_fingerprints.json', blob['fingerprints'])


def label_image(frame, label):
    image = Image.fromarray(frame).resize((PANEL_W, PANEL_H), Image.Resampling.LANCZOS)
    canvas = Image.new('RGB', (PANEL_W, PANEL_H + 32), '#111111')
    canvas.paste(image, (0, 32))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), label, fill='white')
    return canvas


def assemble(cache_path: Path, out_dir: Path, vae=None, device='cpu', identity=None):
    """Use verified RGB caches first; load a VAE only for missing signals."""
    identity = current_identity() if identity is None else identity
    blob, manifest = load_verified_cache(cache_path.parent, identity)
    latents, stats = blob['latents'], blob['stats']
    diff = (latents['stream'] - latents['full']).abs()

    rgb = {}
    for k in SIGNALS:
        path = out_dir / 'decoded' / f'{k}.npy'
        if k in manifest['decoded']:
            rgb[k] = np.load(path, mmap_mode='r', allow_pickle=False)
            print(json.dumps({'decoded_cache_reused': k}), flush=True)
            continue
        if vae is None:
            device = select_device(str(device))
            vae = load_video_vae(BASE_MODEL, device)
        # Reference is an encoded HR latent too: it MUST be decoded.
        src = blob['reference'] if k == 'reference' else latents[k]
        pixels = validate_rgb(decode_latent(vae, src, device, keep_frames=PX_T), k)
        atomic_write(path, lambda handle: np.save(handle, pixels, allow_pickle=False))
        manifest['decoded'][k] = {
            'path': f'decoded/{k}.npy', 'sha256': file_sha256(path),
            'dtype': 'uint8', 'shape': list(pixels.shape),
        }
        manifest['stages']['decoded'] = sorted(manifest['decoded'])
        atomic_json(out_dir / 'manifest.json', manifest)
        del pixels
        rgb[k] = np.load(path, mmap_mode='r', allow_pickle=False)
        empty_cache(device)
        print(json.dumps({'decoded': k}), flush=True)

    # error map: mean over 24 latent channels -> grayscale, x50 clip
    # DISPLAY resampling only: 37->120. The VAE decodes 124 then crops to 120;
    # the pointwise VAE time map is unverified, so no pixel boundary is claimed.
    amp = np.clip(diff.numpy() * 50.0, 0, 1).mean(axis=0)          # [37,68,120]
    idx = np.floor(np.arange(PX_T) * LATENT_T / PX_T).astype(int).clip(0, LATENT_T - 1)
    amp_px = amp[idx]                                              # [120,68,120]
    amp_rgb = np.repeat((amp_px * 255).astype(np.uint8)[..., None], 3, axis=-1)

    frames = []
    t0 = time.perf_counter()
    for i in range(PX_T):
        canvas = Image.new('RGB', (PANEL_W * 3 + 8, (PANEL_H + 32) * 2 + 8), '#111111')
        canvas.paste(label_image(rgb['stream'][i], LABELS['stream']), (0, 0))
        canvas.paste(label_image(rgb['full'][i], LABELS['full']), (PANEL_W + 4, 0))
        canvas.paste(label_image(rgb['teacher'][i], LABELS['teacher']), (2 * PANEL_W + 8, 0))
        canvas.paste(label_image(rgb['noised'][i], LABELS['noised']), (0, PANEL_H + 40))
        canvas.paste(label_image(amp_rgb[i], LABELS['error']), (PANEL_W + 4, PANEL_H + 40))
        canvas.paste(label_image(rgb['reference'][i], LABELS['reference']),
                     (2 * PANEL_W + 8, PANEL_H + 40))
        frames.append(torch.from_numpy(np.asarray(canvas).copy()))
    frames = torch.stack(frames)
    codec = write_mp4(frames, out_dir / f'causal_stream_test_{SAMPLE_ID}.mp4', fps=24., crf=16)
    Image.fromarray(frames[60].numpy()).save(out_dir / 'frame060.png')
    atomic_json(out_dir / 'stats.json', stats)
    atomic_json(out_dir / 'teacher_v4g_fingerprints.json', blob['fingerprints'])
    manifest['stages']['assembly'] = 'complete'
    manifest['outputs'] = {
        'video': file_identity(out_dir / f'causal_stream_test_{SAMPLE_ID}.mp4'),
        'preview': file_identity(out_dir / 'frame060.png'),
    }
    atomic_json(out_dir / 'manifest.json', manifest)
    print(json.dumps({'video': codec, 'seconds': round(time.perf_counter() - t0, 1),
                      'status': 'completed'}), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='npu:0')
    ap.add_argument('--tag', required=True, help='unique attempt tag; output dir gets it')
    ap.add_argument('--from-cache', type=Path, default=None,
                    help='verified cache/manifest dir; reuse RGB or decode missing signals into a NEW tag')
    args = ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', args.tag):
        ap.error('--tag must be a simple filename suffix, without slashes')
    out_dir = ROOT / f'runs/v4g_causal_stream_viz_{SAMPLE_ID}_{args.tag}'
    if out_dir.exists():
        raise FileExistsError(f'{out_dir} (use a new --tag; never overwrite old attempts)')
    identity = current_identity()
    out_dir.mkdir(parents=True, exist_ok=False)

    if args.from_cache:
        recover_cache(args.from_cache, out_dir, identity)
        assemble(out_dir / 'cache.pt', out_dir, device=args.device, identity=identity)
        return

    device = select_device(args.device)
    lr, hr = prep.read_first_120_hr(SOURCE, out_dir, SAMPLE_ID)
    vae = load_video_vae(BASE_MODEL, device)
    from data_utils import encode_pixels
    draft = encode_pixels(vae, lr, device, 42)
    reference = encode_pixels(vae, hr, device, 42)
    assert tuple(draft.shape) == (24, 37, 44, 80), tuple(draft.shape)
    print(json.dumps({'pair': 'ready'}), flush=True)

    node = load_author_upscaler(UPSTREAM, UPSCALER, device)
    model = node.load_model('upscaler', device, 'fp32')
    # FIX 1: fingerprint + save ORIGINAL teacher weights BEFORE v4g load
    teacher_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    teacher_fp = fingerprint(teacher_sd)
    model.eval().requires_grad_(False)
    draft_b = draft[None].to(device, torch.float32)
    target = (37, 68, 120)

    latents, fp = {}, {}
    with torch.no_grad():
        saved = tensor_load(CKPT)
        model.load_state_dict(saved['model_state_dict'], strict=True)
        model._aligned_past_initialized = True
        del saved
        adapter = CausalUpscalerAdapter(node, model)
        seqs = [adapter.stream_step(draft_b[:, :, i:i + 1], (68, 120)) for i in range(37)]
        latents['stream'] = torch.cat(seqs, dim=2)[0].float().cpu()
        del seqs
        state = adapter.stream_state()
        latents['full'] = adapter.native_forward(draft_b, target)[0].float().cpu()
        rng = torch.Generator(device='cpu').manual_seed(20261107)
        noise = torch.randn(draft_b.shape, generator=rng, dtype=torch.float32).to(device) * 3.0
        corrupted = torch.cat([draft_b[:, :, :CUT_LATENT], noise[:, :, CUT_LATENT:]], dim=2)
        latents['noised'] = adapter.native_forward(corrupted, target)[0].float().cpu()
        v4g_fp = fingerprint(model)
        adapter.remove()
        # FIX 1 (cont.): restore original teacher weights, VERIFY fingerprint
        model.load_state_dict(teacher_sd, strict=True)
        restored_fp = fingerprint(model)
        assert restored_fp == teacher_fp, \
            f'teacher restore mismatch: {restored_fp} != {teacher_fp}'
        latents['teacher'] = wrapper_forward(node, model, draft_b, target)[0].float().cpu()
    fp = {'teacher_sd_sha256': teacher_fp, 'teacher_restored_verified': True,
          'v4g_ckpt_sha256_after_load': v4g_fp}
    print(json.dumps(fp), flush=True)
    del adapter, node, model, teacher_sd, draft_b, noise, corrupted
    empty_cache(device)

    diff = (latents['stream'] - latents['full']).abs()
    stats = {'stream_vs_full_max_abs': float(diff.max()),
             'stream_vs_full_mean_abs': float(diff.mean()),
             'noised_vs_full_past_max_abs': float(
                 (latents['noised'][:, :CUT_LATENT] - latents['full'][:, :CUT_LATENT]).abs().max()),
             'stream_state': state, 'fingerprints': fp,
             'caveats': [
                 'teacher panel now uses RESTORED original teacher weights (v1 did not)',
                 '37 latent frames decode to 124 frames, then first 120 are kept',
                 'no pixel cut line: pointwise VAE temporal mapping unverified',
                 'error panel time axis approximate (display resampling only, 37->120)']}
    print(json.dumps({k: v for k, v in stats.items() if k != 'stream_state'}), flush=True)

    # FIX 4: cache BEFORE decode/assembly so a crash there is cheap to recover
    blob = {'schema': CACHE_SCHEMA, 'identity': identity,
            'latents': latents, 'stats': stats, 'reference': reference,
            'fingerprints': fp, 'cut_latent': CUT_LATENT}
    save_inference_cache(blob, out_dir)
    assemble(out_dir / 'cache.pt', out_dir, vae, device, identity)


if __name__ == '__main__':
    main()
