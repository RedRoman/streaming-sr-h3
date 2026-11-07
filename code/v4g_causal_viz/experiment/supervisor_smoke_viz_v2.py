"""CPU-only structural smoke test. Decoder and video writer are intentionally fake."""
import ast
import json
import sys
import tempfile
from pathlib import Path

source = Path(sys.argv[1])
tree = ast.parse(source.read_text())
kept = []
for node in tree.body:
    if isinstance(node, ast.Import):
        if all(a.name != 'prepare_upscaler_prod_pairs' for a in node.names):
            kept.append(node)
    elif isinstance(node, ast.ImportFrom):
        if not (node.module or '').startswith('experiment.'):
            kept.append(node)
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef)):
        kept.append(node)
ns = {'__file__': str(source), '__name__': 'cpu_smoke_only'}
exec(compile(ast.Module(body=kept, type_ignores=[]), str(source), 'exec'), ns)
torch, np = ns['torch'], ns['np']
torch.set_num_threads(1)
ns.update(LATENT_SHAPE=(24, 37, 2, 3), RGB_SHAPE=(120, 4, 6, 3), PANEL_W=6, PANEL_H=4)
identity = {'fixture': 'CPU synthetic smoke; not a model evaluation'}
calls = []
def fake_decode(vae, latent, device, keep_frames):
    assert device == 'cpu' and keep_frames == 120
    value = int(latent.flatten()[0])
    calls.append(value)
    return np.full(ns['RGB_SHAPE'], value * 20, dtype=np.uint8)
def fake_writer(frames, path, **kwargs):
    assert tuple(frames.shape) == (120, 80, 26, 3)
    assert frames.dtype == torch.uint8
    Path(path).write_bytes(b'SYNTHETIC_SMOKE_NOT_A_VIDEO')
    return 'synthetic_smoke_writer'
def forbid(*args, **kwargs):
    raise AssertionError('Unexpected model/device/decoder call during cached assembly')
ns.update(decode_latent=fake_decode, write_mp4=fake_writer,
          empty_cache=lambda device: None, load_video_vae=forbid, select_device=forbid)
fp = {'teacher_sd_sha256': 'a' * 64, 'teacher_restored_verified': True,
      'v4g_ckpt_sha256_after_load': 'b' * 64}
with tempfile.TemporaryDirectory(prefix='glm-v4g-cpu-smoke-') as tmp:
    first = Path(tmp) / 'first'
    first.mkdir()
    blob = {'schema': ns['CACHE_SCHEMA'], 'identity': identity,
            'latents': {k: torch.full(ns['LATENT_SHAPE'], float(i), dtype=torch.float32)
                        for i, k in enumerate(ns['SIGNALS'][:-1])},
            'reference': torch.full(ns['LATENT_SHAPE'], 4.0, dtype=torch.float32),
            'stats': {'fingerprints': fp}, 'fingerprints': fp, 'cut_latent': ns['CUT_LATENT']}
    ns['save_inference_cache'](blob, first)
    ns['assemble'](first / 'cache.pt', first, vae=object(), device='cpu', identity=identity)
    assert calls == [0, 1, 2, 3, 4], calls
    manifest = json.loads((first / 'manifest.json').read_text())
    assert len(manifest['decoded']) == 5 and manifest['stages']['assembly'] == 'complete'
    before = (first / 'manifest.json').read_bytes()
    second = Path(tmp) / 'recovered'
    second.mkdir()
    ns['recover_cache'](first, second, identity)
    ns['decode_latent'] = forbid
    ns['assemble'](second / 'cache.pt', second, vae=None, device='npu:0', identity=identity)
    assert (first / 'manifest.json').read_bytes() == before
    try:
        ns['load_verified_cache'](first, {'fixture': 'wrong source'})
    except ValueError:
        pass
    else:
        raise AssertionError('Wrong input identity was accepted')
    try:
        ns['validate_rgb'](np.zeros((24, 37, 2, 3), dtype=np.float32), 'reference')
    except ValueError:
        pass
    else:
        raise AssertionError('Raw reference latent was accepted as RGB')
    target = second / 'decoded' / 'reference.npy'
    with target.open('r+b') as handle:
        handle.seek(-1, 2)
        value = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([value[0] ^ 1]))
    try:
        ns['load_verified_cache'](second, identity)
    except ValueError as exc:
        assert 'checksum' in str(exc)
    else:
        raise AssertionError('Corrupt RGB cache was accepted')
print(json.dumps({'cpu_smoke': 'passed', 'fake_decoder_calls': calls,
                  'cached_redraw_without_device_or_decoder': True,
                  'rejects_wrong_identity_raw_latent_and_corrupt_cache': True,
                  'real_model_or_NPU_execution': False}))
