"""Frozen MiniMax-H3 (ComfyUI implementation) on Ascend NPU: guide encode, 49-step sampling, sampling-inner DC, decode.

Ported from the colleague's H100 scripts (experiments/h3_backbone/sample_inner_dc*.py in
RedRoman/streaming-sr-h3). Only device calls change: CUDA -> NPU. Same weights, ComfyUI
helpers, sigma grid, seed, guide preprocessing, correction indices and pixel projection.
"""
import ast
import hashlib
import json
import math
import sys
import time
import types
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

ROOT = Path('/cache/zihaohe/streaming_sr0')
COMFY = ROOT / 'code/ComfyUI'
ASSETS = ROOT / 'assets/comfy_h3'
DIT_PATHS = {'fl2va': ASSETS / 'minimax_h3_fl2va_pruned_bf16.safetensors', 'ref2va': ASSETS / 'minimax_h3_ref2va_pruned_bf16.safetensors'}
DIT_PATH = DIT_PATHS['fl2va']
VAE_PATH = ASSETS / 'minimax_h3_video_vae_fp16.safetensors'
CONTEXT_PATH = ASSETS / 'h3_qwen_context_neutral_restore.pt'
SHA = {'dit': 'a32572fb90b5508b201ec7c2eddcc184b13ddfd3c6f6d2cf06a0b46535d541b4',
       'ref2va': '37c0da793e20ca735272ec2be655f08a2e10f97a3ec8fdfb40f5b39a736ed6fe',
       'vae': '7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522',
       'context': 'fb03abcc1eb569d2fcd360fe512f2ae0d707298f4a7ad8b6a35b609e1d97d5d2'}
SIZE = {'dit': 40225724176, 'vae': 5207808496}
POINTS = [28, 45, 48]
SEED = 101
GRID = torch.linspace(1, 0, 50)
SIGMAS = 12 * GRID / (1 + 11 * GRID)

sys.path.insert(0, str(COMFY))
sys.path.insert(0, str(ROOT / 'code'))


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(64 << 20):
            h.update(b)
    return h.hexdigest()


def sync():
    torch.npu.synchronize()


def patch_module_to():
    """torch_npu replaces nn.Module.to with a version that calls self.cast_weight(device); ComfyUI ops define an
    unrelated cast_weight(input), so the call crashes. With ND matmul format and jit_compile=False, torch_npu's
    cast_weight performs no format cast at all, so dropping the call is behaviour-preserving (asserted here)."""
    assert torch.npu.get_mm_bmm_format_nd() and torch.npu.is_jit_compile_false()

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if dtype is not None and not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError(f'nn.Module.to only accepts floating point or complex dtypes, but got desired dtype={dtype}')

        def convert(t):
            if convert_to_format is not None and t.dim() == 4:
                return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking, memory_format=convert_to_format)
            return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
        return self._apply(convert)
    torch.nn.Module.to = to
    return {'torch_npu_module_to_cast_weight_dropped': True, 'mm_bmm_format_nd': True, 'jit_compile_false': True}


def set_precision():
    """Disable HF32 (the NPU analogue of TF32); return the initial flags for the record."""
    initial = {'matmul_allow_hf32': bool(torch.npu.matmul.allow_hf32), 'conv_allow_hf32': bool(torch.npu.conv.allow_hf32)}
    torch.npu.matmul.allow_hf32 = False
    torch.npu.conv.allow_hf32 = False
    return initial


@contextmanager
def native_training_math(functional_residuals=True):
    """Same scoped eager math the colleague used for sampling on H100 (training_math.py)."""
    import comfy.model_management as mm
    import comfy.quant_ops
    import comfy.ldm.minimax.model as implementation
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half
    saved = (mm.in_training, comfy.quant_ops.ck.rms_rope_split_half, implementation._mod_scale_shift, implementation._mod_gate)

    def scale_shift(h, shift, scale, segments):
        return torch.cat([h[a:b] * (1.0 + implementation._mod_row(scale, row, h.dtype)) + implementation._mod_row(shift, row, h.dtype)
                          for a, b, row in segments], dim=0)

    def gate(x, gates, other, segments):
        return torch.cat([torch.addcmul(x[a:b], other[a:b], implementation._mod_row(gates, row, x.dtype)) for a, b, row in segments], dim=0)
    try:
        mm.in_training = True
        comfy.quant_ops.ck.rms_rope_split_half = rms_rope_split_half
        if functional_residuals:
            implementation._mod_scale_shift = scale_shift
            implementation._mod_gate = gate
        yield
    finally:
        mm.in_training, comfy.quant_ops.ck.rms_rope_split_half, implementation._mod_scale_shift, implementation._mod_gate = saved


def official_resize():
    """The node file's _resize (lanczos via comfy.utils.common_upscale), extracted without importing the node module."""
    import comfy.utils
    node_src = COMFY / 'comfy_extras/nodes_minimax_h3.py'
    node = next(n for n in ast.parse(node_src.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == '_resize')
    ns = {'comfy': types.SimpleNamespace(utils=comfy.utils)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(node_src), 'exec'), ns)
    return ns['_resize']


def ref_resize(img_hw3, hw, resize):
    """Official MiniMaxH3ReferenceToVideo 'match' sizing: down only, keep aspect, generation pixel area, multiples of 32.
    img_hw3: [H,W,3] float in [0,1]; hw: generation (height, width). Returns [1,th,tw,3]."""
    h, w = img_hw3.shape[0], img_hw3.shape[1]
    scale = min(1.0, math.sqrt((hw[1] * hw[0]) / (w * h)))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    return resize(img_hw3[None].float(), tw, th, 'disabled')


class H3:
    def __init__(self, load_dit=True, verify_sha=False, dit='fl2va', lora_path=None, lora_rank=32):
        self.precision_initial = set_precision()
        module_to_patch = patch_module_to()
        from comfy.cli_args import args as ca
        ca.reserve_vram = 8.
        import comfy.sd
        import comfy.utils
        self.comfy_sd, self.cu = comfy.sd, comfy.utils
        self.dev = torch.device('npu', 0)
        assert VAE_PATH.stat().st_size == SIZE['vae']
        if verify_sha:
            assert sha(VAE_PATH) == SHA['vae']
        assert sha(CONTEXT_PATH) == SHA['context']
        self.record = {'precision_initial': self.precision_initial, 'precision_used': {'matmul_hf32': False, 'conv_hf32': False},
                       'torch': torch.__version__, 'torch_npu': torch_npu.__version__, **module_to_patch,
                       'comfy_commit': (COMFY / 'COMMIT').read_text().strip()}
        t = time.perf_counter()
        sd, meta = comfy.utils.load_torch_file(str(VAE_PATH), return_metadata=True)
        self.vae = comfy.sd.VAE(sd=sd, device=self.dev, dtype=torch.float16, metadata=meta)
        del sd
        self.vae.first_stage_model.eval().requires_grad_(False)
        self.record['vae_load_seconds'] = time.perf_counter() - t
        self.context = torch.load(CONTEXT_PATH, map_location='cpu', weights_only=True)
        self.patcher = None
        if load_dit:
            dit_path = DIT_PATHS[dit]
            assert dit_path.stat().st_size == SIZE['dit']
            if verify_sha:
                assert sha(dit_path) == SHA['dit' if dit == 'fl2va' else dit]
            self.record['dit'] = dit
            t = time.perf_counter()
            self.patcher = comfy.sd.load_diffusion_model(str(dit_path), model_options={'dtype': torch.bfloat16})
            model = self.patcher.model.diffusion_model.eval().requires_grad_(False)
            assert len(model.blocks) == 50 and self.patcher.model.latent_format.scale_factor == 1
            if lora_path is not None:
                import lora as L
                from safetensors.torch import load_file
                L.inject(model, lora_rank)
                n_loaded = L.load(model, load_file(str(lora_path)))
                self.record['lora'] = {'path': str(lora_path), 'rank': lora_rank, 'tensors_loaded': n_loaded,
                                       'sha256': sha(lora_path)}
            self.frozen = [(n, v, v._version) for n, v in model.named_parameters() if not (n.endswith('.lora_a') or n.endswith('.lora_b'))]
            self.record['dit_load_seconds'] = time.perf_counter() - t
        self._resize = official_resize()
        self.record['guide_helper_sha256'] = sha(COMFY / 'comfy_extras/nodes_minimax_h3.py')

    def freeze_audit(self):
        assert all(v._version == ver and v.grad is None and not v.requires_grad for _, v, ver in self.frozen)
        return {'base_parameter_versions_unchanged': True, 'frozen_base_tensors': len(self.frozen)}

    @torch.no_grad()
    def encode_guide(self, low, hw):
        """low: [T,3,h,w] float in [0,1] (observed LR only). Official AddGuide-style preprocessing."""
        up = torch.nn.functional.interpolate(low, size=hw, mode='bicubic', align_corners=False, antialias=True).clamp(0, 1)
        pixels = self._resize(up.permute(0, 2, 3, 1), hw[1], hw[0], 'center')
        return self.vae.encode(pixels).cpu().contiguous()

    @torch.no_grad()
    def encode_guide_pixels(self, pixels_thw3, hw):
        """Guide from already-built HR-size pixels [T,H,W,3] in [0,1] (e.g. U(LR) plus written memory detail);
        same official _resize and VAE encode as encode_guide."""
        pixels = self._resize(pixels_thw3.float(), hw[1], hw[0], 'center')
        return self.vae.encode(pixels).cpu().contiguous()

    @torch.no_grad()
    def decode(self, latent):
        px = self.vae.decode(latent).float().cpu()
        return px[0] if px.ndim == 5 else px  # [T,H,W,3]

    @torch.no_grad()
    def encode(self, rgb_thw3):
        return self.vae.encode(rgb_thw3.contiguous().clone())

    @torch.no_grad()
    def ref_block(self, resized):
        """DiT ref block for one already-resized reference image [1,th,tw,3] (see ref_resize)."""
        th, tw = resized.shape[1], resized.shape[2]
        z = self.vae.encode(resized.float()).cpu().contiguous()
        return {'kind': 'image', 'latent_h': th // 16, 'latent_w': tw // 16, 'latent': z}

    def sample(self, guide, mode, low=None, hw=None, log=None, seed=SEED, context=None, refs=None):
        """mode: 'base' (unadapted 49-step with LR guide) or 'innerdc' (post-CFG z0 += E(P25(D z0)) - E(D z0) at POINTS)."""
        import comfy.nested_tensor
        import comfy.sample
        from pixel_consistency import BicubicConsistency
        assert mode in ('base', 'innerdc')
        T = guide.shape[2]
        h16, w16 = guide.shape[3], guide.shape[4]
        frames = (T - 2) // 5 * 17 + 5
        audio_t = round(frames / 24 * 40)
        ctx = context if context is not None else self.context
        extra = {'minimax_token_tags': ctx['text_token_tags'], 'minimax_keyframes': [{'resolved_frame_index': 0, 'latent': guide}]}
        if refs:
            extra['minimax_refs'] = refs
        positive = [[ctx['prompt_embeds'][None], extra]]
        latent = comfy.nested_tensor.NestedTensor((torch.zeros(1, 24, T, h16, w16), torch.zeros(1, 32, 2, audio_t)))
        _, shapes = self.cu.pack_latents(list(latent.unbind()))
        events, state = [], {'calls': 0, 'hooks': 0}

        def observer(module, args, kwargs, output):
            if state['calls'] in [0, *POINTS]:
                actual = kwargs['minimax_payload']['cond_video_latents'][0].detach().cpu()
                assert float((actual - guide).abs().max()) == 0
            state['calls'] += 1
        handle = self.patcher.model.diffusion_model.register_forward_hook(observer, with_kwargs=True)
        current = self.patcher.clone()
        # Same Dykstra math as the colleague's CPU float64 run, on NPU in float32: on a 22x480x832 block the
        # max abs difference is 3.6e-7 (results/proj_precision.json), 27x faster.
        op = BicubicConsistency(tuple(hw), (hw[0] // 4, hw[1] // 4), dtype=torch.float32, device=self.dev) if mode == 'innerdc' else None
        low_dev = low.to(self.dev) if mode == 'innerdc' else None

        def hook(args):
            index = state['hooks']
            state['hooks'] += 1
            assert state['calls'] == index + 1 and abs(float(args['sigma'].flatten()[0]) - float(SIGMAS[index])) < 1e-6
            if index not in POINTS:
                return args['denoised']
            begin = time.perf_counter()
            z0, audio = self.cu.unpack_latents(args['denoised'], shapes)
            rgb = self.decode(z0)
            assert rgb.shape == (frames, hw[0], hw[1], 3) and torch.isfinite(rgb).all()
            zbase = self.encode(rgb)
            projected, residual = op.project(rgb.permute(0, 3, 1, 2).to(self.dev), low_dev, iterations=25)
            zc = self.encode(projected.permute(0, 2, 3, 1).cpu())
            delta = (zc - zbase).to(device=z0.device, dtype=z0.dtype)
            assert delta.shape == z0.shape and torch.isfinite(delta).all()
            packed, _ = self.cu.pack_latents([z0 + delta, audio])
            ev = {'index': index, 'sigma': float(args['sigma'].flatten()[0]), 'delta_rms': float(delta.float().square().mean().sqrt()),
                  'projected_lr_rmse_max': float(residual.max()), 'hook_seconds': time.perf_counter() - begin}
            events.append(ev)
            if log:
                log(ev)
            return packed
        if mode == 'innerdc':
            current.set_model_sampler_post_cfg_function(hook)
        t = time.perf_counter()
        try:
            with torch.no_grad(), native_training_math():
                samples = comfy.sample.sample(current, comfy.sample.prepare_noise(latent, seed), 49, 1., 'euler', 'simple', positive, [], latent,
                                              seed=seed, disable_pbar=True, sigmas=SIGMAS)
            sync()
        finally:
            handle.remove()
        video, audio = samples.unbind()
        assert state['calls'] == 49 and torch.isfinite(video).all() and torch.isfinite(audio).all()
        if mode == 'innerdc':
            assert state['hooks'] == 49 and [e['index'] for e in events] == POINTS
        return {'video': video.cpu(), 'audio': audio.cpu()}, {'mode': mode, 'seed': seed, 'steps': 49, 'cfg': 1.,
                                                              'seconds': time.perf_counter() - t, 'correction_events': events, 'projection_device_dtype': 'npu/float32' if mode == 'innerdc' else None,
                                                              **self.freeze_audit()}


def to_u8(px):
    return np.uint8(px.numpy().clip(0, 1) * 255 + .5)
